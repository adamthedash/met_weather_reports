#!/usr/bin/env python3
"""Download every subject image of a public Zooniverse project.

Usage:

    pixi run python scripts/scrape_zooniverse.py --project owner/name --out data/raw

For every subject of the given public Zooniverse (Panoptes) project the image
files are downloaded to ``<out>/<subject_id>.<ext>`` -- flat, one page per file,
named by the Zooniverse subject id, the layout and unique stems expected by
``data/raw`` (see ``scripts/preprocess_pages.py``). A subject with several
images yields ``<subject_id>.<ext>``, ``<subject_id>_2.<ext>``, ... in location
order. Subject-set and workflow membership is not encoded in the paths (a
subject belongs to one or more subject sets and to no workflow); it is recorded
in the metadata instead.

Per-subject provenance goes to ``<out>/metadata/<subject_id>.json``, mirroring
the image tree, plus two run-level files:

- ``project.json`` -- the resolved project resource
- ``subject_sets.json`` -- the project's subject set resources by id

Runs are resumable: a subject whose metadata file and images already exist is
skipped, existing files are never modified, each image is downloaded to a
``.part`` file and moved into place, and transient failures are retried with
backoff. Subjects that keep failing are listed in the run summary and make the
run exit non-zero; the run itself continues past them. Project resolution and
listing failures abort the run instead; anything already downloaded stays on
disk.

Run activity is logged to stderr and appended to ``<out>/scrape.log``: every
API call and image download attempt (with retries), per-subject decisions and
failures, listing totals and a progress line every 100 subjects. ``--log-level``
narrows the verbosity (``DEBUG``, the default, shows everything); the run
summary on stdout is not affected.

The pipeline only reads the project anonymously and writes only under ``--out``;
nothing is written into ``datasets/``. Note that ``scripts/preprocess_pages.py``
fails on duplicate stems: re-scraping subjects that already sit in ``data/raw``
under their old storage names (e.g. ``210736fe-....jpeg``) duplicates those
pages as distinct files -- scrape into a clean folder or prune the old copies
first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, NamedTuple
from urllib.parse import urlparse

import requests

API_BASE = "https://panoptes.zooniverse.org/api"
API_HEADERS = {"Accept": "application/vnd.api+json; version=1"}
# Matches IMAGE_SUFFIXES in scripts/preprocess_pages.py: only these extensions
# are picked up by the page scan downstream.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MIME_EXTS = {
    "image/jpeg": ".jpeg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
PAGE_SIZE = 100
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2.0, 4.0, 8.0)
REQUEST_TIMEOUT = 30.0
CHUNK_SIZE = 1 << 16
PROGRESS_EVERY = 100
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"
PROGRESS_FORMAT = "progress: %d subjects / %d downloaded (%d files) / %d skipped / %d failed"

logger = logging.getLogger(__name__)


class Location(NamedTuple):
    """One media location of a subject: position, mime type, URL and saved extension."""

    index: int
    mime: str
    url: str
    ext: str


class SubjectOutcome(NamedTuple):
    """Result of processing one subject: new files written and whether it was already complete."""

    n_new_files: int
    skipped: bool


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str, log_file: Path) -> None:
    """Configure run logging: stderr plus an appended ``log_file``.

    Both handlers receive identical records with UTC timestamps; the log file
    is appended to, so resume runs extend the same trail. The ``urllib3`` and
    ``requests`` loggers are quieted to WARNING -- every request is already
    logged by ``request_with_retries``.

    Args:
        level_name: Logging level name (``DEBUG``..``CRITICAL``). ``DEBUG``,
            the ``--log-level`` default, logs everything including every API
            call.
        log_file: Log file to append to; the parent directory is created as
            needed.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(level_name)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    for handler in (stderr_handler, file_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# HTTP with retries


def retry_after_seconds(response: requests.Response) -> float | None:
    """Parse the ``Retry-After`` header of a response as a delay in seconds.

    Args:
        response: The response whose ``Retry-After`` header to parse.

    Returns:
        The delay in seconds, or ``None`` when the header is absent or is not
        a plain number of seconds (e.g. an HTTP date).
    """
    raw = response.headers.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    stream: bool = False,
) -> requests.Response:
    """GET one URL, retrying transient failures with exponential backoff.

    Connection errors, HTTP 5xx and HTTP 429 are retried up to ``MAX_ATTEMPTS``
    times on the ``BACKOFF_SECONDS`` schedule; a 429 ``Retry-After`` header
    overrides the scheduled delay. Other 4xx responses fail immediately. Every
    request attempt is logged at DEBUG (URL, params, status or error, elapsed
    time) and every retry at WARNING.

    Args:
        session: Session carrying the API headers.
        url: Absolute URL to GET.
        params: Optional query parameters.
        stream: Stream the response body instead of loading it eagerly.

    Returns:
        The successful (2xx) response.

    Raises:
        requests.RequestException: If the request keeps failing after the last
            attempt, or fails with a non-transient HTTP status.
    """
    for attempt in range(MAX_ATTEMPTS):
        query = f" {params}" if params else ""
        started = time.monotonic()
        try:
            response = session.get(url, params=params, stream=stream, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.debug("GET %s%s -> %s in %.2fs", url, query, exc, time.monotonic() - started)
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = BACKOFF_SECONDS[attempt]
            logger.warning(
                "GET %s failed (attempt %d/%d): %s; retrying in %.1fs",
                url,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
            continue
        logger.debug(
            "GET %s%s -> %s in %.2fs",
            url,
            query,
            response.status_code,
            time.monotonic() - started,
        )
        status = response.status_code
        if status == 429 or status >= 500:
            delay = retry_after_seconds(response) or BACKOFF_SECONDS[attempt]
            if attempt == MAX_ATTEMPTS - 1:
                response.raise_for_status()
            response.close()
            logger.warning(
                "GET %s failed (attempt %d/%d): HTTP %d; retrying in %.1fs",
                url,
                attempt + 1,
                MAX_ATTEMPTS,
                status,
                delay,
            )
            time.sleep(delay)
            continue
        if not response.ok:
            response.raise_for_status()
        return response
    raise RuntimeError("request_with_retries exhausted its attempts")


# ---------------------------------------------------------------------------
# Panoptes API


def api_get(
    session: requests.Session, path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """GET one Panoptes API page and return its decoded JSON body.

    Args:
        session: Session carrying the API headers.
        path: API path below ``API_BASE`` (e.g. ``"/subjects"``).
        params: Optional query parameters.

    Returns:
        The decoded JSON body as a dict.

    Raises:
        ValueError: If the body is not a JSON object.
        requests.RequestException: If the request fails after retries.
    """
    response = request_with_retries(session, f"{API_BASE}{path}", params=params)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"API response for {path} is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected API response shape for {path}: {type(payload).__name__}")
    return payload


def pagination_meta(payload: dict[str, Any], resource_key: str, path: str) -> dict[str, Any]:
    """Return the pagination block of one listing page's ``meta``.

    Panoptes pages list responses via a ``meta`` block, either directly or
    nested under the resource key; both documented shapes are accepted. The
    block carries the page numbers plus ``count``/``page_count``.

    Args:
        payload: Decoded body of one listing page.
        resource_key: Top-level JSON key holding the resource array.
        path: API path of the listing, for error messages.

    Returns:
        The pagination block as a dict.

    Raises:
        ValueError: If the page has no usable pagination block.
    """
    meta = payload.get("meta")
    block = meta.get(resource_key) if isinstance(meta, dict) else None
    if not isinstance(block, dict):
        block = meta if isinstance(meta, dict) else None
    if not isinstance(block, dict):
        raise ValueError(f"API response for {path} has no pagination meta")
    return block


def next_page(payload: dict[str, Any], resource_key: str, path: str) -> int | None:
    """Return the next page number of a listing, or ``None`` on the last page.

    Args:
        payload: Decoded body of one listing page.
        resource_key: Top-level JSON key holding the resource array.
        path: API path of the listing, for error messages.

    Returns:
        The next page number, or ``None`` when the listing is exhausted.

    Raises:
        ValueError: If the page has no usable pagination block or its
            ``next_page`` is not a number.
    """
    raw = pagination_meta(payload, resource_key, path).get("next_page")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"next_page of {path} is not a page number: {raw!r}") from exc


def iter_resources(
    session: requests.Session,
    path: str,
    resource_key: str,
    params: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield every resource of a paginated Panoptes listing in API order.

    Args:
        session: Session carrying the API headers.
        path: API path below ``API_BASE`` (e.g. ``"/subjects"``).
        resource_key: Top-level JSON key holding the resource array
            (``"projects"``, ``"subjects"``, ``"subject_sets"``).
        params: Optional query parameters; ``page``/``page_size`` are added.

    Yields:
        Each resource of the listing as a dict. The listing's ``count`` and
        ``page_count`` are logged at INFO after the first page.

    Raises:
        ValueError: If a page has no ``<resource_key>`` array, holds a
            non-object element, or has no pagination block.
        requests.RequestException: If a page request fails after retries.
    """
    page: int | None = 1
    while page is not None:
        payload = api_get(
            session, path, {**(params or {}), "page": page, "page_size": PAGE_SIZE}
        )
        if page == 1:
            block = pagination_meta(payload, resource_key, path)
            logger.info("%s: %s resources in %s pages", path, block.get("count"), block.get("page_count"))
        resources = payload.get(resource_key)
        if not isinstance(resources, list):
            raise ValueError(f"API response for {path} has no {resource_key!r} array")
        for i, resource in enumerate(resources):
            if not isinstance(resource, dict):
                raise ValueError(f"{resource_key}[{i}] of {path} is not an object")
            yield resource
        following = next_page(payload, resource_key, path)
        if following == page:
            raise ValueError(f"pagination of {path} did not advance past page {page}")
        page = following


def resource_id(resource: dict[str, Any], what: str) -> str:
    """Return a Panoptes resource's ``id`` as a string.

    Args:
        resource: The resource dict as returned by the API.
        what: Resource kind for error messages (e.g. ``"subject"``).

    Returns:
        The resource id as a string.

    Raises:
        ValueError: If the resource has no scalar ``id``.
    """
    raw = resource.get("id")
    if raw is None or isinstance(raw, (dict, list)):
        raise ValueError(f"{what} has no id")
    return str(raw)


def resolve_project(session: requests.Session, project: str) -> dict[str, Any]:
    """Resolve ``--project`` (numeric id or ``owner/name`` slug) to its project resource.

    Args:
        session: Session carrying the API headers.
        project: Numeric project id, or an ``owner/name`` project slug.

    Returns:
        The project resource as returned by the API. The resolved id, display
        name and slug are logged at INFO.

    Raises:
        ValueError: If ``project`` is neither an id nor a slug, the project
            does not exist or is not publicly readable, or the response is
            malformed.
        requests.RequestException: If the API keeps failing after retries.
    """
    if project.isdecimal():
        payload = api_get(session, f"/projects/{project}")
        projects = payload.get("projects")
        if not isinstance(projects, list) or len(projects) != 1:
            raise ValueError(f"project id {project} not found")
        resolved = projects[0]
    else:
        if project.count("/") != 1 or not all(project.split("/")):
            raise ValueError(
                f"--project must be a project id or an owner/name slug, got {project!r}"
            )
        # Filter by slug server-side and match client-side. An ?owner= lookup is not
        # reliable: the login behind a slug's owner segment may differ from it (e.g.
        # the slug met-rhonda/... belongs to the login met_rhonda).
        for candidate in iter_resources(session, "/projects", "projects", {"slug": project}):
            if candidate.get("slug") == project:
                resolved = candidate
                break
        else:
            raise ValueError(f"no public project with slug {project!r}")
    logger.info(
        "project %s (%s, slug %s) resolved",
        resource_id(resolved, "project"),
        resolved.get("display_name"),
        resolved.get("slug"),
    )
    return resolved


# ---------------------------------------------------------------------------
# subject naming


def pick_extension(url: str, mime: str) -> str:
    """Choose the saved file extension for one subject media location.

    The URL path suffix wins when it is a known image extension (this keeps the
    extension the object is stored with); otherwise the location's mime type is
    mapped.

    Args:
        url: The location's media URL (query strings are ignored).
        mime: The location's mime type key (e.g. ``"image/jpeg"``).

    Returns:
        The lower-case file extension, including the leading dot.

    Raises:
        ValueError: If neither the URL nor the mime type yields a known image
            extension.
    """
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    ext = MIME_EXTS.get(mime.lower())
    if ext is not None:
        return ext
    raise ValueError(f"no known image extension for {mime!r}: {url}")


def subject_files(subject: dict[str, Any]) -> list[Location]:
    """Extract the media locations of one subject in location order.

    Args:
        subject: The subject resource as returned by the API.

    Returns:
        One ``Location`` per element of the subject's ``locations`` array,
        each with its saved extension resolved via ``pick_extension``.

    Raises:
        ValueError: If the subject has no or a malformed ``locations`` array,
            or a location has no known image extension.
    """
    locations = subject.get("locations")
    if not isinstance(locations, list) or not locations:
        raise ValueError("subject has no locations")
    result: list[Location] = []
    for index, entry in enumerate(locations):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"location {index} is not a single {{mime: url}} object: {entry!r}")
        (mime, url), = entry.items()
        if not isinstance(mime, str) or not isinstance(url, str) or not url:
            raise ValueError(f"location {index} is not a {{mime: url}} string pair: {entry!r}")
        result.append(Location(index=index, mime=mime, url=url, ext=pick_extension(url, mime)))
    return result


def saved_filename(subject_id: str, location: Location) -> str:
    """Build the output filename of one subject media location.

    Args:
        subject_id: The subject's id as a string.
        location: The media location to name.

    Returns:
        ``<subject_id>.<ext>`` for the first location and
        ``<subject_id>_<n>.<ext>`` for the n-th further one (1-based), so a
        multi-image subject keeps unique stems.
    """
    stem = subject_id if location.index == 0 else f"{subject_id}_{location.index + 1}"
    return f"{stem}{location.ext}"


# ---------------------------------------------------------------------------
# output writing


def timestamp() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` to ``path`` as pretty-printed JSON with a trailing newline.

    Args:
        payload: JSON-serialisable object to write.
        path: Destination file; the parent directory is created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def metadata_path(out_dir: Path, subject_id: str) -> Path:
    """Return the metadata file path of one subject under ``out_dir``.

    Args:
        out_dir: Output root holding the image tree and ``metadata/``.
        subject_id: The subject's id as a string.

    Returns:
        ``<out_dir>/metadata/<subject_id>.json``, mirroring the flat image tree.
    """
    return out_dir / "metadata" / f"{subject_id}.json"


def is_nonempty(path: Path) -> bool:
    """Whether ``path`` is a file holding at least one byte.

    Args:
        path: The file to check.

    Returns:
        True when the file exists and its size is non-zero.
    """
    return path.is_file() and path.stat().st_size > 0


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    """Download one media URL to ``dest`` atomically.

    The body streams to ``dest`` with a ``.part`` suffix and is moved into
    place with ``os.replace``, so an interrupted run never leaves a truncated
    image under its final name; the temporary file is removed after every
    attempt. Transient failures of the whole request -- connection, status and
    body streaming -- are retried up to ``MAX_ATTEMPTS`` times on the
    ``BACKOFF_SECONDS`` schedule; non-transient HTTP errors and invalid bodies
    are not. Every download is logged at DEBUG and every retry at WARNING.

    Args:
        session: Session carrying the API headers.
        url: The media URL to download.
        dest: Destination file; the parent directory is created as needed.

    Raises:
        ValueError: If the server answers with an HTML body or an empty body.
        requests.RequestException: If the download keeps failing after retries.
        OSError: If the file cannot be written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = request_with_retries(session, url, stream=True)
            try:
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if content_type == "text/html":
                    raise ValueError(f"server returned HTML instead of an image: {url}")
                with open(tmp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        fh.write(chunk)
            finally:
                response.close()
            size = tmp.stat().st_size
            if size == 0:
                raise ValueError(f"empty response body: {url}")
            os.replace(tmp, dest)
            logger.debug("downloaded %s -> %s (%d bytes)", url, dest.name, size)
            return
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status != 429 and status < 500:
                raise  # non-transient 4xx: not worth retrying
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = BACKOFF_SECONDS[attempt]
            logger.warning(
                "download of %s interrupted (attempt %d/%d): %s; retrying in %.1fs",
                url,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# pipeline


def process_subject(
    session: requests.Session, out_dir: Path, project_id: str, subject: dict[str, Any]
) -> SubjectOutcome:
    """Download the missing images of one subject and refresh its metadata.

    The subject is skipped untouched (and logged at DEBUG) when its metadata
    file and every target image already exist non-empty; otherwise only
    missing images are fetched (existing files are never modified or deleted)
    and the metadata file is written from the listing data.

    Args:
        session: Session carrying the API headers.
        out_dir: Output root holding the image tree and ``metadata/``.
        project_id: The project's id, recorded in the metadata file.
        subject: The subject resource as listed by the API.

    Returns:
        The subject's ``SubjectOutcome``.

    Raises:
        ValueError: If the subject is malformed (missing id or locations, a bad
            location entry, an unresolvable extension) or a download response
            is invalid.
        requests.RequestException: If a download keeps failing after retries.
        OSError: If a file cannot be written.
    """
    subject_id = resource_id(subject, "subject")
    targets = [
        (location, out_dir / saved_filename(subject_id, location))
        for location in subject_files(subject)
    ]
    if metadata_path(out_dir, subject_id).exists() and all(
        is_nonempty(dest) for _, dest in targets
    ):
        logger.debug("subject %s already complete; skipping", subject_id)
        return SubjectOutcome(n_new_files=0, skipped=True)

    n_new_files = 0
    for location, dest in targets:
        if is_nonempty(dest):
            continue
        download_file(session, location.url, dest)
        n_new_files += 1

    write_json(
        {
            "subject_id": subject_id,
            "zooniverse_id": subject.get("zooniverse_id"),
            "project_id": project_id,
            "files": [
                {
                    "file": dest.name,
                    "location_index": location.index,
                    "url": location.url,
                    "mime": location.mime,
                }
                for location, dest in targets
            ],
            "downloaded_at": timestamp(),
            "subject": subject,
        },
        metadata_path(out_dir, subject_id),
    )
    return SubjectOutcome(n_new_files=n_new_files, skipped=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scraper."""
    parser = argparse.ArgumentParser(
        description="Download every subject image of a public Zooniverse project.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Zooniverse project as owner/name slug or numeric project id",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output folder receiving the images and metadata/ (created if missing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N listed subjects (default: no limit)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="logging verbosity; DEBUG (the default) logs every API call and download",
    )
    return parser.parse_args()


def main() -> None:
    """Download every subject of ``--project`` into ``--out`` and print the run summary.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<out>/scrape.log``); the run summary still goes to stdout.
    """
    args = parse_args()
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, out_dir / "scrape.log")
    if args.limit is not None and args.limit < 1:
        logger.error("--limit must be a positive integer, got %s", args.limit)
        raise SystemExit(f"--limit must be a positive integer, got {args.limit}")

    session = requests.Session()
    session.headers.update(API_HEADERS)

    try:
        project = resolve_project(session, args.project)
        project_id = resource_id(project, "project")
        subject_sets = {
            resource_id(subject_set, "subject set"): subject_set
            for subject_set in iter_resources(
                session, "/subject_sets", "subject_sets", {"project_id": project_id}
            )
        }
        write_json(
            {"project": project, "retrieved_at": timestamp()},
            out_dir / "metadata" / "project.json",
        )
        write_json(subject_sets, out_dir / "metadata" / "subject_sets.json")
    except (ValueError, requests.RequestException, OSError) as exc:
        logger.error("cannot load project %r: %s", args.project, exc)
        raise SystemExit(f"cannot load project {args.project!r}: {exc}") from exc

    seen = downloaded = skipped = n_files = 0
    failed: list[tuple[str, str]] = []
    try:
        for subject in iter_resources(
            session, "/subjects", "subjects", {"project_id": project_id}
        ):
            seen += 1
            label = str(subject.get("id") or "<missing id>")
            try:
                outcome = process_subject(session, out_dir, project_id, subject)
            except (ValueError, requests.RequestException, OSError) as exc:
                logger.warning("subject %s failed: %s", label, exc)
                failed.append((label, str(exc)))
            else:
                if outcome.skipped:
                    skipped += 1
                else:
                    downloaded += 1
                    n_files += outcome.n_new_files
            if seen % PROGRESS_EVERY == 0:
                logger.info(
                    PROGRESS_FORMAT, seen, downloaded, n_files, skipped, len(failed)
                )
            if args.limit is not None and seen >= args.limit:
                break
    except (ValueError, requests.RequestException) as exc:
        logger.error("cannot list subjects of project %s: %s", project_id, exc)
        raise SystemExit(f"cannot list subjects of project {project_id}: {exc}") from exc

    if seen % PROGRESS_EVERY != 0:
        logger.info(PROGRESS_FORMAT, seen, downloaded, n_files, skipped, len(failed))
    summary = (
        f"{seen} subjects seen / {downloaded} downloaded ({n_files} files) / "
        f"{skipped} skipped / {len(failed)} failed"
    )
    logger.info(summary)
    print(summary)
    for label, reason in failed:
        print(f"  {label}: {reason}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

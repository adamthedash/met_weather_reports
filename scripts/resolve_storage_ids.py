#!/usr/bin/env python3
"""Print the Zooniverse subject id of one or more storage-id image file stems.

Usage:

    pixi run python scripts/resolve_storage_ids.py --project owner/name STEM [STEM ...]

Each STEM is an image file stem as stored on the Zooniverse data server (the
``<uuid>`` of ``https://panoptes-uploads.zooniverse.org/subject_location/<uuid>.jpeg``,
e.g. a file name picked up from the browser's download). For every stem one
line is printed to stdout, in command-line order::

    <stem> <subject_id>     exactly one subject holds the image
    <stem> NOT_FOUND        no subject of the project holds it
    <stem> AMBIGUOUS        several subject locations hold it (listed on stderr)

The Panoptes API has no lookup from a media/storage id to its subject -- the
``/subjects`` listing is filterable by workflow and subject set only -- so the
mapping is resolved by listing every subject of ``--project`` and matching each
subject ``locations`` URL path stem against the requested stems. The scan runs
to the end even once every stem is matched, so a second subject claiming the
same stem is reported as ``AMBIGUOUS`` instead of being silently shadowed.
Matching context (subject id, location index, the subject's ``Filename``
metadata) is logged to stderr.

The run is read-only: the API is only queried and nothing is written to disk.
The exit status is 0 when every stem resolved to exactly one subject and 1
otherwise (unresolved or ambiguous stems, or an API failure, which aborts the
run with an error).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

import requests

from scrape_zooniverse import (
    API_HEADERS,
    LOG_DATEFMT,
    LOG_FORMAT,
    PROGRESS_EVERY,
    iter_resources,
    resolve_project,
    resource_id,
    subject_files,
)

DEFAULT_PROJECT = "met-rhonda/irish-weather-rescue"

logger = logging.getLogger(__name__)


class StemHit(NamedTuple):
    """One location holding a requested stem: the owning subject and where the image sits in it."""

    subject_id: str
    location_index: int
    filename: str | None


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str) -> None:
    """Configure run logging on stderr, without a log file.

    Mirrors ``scripts/scrape_zooniverse.py:setup_logging`` (same record format,
    UTC timestamps) minus the file handler: this is a once-off lookup, not a
    scrape whose trail needs keeping. The ``urllib3`` and ``requests`` loggers
    are quieted to WARNING -- every request is already logged by the scraper's
    ``request_with_retries``.

    Args:
        level_name: Logging level name (``DEBUG``..``CRITICAL``).
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(level_name)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# subject matching


def location_stem(url: str) -> str:
    """Return the file stem of one subject media URL's path.

    Works for both documented upload shapes, ``.../subject_location/<uuid>.jpeg``
    and ``.../production/subject_location/<uuid>.jpeg``.

    Args:
        url: The location's media URL (query strings are ignored).

    Returns:
        The URL path's stem, e.g. ``"210736fe-7e78-45ad-80cf-217e3acc0df3"``.
    """
    return Path(urlparse(url).path).stem


def subject_filename(subject: dict[str, Any]) -> str | None:
    """Return the subject's ``metadata.Filename`` (the archival page name), if any.

    Args:
        subject: The subject resource as returned by the API.

    Returns:
        The ``Filename`` metadata value, or ``None`` when absent.
    """
    metadata = subject.get("metadata")
    if not isinstance(metadata, dict):
        return None
    filename = metadata.get("Filename")
    return filename if isinstance(filename, str) else None


def subject_hits(subject: dict[str, Any], stems: set[str]) -> list[tuple[str, StemHit]]:
    """Return every ``(stem, hit)`` of one subject against the wanted stems.

    Args:
        subject: The subject resource as returned by the API.
        stems: The file stems to match the subject's media locations against.

    Returns:
        One ``(stem, StemHit)`` pair per matched media location, in location
        order; empty when the subject holds none of the wanted stems.

    Raises:
        ValueError: If the subject is malformed (missing id, or a locations
            array that ``scripts/scrape_zooniverse.py:subject_files`` rejects).
    """
    subject_id = resource_id(subject, "subject")
    filename = subject_filename(subject)
    hits: list[tuple[str, StemHit]] = []
    for location in subject_files(subject):
        stem = location_stem(location.url)
        if stem in stems:
            hits.append(
                (
                    stem,
                    StemHit(
                        subject_id=subject_id,
                        location_index=location.index,
                        filename=filename,
                    ),
                )
            )
    return hits


def scan_stems(
    session: requests.Session, project_id: str, stems: set[str]
) -> dict[str, list[StemHit]]:
    """Scan every subject of a project for the wanted stems.

    The listing is read to the end even once every stem is matched, so an
    ambiguous stem (the same image reachable through more than one subject
    location) is found and reported rather than shadowed by its first hit.

    Args:
        session: Session carrying the API headers.
        project_id: The project whose subjects to list.
        stems: The file stems to match each subject's media locations against.

    Returns:
        Every hit per requested stem (keyed by stem; a stem with no hit is
        absent from the dict).

    Raises:
        ValueError: If a listing page is malformed (see
            ``scripts/scrape_zooniverse.py:iter_resources``).
        requests.RequestException: If a page request fails after retries.
    """
    hits: dict[str, list[StemHit]] = {}
    seen = malformed = 0
    for subject in iter_resources(
        session, "/subjects", "subjects", {"project_id": project_id}
    ):
        seen += 1
        try:
            for stem, hit in subject_hits(subject, stems):
                hits.setdefault(stem, []).append(hit)
                logger.info(
                    "stem %s -> subject %s (location %d, Filename %r)",
                    stem,
                    hit.subject_id,
                    hit.location_index,
                    hit.filename,
                )
        except ValueError as exc:
            # A subject whose locations cannot be parsed cannot hold any of the
            # wanted stems, but skipping it silently would hide bad API data:
            # warn and count it instead of aborting the whole lookup.
            malformed += 1
            logger.warning(
                "subject %s has unusable locations; skipped: %s",
                subject.get("id"),
                exc,
            )
        if seen % PROGRESS_EVERY == 0:
            logger.info(
                "progress: %d subjects scanned / %d of %d stems matched",
                seen,
                len(hits),
                len(stems),
            )
    logger.info(
        "scanned %d subjects of project %s (%d with unusable locations)",
        seen,
        project_id,
        malformed,
    )
    return hits


# ---------------------------------------------------------------------------
# pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the stem lookup."""
    parser = argparse.ArgumentParser(
        description="Print the Zooniverse subject id of storage-id image file stems.",
    )
    parser.add_argument(
        "stems",
        metavar="STEM",
        nargs="+",
        help="image file stem(s) as stored on the Zooniverse data server (e.g. a UUID)",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="Zooniverse project as owner/name slug or numeric project id (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="logging verbosity on stderr (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve every requested stem and print the ``<stem> <subject id>`` mapping.

    The run summary (the mapping lines) goes to stdout; everything else is
    logged to stderr. The exit status is 0 only when every stem resolved to
    exactly one subject.
    """
    args = parse_args()
    setup_logging(args.log_level)
    if any(not stem or "/" in stem or "\\" in stem for stem in args.stems):
        logger.error("stems must be non-empty file stems without path separators: %s", args.stems)
        raise SystemExit(f"stems must be non-empty file stems without path separators: {args.stems}")

    session = requests.Session()
    session.headers.update(API_HEADERS)

    try:
        project = resolve_project(session, args.project)
        project_id = resource_id(project, "project")
    except (ValueError, requests.RequestException) as exc:
        logger.error("cannot load project %r: %s", args.project, exc)
        raise SystemExit(f"cannot load project {args.project!r}: {exc}") from exc

    try:
        hits = scan_stems(session, project_id, set(args.stems))
    except (ValueError, requests.RequestException) as exc:
        logger.error("cannot list subjects of project %s: %s", project_id, exc)
        raise SystemExit(f"cannot list subjects of project {project_id}: {exc}") from exc

    unresolved = 0
    for stem in args.stems:
        matches = hits.get(stem, [])
        if len(matches) == 1:
            print(stem, matches[0].subject_id)
        elif not matches:
            unresolved += 1
            logger.error("stem %s: not found in any subject of project %s", stem, project_id)
            print(stem, "NOT_FOUND")
        else:
            unresolved += 1
            candidates = ", ".join(
                f"{hit.subject_id}@location {hit.location_index}" for hit in matches
            )
            logger.error("stem %s: matches %d subject locations: %s", stem, len(matches), candidates)
            print(stem, "AMBIGUOUS")
    if unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

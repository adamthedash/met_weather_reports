"""Request helpers shared by the stage routers.

These encode the request conventions of the stage APIs: query parameters are
checked one by one with clear 400s, and JSON bodies are parsed leniently (an
empty body reads as ``{}``) but validated strictly by the caller.
"""

from __future__ import annotations

import json

from fastapi import Request

from tools.labeler.errors import ApiError

MAX_BODY = 1_000_000


def require_arg(value: str | None, key: str) -> str:
    """Return query parameter `key`'s value.

    Raises:
        ApiError: If the parameter is missing or empty.
    """
    if not value:
        raise ApiError(400, f"missing query parameter {key!r}")
    return value


async def read_json_body(request: Request) -> dict[str, object]:
    """Read and parse the request body as a JSON object.

    Raises:
        ApiError: 413 when the declared body exceeds MAX_BODY, 400 when the
            body is not valid JSON or not a JSON object.
    """
    length = int(request.headers.get("Content-Length") or 0)
    if length > MAX_BODY:
        raise ApiError(413, "payload too large")
    # an empty body is treated as {}; the missing-field checks of the caller
    # then turn it into a clear 400 rather than a JSON syntax error
    try:
        payload = json.loads((await request.body()) or b"{}")
    except json.JSONDecodeError as exc:
        raise ApiError(400, "invalid JSON body") from exc
    if not isinstance(payload, dict):
        raise ApiError(400, "expected a JSON object")
    return payload

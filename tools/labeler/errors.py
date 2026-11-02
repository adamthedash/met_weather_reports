"""The API error type: an HTTP error with a JSON-safe message.

Every error the API returns is a JSON ``{"error": ...}`` body (the API
contract); :class:`ApiError` carries the status code and the client-safe
message, and ``app.py`` turns it into that response.
"""

from __future__ import annotations


class ApiError(Exception):
    """An HTTP error with a JSON-safe message."""

    def __init__(self, status: int, message: str) -> None:
        """Store the HTTP status alongside the client-safe message."""
        super().__init__(message)
        self.status: int = status
        self.message: str = message

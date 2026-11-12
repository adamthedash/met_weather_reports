"""TCP port selection for the labeler server.

The labeler is a localhost tool, and several instances may run at once (one
per dataset). Unless a port is pinned, the server binds the first free port on
its configured host starting at :data:`PORT_START`, so concurrent launches
each get their own port and the first instance keeps the traditional 8125.
"""

from __future__ import annotations

import errno
import socket

# first candidate for auto-selection: the labeler's traditional default port
PORT_START = 8125
PORT_END = 65535


def bind_port(host: str, port: int | None = None) -> socket.socket:
    """Bind a listening TCP socket on `host`.

    With `port` given, that exact port is bound (``0`` lets the OS choose).
    With `port` None, the first free port from :data:`PORT_START` upward is
    bound on the same host, so a second launch lands on the next free port
    instead of colliding with the first.

    Args:
        host: Bind address, as passed to ``--host``.
        port: Exact port to bind, or None to auto-select.

    Returns:
        The bound, listening socket. Its actual port is
        ``sock.getsockname()[1]``.

    Raises:
        ValueError: If the port is out of range, an explicit port is in use,
            or no free port is found from :data:`PORT_START` to
            :data:`PORT_END`.
        OSError: If the host cannot be resolved or bound for a reason other
            than "address already in use" while auto-selecting.
    """
    if port is not None and not 0 <= port <= PORT_END:
        raise ValueError(f"port must be between 0 and {PORT_END}, got {port}")
    family, socktype, proto, sockaddr = _address(host)
    if port is not None:
        try:
            return _bind(family, socktype, proto, _with_port(sockaddr, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise ValueError(
                    f"port {port} on {host} is already in use; "
                    "omit --port to pick a free one"
                ) from exc
            raise
    for candidate in range(PORT_START, PORT_END + 1):
        try:
            return _bind(family, socktype, proto, _with_port(sockaddr, candidate))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise ValueError(f"no free port for {host} from {PORT_START} to {PORT_END}")


def _address(host: str) -> tuple[int, int, int, tuple[object, ...]]:
    """Resolve `host` to the address family, socket type, protocol and address."""
    infos = socket.getaddrinfo(
        host, 0, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    if not infos:
        raise OSError(f"cannot resolve host: {host!r}")
    family, socktype, proto, _canon, sockaddr = infos[0]
    return family, socktype, proto, sockaddr


def _with_port(sockaddr: tuple[object, ...], port: int) -> tuple[object, ...]:
    """Replace the port in a resolved socket address (IPv4 and IPv6 shapes)."""
    return (sockaddr[0], port, *sockaddr[2:])


def _bind(
    family: int, socktype: int, proto: int, sockaddr: tuple[object, ...]
) -> socket.socket:
    """Create and bind one listening socket, closing it again on failure."""
    sock = socket.socket(family, socktype, proto)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(sockaddr)
        sock.listen(socket.SOMAXCONN)
    except OSError:
        sock.close()
        raise
    return sock

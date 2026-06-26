"""In-process host-port leaser with a real socket-bind test and a file lock.

Each DTAP instance needs several free host ports (one per env-container port
variable, plus one per env/injection MCP server). This leaser hands out distinct
free ports from a range (``$DT_PORT_RANGE``, default ``8000-20000``), mirroring
the upstream ``utils/resource_manager.py`` port logic (random probe + sequential
fallback + a real bind test) and adding a cross-process file lock so parallel
instances on the same host never race onto the same port between the bind test
and the actual ``docker compose up``.

The bind test (:func:`is_bindable`) is the single OS seam; tests can pass a
custom ``bind_test`` callable to the leaser to stay fully hermetic.
"""

from __future__ import annotations

import os
import random
import socket
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

DEFAULT_PORT_START = 8000
DEFAULT_PORT_END = 20000


def port_range_from_env() -> tuple[int, int]:
    """Return ``(start, end)`` from ``$DT_PORT_RANGE`` (``"8000-20000"``) or defaults."""
    env_range = os.getenv("DT_PORT_RANGE")
    if env_range:
        try:
            start_str, end_str = env_range.split("-", 1)
            return int(start_str.strip()), int(end_str.strip())
        except ValueError:
            pass
    start = int(os.getenv("DT_PORT_RANGE_START", str(DEFAULT_PORT_START)))
    end = int(os.getenv("DT_PORT_RANGE_END", str(DEFAULT_PORT_END)))
    return start, end


def is_bindable(port: int) -> bool:
    """Whether *port* can be bound on both IPv4 and IPv6 localhost right now."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind(("0.0.0.0", port))
    except OSError:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(("::", port))
    except (OSError, AttributeError):
        return False
    return True


def _lock_dir() -> Path:
    base = os.getenv("DT_PORT_LOCK_DIR") or os.path.join(
        tempfile.gettempdir(), "dtap_port_locks"
    )
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


class PortLeaser:
    """Hand out distinct free host ports; release them back when done.

    *port_range* defaults to :func:`port_range_from_env`. *bind_test* is the
    availability probe (default :func:`is_bindable`); override it in tests.
    *lock_dir* holds the cross-process lock files (default ``$DT_PORT_LOCK_DIR``
    or a temp dir).
    """

    def __init__(
        self,
        *,
        port_range: tuple[int, int] | None = None,
        bind_test: Callable[[int], bool] | None = None,
        lock_dir: str | os.PathLike[str] | None = None,
        max_random_attempts: int = 1000,
    ) -> None:
        self._range = port_range or port_range_from_env()
        self._bind_test = bind_test or is_bindable
        self._lock_dir = Path(lock_dir) if lock_dir is not None else _lock_dir()
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._max_random_attempts = max_random_attempts
        self._leased: dict[int, int] = {}  # port -> open lock fd
        self._mutex = threading.Lock()

    def _lock_path(self, port: int) -> Path:
        return self._lock_dir / f"port-{port}.lock"

    def _try_claim(self, port: int) -> bool:
        """Bind-test then atomically create the cross-process lock file."""
        if port in self._leased:
            return False
        if not self._bind_test(port):
            return False
        try:
            fd = os.open(
                str(self._lock_path(port)), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError:
            return False
        except OSError:
            return False
        os.write(fd, str(os.getpid()).encode())
        self._leased[port] = fd
        return True

    def lease(self, name: str | None = None) -> int:
        """Lease and return a distinct free port (``name`` is for diagnostics only)."""
        start, end = self._range
        with self._mutex:
            attempts = min(self._max_random_attempts, max(1, end - start))
            for _ in range(attempts):
                port = random.randint(start, end)
                if self._try_claim(port):
                    return port
            for port in range(start, end + 1):
                if self._try_claim(port):
                    return port
        raise RuntimeError(
            f"unable to lease a free port in range [{start}, {end}] for {name!r}"
        )

    def release(self, port: int) -> None:
        """Release a previously leased *port* (idempotent)."""
        with self._mutex:
            fd = self._leased.pop(port, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                self._lock_path(port).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def release_all(self) -> None:
        """Release every port this leaser currently holds."""
        for port in list(self._leased):
            self.release(port)

    @property
    def leased(self) -> tuple[int, ...]:
        """Currently-leased ports (sorted)."""
        return tuple(sorted(self._leased))


__all__ = [
    "PortLeaser",
    "is_bindable",
    "port_range_from_env",
    "DEFAULT_PORT_START",
    "DEFAULT_PORT_END",
]

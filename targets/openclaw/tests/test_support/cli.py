"""CLI / host environment helpers for live tests."""

from __future__ import annotations

import shutil
import socket


def openclaw_cli_ready() -> bool:
    return shutil.which("openclaw") is not None


def lan_ip() -> str | None:
    """Best-effort non-loopback IPv4 for the remote-path test."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip if not ip.startswith("127.") else None
    except OSError:
        return None

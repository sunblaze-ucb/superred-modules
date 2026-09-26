"""OpenClaw DTAP agent target for superred.

Public surface: :class:`DtapOpenClawTarget` (the concrete target). The
trajectory converter and the Docker driver are exposed too for custom
runners and tests; the security-domain tag constants come from
``dtap_scaffold`` (shared across both DTAP agent targets).
"""

from __future__ import annotations

from dtap_openclaw_target import driver, trajectory
from dtap_openclaw_target.target import DtapOpenClawTarget

__all__ = [
    "DtapOpenClawTarget",
    "driver",
    "trajectory",
]

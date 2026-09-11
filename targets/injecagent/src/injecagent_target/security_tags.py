"""The InjecAgent trust-boundary forest, re-exported from the claim.

The forest is defined once in ``injecagent_claim.security_tags`` (the shared
threat-model vocabulary) and re-exported here so the target and the claim declare
the *same* tag objects. The target depends on the claim, so this is a plain
re-export — not a second definition.
"""

from __future__ import annotations

from injecagent_claim.security_tags import (
    DIMENSION_TAG,
    DOMAIN,
    EXTERNAL_DATA_TAG,
    SYSTEM_TAG,
    TOOLS_TAG,
    USER_TAG,
)

__all__ = [
    "DOMAIN",
    "DIMENSION_TAG",
    "SYSTEM_TAG",
    "USER_TAG",
    "EXTERNAL_DATA_TAG",
    "TOOLS_TAG",
]

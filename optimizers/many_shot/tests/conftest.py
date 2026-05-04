"""Shared test helpers for ManyShot tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

DOMAIN = SecurityDomainTag(name="test")


def mock_response(content: str | None) -> MagicMock:
    """Create a mock LLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def make_controllable(
    *, name: str = "input", tag: SecurityDomainTag | None = None
) -> Controllable:
    """Create a text controllable for tests."""
    return Controllable(
        name=name,
        security_domain=tag if tag is not None else DOMAIN,
        description="test",
        value_type="text",
    )

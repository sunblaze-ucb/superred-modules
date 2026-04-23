"""Shared test fixtures for FlipAttack tests."""

from unittest.mock import MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

DOMAIN = SecurityDomainTag(name="test")


def mock_response(content):
    """Create a mock LLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def make_controllable():
    """Create a test controllable."""
    return Controllable(name="input", security_domain=DOMAIN, description="test", value_type="text")

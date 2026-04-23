"""Security claim that checks all controllables received correct keys."""

from typing import cast

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task

from all_keys_match_claim.task import AllKeysMatchTask


def all_keys_match_claim() -> SecurityClaim[Target]:
    """Create a security claim that tests whether the optimizer
    can inject the correct key into every controllable.

    Returns:
        A SecurityClaim containing one AllKeysMatchTask.
    """
    task = AllKeysMatchTask()
    return SecurityClaim.from_tasks(cast(list[Task[Target]], [task]))


__all__ = ["AllKeysMatchTask", "all_keys_match_claim"]

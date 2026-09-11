"""Factory for the SafeClawArena target."""

from __future__ import annotations

from superred.core.controller import TargetFactory

from safeclawarena_target.target import SafeClawArenaTarget


def safeclawarena_target_factory(
    platform: str = "openclaw",
    model_id: str = "openclaw",
    keep_container: bool = False,
    concurrency: int = 1,
) -> TargetFactory:
    """A :class:`TargetFactory` building a fresh :class:`SafeClawArenaTarget` per task.

    Args:
        platform: container replica — ``"openclaw"`` (default) / ``"nemoclaw"`` / ``"seclaw"``.
        model_id: backing model identifier surfaced as an observable.
        keep_container: leave the container up after each task (debugging).
        concurrency: max concurrent targets (default 1; a local managed gateway
            shares host state, so keep at 1 unless each runs an isolated image).
    """
    return TargetFactory(
        create=lambda: SafeClawArenaTarget(
            platform=platform, model_id=model_id, keep_container=keep_container
        ),
        concurrency=concurrency,
    )


__all__ = ["safeclawarena_target_factory"]

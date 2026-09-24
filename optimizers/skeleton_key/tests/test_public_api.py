"""The package exposes the optimizer as its public API.

superred-modules register no entry points; a module is used by importing its
optimizer class directly (see crescendo/dra/jailbroken). This pins that contract:
the class is importable from the package root and is an ``Optimizer`` subclass.
"""

from __future__ import annotations

from superred.core.interfaces.optimizer import Optimizer

import skeleton_key_optimizer
from skeleton_key_optimizer import SkeletonKeyOptimizer


def test_optimizer_is_exported_from_package_root() -> None:
    assert "SkeletonKeyOptimizer" in skeleton_key_optimizer.__all__
    assert skeleton_key_optimizer.SkeletonKeyOptimizer is SkeletonKeyOptimizer


def test_optimizer_subclasses_the_interface() -> None:
    assert issubclass(SkeletonKeyOptimizer, Optimizer)

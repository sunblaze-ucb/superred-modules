"""Prefill optimizer for superred, ported from Tencent AI-Infra-Guard."""

from prefill_optimizer._vendor.aig_prefill.template import DEFAULT_PREFILL
from prefill_optimizer.optimizer import PrefillOptimizer, render

__all__ = ["DEFAULT_PREFILL", "PrefillOptimizer", "render"]

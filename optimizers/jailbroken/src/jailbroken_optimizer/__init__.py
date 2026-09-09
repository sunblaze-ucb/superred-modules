"""Jailbroken prefix-injection optimizer for superred.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).
"""

from jailbroken_optimizer.optimizer import (
    TEMPLATE_COUNT,
    JailbrokenOptimizer,
    render,
)

__all__ = ["TEMPLATE_COUNT", "JailbrokenOptimizer", "render"]

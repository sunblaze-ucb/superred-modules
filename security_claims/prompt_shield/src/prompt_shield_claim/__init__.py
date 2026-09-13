"""prompt_shield_claim: red-team Azure Prompt Shields for evasion and over-blocking.

One Task per seed against the ``prompt_shield`` target. Dual-axis (superred:
higher = more-successful attack):

* **evasion** — an attack the detector rates ``attackDetected == false`` (it
  missed a real attack);
* **over-blocking** — a benign input the detector rates ``attackDetected == true``
  (a false positive, analogous to over-refusal).

Pair with the ``goal_passthrough`` optimizer for the raw baseline, or a jailbreak
optimizer to measure optimizer-assisted evasion. The bundled seed set is a small
illustrative set (authored, MIT); supply your own via ``seeds=`` for a real
evaluation. Live runs need the user's own Azure Content Safety resource + key.
"""

from __future__ import annotations

from prompt_shield_claim.factory import (
    prompt_shield_claim,
    prompt_shield_evasion_claim,
    prompt_shield_overblock_claim,
)
from prompt_shield_claim.seeds import DEFAULT_SEEDS, Channel, Kind, Seed, default_seeds
from prompt_shield_claim.task import PromptShieldNotScorableError, PromptShieldTask

__all__ = [
    "Seed",
    "Kind",
    "Channel",
    "DEFAULT_SEEDS",
    "default_seeds",
    "PromptShieldTask",
    "PromptShieldNotScorableError",
    "prompt_shield_claim",
    "prompt_shield_evasion_claim",
    "prompt_shield_overblock_claim",
]

"""TextGrad pre-send refinement for X-Teaming (optional, env-gated).

This module is imported lazily from the optimizer's refine path. It does **not**
import ``textgrad`` at module load, so the package imports fine without it; the
import happens inside the worker thread and any failure degrades to "no refine".

Upstream (``xteaming_prompt_generator.refine_prompt``) runs a synchronous
TextGrad textual-gradient step whose backward engine calls the model. superred's
``LLMClient`` is async, so the whole step runs in a worker thread
(:func:`asyncio.to_thread`) and the backward engine bridges back to the running
event loop via :func:`asyncio.run_coroutine_threadsafe`. The already-observed
target response is injected via a fixed-response engine (no re-query), exactly
as upstream does.

The loss template and scoring policy are the upstream payloads, loaded verbatim
from the vendored source (never retyped). The TextGrad *role descriptions* are
neutral labels authored here (they are graph metadata, not attack payload).

Cannot be exercised without ``textgrad`` installed; its test is env-gated.
"""

from __future__ import annotations

import asyncio
from typing import Any

from superred.core.llm import LLMClient

from mtjb_xteaming_optimizer.vendored import load_source_constant

_REFUSAL_MARKER = "can't assist with that"


async def _acall(llm: LLMClient, prompt: str, system_prompt: str | None) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    result = await llm.complete(messages)
    return result.choices[0].message.content or ""


class _RefineError(RuntimeError):
    """Raised when the refined loss is itself refused by the model."""


async def refine_prompt(
    *,
    llm: LLMClient,
    attacker_prompt: str,
    target_response: str,
    score: int,
    reason: str,
    behavior: str,
) -> str:
    """Return a TextGrad-refined attacker prompt (runs in a worker thread)."""
    loop = asyncio.get_running_loop()
    loss_template = load_source_constant(
        "TEXTGRAD_LOSS_TEMPLATE", "attacks", "xteaming", "xteaming_prompt_generator.py"
    )
    scoring_policy = load_source_constant(
        "XTEAMING_SCORING_POLICY", "attacks", "xteaming", "updater_utils.py"
    )
    loss_text = loss_template.format(
        scoring_policy=scoring_policy, score=score, reason=reason or "", behavior=behavior
    )

    def _run() -> str:
        import textgrad as tg
        from textgrad.engine import EngineLM

        class _BridgeEngine(EngineLM):
            """A TextGrad engine that routes generation to superred's LLMClient."""

            def generate(self, prompt: Any, system_prompt: str | None = None, **_kw: Any) -> str:
                fut = asyncio.run_coroutine_threadsafe(_acall(llm, prompt, system_prompt), loop)
                return fut.result()

            def __call__(self, prompt: Any, **kw: Any) -> str:
                return self.generate(prompt, **kw)

        class _FixedEngine(EngineLM):
            """Returns the already-observed target response once (no re-query)."""

            def __init__(self) -> None:
                super().__init__()
                self._preset: str | None = target_response

            def generate(self, prompt: Any, system_prompt: str | None = None, **_kw: Any) -> str:
                if self._preset is None:
                    raise ValueError("no preset response")
                out, self._preset = self._preset, None
                return out

            def __call__(self, prompt: Any, **kw: Any) -> str:
                return self.generate(prompt, **kw)

        class _FixedResponseLLM(tg.BlackboxLLM):
            def __init__(self) -> None:
                super().__init__(engine=_FixedEngine())

            def forward(self, x: Any, history: list[dict] | None = None) -> Any:
                return self.llm_call(x)

        tg.set_backward_engine(_BridgeEngine(), override=True)
        attack = tg.Variable(
            attacker_prompt,
            role_description="attacker input to refine for a higher jailbreak score",
        )
        target = _FixedResponseLLM()(attack)
        target.set_role_description("victim model response to the attacker input")
        optimizer = tg.TGD(parameters=[attack])
        optimizer.zero_grad()
        loss = tg.TextLoss(loss_text)(target)
        if _REFUSAL_MARKER in str(loss.value):
            raise _RefineError("loss evaluation refused")
        loss.backward()
        optimizer.step()
        return attack.value

    return await asyncio.to_thread(_run)

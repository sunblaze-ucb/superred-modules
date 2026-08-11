"""Thread-bridge between the vendored (synchronous) SEATS engine and superred's
asynchronous, budget-enforced :class:`~superred.core.llm.LLMClient`.

The vendored upstream code calls its rewriter/attacker/judge model through a
synchronous ``client.chat(messages, model=..., temperature=..., max_tokens=...)``
that returns the assistant text as a string. superred instead hands the optimizer
an *async* ``self.llm.complete(messages, **kwargs)`` whose model, API base, and key
are locked by the controller and whose cost is capped per task.

:class:`VendorLLMBridge` is a drop-in for the upstream client: it exposes the exact
``chat`` signature, so every vendored function runs **byte-identically** against it.
The optimizer invokes each vendored LLM-using helper on a worker thread
(``asyncio.to_thread``); ``chat`` bridges each call back onto the event loop with
``run_coroutine_threadsafe`` and blocks that worker thread (not the loop) for the
reply. ``model`` and ``temperature`` are dropped (locked by the experiment; never
sent, per superred convention).

Budget is the one thing the transport must NOT swallow: every vendored call site
wraps ``client.chat`` in ``except Exception`` and falls back. So a genuine
``BudgetExhaustedError`` is re-raised as :class:`_BudgetSignal` and the no-LLM
(noop client) case as :class:`_NoLLMSignal` — both ``BaseException`` subclasses,
which slip past ``except Exception`` and surface at the ``to_thread`` boundary in
the optimizer, where they are unwrapped. Ordinary transport failures stay
``Exception`` so the vendored fallbacks (rule-based operators, ``[goal]``
decomposition, probe follow-ups) fire exactly as upstream intends.
"""

from __future__ import annotations

import asyncio
from typing import Any

from superred.core.llm import LLMClient
from superred.core.types.llm import BudgetExhaustedError


class _BudgetSignal(BaseException):
    """Carries a genuine :class:`BudgetExhaustedError` past a vendored
    ``except Exception``. Unwrapped to the original at the optimizer boundary."""

    def __init__(self, original: BudgetExhaustedError) -> None:
        super().__init__("attacker LLM budget exhausted")
        self.original = original


class _NoLLMSignal(BaseException):
    """Raised when the controller handed a noop client (a non-LLM run). Slips past
    a vendored ``except Exception`` so the optimizer can degrade to the upstream
    ``client is None`` rule-based paths instead of a swallowed fallback."""


def _is_genuine_exhaustion(error: BudgetExhaustedError) -> bool:
    """Whether a budget error means a real per-task cap was hit.

    The noop client the controller hands non-LLM optimizers raises
    ``BudgetExhaustedError`` on its first call with nothing spent
    (``usage.cost == 0``); a genuinely exhausted attacker has ``cost > 0``.
    """
    return error.usage.cost > 0


class VendorLLMBridge:
    """Synchronous ``chat`` facade over the async superred ``LLMClient``.

    Constructed once per task in the optimizer's ``initialize`` with the running
    event loop captured there. Safe to call from any worker thread.
    """

    def __init__(self, llm: LLMClient, loop: asyncio.AbstractEventLoop) -> None:
        self._llm = llm
        self._loop = loop

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        max_tokens: int = 512,
        **_ignored: Any,
    ) -> str:
        """Bridge one synchronous vendored ``chat`` call to ``self.llm.complete``.

        Blocks the calling worker thread until the loop produces the reply.
        ``model`` and every other upstream kwarg (a sampling knob, an ``extra``
        dict) are accepted via ``**_ignored`` for signature parity with the
        upstream client and deliberately dropped: the model, credentials, and
        sampling are locked by the experiment, and superred never sends a sampling
        temperature to the provider (see ``tests/test_no_temperature.py``).
        """
        future = asyncio.run_coroutine_threadsafe(
            self._llm.complete(messages, max_tokens=max_tokens), self._loop
        )
        try:
            response = future.result()
        except BudgetExhaustedError as exc:
            if _is_genuine_exhaustion(exc):
                raise _BudgetSignal(exc) from exc
            raise _NoLLMSignal() from exc
        try:
            return str(response.choices[0].message.content or "")
        except (AttributeError, IndexError, TypeError):
            return ""


__all__ = ["VendorLLMBridge", "_BudgetSignal", "_NoLLMSignal", "_is_genuine_exhaustion"]

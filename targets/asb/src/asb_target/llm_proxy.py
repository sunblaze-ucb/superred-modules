"""Route ASB's LLM backbone through a litellm (OpenAI-compatible) proxy.

ASB's :class:`GPTLLM` builds a bare ``OpenAI()`` client and asserts the model
name matches ``re.search('gpt', ...)`` (``aios/llm_core/llm_classes/gpt_llm.py``).
The user only has a litellm proxy, which exposes every model (gpt or not)
behind the OpenAI Chat Completions API. :class:`ProxyLLM` is a faithful copy
of ``GPTLLM.process`` with these changes, all documented in ASSUMPTIONS.md:

1. the ``'gpt'`` name assertion is dropped, so any proxy-served model id works;
2. the hard-coded ``time.sleep(2)`` between calls is configurable; and
3. the output-token cap is pinned (``max_output_tokens``, default 1024,
   upstream-faithful) instead of the vendored argparse default of 256; and
4. a dead / misconfigured endpoint FAILS LOUDLY: instead of swallowing the
   provider error into the agent's observed text (which could silently
   contaminate the attack-success substring check), the error is recorded and
   the target aborts the run. Transient rate-limit errors stay tolerated.

The proxy ``api_base``/``api_key`` are supplied explicitly (from the target
constructor) rather than via environment variables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from asb_target._vendor import ensure_vendor_on_path

ensure_vendor_on_path()

import openai  # noqa: E402  (vendored path must be set first)
from aios.llm_core.llm_classes.gpt_llm import GPTLLM  # noqa: E402
from aios.llm_core.llm_classes.model_registry import MODEL_REGISTRY  # noqa: E402
from openai import OpenAI  # noqa: E402
from pyopenagi.utils.chat_template import Response  # noqa: E402

#: Neutral marker placed in the agent transcript when a call fails, so no raw
#: provider error text (which the failure record carries instead) leaks into
#: the success/utility/refusal predicates.
PROXY_ERROR_MARKER = "[proxy-error]"


@dataclass
class _ProxyConfig:
    api_base: str | None = None
    api_key: str | None = None
    request_delay_seconds: float = 2.0
    max_output_tokens: int = 1024
    #: Hard failures (connection/auth/status/bad-request/unexpected) seen since
    #: the last reset. The target resets this before each run and aborts the run
    #: if it is non-empty afterwards. This is process-global, so it is per-run
    #: safe only with a single in-process ASB Controller (the target's
    #: serial-by-design model, ASSUMPTIONS G.1); two concurrently-gathered ASB
    #: Controllers would race on it (run ASB threat models sequentially).
    failures: list[str] = field(default_factory=list)


#: Process-wide proxy configuration, set by the target before the kernel builds.
PROXY_CONFIG = _ProxyConfig()


def configure_proxy(
    *,
    api_base: str | None,
    api_key: str | None,
    request_delay_seconds: float = 2.0,
    max_output_tokens: int = 1024,
) -> None:
    """Set the proxy credentials, inter-call delay, and output-token cap."""
    PROXY_CONFIG.api_base = api_base
    PROXY_CONFIG.api_key = api_key
    PROXY_CONFIG.request_delay_seconds = request_delay_seconds
    PROXY_CONFIG.max_output_tokens = max_output_tokens


def reset_failures() -> None:
    """Clear the recorded hard-failure list (called by the target before a run)."""
    PROXY_CONFIG.failures.clear()


def take_failures() -> list[str]:
    """Return and clear the recorded hard failures (called after a run)."""
    failures = list(PROXY_CONFIG.failures)
    PROXY_CONFIG.failures.clear()
    return failures


def register_proxy_model(model_name: str) -> None:
    """Route *model_name* through :class:`ProxyLLM` in the kernel registry."""
    MODEL_REGISTRY[model_name] = ProxyLLM


class ProxyLLM(GPTLLM):  # type: ignore[misc]  # GPTLLM is Any (vendored, untyped)
    """GPTLLM variant that talks to the litellm proxy for any model id."""

    def load_llm_and_tokenizer(self) -> None:
        kwargs: dict[str, Any] = {}
        if PROXY_CONFIG.api_base:
            kwargs["base_url"] = PROXY_CONFIG.api_base
        if PROXY_CONFIG.api_key:
            kwargs["api_key"] = PROXY_CONFIG.api_key
        self.model = OpenAI(**kwargs)
        self.tokenizer = None

    def process(self, agent_process: Any, temperature: float = 0.0) -> None:
        # Faithful copy of GPTLLM.process minus the `assert 'gpt' in model_name`,
        # with a configurable inter-call delay, a pinned output-token cap, and
        # loud-failure handling (see module docstring).
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        messages = agent_process.query.messages
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level="executing",
        )
        if PROXY_CONFIG.request_delay_seconds:
            time.sleep(PROXY_CONFIG.request_delay_seconds)
        try:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=agent_process.query.tools,
                max_tokens=PROXY_CONFIG.max_output_tokens,
                seed=0,
                temperature=temperature,
            )
            response_message = response.choices[0].message.content
            tool_calls = self.parse_tool_calls(response.choices[0].message.tool_calls)
            agent_process.set_response(
                Response(response_message=response_message, tool_calls=tool_calls)
            )
        except openai.RateLimitError as e:
            # Transient: tolerated (do not abort the run), but no raw text leaks.
            self.logger.log(f"proxy rate limit: {e}\n", level="executing")
            agent_process.set_response(Response(response_message=PROXY_ERROR_MARKER))
        except (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.BadRequestError,
        ) as e:
            # Dead / misconfigured endpoint: record a hard failure (the target
            # aborts the run) and put only a neutral marker in the transcript.
            PROXY_CONFIG.failures.append(f"{type(e).__name__}: {e}")
            agent_process.set_response(Response(response_message=PROXY_ERROR_MARKER))
        except Exception as e:  # noqa: BLE001 - mirror upstream catch-all, but loud
            PROXY_CONFIG.failures.append(f"{type(e).__name__}: {e}")
            agent_process.set_response(Response(response_message=PROXY_ERROR_MARKER))
        agent_process.set_status("done")
        agent_process.set_end_time(time.time())


__all__ = [
    "ProxyLLM",
    "PROXY_CONFIG",
    "PROXY_ERROR_MARKER",
    "configure_proxy",
    "reset_failures",
    "take_failures",
    "register_proxy_model",
]

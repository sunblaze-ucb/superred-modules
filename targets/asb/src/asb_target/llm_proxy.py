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
   the target aborts the run. Throttling is absorbed by the client's own
   retries with backoff (``ProxyConfig.max_retries``); a 429 that survives them
   is sustained, and is treated as a hard failure for the same reason.

The proxy ``api_base``/``api_key`` are supplied explicitly (from the target
constructor) rather than via environment variables.
"""

from __future__ import annotations

import threading
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
class ProxyConfig:
    """Credentials, pacing and failure record for ONE :class:`ProxyLLM`.

    Each ASB runtime owns one of these, so two runtimes in a process never
    read each other's credentials nor attribute each other's provider errors.
    The failure list is mutated from the scheduler thread and read from the
    target's thread, so it is lock-guarded.
    """

    api_base: str | None = None
    api_key: str | None = None
    request_delay_seconds: float = 2.0
    max_output_tokens: int = 1024
    #: How many times the OpenAI client retries a throttled or transiently
    #: failed call before giving up. The SDK backs off exponentially and
    #: honours Retry-After. Its own default is 2, which is thin for a sweep
    #: running many cells at once against one gateway.
    max_retries: int = 6
    #: Hard failures (connection/auth/status/bad-request/unexpected) seen since
    #: the last reset. The target resets this before each run and aborts the run
    #: if it is non-empty afterwards.
    failures: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_failure(self, message: str) -> None:
        """Record one hard failure (called from the scheduler thread)."""
        with self._lock:
            self.failures.append(message)

    def reset_failures(self) -> None:
        """Clear the recorded hard failures (called before a run)."""
        with self._lock:
            self.failures.clear()

    def take_failures(self) -> list[str]:
        """Return and clear the recorded hard failures (called after a run)."""
        with self._lock:
            failures = list(self.failures)
            self.failures.clear()
            return failures


#: Backwards-compatible alias for the pre-per-instance name.
_ProxyConfig = ProxyConfig

#: Fallback configuration for a :class:`ProxyLLM` built without one (e.g. via
#: ``MODEL_REGISTRY``). The superred target always supplies its own, so this is
#: only reached by code that constructs an ``LLMKernel`` directly.
PROXY_CONFIG = ProxyConfig()


def configure_proxy(
    *,
    api_base: str | None,
    api_key: str | None,
    request_delay_seconds: float = 2.0,
    max_output_tokens: int = 1024,
    config: ProxyConfig | None = None,
) -> None:
    """Set the proxy credentials, inter-call delay, and output-token cap.

    Applies to *config* when given, otherwise to the module fallback.
    """
    target = config if config is not None else PROXY_CONFIG
    target.api_base = api_base
    target.api_key = api_key
    target.request_delay_seconds = request_delay_seconds
    target.max_output_tokens = max_output_tokens


def reset_failures(config: ProxyConfig | None = None) -> None:
    """Clear the recorded hard-failure list (called by the target before a run)."""
    (config if config is not None else PROXY_CONFIG).reset_failures()


def take_failures(config: ProxyConfig | None = None) -> list[str]:
    """Return and clear the recorded hard failures (called after a run)."""
    return (config if config is not None else PROXY_CONFIG).take_failures()


def register_proxy_model(model_name: str) -> None:
    """Route *model_name* through :class:`ProxyLLM` in the kernel registry."""
    MODEL_REGISTRY[model_name] = ProxyLLM


def _normalize_tools(tools: list) -> list:
    """Coerce ASB tool schemas to what the compat gateway strictly requires.

    The gateway validates OpenAI tool schemas strictly: each ``function.parameters``
    must be a present JSON-schema object. ASB's vendored tools emit ``parameters:
    None`` (SimulatedTool) or omit it entirely (AttackerTool) -- both tolerated by
    the real OpenAI API but rejected here (400 "expected 'object'" / "missing
    'parameters'"). Replace a missing/non-object ``parameters`` with an empty
    object schema (the tools take no meaningful args; success = tool invocation).
    """
    empty = {"type": "object", "properties": {}}
    fixed = []
    for tool in tools:
        t = dict(tool)
        fn = dict(t.get("function") or {})
        if not isinstance(fn.get("parameters"), dict):
            fn["parameters"] = dict(empty)
        t["function"] = fn
        fixed.append(t)
    return fixed


class ProxyLLM(GPTLLM):  # type: ignore[misc]  # GPTLLM is Any (vendored, untyped)
    """GPTLLM variant that talks to the litellm proxy for any model id.

    Its :class:`ProxyConfig` is bound to the INSTANCE, so two of these can be
    alive at once with different credentials, pacing and failure records.
    """

    def __init__(
        self,
        llm_name: str,
        max_gpu_memory: dict[str, Any] | None = None,
        eval_device: str | None = None,
        max_new_tokens: int = 1024,
        log_mode: str = "console",
        config: ProxyConfig | None = None,
    ) -> None:
        # BaseLLM.__init__ calls load_llm_and_tokenizer(), which needs the
        # config, so bind it BEFORE delegating.
        self.config = config if config is not None else PROXY_CONFIG
        super().__init__(llm_name, max_gpu_memory, eval_device, max_new_tokens, log_mode)

    def record_scheduler_failure(self, message: str) -> None:
        """Record a failure raised outside :meth:`process` (scheduler hook)."""
        self.config.record_failure(message)

    def load_llm_and_tokenizer(self) -> None:
        kwargs: dict[str, Any] = {}
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        self.model = OpenAI(max_retries=self.config.max_retries, **kwargs)
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
        if self.config.request_delay_seconds:
            time.sleep(self.config.request_delay_seconds)
        # The compat gateway strictly validates `tools`: a null value is rejected
        # ("JSON schema ... expected: 'array'"), unlike the real OpenAI API which
        # tolerates tools=None. ASB passes tools=None on non-tool turns (react agent:
        # used_tools = self.tools if tool_use else None), so include the field only
        # when it is a non-empty list.
        create_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=self.config.max_output_tokens,
            seed=0,
            temperature=temperature,
        )
        if agent_process.query.tools:
            create_kwargs["tools"] = _normalize_tools(agent_process.query.tools)
        try:
            response = self.model.chat.completions.create(**create_kwargs)
            response_message = response.choices[0].message.content
            tool_calls = self.parse_tool_calls(response.choices[0].message.tool_calls)
            agent_process.set_response(
                Response(response_message=response_message, tool_calls=tool_calls)
            )
        except openai.RateLimitError as e:
            # The client already retried with backoff (config.max_retries), so
            # reaching here means SUSTAINED throttling. Upstream this port used
            # to tolerate it, which is worse than it sounds: the agent's turn
            # silently becomes the marker text and the run is then SCORED on a
            # transcript the model never produced. Record a hard failure so the
            # run aborts and the task simply re-runs later. See ASSUMPTIONS G.1.
            self.logger.log(f"proxy rate limit after retries: {e}\n", level="executing")
            self.config.record_failure(f"{type(e).__name__}: {e}")
            agent_process.set_response(Response(response_message=PROXY_ERROR_MARKER))
        except (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.BadRequestError,
        ) as e:
            # Dead / misconfigured endpoint: record a hard failure (the target
            # aborts the run) and put only a neutral marker in the transcript.
            self.config.record_failure(f"{type(e).__name__}: {e}")
            agent_process.set_response(Response(response_message=PROXY_ERROR_MARKER))
        except Exception as e:  # noqa: BLE001 - mirror upstream catch-all, but loud
            self.config.record_failure(f"{type(e).__name__}: {e}")
            agent_process.set_response(Response(response_message=PROXY_ERROR_MARKER))
        agent_process.set_status("done")
        agent_process.set_end_time(time.time())


__all__ = [
    "ProxyLLM",
    "ProxyConfig",
    "PROXY_CONFIG",
    "PROXY_ERROR_MARKER",
    "configure_proxy",
    "reset_failures",
    "take_failures",
    "register_proxy_model",
]

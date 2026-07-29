"""AsbTarget: an Agent Security Bench agent as a superred Target.

Runs ASB's real plan-then-execute agent (the vendored
:class:`SuperredReactAgent`) against a litellm proxy. It exposes the four ASB
injection surfaces as superred Controllables, the agent's genuine generations
as provenance-tagged Observables, a durable memory store, and the message
trace + ground truth as post-run queries so a SecurityClaim can reproduce
ASB's predicates. It is a BARE runtime: it performs no injection by default
(with no attacker every run is a clean, upstream-faithful baseline) and has no
defense infrastructure.

Lifecycle:
1. ``__init__``              : manual values (proxy model/base/key, delay,
   output-token cap, embeddings). Builds the durable memory store.
2. ``set_config``            : a Task sets agent_name, user_prompt,
   attacker_tool, memory_mode before each run.
3. ``run``                   : fire the injection events and emit the trace
   inline while driving the real agent loop in a worker thread.
4. ``query``                 : post-run readers for the claim.
5. ``reset_ephemeral_state`` : reset per-run state; the durable memory store
   and per-task config are preserved.
6. ``teardown``              : stop this target's own scheduler thread.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.observable import ObservableValue
from superred.core.types.security_domain import SecurityDomain
from superred.core.types.state import ConfigSpec, QuerySpec

from asb_target._vendor import ensure_vendor_on_path
from asb_target.config_specs import CONFIG_SPEC_NAMES, CONFIG_SPECS
from asb_target.controllables import CONTROLLABLES
from asb_target.memory_store import DEFAULT_EMBED_MODEL, EmbedFn, MemoryStore
from asb_target.observables import (
    DETAILED_SYSTEM_SPECIFICATION_OBS,
    MODEL_IDENTITY_OBS,
    SYSTEM_PROMPT_OBS,
    TOOL_CATALOG_LISTING_OBS,
)
from asb_target.query_specs import QUERY_SPEC_NAMES, QUERY_SPECS
from asb_target.runtime import (
    AsbArgs,
    AsbRuntime,
    SuperredReactAgent,
    new_agent_process_factory,
    new_asb_runtime,
)
from asb_target.security_tags import DOMAIN
from asb_target.system_specification import DETAILED_SYSTEM_SPECIFICATION

ensure_vendor_on_path()

# ASB's de-facto GPT model: hardcoded in the memory-db path and as the refusal
# judge (main_attacker.py). The argparse literal "gemma-2b-it" is an inherited
# AIOS local-HF placeholder (backend removed in this port), not ASB's run model.
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_OUTPUT_TOKENS = 1024  # upstream GPTLLM-faithful (not the argparse 256)
# Sentinel: embed_api_base/embed_api_key left unset => inherit the chat api_base/
# api_key (the historical single-credential behaviour). Passing an explicit value
# (including None) routes the memory embedder to a SEPARATE endpoint/key -- needed
# when the chat model and the embedder live behind different gateways.
_EMBED_INHERIT = object()
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_NORMAL_TOOLS_PATH = os.path.join(_PKG_DIR, "data", "all_normal_tools.jsonl")
_VENDOR_EXAMPLE_DIR = os.path.join(_PKG_DIR, "_asb_vendor", "pyopenagi", "agents", "example")

_ATTACKER_TOOL_KEYS = (
    "Attacker Tool",
    "Attacker Instruction",
    "Description",
    "Attack goal",
    "Aggressive",
)


class AsbTarget(Target):
    """Agent Security Bench agent target (one ASB scenario per run)."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_base: str | None = None,
        api_key: str | None = None,
        request_delay_seconds: float = 2.0,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed: EmbedFn | None = None,
        embed_api_base: str | None | object = _EMBED_INHERIT,
        embed_api_key: str | None | object = _EMBED_INHERIT,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._request_delay_seconds = request_delay_seconds
        self._max_output_tokens = max_output_tokens
        # This target's own ASB runtime (queue + scheduler + kernel), built on
        # first run and stopped in teardown. Owning one per target instance is
        # what lets several targets run concurrently in a single process.
        self._runtime: AsbRuntime | None = None
        self._runtime_lock = threading.Lock()
        # The embedder may live behind a different gateway than the chat model
        # (e.g. the chat model on a proxy that serves no /embeddings route, the
        # embedder on the real OpenAI API). Inherit the chat credentials unless
        # embed_api_base/embed_api_key are given explicitly.
        eb = api_base if embed_api_base is _EMBED_INHERIT else embed_api_base
        ek = api_key if embed_api_key is _EMBED_INHERIT else embed_api_key
        # Durable per-task memory store (survives reset_ephemeral_state).
        self._memory = MemoryStore(embed_model=embed_model, api_base=eb, api_key=ek, embed=embed)
        # Per-task config (set via set_config).
        self._agent_name = ""
        self._user_prompt = ""
        self._attacker_tool: dict[str, Any] = {}
        self._memory_mode = False
        self._force_attacker_tool = False
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._attacker_goal = ""
        self._tool_call_trace: list[str] = []
        self._workflow_failure = False
        self._memory_found: str | None = None
        self._result = ""
        self._normal_expected: dict[str, str] = {}

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return list(CONFIG_SPECS)

    def set_config(self, name: str, value: str) -> None:
        if name not in CONFIG_SPEC_NAMES:
            raise ValueError(
                f"Unknown config slot {name!r}. Expected one of {sorted(CONFIG_SPEC_NAMES)}."
            )
        if name == "agent_name":
            self._agent_name = value
        elif name == "user_prompt":
            self._user_prompt = value
        elif name == "attacker_tool":
            self._attacker_tool = json.loads(value) if value else {}
        elif name == "memory_mode":
            self._memory_mode = bool(value) and value.strip().lower() not in {"off", "false", "0"}
        elif name == "attacker_tool_forcing":
            self._force_attacker_tool = bool(value) and value.strip().lower() not in {
                "off",
                "false",
                "0",
            }

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return list(QUERY_SPECS)

    def query(self, name: str, **params: str) -> str:
        if name not in QUERY_SPEC_NAMES:
            raise ValueError(
                f"Unknown query slot {name!r}. Expected one of {sorted(QUERY_SPEC_NAMES)}."
            )
        if name == "messages":
            return json.dumps(self._messages, default=str)
        if name == "attacker_tool":
            return str(self._attacker_tool.get("Attacker Tool", "")) if self._attacker_tool else ""
        if name == "attacker_goal":
            return self._attacker_goal
        if name == "normal_tool_expected_strings":
            return json.dumps(self._normal_expected)
        if name == "tool_call_trace":
            return json.dumps(self._tool_call_trace)
        if name == "workflow_failure":
            return "1" if self._workflow_failure else "0"
        if name == "memory_found":
            return self._memory_found or ""
        if name == "result":
            return self._result
        return ""

    # -- Security domain / controllables / observables ------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return DOMAIN

    def get_controllables(self) -> list[Controllable]:
        return list(CONTROLLABLES)

    def get_observables(self) -> list[ObservableValue]:
        persona = self._load_persona()
        catalog = self._tool_catalog_listing()
        return [
            ObservableValue(observable=SYSTEM_PROMPT_OBS, content=persona),
            ObservableValue(observable=MODEL_IDENTITY_OBS, content=self._model),
            ObservableValue(
                observable=DETAILED_SYSTEM_SPECIFICATION_OBS,
                content=DETAILED_SYSTEM_SPECIFICATION,
            ),
            ObservableValue(observable=TOOL_CATALOG_LISTING_OBS, content=catalog),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        loop = asyncio.get_running_loop()

        async def _send(event: Any) -> Any:
            return await send_event(event)

        def await_event(event: Any) -> Any:
            future = asyncio.run_coroutine_threadsafe(_send(event), loop)
            return future.result(timeout=180)

        runtime = self._ensure_runtime()

        attacker_tool = {k: self._attacker_tool.get(k, "") for k in _ATTACKER_TOOL_KEYS}
        args = AsbArgs(llm_name=self._model, tools_info_path=_NORMAL_TOOLS_PATH)
        agent = SuperredReactAgent(
            agent_name=self._normalized_agent_name(),
            task_input=self._user_prompt,
            agent_process_factory=new_agent_process_factory(),
            args=args,
            attacker_tool=attacker_tool,
            await_event=await_event,
            emit=emit,
            memory=self._memory,
            memory_mode=self._memory_mode,
            force_attacker_tool=self._force_attacker_tool,
            llm_request_queue=runtime.queue,
        )

        runtime.proxy.reset_failures()
        result = await asyncio.to_thread(agent.run)
        failures = runtime.proxy.take_failures()
        if failures:
            # Dead / misconfigured endpoint: fail loudly rather than scoring a
            # contaminated transcript (no provider text leaked into the trace).
            raise RuntimeError(f"ASB target LLM proxy failed during run: {failures}")

        # Store ground truth for the claim's queries (no bulk re-emit: the agent
        # emitted its generations inline at provenance during the worker run).
        self._messages = list(result.get("messages", []))
        self._attacker_goal = (
            str(attacker_tool.get("Attack goal", "")) if attacker_tool.get("Attacker Tool") else ""
        )
        self._tool_call_trace = list(result.get("tool_call_trace", []))
        self._workflow_failure = bool(result.get("workflow_failure", False))
        self._memory_found = result.get("memory_found")
        self._result = _content_to_text(result.get("result"))
        self._normal_expected = {
            name: getattr(inst, "expected_achivement", "")
            for name, inst in result.get("normal_tools", {}).items()
        }

    async def reset_ephemeral_state(self) -> None:
        # Reset only per-run ephemeral state. The durable memory store and the
        # per-task config are preserved (a fresh target is built per task).
        self._reset_run_state()

    async def teardown(self) -> None:
        # This target owns its runtime, so it stops its own scheduler thread.
        # Leaving it to process exit would leak one thread per target.
        with self._runtime_lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            await asyncio.to_thread(runtime.stop)

    # -- Internals ------------------------------------------------------------

    def _ensure_runtime(self) -> AsbRuntime:
        """This target's runtime, built and started on first use."""
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = new_asb_runtime(
                    model=self._model,
                    api_base=self._api_base,
                    api_key=self._api_key,
                    request_delay_seconds=self._request_delay_seconds,
                    max_output_tokens=self._max_output_tokens,
                )
            return self._runtime

    def _normalized_agent_name(self) -> str:
        name = self._agent_name
        return name if "/" in name else f"example/{name}"

    def _load_persona(self) -> str:
        cfg = self._load_config_json()
        desc = cfg.get("description", []) if cfg else []
        return "".join(desc) if isinstance(desc, list) else str(desc)

    def _tool_catalog_listing(self) -> list[dict[str, str]]:
        cfg = self._load_config_json()
        listing: list[dict[str, str]] = []
        for tool_ref in cfg.get("tools", []) if cfg else []:
            tool_name = tool_ref.split("/")[-1]
            listing.append({"name": tool_name})
        if self._attacker_tool.get("Attacker Tool"):
            listing.append(
                {
                    "name": self._attacker_tool["Attacker Tool"],
                    "description": self._attacker_tool.get("Description", ""),
                }
            )
        return listing

    def _load_config_json(self) -> dict[str, Any]:
        name = self._normalized_agent_name().split("/")[-1]
        path = os.path.join(_VENDOR_EXAMPLE_DIR, name, "config.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return data


def _content_to_text(content: Any) -> str:
    """Render an ASB final-result (a message dict or string) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        value = content.get("content", "")
        return value if isinstance(value, str) else json.dumps(value, default=str)
    return str(content)


__all__ = ["AsbTarget"]

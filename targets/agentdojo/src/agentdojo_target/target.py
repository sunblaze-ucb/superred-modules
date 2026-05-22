"""AgentDojoTarget: composite multi-suite target for superred.

Wires together every other module in this package: composite env,
wrapped runtime, tool catalog, controllables, observables, config and
query specs, pipeline bridge.

Lifecycle:

1. ``__init__``: build the canonical seed env + tool list once.
2. ``set_config``: Tasks set ``system_prompt``, ``user_prompt``, the
   per-suite ``seed_yaml_override__*`` slots, and optionally
   ``pipeline_model`` before each run.
3. ``run(emit, send_event)``: five-phase execution.

   - Phase 1: system_prompt controllable event; optimizer may override.
   - Phase 2: user_prompt controllable event; optimizer may override.
   - Phase 3: apply seed overlays and build a fresh per-run composite
     env + per-run tool catalog.
   - Phase 4: build wrapped runtime + pipeline (the catalog hook fires
     the four tool-catalog Controllables before every LLM turn) and run
     the pipeline with up to three retries, mirroring AgentDojo's outer
     retry loop.
   - Phase 5: emit final observables (composite env snapshot, agent
     trace tool-calls).

4. ``query``: post-run readers (last_response, function_call_trace,
   pre/post env snapshot, conversation_history, tool catalog snapshot,
   write_calls_made).
5. ``cleanup``: reset per-run state so the next call sees a fresh env
   and a seeded catalog.
6. ``teardown``: no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.task_suite import (
    model_output_from_messages as _upstream_model_output_from_messages,
)
from agentdojo.types import ChatMessage, MessageContentBlock
from pydantic import BaseModel
from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain
from superred.core.types.state import ConfigSpec, QuerySpec

from agentdojo_target.config_specs import (
    CONFIG_SPEC_NAMES,
    CONFIG_SPECS,
    PIPELINE_MODEL_SPEC,
    SEED_OVERRIDE_SPECS,
    SYSTEM_PROMPT_SPEC,
    USER_PROMPT_SPEC,
)
from agentdojo_target.controllables import (
    CONTROLLABLES,
    SYSTEM_PROMPT_CTRL,
    USER_PROMPT_CTRL,
)
from agentdojo_target.env import CompositeEnvironment, sync_initial_fields
from agentdojo_target.observables import (
    COMPOSITE_ENV_SNAPSHOT_OBS,
    MODEL_IDENTITY_OBS,
    SYSTEM_PROMPT_OBS,
    TOOL_CATALOG_LISTING_OBS,
    agent_tool_call_observable,
)
from agentdojo_target.pipeline_bridge import build_pipeline
from agentdojo_target.query_specs import QUERY_SPEC_NAMES, QUERY_SPECS
from agentdojo_target.runtime_wrapper import WrappedFunctionsRuntime
from agentdojo_target.security_tags import DOMAIN
from agentdojo_target.seed_loader import load_composite_seed, merge_yaml_overlay
from agentdojo_target.system_prompt import default_system_prompt
from agentdojo_target.tool_catalog import ToolCatalog
from agentdojo_target.tool_registry import (
    ALL_FUNCTIONS,
    SUITE_NAMES,
    WRITE_FUNCTION_NAMES,
)

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE_MODEL: str = "openai/gpt-4o-2024-05-13"


class AgentDojoTarget(Target):
    """Composite AgentDojo target exposing all four suites simultaneously.

    Args:
        pipeline_model: litellm-style model id for the underlying
            AgentDojo agent pipeline (e.g. ``openai/gpt-4o-2024-05-13``).
            Tasks may override per-run via the ``pipeline_model`` config
            slot.
        api_base: Optional API base URL (e.g. for a litellm proxy).
        api_key: Optional API key.  Falls back to the provider client's
            environment variables when unset.
    """

    def __init__(
        self,
        *,
        pipeline_model: str = _DEFAULT_PIPELINE_MODEL,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_base = api_base
        self._api_key = api_key
        # Per-task config (Tasks set these via set_config before run()).
        self._pipeline_model: str = pipeline_model
        self._system_prompt: str = default_system_prompt()
        self._user_prompt: str = ""
        self._seed_overrides: dict[str, str] = {s: "" for s in SUITE_NAMES}
        # Per-run state, populated by run().
        self._env: CompositeEnvironment | None = None
        self._pre_env: CompositeEnvironment | None = None
        self._catalog: ToolCatalog | None = None
        self._wrapped_runtime: WrappedFunctionsRuntime | None = None
        self._messages: Sequence[ChatMessage] = []
        self._last_response: str = ""
        self._function_call_trace: list[FunctionCall] = []

    # ------------------------------------------------------------------
    # Pre-run configuration
    # ------------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return list(CONFIG_SPECS)

    def set_config(self, name: str, value: str) -> None:
        if name not in CONFIG_SPEC_NAMES:
            raise ValueError(
                f"Unknown config slot {name!r}.  Expected one of "
                f"{sorted(CONFIG_SPEC_NAMES)}."
            )
        if name == SYSTEM_PROMPT_SPEC.name:
            self._system_prompt = value
        elif name == USER_PROMPT_SPEC.name:
            self._user_prompt = value
        elif name == PIPELINE_MODEL_SPEC.name:
            self._pipeline_model = value
        else:
            # Per-suite seed overlay.  The slot name is
            # ``seed_yaml_override__{suite}``.
            for spec in SEED_OVERRIDE_SPECS:
                if name == spec.name:
                    suite = name.removeprefix("seed_yaml_override__")
                    self._seed_overrides[suite] = value
                    return
            raise ValueError(  # pragma: no cover - guarded by CONFIG_SPEC_NAMES
                f"set_config slot {name!r} matched no known dispatch case"
            )

    # ------------------------------------------------------------------
    # Post-run queries
    # ------------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return list(QUERY_SPECS)

    def query(self, name: str, **params: str) -> str:
        if name not in QUERY_SPEC_NAMES:
            raise ValueError(
                f"Unknown query slot {name!r}.  Expected one of "
                f"{sorted(QUERY_SPEC_NAMES)}."
            )
        if name == "last_response":
            return self._last_response
        if name == "function_call_trace":
            return json.dumps(
                [_function_call_to_dict(fc) for fc in self._function_call_trace]
            )
        if name == "pre_environment_snapshot":
            return _dump_env_or_empty(self._pre_env)
        if name == "post_environment_snapshot":
            return _dump_env_or_empty(self._env)
        if name == "conversation_history":
            return json.dumps(_messages_to_jsonable(self._messages))
        if name == "tool_catalog_snapshot":
            if self._catalog is None:
                return json.dumps([])
            return json.dumps(self._catalog.snapshot())
        if name == "write_calls_made":
            writes = [
                fc.function for fc in self._function_call_trace
                if fc.function in WRITE_FUNCTION_NAMES
            ]
            return json.dumps(writes)
        raise ValueError(  # pragma: no cover
            f"query slot {name!r} matched no dispatch case"
        )

    # ------------------------------------------------------------------
    # Security domain + controllables + observables
    # ------------------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return DOMAIN

    def get_controllables(self) -> list[Controllable]:
        return list(CONTROLLABLES)

    def get_observables(self) -> list[ObservableValue]:
        """Static observables with current pre-run content snapshots.

        Returned at ``initialize()`` time so optimizers know about the
        model identity, current system prompt, seeded tool catalog, and
        full composite env *before* any tool call.  Out-of-scope items
        are filtered by the controller.
        """
        # Build a transient catalog + env snapshot to expose pre-run.
        seed_env = self._build_seed_env_with_overrides()
        seed_catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
        return [
            ObservableValue(observable=MODEL_IDENTITY_OBS, content=self._pipeline_model),
            ObservableValue(observable=SYSTEM_PROMPT_OBS, content=self._system_prompt),
            ObservableValue(
                observable=TOOL_CATALOG_LISTING_OBS,
                content=seed_catalog.snapshot(),
            ),
            ObservableValue(
                observable=COMPOSITE_ENV_SNAPSHOT_OBS,
                content=seed_env.model_dump(),
            ),
        ]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        emit: EventHandler,
        send_event: EventResponseHandler,
    ) -> None:
        loop = asyncio.get_running_loop()

        # --- Phase 1: system prompt ---
        sp_resp = await send_event(
            ControllablePreCallEvent(
                controllable=SYSTEM_PROMPT_CTRL,
                request=self._system_prompt,
            )
        )
        effective_system = (
            sp_resp.value if isinstance(sp_resp, ControllableInjection)
            else self._system_prompt
        )

        # --- Phase 2: user prompt ---
        up_resp = await send_event(
            ControllablePreCallEvent(
                controllable=USER_PROMPT_CTRL,
                request=self._user_prompt,
            )
        )
        effective_user = (
            up_resp.value if isinstance(up_resp, ControllableInjection)
            else self._user_prompt
        )

        # --- Phase 3: build per-run env + catalog ---
        self._env = self._build_seed_env_with_overrides()
        self._catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)

        # --- Phase 4: wrapped runtime + pipeline + 3-retry loop ---
        self._wrapped_runtime = WrappedFunctionsRuntime(
            catalog=self._catalog,
            send_event=send_event,
            emit=emit,
            loop=loop,
        )
        pipeline = build_pipeline(
            pipeline_model=self._pipeline_model,
            system_prompt=effective_system,
            catalog=self._catalog,
            wrapper=self._wrapped_runtime,
            send_event=send_event,
            emit=emit,
            loop=loop,
            api_base=self._api_base,
            api_key=self._api_key,
        )

        # Snapshot pre-environment AFTER phases 1+2 but BEFORE any tools run.
        self._pre_env = self._env.model_copy(deep=True)

        model_output: list[MessageContentBlock] | None = None
        for attempt in range(3):
            try:
                _q, _runtime, new_env, messages, _extra = await asyncio.to_thread(
                    pipeline.query,
                    effective_user,
                    self._wrapped_runtime,
                    self._env,
                )
            except Exception:  # pragma: no cover - the pipeline can raise
                logger.exception(
                    "AgentDojo pipeline.query raised on attempt %d", attempt + 1
                )
                break
            self._env = new_env
            self._messages = messages
            model_output = _model_output_from_messages(messages)
            if model_output is not None:
                break

        self._last_response = _content_blocks_to_text(model_output)
        # The wrapped runtime's trace mirrors AgentDojo's
        # functions_stack_trace_from_messages but is recorded eagerly so
        # it survives mid-run exceptions.
        self._function_call_trace = (
            self._wrapped_runtime.trace if self._wrapped_runtime is not None else []
        )

        # --- Phase 5: final observables ---
        emit(
            ObservableEvent(
                observable=COMPOSITE_ENV_SNAPSHOT_OBS,
                content=self._env.model_dump() if self._env is not None else {},
            )
        )
        for idx, fc in enumerate(self._function_call_trace):
            emit(
                ObservableEvent(
                    observable=agent_tool_call_observable(idx),
                    content=_function_call_to_dict(fc),
                )
            )

    # ------------------------------------------------------------------
    # Cleanup / teardown
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Reset per-run state so the next run starts from a clean baseline.

        Per-task config slots (system_prompt, user_prompt, seed
        overrides) are intentionally NOT reset; the controller may
        invoke ``run`` multiple times against the same configured Task
        within a single ``Controller.run`` invocation (multi-run loop),
        and the Task only calls ``configure_target`` once per task.
        """
        self._env = None
        self._pre_env = None
        self._catalog = None
        self._wrapped_runtime = None
        self._messages = []
        self._last_response = ""
        self._function_call_trace = []

    async def teardown(self) -> None:
        # No long-lived resources to release.
        pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_seed_env_with_overrides(self) -> CompositeEnvironment:
        """Construct the per-run composite env, applying any non-empty
        ``seed_yaml_override__{suite}`` overlays in stable order."""
        env = load_composite_seed()
        for suite in SUITE_NAMES:
            overlay = self._seed_overrides.get(suite, "")
            if overlay:
                env = merge_yaml_overlay(env, suite, overlay)
        return env


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _function_call_to_dict(fc: FunctionCall) -> dict[str, Any]:
    """Serialise a FunctionCall to a JSON-friendly dict."""
    return {
        "function": fc.function,
        "args": dict(fc.args),
        "id": fc.id,
        "placeholder_args": dict(fc.placeholder_args) if fc.placeholder_args else None,
    }


def _dump_env_or_empty(env: CompositeEnvironment | None) -> str:
    """Serialise *env* to a JSON string suitable for a round-trip through
    :meth:`CompositeEnvironment.model_validate`.

    AgentDojo's :class:`Inbox`, :class:`Calendar`, and :class:`CloudDrive`
    rebuild their dict fields (``emails`` / ``events`` / ``files``) from
    matching ``initial_*`` lists in a pydantic ``@model_validator``.  An
    in-memory mutation (the agent deletes an email, cancels an event,
    creates a file) is therefore lost when the env round-trips through
    JSON unless we first sync the ``initial_*`` lists from the live
    dicts.  :func:`sync_initial_fields` does that in-place before we
    dump.  See ASSUMPTIONS.md §C.4.
    """
    if env is None:
        return json.dumps({})
    sync_initial_fields(env)
    return env.model_dump_json()


def _messages_to_jsonable(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage TypedDicts into JSON-friendly dicts.

    ChatMessage is a TypedDict union so it serialises directly via
    ``json.dumps``; the only complication is FunctionCall nested under
    assistant ``tool_calls``, which is a pydantic model.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = dict(msg)
        tool_calls = msg_dict.get("tool_calls")
        if isinstance(tool_calls, list):
            msg_dict["tool_calls"] = [
                _function_call_to_dict(c) if isinstance(c, FunctionCall) else c
                for c in tool_calls
            ]
        tool_call = msg_dict.get("tool_call")
        if isinstance(tool_call, FunctionCall):
            msg_dict["tool_call"] = _function_call_to_dict(tool_call)
        out.append(msg_dict)
    return out


def _model_output_from_messages(
    messages: Sequence[ChatMessage],
) -> list[MessageContentBlock] | None:
    """Extract the final assistant content blocks, matching AgentDojo's
    ``model_output_from_messages`` shape exactly.

    Upstream raises ``ValueError`` when the last message is not an
    assistant message; we mirror that and treat the case as "no output
    yet" by returning ``None`` so the outer 3-retry loop can re-query.
    Returning the raw content list (rather than a joined string)
    preserves the upstream contract for thinking-capable models that may
    produce only thinking blocks before an answer.
    """
    if not messages:
        return None
    try:
        return _upstream_model_output_from_messages(messages)
    except ValueError:
        return None


def _content_blocks_to_text(
    content: list[MessageContentBlock] | str | None,
) -> str:
    """Render content (block list or plain string) into a flat string
    for ``Target.query("last_response")``.  Thinking blocks are skipped
    so an optimizer querying the post-run response gets only the
    user-facing text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            value = block.get("content", "")
            if isinstance(value, str):
                texts.append(value)
    return "".join(texts)


__all__ = ["AgentDojoTarget"]

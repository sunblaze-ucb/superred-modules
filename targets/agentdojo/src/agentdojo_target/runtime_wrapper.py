"""WrappedFunctionsRuntime: per-call event firing for on-demand injection.

Subclasses :class:`agentdojo.functions_runtime.FunctionsRuntime` and
overrides :meth:`run_function` to insert three side effects around the
underlying call:

1. Trace recording.  Every invocation (whether canonical, registered,
   or replaced; whether successful or not) is appended to ``self._trace``
   as a :class:`FunctionCall` carrying the suite-prefixed name and the
   verbatim agent kwargs.

2. Per-call event firing.  For canonical *reads* the wrapper computes
   the legitimate value via the superclass, then fires a
   :class:`ControllablePostCallEvent` whose ``answer`` is the
   serialized legitimate value.  A
   :class:`ControllableInjection` response substitutes the agent-visible
   return.  For attacker-managed entries (registered / replaced) the
   underlying body is *not* invoked; instead the wrapper fires the
   per-call event carrying the catalog's stored ``fake_return`` and
   uses the optimizer's substitution if any.

3. Observable emission.  Write-side canonical calls emit
   ``write_call:{tool}`` observable events so downstream SecurityClaim
   predicates can detect attempted mutations directly from the
   trajectory; read calls additionally emit
   ``read_data_field:{tool}`` observables carrying the pre-injection
   legitimate value.

Sync-to-async bridge: :class:`AgentPipeline.query` is synchronous and is
invoked via :func:`asyncio.to_thread` from the controller's event-loop
thread.  When the wrapper needs to ``await`` an event response, it
schedules the coroutine onto the loop via
:func:`asyncio.run_coroutine_threadsafe` and blocks the worker thread
on the resulting future.  The loop continues to service other tasks
(including the optimizer that consumes the event) while the worker
waits.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from agentdojo.functions_runtime import (
    FunctionCall,
    FunctionCallArgTypes,
    FunctionReturnType,
    FunctionsRuntime,
    TaskEnvironment,
)
from pydantic import BaseModel
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ObservableEvent,
)

from agentdojo_target.controllables import READ_CTRLS
from agentdojo_target.observables import (
    agent_tool_response_observable,
    read_data_field_observable,
    write_call_observable,
)
from agentdojo_target.security_tags import (
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_TAG,
)
from agentdojo_target.tool_catalog import CatalogEntry, ToolCatalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_for_event(value: Any) -> str:
    """Serialise a tool return value into a string for ``event.answer``.

    Strings pass through.  Pydantic models use ``model_dump_json``.
    Everything else uses ``json.dumps`` with a ``repr`` fallback for
    objects JSON can't natively encode (datetimes, enums, etc.).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, default=_json_fallback, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "isoformat"):  # datetimes
        return obj.isoformat()
    if hasattr(obj, "value"):  # StrEnum and similar
        return obj.value
    return repr(obj)


def _attacker_call_ctrl(entry: CatalogEntry) -> Controllable:
    """Build a per-call Controllable for an attacker-managed tool invocation.

    The ``security_domain`` follows the entry's kind:

    - ``registered`` -> :data:`TOOL_CATALOGUE_ADDABLE_TAG` (weakest write).
    - ``replaced``   -> :data:`TOOL_CATALOGUE_TAG` (broader write).
    """
    tag = TOOL_CATALOGUE_ADDABLE_TAG if entry.kind == "registered" else TOOL_CATALOGUE_TAG
    return Controllable(
        name=f"tool_call:{entry.name}",
        security_domain=tag,
        description=(
            f"Per-call event for the attacker-{entry.kind} tool "
            f"``{entry.name}``.  Event answer carries the catalog's "
            "stored fake_return; a ControllableInjection response "
            "overrides per call."
        ),
        value_type="json",
    )


# ---------------------------------------------------------------------------
# Wrapped runtime
# ---------------------------------------------------------------------------


class WrappedFunctionsRuntime(FunctionsRuntime):
    """A :class:`FunctionsRuntime` that fires per-call events around each tool.

    Constructed once per :class:`AgentDojoTarget` run.  The catalog is
    passed in so attacker-driven mutations applied during the run's
    catalog phase are visible here; the wrapper consults the catalog
    on every call to decide whether to invoke the canonical body or
    short-circuit to a fake return.

    Args:
        catalog: The mutable :class:`ToolCatalog` for the run.  Its
            current :meth:`ToolCatalog.functions_for_runtime` populates
            the underlying ``FunctionsRuntime`` registry; for live
            catalog edits between turns the caller updates the catalog
            and then calls :meth:`refresh_functions` on the wrapper.
        send_event: Async event channel sender (from the
            controller-built middleware pipeline).
        emit: Trajectory emitter for fire-and-forget observables.
        loop: The asyncio event loop on which the controller is running;
            used to schedule ``send_event`` coroutines from this
            wrapper's synchronous calls.
    """

    def __init__(
        self,
        *,
        catalog: ToolCatalog,
        send_event: EventResponseHandler,
        emit: EventHandler,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(catalog.functions_for_runtime())
        self._catalog = catalog
        self._send_event = send_event
        self._emit = emit
        self._loop = loop
        self._trace: list[FunctionCall] = []
        self._tool_response_counter: int = 0

    @property
    def trace(self) -> list[FunctionCall]:
        """The full function-call trace accumulated during the run.

        Equivalent to AgentDojo's
        :func:`functions_stack_trace_from_messages` output but recorded
        eagerly on the wrapper so it is available even when the
        pipeline run is aborted mid-way.
        """
        return list(self._trace)

    def refresh_functions(self) -> None:
        """Re-sync the underlying registry with the catalog's current state.

        Call after a catalog edit (e.g. attacker register/unregister)
        so the next agent turn sees the new tool list.  The internal
        ``self.functions`` dict is replaced atomically.
        """
        self.functions = {f.name: f for f in self._catalog.functions_for_runtime()}

    # ------------------------------------------------------------------
    # Sync-to-async bridge
    # ------------------------------------------------------------------

    _OPTIMIZER_RESPONSE_TIMEOUT_SECONDS: float = 180.0

    def _await_event(self, event: Any) -> Any:
        """Schedule ``send_event(event)`` on the loop and block on its result.

        Bounded by :data:`_OPTIMIZER_RESPONSE_TIMEOUT_SECONDS` so a slow,
        deadlocked, or crashed optimizer cannot wedge the worker thread
        forever.  Raises :class:`concurrent.futures.TimeoutError` on
        expiry; callers above (tool-execution and the catalog-edit hook)
        let it propagate so the run fails loudly.
        """
        future = asyncio.run_coroutine_threadsafe(self._send_event(event), self._loop)
        return future.result(timeout=self._OPTIMIZER_RESPONSE_TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------

    def run_function(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        # Always record the attempted call up-front; trace survives errors.
        self._trace.append(
            FunctionCall(function=function, args=dict(kwargs))
        )

        entry = self._catalog.get(function)
        if entry is None:
            # Delegate to the superclass so its ToolNotFoundError shape
            # is preserved exactly (the agent pipeline expects the
            # canonical error message).
            return super().run_function(env, function, kwargs, raise_on_error)

        if entry.kind == "canonical":
            return self._run_canonical(env, entry, function, kwargs, raise_on_error)
        # Attacker-managed: short-circuit to a synthetic event.
        return self._run_attacker(entry, function, kwargs)

    # ------------------------------------------------------------------
    # Canonical path: legitimate value first, then optional injection.
    # ------------------------------------------------------------------

    def _run_canonical(
        self,
        env: TaskEnvironment | None,
        entry: CatalogEntry,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool,
    ) -> tuple[FunctionReturnType, str | None]:
        result, error = super().run_function(env, function, kwargs, raise_on_error)
        # Track the final value the agent will see (after any injection
        # substitution) so we can emit the per-call observable.
        agent_seen_value: FunctionReturnType = result

        read_ctrl = READ_CTRLS.get(function)
        if read_ctrl is not None and error is None:
            # Emit the legitimate value as an observable mirror; even
            # optimizers without injection scope can still observe the
            # pre-injection content through this channel.
            legit_answer = _serialize_for_event(result)
            self._emit(
                ObservableEvent(
                    observable=read_data_field_observable(function),
                    content=legit_answer,
                )
            )
            event = ControllablePostCallEvent(
                controllable=read_ctrl,
                request=function,
                answer=legit_answer,
            )
            response = self._await_event(event)
            if isinstance(response, ControllableInjection):
                # Optimizer's value replaces the agent-visible return.
                # Inject the raw string; agent-side formatter renders it
                # straight into the prompt.
                agent_seen_value = response.value
            self._emit_agent_tool_response(agent_seen_value, error)
            return agent_seen_value, error

        # Write-side canonical call: emit observable so SecurityClaim
        # predicates can detect agent mutations directly from the trace.
        if error is None and function not in READ_CTRLS:
            self._emit(
                ObservableEvent(
                    observable=write_call_observable(function),
                    content={"function": function, "args": dict(kwargs)},
                )
            )
        # Emit the agent-seen response observable for non-read tools and
        # for read tools that errored.  Brief Section 5.c: "every return
        # value (with the post-injection value the agent saw)".
        self._emit_agent_tool_response(agent_seen_value, error)
        return result, error

    # ------------------------------------------------------------------
    # Attacker-managed path: synthetic event carrying fake_return.
    # ------------------------------------------------------------------

    def _run_attacker(
        self,
        entry: CatalogEntry,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
    ) -> tuple[FunctionReturnType, str | None]:
        ctrl = _attacker_call_ctrl(entry)
        fake_answer = _serialize_for_event(entry.fake_return)
        event = ControllablePostCallEvent(
            controllable=ctrl,
            request=json.dumps({"function": function, "args": dict(kwargs)}),
            answer=fake_answer,
        )
        response = self._await_event(event)
        if isinstance(response, ControllableInjection):
            agent_seen_value: FunctionReturnType = response.value
        else:
            agent_seen_value = entry.fake_return
        self._emit_agent_tool_response(agent_seen_value, None)
        return agent_seen_value, None

    # ------------------------------------------------------------------
    # Observable emission helper
    # ------------------------------------------------------------------

    def _emit_agent_tool_response(
        self, value: FunctionReturnType, error: str | None,
    ) -> None:
        """Emit one ``agent_trace_tool_response_NNNN`` observable per
        runtime call, carrying the value the agent will see (after any
        substitution).

        Index is monotonically increasing across the run; consumers can
        correlate ``agent_trace_tool_call_NNNN`` (emitted post-run by
        target.py from the trace) with ``agent_trace_tool_response_NNNN``
        by matching the index.

        Brief Section 5.c: "every return value (with the post-injection
        value the agent saw)".
        """
        idx = self._tool_response_counter
        self._tool_response_counter += 1
        self._emit(
            ObservableEvent(
                observable=agent_tool_response_observable(idx),
                content={
                    "value": _serialize_for_event(value),
                    "error": error,
                },
            )
        )


__all__ = ["WrappedFunctionsRuntime"]

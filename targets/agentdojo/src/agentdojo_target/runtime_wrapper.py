"""WrappedFunctionsRuntime: per-call event firing for on-demand injection.

Subclasses :class:`agentdojo.functions_runtime.FunctionsRuntime` and
overrides :meth:`run_function` to insert three side effects around the
underlying call:

1. Trace recording + live call emission.  Every invocation (whether
   canonical, registered, or replaced; whether successful or not) is
   appended to ``self._trace`` as a :class:`FunctionCall` carrying the
   suite-prefixed name and the verbatim agent kwargs, and emitted
   immediately as an ``agent_trace_tool_call_NNNN`` observable so
   trace-scoped optimizers see calls as they happen, not only at run
   end.

2. Per-call event firing.  For canonical *reads* the wrapper computes
   the legitimate value via the superclass, then fires a
   :class:`ControllablePostCallEvent` whose ``answer`` is the
   serialized legitimate value.  A
   :class:`ControllableInjection` response substitutes the agent-visible
   return.  For attacker-managed entries (registered / replaced) the
   underlying body is *not* invoked; instead the wrapper fires the
   per-call event carrying the catalog's stored ``fake_return`` and
   uses the optimizer's substitution if any.

3. Observable emission.  Every runtime call emits one
   ``agent_trace_tool_response_NNNN`` observable carrying the value the
   agent actually saw (post-injection).  The pre-injection legitimate
   value is carried exactly once, on the per-read
   ``ControllablePostCallEvent`` — there is no observable mirror; a
   Controller that lists the quadrant tag under ``read_only`` reads it
   from the trajectory without being able to inject.

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

from agentdojo_target.controllables import READ_CTRLS, WRITE_STORE_MAP
from agentdojo_target.observables import (
    agent_tool_call_observable,
    agent_tool_response_observable,
    tool_menu_rebuild_observable,
    write_observation_observable,
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


def function_call_to_jsonable(fc: FunctionCall) -> dict[str, Any]:
    """Serialise a FunctionCall to a JSON-friendly dict."""
    return {
        "function": fc.function,
        "args": dict(fc.args),
        "id": fc.id,
        "placeholder_args": dict(fc.placeholder_args) if fc.placeholder_args else None,
    }


def _attacker_call_ctrl(entry: CatalogEntry) -> Controllable:
    """Build a per-call Controllable for an attacker-managed tool invocation.

    The ``security_domain`` follows the entry's kind:

    - ``registered`` -> :data:`TOOL_CATALOGUE_ADDABLE_TAG` (weakest write).
    - ``replaced``   -> :data:`TOOL_CATALOGUE_TAG` (broader write).
    """
    tag = (
        TOOL_CATALOGUE_ADDABLE_TAG if entry.kind == "registered" else TOOL_CATALOGUE_TAG
    )
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
            the underlying ``FunctionsRuntime`` registry; to apply a
            catalog edit during a run the caller updates the catalog
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
        self._write_observation_counter: int = 0
        self._menu_rebuild_counter: int = 0

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

        After the swap, emits a one-way ``tool_menu_rebuild_NNNN``
        observable so a trace-scoped reader sees the menu changed.  The
        emit is fire-and-forget and happens after the assignment, so it
        cannot perturb ``self.functions``.
        """
        self.functions = {f.name: f for f in self._catalog.functions_for_runtime()}
        self._emit(
            ObservableEvent(
                observable=tool_menu_rebuild_observable(self._menu_rebuild_counter),
                content={
                    "tool_names": sorted(self.functions),
                    "count": len(self.functions),
                },
            )
        )
        self._menu_rebuild_counter += 1

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
        call = FunctionCall(function=function, args=dict(kwargs))
        self._trace.append(call)
        # Live call observable: one per runtime call (reads, writes,
        # attacker-managed, and unknown tools alike), emitted at
        # invocation so trace-scoped optimizers can react mid-run.  The
        # index follows trace order and correlates with
        # ``agent_trace_tool_response_NNNN``.
        self._emit(
            ObservableEvent(
                observable=agent_tool_call_observable(len(self._trace) - 1),
                content=function_call_to_jsonable(call),
            )
        )

        entry = self._catalog.get(function)
        if entry is None:
            # Delegate to the superclass so its ToolNotFoundError shape
            # is preserved exactly (the agent pipeline expects the
            # canonical error message).
            return super().run_function(env, function, kwargs, raise_on_error)

        if entry.kind == "canonical":
            return self._run_canonical(
                env, entry, function, kwargs, raise_on_error, call
            )
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
        call: FunctionCall,
    ) -> tuple[FunctionReturnType, str | None]:
        result, error = super().run_function(env, function, kwargs, raise_on_error)
        # Track the final value the agent will see (after any injection
        # substitution) so we can emit the per-call observable.
        agent_seen_value: FunctionReturnType = result

        read_ctrl = READ_CTRLS.get(function)
        if read_ctrl is not None and error is None:
            # The legitimate value is carried once, on the post-call
            # event; a Controller that lists the quadrant tag under
            # ``read_only`` observes it from the trajectory without
            # being able to inject.
            legit_answer = _serialize_for_event(result)
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
            self._emit_agent_tool_response(function, agent_seen_value, error)
            return agent_seen_value, error

        # Write-side canonical calls and errored reads: the call itself
        # was already emitted live as agent_trace_tool_call_NNNN at
        # invocation (SecurityClaim predicates use the write_calls_made
        # query for mutation detection).
        self._emit_agent_tool_response(function, agent_seen_value, error)
        # For a successful write, also emit a one-way observation tagged at
        # the store the write mutates, so a service-scoped attacker sees the
        # action it provoked under the same boundary it reads from (reading
        # and acting on a store share a label).  Fire-and-forget after the
        # response observable; never gates the call or alters its result.
        store_tag = WRITE_STORE_MAP.get(function)
        if store_tag is not None and error is None:
            self._emit(
                ObservableEvent(
                    observable=write_observation_observable(
                        self._write_observation_counter, store_tag
                    ),
                    content=function_call_to_jsonable(call),
                )
            )
            self._write_observation_counter += 1
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
        self._emit_agent_tool_response(function, agent_seen_value, None)
        return agent_seen_value, None

    # ------------------------------------------------------------------
    # Observable emission helper
    # ------------------------------------------------------------------

    def _emit_agent_tool_response(
        self,
        function: str,
        value: FunctionReturnType,
        error: str | None,
    ) -> None:
        """Emit one ``agent_trace_tool_response_NNNN`` observable per
        runtime call, carrying the value the agent will see (after any
        substitution).

        The record also carries ``function`` (the suite-prefixed tool name
        that produced the value) so a single response record is
        self-identifying: a consumer can tell which tool produced an output
        without cross-referencing the ``agent_trace_tool_call_NNNN``
        observable by index.  The index correlation is preserved as well;
        ``function`` is additive.

        Index is monotonically increasing across the run; consumers can
        correlate ``agent_trace_tool_call_NNNN`` (emitted live per call by
        :meth:`run_function`) with ``agent_trace_tool_response_NNNN`` by
        matching the index.

        Brief Section 5.c: "every return value (with the post-injection
        value the agent saw)".
        """
        idx = self._tool_response_counter
        self._tool_response_counter += 1
        self._emit(
            ObservableEvent(
                observable=agent_tool_response_observable(idx),
                content={
                    "function": function,
                    "value": _serialize_for_event(value),
                    "error": error,
                },
            )
        )


__all__ = ["WrappedFunctionsRuntime", "function_call_to_jsonable"]

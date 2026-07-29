"""Model-aware replay optimizer for Pliny's L1B3RT4S prompt corpus."""

from __future__ import annotations

from collections.abc import Iterable

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from libertas_optimizer.corpus import (
    PromptTemplate,
    detect_provider,
    load_prompt_templates,
    render_prompt,
)

_SYSTEM_PROMPT_NAME = "system_prompt"
_MODEL_OBSERVABLE_NAMES = frozenset(
    {"model", "model_id", "model_identity", "target_model", "victim_model"}
)
_USER_CONTROLLABLE_NAMES = frozenset(
    {
        "instruction",
        "prompt",
        "query",
        "user_input",
        "user_message",
        "user_prompt",
        "user_query",
    }
)


class LibertasOptimizer(Optimizer):
    """Replay byte-faithful L1B3RT4S templates, one template per run.

    By default only upstream sections containing an explicit generic goal
    marker are used.  The target's model identity selects the matching vendor
    file family; if no identity can be inferred, all vendor families are tried
    in deterministic source order.

    No attacker LLM or internal judge is used.  The SecurityClaim's
    ``RunEndEvent.evaluation.success`` is authoritative.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        max_attempts: int | None = None,
        include_untemplated: bool = False,
        include_system_templates: bool = False,
        source_files: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
    ) -> None:
        super().__init__()
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if provider is not None and not provider.strip():
            raise ValueError("provider must not be empty")
        if target_controllable_name is not None and not target_controllable_name:
            raise ValueError("target_controllable_name must not be empty")

        self._provider_override = provider.strip().lower() if provider is not None else None
        self._max_attempts = max_attempts
        self._include_untemplated = include_untemplated
        self._include_system_templates = include_system_templates
        self._source_files = tuple(source_files) if source_files is not None else None
        self._target_controllable_name_override = target_controllable_name

        self._goal: Goal | None = None
        self._resolved_provider: str | None = None
        self._templates: tuple[PromptTemplate, ...] = ()
        self._template_index = 0
        self._current: PromptTemplate | None = None
        self._user_controllable_name: str | None = None
        self._has_system_controllable = False
        self._injected_primary = False
        self._injected_system_trigger = False
        self._succeeded = False

    @property
    def resolved_provider(self) -> str | None:
        """Provider selected from the override or target model observable."""

        return self._resolved_provider

    @property
    def templates(self) -> tuple[PromptTemplate, ...]:
        """The deterministic task-local attack schedule."""

        return self._templates

    @property
    def current_template(self) -> PromptTemplate | None:
        """Template selected for the active run."""

        return self._current

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._resolved_provider = self._provider_override or self._provider_from_observables(
            observables
        )
        self._user_controllable_name = self._resolve_user_controllable(controllables)
        self._has_system_controllable = any(
            controllable.name == _SYSTEM_PROMPT_NAME for controllable in controllables
        )

        templates = load_prompt_templates(
            provider=self._resolved_provider,
            include_untemplated=self._include_untemplated,
            include_system_templates=self._include_system_templates,
            source_files=self._source_files,
        )
        templates = tuple(
            template
            for template in templates
            if self._template_has_delivery_surface(template)
        )
        if self._max_attempts is not None:
            templates = templates[: self._max_attempts]
        if not templates:
            provider_text = self._resolved_provider or "all providers"
            raise ValueError(
                "no runnable L1B3RT4S templates for "
                f"{provider_text}; check scope, source_files, and template options"
            )

        self._templates = templates
        self._template_index = 0
        self._current = None
        self._succeeded = False
        self._reset_run_state()

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._reset_run_state()
            self._current = (
                self._templates[self._template_index]
                if self._template_index < len(self._templates)
                else None
            )
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if isinstance(event, RunEndEvent):
            if event.evaluation is not None and event.evaluation.success:
                self._succeeded = True
            self._template_index += 1
            done = (
                self._succeeded
                or self._template_index >= len(self._templates)
                or self._current is None
            )
            return RunEndResponse(event=event, done=done)

        return EventResponse(event=event)

    def _handle_pre_call(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        if self._current is None or self._goal is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._target_controllable_name_override is not None:
            if event.controllable.name != self._target_controllable_name_override:
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if self._injected_primary:
                return ControllableNoInjection(event=event, controllable=event.controllable)
            self._injected_primary = True
            return self._render_injection(event)

        if self._current.delivery == "system":
            if event.controllable.name == _SYSTEM_PROMPT_NAME:
                if self._injected_primary:
                    return ControllableNoInjection(
                        event=event, controllable=event.controllable
                    )
                self._injected_primary = True
                return self._render_injection(event)

            # A ChatbotTarget needs a user turn after the custom/system
            # instruction is installed.  Upstream's UI/custom-instruction
            # workflow likewise has a subsequent user query.
            if (
                event.controllable.name == self._user_controllable_name
                and self._injected_primary
                and not self._injected_system_trigger
            ):
                self._injected_system_trigger = True
                return ControllableInjection(
                    event=event,
                    controllable=event.controllable,
                    value=self._goal.description,
                )

            return ControllableNoInjection(event=event, controllable=event.controllable)

        if event.controllable.name != self._user_controllable_name:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if self._injected_primary:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        self._injected_primary = True
        return self._render_injection(event)

    def _render_injection(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection:
        assert self._current is not None
        assert self._goal is not None
        value = render_prompt(
            self._current,
            self._goal.description,
            append_untemplated=self._include_untemplated,
        )
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=value,
        )

    def _template_has_delivery_surface(self, template: PromptTemplate) -> bool:
        if self._target_controllable_name_override is not None:
            return True
        if template.delivery == "system":
            return self._has_system_controllable and self._user_controllable_name is not None
        return self._user_controllable_name is not None

    @staticmethod
    def _provider_from_observables(observables: list[ObservableValue]) -> str | None:
        # Prefer explicitly model-named observables, then fall back to any
        # string-valued observable.  Targets are not required to use one naming
        # convention, but model IDs are distinctive enough for a conservative
        # provider detector.
        ordered = sorted(
            observables,
            key=lambda item: item.observable.name.casefold() not in _MODEL_OBSERVABLE_NAMES,
        )
        for observable_value in ordered:
            content = observable_value.content
            if not isinstance(content, str):
                continue
            provider = detect_provider(content)
            if provider is not None:
                return provider
        return None

    def _resolve_user_controllable(
        self,
        controllables: list[Controllable],
    ) -> str | None:
        if self._target_controllable_name_override is not None:
            available = {controllable.name for controllable in controllables}
            if self._target_controllable_name_override not in available:
                raise ValueError(
                    f"target controllable {self._target_controllable_name_override!r} "
                    f"is unavailable; got {sorted(available)}"
                )
            return self._target_controllable_name_override

        for controllable in controllables:
            if controllable.name.casefold() in _USER_CONTROLLABLE_NAMES:
                return controllable.name
        remaining = [
            controllable.name
            for controllable in controllables
            if controllable.name != _SYSTEM_PROMPT_NAME
        ]
        if len(remaining) == 1:
            return remaining[0]
        return None

    def _reset_run_state(self) -> None:
        self._injected_primary = False
        self._injected_system_trigger = False

    async def teardown(self) -> None:
        return None

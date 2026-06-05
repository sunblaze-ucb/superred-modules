"""FilterTestTarget: deterministic target with controllables at three security domains.

No LLM calls. Three controllables (alpha, beta, root), three observables
with hints telling the optimizer what to inject. The optimizer can only
succeed for controllables whose hints are visible within its scope.

Used to validate that all five security domain filters work correctly:
controllables, observables, events, trajectory entries, and feedback sub_scores.
"""

from __future__ import annotations

import json

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag
from superred.core.types.state import ConfigSpec, QuerySpec

# Security domain tree: root -> alpha, beta
ROOT_TAG = SecurityDomainTag("root")
ALPHA_TAG = SecurityDomainTag("alpha", parent=ROOT_TAG)
BETA_TAG = SecurityDomainTag("beta", parent=ROOT_TAG)
_DOMAIN = SecurityDomain([ROOT_TAG, ALPHA_TAG, BETA_TAG])

# The correct keys each controllable must receive
EXPECTED_KEYS = {
    "alpha_input": "ALPHA_KEY",
    "beta_input": "BETA_KEY",
    "root_input": "ROOT_KEY",
}

# Map controllable name -> security domain tag
_CTRL_DOMAINS = {
    "alpha_input": ALPHA_TAG,
    "beta_input": BETA_TAG,
    "root_input": ROOT_TAG,
}


class FilterTestTarget(Target):
    """Deterministic target with three controllables at different domains.

    Each controllable expects a specific key. The correct key is conveyed
    via an observable hint at the same security domain. The optimizer can
    only succeed for controllables whose hints are visible within its scope.
    """

    def __init__(self) -> None:
        self._scenario: str = ""
        self._injected: dict[str, str] = {}

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(name="scenario", security_domain=ROOT_TAG, description="Scenario ID"),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "scenario":
            self._scenario = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="results",
                description="JSON: {controllable_name: {injected, expected, correct}}",
            ),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "results":
            results = {}
            for ctrl_name, expected in EXPECTED_KEYS.items():
                injected = self._injected.get(ctrl_name, "")
                results[ctrl_name] = {
                    "injected": injected,
                    "expected": expected,
                    "correct": injected == expected,
                }
            return json.dumps(results)
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables --------------------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [
            Controllable(
                name=name, security_domain=domain,
                description=f"Expects '{EXPECTED_KEYS[name]}'",
            )
            for name, domain in _CTRL_DOMAINS.items()
        ]

    # -- Observables (hints) --------------------------------------------------

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="alpha_hint", security_domain=ALPHA_TAG,
                    description="Hint for alpha_input",
                ),
                content="inject ALPHA_KEY into alpha_input",
            ),
            ObservableValue(
                observable=Observable(
                    name="beta_hint", security_domain=BETA_TAG,
                    description="Hint for beta_input",
                ),
                content="inject BETA_KEY into beta_input",
            ),
            ObservableValue(
                observable=Observable(
                    name="root_hint", security_domain=ROOT_TAG,
                    description="Hint for root_input",
                ),
                content="inject ROOT_KEY into root_input",
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._injected.clear()

        for ctrl_name, domain in _CTRL_DOMAINS.items():
            ctrl = Controllable(
                name=ctrl_name, security_domain=domain,
            )
            resp = await send_event(
                ControllablePreCallEvent(controllable=ctrl, request=f"key for {ctrl_name}"),
            )
            value = resp.value if isinstance(resp, ControllableInjection) else ""
            self._injected[ctrl_name] = value

            # Emit trajectory entry at the controllable's domain
            emit(ObservableEvent(
                observable=Observable(name=ctrl_name, security_domain=domain),
                content=f"{ctrl_name}={value}",
            ))

    async def reset_ephemeral_state(self) -> None:
        self._injected.clear()

    async def teardown(self) -> None:
        pass

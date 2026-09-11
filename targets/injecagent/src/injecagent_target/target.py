"""InjecAgentTarget: a superred Target driving the agent-under-test through one
InjecAgent test case.

The claim (``injecagent_claim``) hands the verbatim test case via
``set_config("case", ...)`` plus its ``attack`` (dh/ds) and ``setting``. ``run()``
optionally lets an optimizer vary the injected ``attacker_instruction`` (rebuilt
into the poisoned tool observation) or the benign ``user_instruction``, builds the
ReAct / function-calling prompt, calls the model once (twice for a succeeded ds
case), and captures the outputs the claim's pure judge reads back via
``query("result")``.

The environment is fully simulated (no Docker): the poisoned tool observation is
baked into the test case, and the ds step-2 exfiltration response is served
cache-first from the vendored cache (LLM fallback only on a genuine miss).

Two evaluation modes (upstream's two runners) are supported:

* ``prompted`` — text ReAct (the paper's headline path);
* ``finetuned`` — native function-calling.

The ds step-2 gate reuses the claim's evaluator (``evaluate_output_prompted`` /
``evaluate_output_finetuned``), so the target's "did step 1 succeed?" decision and
the claim's final verdict can never disagree.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from injecagent_claim.judge import evaluate_output_finetuned, evaluate_output_prompted
from injecagent_claim.security_tags import (
    DOMAIN,
    EXTERNAL_DATA_TAG,
    SYSTEM_TAG,
    USER_TAG,
)
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

from injecagent_target import agent as _agent
from injecagent_target import simulator as _sim

# -- Controllables (injection surfaces an optimizer may drive) ---------------
ATTACKER_INSTRUCTION_CTRL = Controllable(
    name="attacker_instruction",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "The attacker instruction embedded in the tool's returned content (the "
        "indirect-prompt-injection surface). Injecting overrides the task's "
        "shipped instruction and is re-substituted into the poisoned tool "
        "observation; left un-injected, the benchmark's own injection runs."
    ),
)
USER_INSTRUCTION_CTRL = Controllable(
    name="user_instruction",
    security_domain=USER_TAG,
    description=(
        "The benign user turn the agent is trying to help with. Injecting "
        "overrides the task's user instruction; the adversarial payload stays in "
        "the tool observation either way."
    ),
)

# -- Observables (static context, read at optimizer init) --------------------
MODEL_IDENTITY_OBS = Observable(
    name="model_identity",
    security_domain=SYSTEM_TAG,
    description="Backing model id (agent under test).",
)
ATTACK_TYPE_OBS = Observable(
    name="attack_type",
    security_domain=SYSTEM_TAG,
    description="Attack type: dh (direct harm) or ds (data stealing).",
)
SETTING_OBS = Observable(
    name="setting", security_domain=SYSTEM_TAG, description="Injection setting: base or enhanced."
)
PROMPT_TYPE_OBS = Observable(
    name="prompt_type",
    security_domain=SYSTEM_TAG,
    description="Agent prompt type: InjecAgent or hwchase17_react.",
)
MODE_OBS = Observable(
    name="mode", security_domain=SYSTEM_TAG, description="Evaluation mode: prompted or finetuned."
)

_PROMPT_TYPES = ("InjecAgent", "hwchase17_react")
_MODES = ("prompted", "finetuned")
# Upstream simulates a ds step-2 exfiltration response with gpt-4-0613 on a cache
# miss (src/utils.py:150); default to the same for faithfulness. Override via
# ``sim_model`` (e.g. to reuse the agent's model when OpenAI is unavailable).
_DEFAULT_SIM_MODEL = "gpt-4-0613"


class InjecAgentTarget(Target):
    """Drives the agent-under-test through one InjecAgent test case.

    Args:
        model: litellm-style model id of the agent under test.
        api_base, api_key: optional litellm routing for ``model``.
        prompt_type: ``"InjecAgent"`` (default) or ``"hwchase17_react"``.
        mode: ``"prompted"`` (text ReAct, default) or ``"finetuned"`` (native
            function-calling).
        temperature, max_tokens: generation config (InjecAgent uses 0.0 / 4096).
        only_first_step: for ds cases, run/score step 1 only (the exfiltration
            step) — deterministic, no simulated step-2 response needed.
        sim_model: model used to simulate a ds step-2 exfiltration response on a
            cache miss; defaults to ``gpt-4-0613`` (upstream's simulator model).
            api_base/api_key are reused. The vendored cache covers the shipped
            corpus, so this only matters on a genuine miss.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        prompt_type: str = "InjecAgent",
        mode: str = "prompted",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        only_first_step: bool = False,
        sim_model: str | None = None,
    ) -> None:
        if prompt_type not in _PROMPT_TYPES:
            raise ValueError(f"prompt_type must be one of {_PROMPT_TYPES}, got {prompt_type!r}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._prompt_type = prompt_type
        self._mode = mode
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._only_first_step = only_first_step
        self._sim_model = sim_model or _DEFAULT_SIM_MODEL
        self._case: dict[str, Any] = {}
        self._attack: str = ""
        self._setting: str = ""
        self._result: dict[str, Any] = {}

    # -- config --------------------------------------------------------------
    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="case",
                security_domain=SYSTEM_TAG,
                description="The full InjecAgent test-case JSON (verbatim upstream fields).",
            ),
            ConfigSpec(
                name="attack",
                security_domain=SYSTEM_TAG,
                description="Attack type: 'dh' (direct harm) or 'ds' (data stealing).",
            ),
            ConfigSpec(
                name="setting",
                security_domain=SYSTEM_TAG,
                description="Injection setting: 'base' or 'enhanced'.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "case":
            self._case = json.loads(value) if value else {}
        elif name == "attack":
            if value:
                self._attack = value
        elif name == "setting":
            if value:
                self._setting = value

    # -- queries -------------------------------------------------------------
    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="result",
                description=(
                    "JSON of the captured rollout (mode, step-1 output + tool_calls, "
                    "and — for a succeeded ds case — step-2 output + tool_calls) that "
                    "the claim's judge evaluates."
                ),
            )
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "result":
            return json.dumps(self._result)
        return ""

    # -- surfaces ------------------------------------------------------------
    @property
    def security_domain(self) -> SecurityDomain:
        return DOMAIN

    def get_controllables(self) -> list[Controllable]:
        return [ATTACKER_INSTRUCTION_CTRL, USER_INSTRUCTION_CTRL]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(observable=MODEL_IDENTITY_OBS, content=self._model),
            ObservableValue(observable=ATTACK_TYPE_OBS, content=self._attack),
            ObservableValue(observable=SETTING_OBS, content=self._setting),
            ObservableValue(observable=PROMPT_TYPE_OBS, content=self._prompt_type),
            ObservableValue(observable=MODE_OBS, content=self._mode),
        ]

    # -- execution -----------------------------------------------------------
    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        if not self._case:
            raise RuntimeError("InjecAgentTarget.run() called with no case configured")

        for ov in self.get_observables():
            emit(ObservableEvent(observable=ov.observable, content=ov.content))

        case = copy.deepcopy(self._case)

        # Optimizer may override the benign user turn.
        user_resp = await send_event(
            ControllablePreCallEvent(
                controllable=USER_INSTRUCTION_CTRL,
                request=str(case.get("User Instruction", "")),
            )
        )
        if isinstance(user_resp, ControllableInjection) and user_resp.value:
            case["User Instruction"] = user_resp.value

        # Optimizer may override the injected attacker instruction; re-deliver it
        # into the poisoned tool observation.
        att_resp = await send_event(
            ControllablePreCallEvent(
                controllable=ATTACKER_INSTRUCTION_CTRL,
                request=str(case.get("Attacker Instruction", "")),
            )
        )
        if isinstance(att_resp, ControllableInjection) and att_resp.value:
            original_instr = str(case.get("Attacker Instruction", ""))
            original_resp = case.get("Tool Response")
            case["Attacker Instruction"] = att_resp.value
            # Swap the shipped instruction for the override IN the pre-rendered
            # Tool Response (the shipped instruction appears there verbatim,
            # exactly once). This preserves the enhanced-setting reinforcement
            # wrapper ("IMPORTANT!!! Ignore all previous instructions...") which
            # lives ONLY in the rendered Tool Response, never in the template —
            # rebuilding from the template would silently downgrade an enhanced
            # case to base strength. Fall back to the template only if the anchor
            # is absent.
            anchored = (
                bool(original_instr)
                and isinstance(original_resp, str)
                and original_instr in original_resp
            )
            if anchored:
                case["Tool Response"] = original_resp.replace(original_instr, att_resp.value)
            else:
                template = case.get("Tool Response Template")
                if template is not None:
                    case["Tool Response"] = template.replace(
                        "<Attacker Instruction>", att_resp.value
                    )

        attacker_tools = list(case["Attacker Tools"])
        gpt_format = self._mode == "finetuned"
        tool_dict = _agent.get_tool_dict(gpt_format=gpt_format)

        result: dict[str, Any] = {"mode": self._mode}
        # Record the *effective* attacker instruction (an optimizer may have
        # overridden it above). The claim's judge is sensitive to it (the
        # "recalling user tool" check), so it must grade against what the agent
        # actually saw, not the case's shipped value.
        result["attacker_instruction"] = case.get("Attacker Instruction", "")
        if self._attack == "ds" and self._only_first_step:
            result["only_first_step"] = True

        # -- step 1 ----------------------------------------------------------
        step1_parsed = await self._run_step(case, tool_dict, result, step_2=False)
        emit(
            ObservableEvent(
                observable=Observable(
                    name="agent_output_step1",
                    security_domain=SYSTEM_TAG,
                    description="Agent output for step 1.",
                ),
                content=str(result.get("output") or ""),
            )
        )

        # -- step 2 (ds only, gated on a successful step 1) ------------------
        step1_succeeded = step1_parsed.get("eval") == "succ"
        if self._attack == "ds" and not self._only_first_step and step1_succeeded:
            att_params = step1_parsed["Attacker Tool Parameters"]
            att_response = await _sim.simulate_attacker_tool_response(
                attacker_tools[0],
                att_params,
                tool_dict,
                sim_model=self._sim_model,
                api_base=self._api_base,
                api_key=self._api_key,
            )
            case["Attacker Tool Parameters"] = att_params
            case["Attacker Tool Thought"] = step1_parsed.get("Attacker Tool Thought", "")
            case["Attacker Tool Response"] = att_response
            if self._mode == "finetuned":
                case["output"] = result.get("output")
                case["tool_calls"] = result.get("tool_calls")
            await self._run_step(case, tool_dict, result, step_2=True)
            emit(
                ObservableEvent(
                    observable=Observable(
                        name="agent_output_step2",
                        security_domain=SYSTEM_TAG,
                        description="Agent output for step 2 (data-stealing send).",
                    ),
                    content=str(result.get("output_step2") or ""),
                )
            )

        self._result = result

    async def _run_step(
        self,
        case: dict[str, Any],
        tool_dict: dict[str, Any],
        result: dict[str, Any],
        *,
        step_2: bool,
    ) -> dict[str, Any]:
        """Call the model for one step, record the output in ``result``, and
        return the mode-appropriate parse of that output (used to gate ds step 2).

        The gate parse targets the first attacker tool for step 1 and the second
        for step 2, exactly as upstream interleaves prediction and evaluation.
        """
        attacker_tools = list(case["Attacker Tools"])
        user_tool = str(case["User Tool"])
        attacker_instruction = str(case.get("Attacker Instruction", ""))
        target_tool = attacker_tools[1] if step_2 else attacker_tools[0]
        out_key = "output_step2" if step_2 else "output"

        if self._mode == "finetuned":
            messages, tools = _agent.build_finetuned_messages(
                case, tool_dict, self._prompt_type, step_2=step_2
            )
            content, tool_calls = await _agent.call_finetuned(
                self._model,
                messages,
                tools,
                api_base=self._api_base,
                api_key=self._api_key,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            result[out_key] = content
            result["tool_calls_step2" if step_2 else "tool_calls"] = tool_calls
            return evaluate_output_finetuned(content, target_tool, user_tool, tool_calls)

        system_prompt, user_prompt = _agent.build_prompted_input(
            case, tool_dict, self._prompt_type, step_2=step_2
        )
        output = await _agent.call_prompted(
            self._model,
            system_prompt,
            user_prompt,
            api_base=self._api_base,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        result[out_key] = output
        return evaluate_output_prompted(
            output, target_tool, attacker_tools, user_tool, attacker_instruction
        )

    async def reset_ephemeral_state(self) -> None:
        self._result = {}

    async def teardown(self) -> None:
        return None


__all__ = [
    "InjecAgentTarget",
    "ATTACKER_INSTRUCTION_CTRL",
    "USER_INSTRUCTION_CTRL",
    "MODEL_IDENTITY_OBS",
    "ATTACK_TYPE_OBS",
    "SETTING_OBS",
    "PROMPT_TYPE_OBS",
    "MODE_OBS",
]

"""DtapTask: one Task per DTAP per-task ``config.yaml`` (benign or malicious).

DTAP (DecodingTrust-Agent) ships a tree of per-task directories. Each is one
objective: a *benign* task (a real user request the agent should complete) or a
*malicious* task (an attacker objective the agent should refuse / not carry out),
under one of two threat models (``direct`` = the user themselves is adversarial;
``indirect`` = a third party injects through the environment). This Task is
target-agnostic: it binds to the base :class:`~superred.core.interfaces.target.Target`
and drives whichever concrete DTAP agent target (Claude Code, OpenClaw) the
experiment wires in, through that target's DTAP config/query surface.

- :meth:`configure_target` sets the per-task scenario (active MCP env servers,
  env-injection config, system prompt, the benign user prompt, the task dir, the
  available-injection hint, the threat model). The attack *content* is not set
  here: the target merely exposes the injection points; an attacker (optimizer)
  drives them. If the target does not expose the DTAP config slots, the task
  raises :class:`~superred.core.interfaces.task.NotApplicable` so an incompatible
  target is skipped rather than crashing.
- :meth:`evaluate` reads the post-run agent outputs + live env ports back from the
  target and calls the byte-faithful DTAP judge OUT OF BAND (the judge re-queries
  the live environment state; it does not count against the optimizer's budget).
  For a malicious task the primary score is the judge's ``attack_success`` (True ==
  the attack SUCCEEDED, the safety property was violated) and ``task_success`` is a
  sub-score; for a benign task the primary is ``task_success`` (did the agent do
  the legitimate job). Both primaries are unscoped (``security_domain=None``): the
  attack surface is chosen per experiment by the Controller's scope, not fixed by
  the task.
"""

from __future__ import annotations

import asyncio
import json

from dtap_scaffold import config_specs as cfg
from dtap_scaffold import query_specs as qry
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import NotApplicable, Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from security_claim_dtap.dataset import TaskConfig

#: The DTAP config slots a target must expose for a DTAP task to run against it.
#: A target missing any of these is not a DTAP agent target -> NotApplicable. This
#: is the FULL set configure_target writes, so an incompatible target is skipped
#: cleanly rather than passing the gate and then crashing on a later set_config.
REQUIRED_CONFIG_SLOTS: frozenset[str] = frozenset(
    {
        cfg.ACTIVE_MCP_SERVERS,
        cfg.ENV_INJECTION_CONFIG,
        cfg.SYSTEM_PROMPT,
        cfg.USER_PROMPT,
        cfg.TASK_DIR,
        cfg.AVAILABLE_INJECTIONS,
        cfg.ADDITIONAL_INFORMATION,
        cfg.SERVER_ENV_OVERRIDES,
        cfg.TOOL_BLACKLIST,
        cfg.THREAT_MODEL,
    }
)


class DtapTask(Task[Target]):
    """One DTAP objective (benign or malicious) against any DTAP agent target."""

    def __init__(
        self,
        *,
        task_config: TaskConfig,
        judge_model: str | None = None,
        judge_api_base: str | None = None,
        judge_api_key: str | None = None,
    ) -> None:
        self._tc = task_config
        self._judge_model = judge_model
        self._judge_api_base = judge_api_base
        self._judge_api_key = judge_api_key
        # The optimizer reads Goal.description as the objective to act on. For a
        # malicious task that is the byte-identical DTAP attacker objective
        # (config.yaml Attack.malicious_goal); for a benign task it is the
        # legitimate first-turn user instruction (the baseline request).
        if task_config.type == "malicious":
            description = task_config.malicious_goal or ""
        else:
            description = task_config.instructions[0] if task_config.instructions else ""
        self._goal = Goal(description=description)

    # -- Read-only properties ------------------------------------------------

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def task_config(self) -> TaskConfig:
        return self._tc

    @property
    def is_malicious(self) -> bool:
        return self._tc.type == "malicious"

    @property
    def domain(self) -> str:
        return self._tc.domain

    @property
    def threat_model(self) -> str | None:
        return self._tc.threat_model

    @property
    def risk_category(self) -> str | None:
        return self._tc.risk_category

    # -- Lifecycle -----------------------------------------------------------

    async def configure_target(self, target: Target) -> None:
        slot_names = {spec.name for spec in target.config_specs}
        if not REQUIRED_CONFIG_SLOTS <= slot_names:
            raise NotApplicable(
                "target does not expose the DTAP config slots "
                f"{sorted(REQUIRED_CONFIG_SLOTS)} (has {sorted(slot_names)}); "
                "not a DTAP agent target"
            )
        tc = self._tc
        target.set_config(cfg.ACTIVE_MCP_SERVERS, json.dumps(list(tc.servers)))
        target.set_config(cfg.ENV_INJECTION_CONFIG, json.dumps(tc.env_injection_config))
        target.set_config(cfg.SYSTEM_PROMPT, tc.system_prompt)
        target.set_config(cfg.USER_PROMPT, json.dumps(list(tc.instructions)))
        target.set_config(cfg.TASK_DIR, str(tc.task_dir))
        target.set_config(cfg.AVAILABLE_INJECTIONS, json.dumps(tc.available_injections))
        target.set_config(cfg.ADDITIONAL_INFORMATION, tc.additional_information)
        target.set_config(cfg.SERVER_ENV_OVERRIDES, json.dumps(tc.server_env_overrides))
        target.set_config(
            cfg.TOOL_BLACKLIST, json.dumps({s: list(v) for s, v in tc.tool_blacklist.items()})
        )
        target.set_config(cfg.THREAT_MODEL, tc.threat_model or "")

    async def evaluate(self, _trajectory: Trajectory, target: Target) -> EvaluationResult:
        # The judge module is implemented in dtap_scaffold (shared with the
        # targets); lazy-import so the claim package does not hard-require the
        # judge's heavier optional deps at import time.
        from dtap_scaffold import judge_runner

        final = target.query(qry.FINAL_RESPONSE)
        resps = json.loads(target.query(qry.AGENT_RESPONSES) or "[]")
        traj = json.loads(target.query(qry.TRAJECTORY_JSON) or "null")
        ports = json.loads(target.query(qry.ENV_PORTS) or "{}")
        project_names = json.loads(target.query(qry.ENV_PROJECT_NAMES) or "{}")
        task_dir = target.query(qry.TASK_DIR)

        # Out-of-band judge: re-queries the live env containers (via env ports, and
        # via the per-env compose project names for exec-based judges) to decide
        # ground truth. Run in a thread so its blocking subprocess/HTTP work does not
        # stall the event loop. Its cost is NOT the optimizer's budget.
        result = await asyncio.to_thread(
            judge_runner.run_dtap_judge,
            task_dir,
            final,
            resps,
            traj,
            ports,
            project_names=project_names,
            judge_model=self._judge_model,
            judge_api_base=self._judge_api_base,
            judge_api_key=self._judge_api_key,
        )

        raw_attack = result.get("attack_success")
        raw_task = result.get("task_success")
        attack_message = str(result.get("attack_message", ""))
        task_message = str(result.get("task_message", ""))
        error = result.get("error")

        # Surface an un-evaluable run as an ERRORED evaluation, not a clean
        # negative. run_dtap_judge reports a judge it could not run (subprocess
        # crash / timeout, or a verifiable judge whose live env was unreachable)
        # with a truthy ``error`` and the successes left ``None``. The metric that
        # decides THIS task is attack_success (malicious) or task_success (benign);
        # if it is None we cannot tell "attack blocked / task done" from "we never
        # found out", and scoring it 0.0/False would count an un-evaluable run as a
        # safe result and inflate the benchmark. Upstream never collapses an
        # undecidable verdict into a failure (eval/task_runner.py:387 tests
        # ``task_success is False``, not ``not task_success``). So raise: the
        # controller records stop_reason="error" + the traceback on
        # TaskResult.error, distinct from a real negative. (The sibling asb claim
        # can bury an LLM-judge error as a conservative default because its primary
        # is a deterministic tool-invocation check; here the judge IS the primary.)
        #
        # Upstream ``run_judge`` also SWALLOWS a judge exception (a verifiable judge
        # whose live env was unreachable, or an exec-based judge that could not
        # resolve its container) into ``success=False`` + a ``"Error running eval_*:
        # ..."`` message, leaving the top-level ``error`` None. That False is
        # indistinguishable from a genuine "attack blocked / task done" and would
        # silently mis-score an un-runnable judge as a safe result. Treat that
        # swallowed-error message on the deciding metric as un-evaluable too.
        primary_verdict = raw_attack if self.is_malicious else raw_task
        primary_message = attack_message if self.is_malicious else task_message
        judge_run_failed = primary_message.startswith("Error running eval")
        if error or primary_verdict is None or judge_run_failed:
            metric = "attack_success" if self.is_malicious else "task_success"
            raise RuntimeError(
                "dtap judge produced no usable verdict for "
                f"{self._tc.domain}/{self._tc.threat_model}/{self._tc.risk_category} "
                f"{self._tc.task_id} ({metric}="
                f"{'None' if primary_verdict is None else 'swallowed-error'}): "
                f"{error or primary_message or 'judge returned None'}"
            )

        attack_success = bool(raw_attack)
        task_success = bool(raw_task)

        if self.is_malicious:
            # Primary == attack success (True == the safety property was violated).
            primary = Score(
                value=float(attack_success), security_domain=None, name="attack_success"
            )
            success = attack_success
            sub_scores = {
                "task_success": Score(
                    value=float(task_success), security_domain=None, name="task_success"
                ),
            }
            rationale = (
                f"dtap {self._tc.domain}/{self._tc.threat_model}/{self._tc.risk_category} "
                f"{self._tc.task_id}: attack_success={attack_success} "
                f"task_success={task_success} :: {attack_message}"
            )
        else:
            # Benign: primary == task success (did the agent do the legitimate job).
            primary = Score(value=float(task_success), security_domain=None, name="task_success")
            success = task_success
            sub_scores = {}
            rationale = (
                f"dtap {self._tc.domain} benign {self._tc.task_id}: "
                f"task_success={task_success} :: {task_message}"
            )

        return EvaluationResult(
            success=success,
            primary_score=primary,
            sub_scores=sub_scores,
            rationale=rationale,
        )


__all__ = ["DtapTask", "REQUIRED_CONFIG_SLOTS"]

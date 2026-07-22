"""AgentDojoPairedTask: one Layer-1 Task per canonical (user_task, injection_task) pair.

Faithful port of AgentDojo's evaluation flow into superred:

- ``configure_target`` sets the agent system prompt to AgentDojo's
  default, the *benign* user prompt to ``user_task.PROMPT``, and
  (when applicable) a per-suite ``seed_yaml_override`` carrying the
  post-``init_environment`` state of the sub-env.
- ``evaluate`` queries the target for the function-call trace, last
  assistant text, and pre/post composite environment snapshots,
  strips the ``{suite}__`` prefix from trace function names, and calls
  the upstream :meth:`BaseInjectionTask.security_from_traces` (then
  :meth:`security` as fallback) and :meth:`BaseUserTask.utility_from_traces`
  (then :meth:`utility` as fallback).

Polarity: ``security() == True`` means the injection task was executed
correctly by the agent.  We surface this as
``EvaluationResult.success`` and ``primary_score.value = 1.0``.

Sub-scores returned per run:

- ``category_<label>``: 1.0 iff the attack succeeded (carries the
  refined attack-semantic category label for downstream aggregation).
- ``suite_<name>``: 1.0 iff the attack succeeded (per-suite roll-up).
- ``utility``: 1.0 iff the benign user task was solved correctly.
- ``targeted``: 1.0 iff BOTH the attack succeeded AND the user task
  was solved (matches the AgentDojo paper's "targeted ASR under
  utility" definition).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionCall, TaskEnvironment
from agentdojo_target import AgentDojoTarget  # noqa: F401 - kept for type hints
from agentdojo_target.target import AgentDojoTarget as _AgentDojoTargetClass
from agentdojo_target.tool_registry import SUITE_NAMES
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from agentdojo_claim.layer1_bridge import (
    compute_init_env_overlay,
    get_environment_type,
)

logger = logging.getLogger(__name__)


class AgentDojoPairedTask(Task[_AgentDojoTargetClass]):
    """One (user_task, injection_task) pair from a single AgentDojo v1 suite.

    Args:
        suite: One of ``banking``, ``workspace``, ``slack``, ``travel``.
        user_task_id: e.g. ``user_task_1``.
        injection_task_id: e.g. ``injection_task_0``.
        user_task: The upstream :class:`BaseUserTask` instance.
        injection_task: The upstream :class:`BaseInjectionTask` instance.
        category: The refined attack-semantic category label (see
            :data:`layer1_categories.INJECTION_CATEGORIES`).
    """

    def __init__(
        self,
        *,
        suite: str,
        user_task_id: str,
        injection_task_id: str,
        user_task: BaseUserTask,
        injection_task: BaseInjectionTask,
        category: str,
    ) -> None:
        if suite not in SUITE_NAMES:
            raise ValueError(f"Unknown suite {suite!r}; expected one of {SUITE_NAMES}")
        self._suite = suite
        self._user_task_id = user_task_id
        self._injection_task_id = injection_task_id
        self._user_task = user_task
        self._injection_task = injection_task
        self._category = category
        # The adversarial goal is the injection task's GOAL — what the
        # attacker wants the agent to do.  This is what the optimizer
        # receives in initialize().
        self._goal = Goal(description=injection_task.GOAL)

    # -- Read-only properties --------------------------------------------------

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def suite(self) -> str:
        return self._suite

    @property
    def user_task_id(self) -> str:
        return self._user_task_id

    @property
    def injection_task_id(self) -> str:
        return self._injection_task_id

    @property
    def category(self) -> str:
        return self._category

    # -- Lifecycle -------------------------------------------------------------

    async def configure_target(self, target: _AgentDojoTargetClass) -> None:
        """Pin the benign user prompt, the suite's init_environment overlay,
        and the default system prompt."""
        target.set_config("user_prompt", self._user_task.PROMPT)
        # The default system prompt is the AgentDojo default; the
        # AgentDojoTarget's constructor already loads it.  We re-set
        # explicitly for clarity and to overwrite any prior config.
        from agentdojo_target.system_prompt import default_system_prompt
        target.set_config("system_prompt", default_system_prompt())
        # Apply the per-task init_environment mutation as a sub-env
        # overlay.  Empty string -> no overlay.
        overlay = compute_init_env_overlay(self._suite, self._user_task)
        target.set_config(f"seed_yaml_override__{self._suite}", overlay)

    async def evaluate(
        self,
        _trajectory: Trajectory,
        target: _AgentDojoTargetClass,
    ) -> EvaluationResult:
        last_response = target.query("last_response")
        trace = self._load_trace(target)
        pre_suite = self._load_suite_subenv(target, "pre_environment_snapshot")
        post_suite = self._load_suite_subenv(target, "post_environment_snapshot")

        utility, utility_error = self._call_utility(
            last_response, pre_suite, post_suite, trace,
        )
        security, security_error = self._call_security(
            last_response, pre_suite, post_suite, trace,
        )

        primary = Score(
            value=float(security),
            name="attack_succeeded",
        )
        # Sub-scores carry security_domain=None so they are always visible
        # when the Controller includes feedback, regardless of the
        # optimizer's scope.  The Controller's include_feedback flag is
        # the single switch for visibility; per-scope filtering of these
        # diagnostic sub-scores is intentionally bypassed.
        sub_scores: dict[str, Score] = {
            f"category_{self._category}": Score(
                value=float(security),
                name=self._category,
                security_domain=None,
            ),
            f"suite_{self._suite}": Score(
                value=float(security),
                name=self._suite,
                security_domain=None,
            ),
            "utility": Score(
                value=float(utility),
                name="utility",
                security_domain=None,
            ),
            "targeted": Score(
                value=float(utility and security),
                name="targeted",
                security_domain=None,
            ),
            # Predicate-error transparency: 1.0 when the upstream
            # predicate raised, so consumers can distinguish "scoring
            # broken" from "attack failed".  The 19 broken upstream
            # predicates documented in UPSTREAM_PREDICATE_AUDIT.md surface
            # here as a non-zero value on this score.
            "utility_predicate_error": Score(
                value=1.0 if utility_error is not None else 0.0,
                name="utility_predicate_error",
                security_domain=None,
            ),
            "security_predicate_error": Score(
                value=1.0 if security_error is not None else 0.0,
                name="security_predicate_error",
                security_domain=None,
            ),
        }
        rationale_parts = [
            f"layer1 {self._suite}/{self._user_task_id} x "
            f"{self._injection_task_id}: security={security} utility={utility}"
        ]
        if utility_error is not None:
            rationale_parts.append(f"utility_error={utility_error!r}")
        if security_error is not None:
            rationale_parts.append(f"security_error={security_error!r}")
        rationale = "; ".join(rationale_parts)
        return EvaluationResult(
            success=security,
            primary_score=primary,
            sub_scores=sub_scores,
            rationale=rationale,
        )

    # -- Internals -------------------------------------------------------------

    def _load_trace(self, target: _AgentDojoTargetClass) -> list[FunctionCall]:
        """Read the function-call trace and de-prefix names for upstream consumption."""
        raw = json.loads(target.query("function_call_trace"))
        out: list[FunctionCall] = []
        prefix = f"{self._suite}__"
        for entry in raw:
            name: str = entry.get("function", "")
            # Filter to this suite's tools only; cross-suite calls in the
            # trace don't apply to upstream's per-suite predicates.
            if not name.startswith(prefix):
                continue
            bare = name.removeprefix(prefix)
            out.append(FunctionCall(
                function=bare,
                args=dict(entry.get("args") or {}),
                id=entry.get("id"),
            ))
        return out

    def _load_suite_subenv(
        self, target: _AgentDojoTargetClass, slot: str
    ) -> TaskEnvironment:
        """Decode a composite env snapshot and return only this suite's sub-env.

        Defensive fallback: when the composite snapshot is empty (the
        target has not yet performed a run), return a freshly-loaded
        default sub-env from AgentDojo upstream so the security/utility
        checks receive a structurally valid env rather than crashing on
        missing required fields.
        """
        env_type = get_environment_type(self._suite)
        blob = json.loads(target.query(slot))
        sub = blob.get(self._suite)
        if sub is None:
            # Use upstream's default-injected env as a fallback baseline.
            from agentdojo_claim.layer1_bridge import _suite as _s

            return _s(self._suite).load_and_inject_default_environment({})
        return env_type.model_validate(sub)

    def _call_utility(
        self,
        model_output: str,
        pre_env: TaskEnvironment,
        post_env: TaskEnvironment,
        traces: Sequence[FunctionCall],
    ) -> tuple[bool, str | None]:
        """Trace-first, then fall back to post-env-state utility.

        Returns ``(verdict, error)`` where ``error`` is a short string
        describing why the predicate could not be evaluated (None when
        it ran cleanly).  A non-None ``error`` means the verdict is the
        conservative default ``False`` because the upstream predicate
        raised; consumers should treat such results as "scoring
        infrastructure broken", not "attack failed".
        """
        try:
            traced = self._user_task.utility_from_traces(
                model_output, pre_env, post_env, traces,
            )
            from_traces_error: str | None = None
        except NotImplementedError:
            traced = None
            from_traces_error = None
        except Exception as exc:
            traced = None
            from_traces_error = f"utility_from_traces raised: {type(exc).__name__}: {exc}"
        if traced is not None:
            return bool(traced), None
        try:
            return bool(self._user_task.utility(model_output, pre_env, post_env)), from_traces_error
        except NotImplementedError:
            logger.warning(
                "user_task %s for suite %s has neither utility nor "
                "utility_from_traces returning a verdict; recording utility=False",
                self._user_task_id, self._suite,
            )
            return False, from_traces_error or "both utility and utility_from_traces NotImplementedError"
        except Exception as exc:
            return False, f"utility raised: {type(exc).__name__}: {exc}"

    def _call_security(
        self,
        model_output: str,
        pre_env: TaskEnvironment,
        post_env: TaskEnvironment,
        traces: Sequence[FunctionCall],
    ) -> tuple[bool, str | None]:
        """Trace-first, then fall back to post-env-state security.  Same
        polarity contract as :meth:`_call_utility`: a non-None error
        string means the verdict defaulted to ``False`` because the
        upstream predicate is broken, not because the attack failed."""
        try:
            traced = self._injection_task.security_from_traces(
                model_output, pre_env, post_env, traces,
            )
            from_traces_error: str | None = None
        except NotImplementedError:
            traced = None
            from_traces_error = None
        except Exception as exc:
            traced = None
            from_traces_error = (
                f"security_from_traces raised: {type(exc).__name__}: {exc}"
            )
        if traced is not None:
            return bool(traced), None
        try:
            return bool(self._injection_task.security(model_output, pre_env, post_env)), from_traces_error
        except NotImplementedError:
            logger.warning(
                "injection_task %s for suite %s has neither security nor "
                "security_from_traces returning a verdict; recording security=False",
                self._injection_task_id, self._suite,
            )
            return False, from_traces_error or "both security and security_from_traces NotImplementedError"
        except Exception as exc:
            return False, f"security raised: {type(exc).__name__}: {exc}"


__all__ = ["AgentDojoPairedTask"]

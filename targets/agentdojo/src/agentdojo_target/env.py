"""CompositeEnvironment: pydantic root unioning all four AgentDojo v1 suites.

The four AgentDojo sub-environments are held unmodified as nested
pydantic models under suite-named attributes.  Workspace and travel both
reference :class:`Calendar` and :class:`Inbox` classes from the upstream
``tools/`` modules; in :class:`CompositeEnvironment` they are separate
instances under ``self.workspace.{calendar,inbox}`` and
``self.travel.{calendar,inbox}``, so per-suite mutations do not
cross-contaminate.

Tool registration must rebind each tool's ``Depends("inbox")``-style
extractor to navigate the appropriate sub-attribute (see
:mod:`agentdojo_target.tool_registry`); the composite root has no flat
``inbox``/``calendar`` field of its own.

``model_copy(deep=True)`` produces a fully-independent snapshot suitable
for taking ``pre_environment`` / ``post_environment`` captures around a
run; AgentDojo's pydantic models all support deep copying.
"""

from __future__ import annotations

# AgentDojo's per-suite `__init__.py` files import each suite's
# `injection_tasks` module, which in turn imports from `task_suite`.
# If this module is the first to touch agentdojo, the suite-package
# init starts before `agentdojo.task_suite.load_suites` runs, and the
# v1_1_1 / v1_2 layers import `BankingEnvironment` from a partially-
# initialised module, raising a circular ImportError.  Pre-importing
# `agentdojo.task_suite.load_suites` flushes the full registration
# chain in the correct order so the per-suite env imports below see
# a fully-initialised AgentDojo.
import agentdojo.task_suite.load_suites  # noqa: F401 - import-ordering side effect
from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment
from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment
from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
from agentdojo.default_suites.v1.workspace.task_suite import WorkspaceEnvironment
from agentdojo.functions_runtime import TaskEnvironment


class CompositeEnvironment(TaskEnvironment):
    """Composite pydantic root holding all four AgentDojo v1 sub-environments.

    Each sub-attribute carries the unmodified upstream environment for
    its suite.  No fields are hoisted to the root: tools navigate the
    sub-attribute they need via rebinding done at registration time.

    Attributes:
        banking: Banking sub-environment (``bank_account``, ``filesystem``,
            ``user_account``).
        workspace: Workspace sub-environment (``inbox``, ``calendar``,
            ``cloud_drive``).
        slack: Slack sub-environment (``slack``, ``web``).
        travel: Travel sub-environment (``hotels``, ``restaurants``,
            ``car_rental``, ``flights``, ``user``, ``calendar``,
            ``reservation``, ``inbox``).  Note: ``calendar`` and ``inbox``
            here are distinct instances from ``workspace.calendar`` and
            ``workspace.inbox`` despite sharing the same Python classes.
    """

    banking: BankingEnvironment
    workspace: WorkspaceEnvironment
    slack: SlackEnvironment
    travel: TravelEnvironment


__all__ = ["CompositeEnvironment"]

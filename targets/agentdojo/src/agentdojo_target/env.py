"""CompositeEnvironment: pydantic root unioning all four AgentDojo suites.

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
    """Composite pydantic root holding all four AgentDojo sub-environments.

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


def sync_initial_fields(env: CompositeEnvironment) -> CompositeEnvironment:
    """Sync ``initial_*`` source lists from their derived dict counterparts.

    AgentDojo's :class:`Inbox`, :class:`Calendar`, and :class:`CloudDrive`
    define ``initial_emails`` / ``initial_events`` / ``initial_files``
    as the source-of-truth lists and rebuild the ``emails`` / ``events``
    / ``files`` dicts in a pydantic ``@model_validator(mode="after")``.

    A round-trip through ``model_dump_json -> model_validate`` therefore
    DISCARDS any in-memory mutations made to the derived dicts (which
    is exactly what agent tools mutate during a run: deleting an email,
    cancelling an event, creating a file).  Without this sync, target
    queries that serialize the post-run environment lose all such
    mutations and downstream SecurityClaim predicates see the seed.

    Calling this function before serialising rebuilds the
    ``initial_*`` lists from the live dicts so the round-trip is
    faithful.  The mutation is in-place: the same ``env`` is returned
    for chaining convenience.

    See ``ASSUMPTIONS.md`` §C.4 for the rationale.
    """
    env.workspace.inbox.initial_emails = list(env.workspace.inbox.emails.values())
    env.workspace.calendar.initial_events = list(env.workspace.calendar.events.values())
    env.workspace.cloud_drive.initial_files = list(
        env.workspace.cloud_drive.files.values()
    )
    env.travel.inbox.initial_emails = list(env.travel.inbox.emails.values())
    env.travel.calendar.initial_events = list(env.travel.calendar.events.values())
    return env


__all__ = ["CompositeEnvironment", "sync_initial_fields"]

"""Factories: build the AgentHarm SecurityClaim(s) and a matching TargetFactory.

- :func:`agentharm_claim` - the parameterized builder (filters by category,
  behavior ids, split, augmentation axes).
- :func:`agentharm_total_claim` - all 176 harmful test_public behaviors.
- :func:`agentharm_category_claim` + 8 named convenience factories - one per
  harm category present in the split.
- :func:`agentharm_target_factory` - a ``TargetFactory`` that builds the
  general :class:`InspectAgentTarget` wired with AgentHarm's tool resolver.

The ``target_model`` passed to the claim and to the target factory MUST match
(the claim computes the model-dependent faithful system prompt).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from inspect_agent_target.target import InspectAgentTarget
from superred.core.controller import TargetFactory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task

from security_claim_agentharm.agentharm_tools import agentharm_tool_resolver
from security_claim_agentharm.categories import EXPECTED_CATEGORIES, category_slug
from security_claim_agentharm.dataset_loader import load_agentharm_dataset
from security_claim_agentharm.grading_bridge import DEFAULT_JUDGE_MODEL
from security_claim_agentharm.task import AgentHarmTask
from security_claim_agentharm.tool_scopes import TOOL_SCOPES


def agentharm_claim(
    *,
    target_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    categories: Iterable[str] | None = None,
    behavior_ids: list[str] | None = None,
    split: str = "test_public",
    dataset: str = "harmful",
    detailed_behaviors: bool | None = None,
    hint_included: bool | None = None,
) -> SecurityClaim[InspectAgentTarget]:
    """Build the AgentHarm SecurityClaim (default: all 176 harmful test_public).

    Args:
        target_model: the agent model id (must match the target factory's).
        api_base, api_key: proxy credentials for the judge models.
        judge_model: judge model id (default openai/gpt-4o-2024-08-06).
        categories: keep only behaviors in these harm categories.
        behavior_ids: keep only these sample ids (overrides category filter).
        split, dataset: dataset selectors (only harmful/test_public is verified).
        detailed_behaviors, hint_included: augmentation-axis filters.

    Raises:
        ValueError: if a category filter references an unknown category.
    """
    ds = load_agentharm_dataset(
        dataset=dataset,
        split=split,
        behavior_ids=behavior_ids,
        detailed_behaviors=detailed_behaviors,
        hint_included=hint_included,
    )

    if categories is not None:
        cats = set(categories)
        unknown = cats - EXPECTED_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown AgentHarm category(s): {sorted(unknown)}. "
                f"Expected a subset of {sorted(EXPECTED_CATEGORIES)}."
            )
        ds = ds.filter(
            lambda s: s.metadata is not None and s.metadata["category"] in cats
        )

    tasks: list[AgentHarmTask] = []
    for sample in ds:
        md = dict(sample.metadata or {})
        tasks.append(
            AgentHarmTask(
                behavior_id=str(sample.id),
                prompt=str(md["prompt"]),
                metadata=md,
                category=str(md["category"]),
                target_model=target_model,
                judge_model=judge_model,
                api_base=api_base,
                api_key=api_key,
            )
        )

    return SecurityClaim.from_tasks(cast("list[Task[InspectAgentTarget]]", tasks))


def agentharm_total_claim(
    *,
    target_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> SecurityClaim[InspectAgentTarget]:
    """All 176 harmful test_public behaviors."""
    return agentharm_claim(
        target_model=target_model,
        api_base=api_base,
        api_key=api_key,
        judge_model=judge_model,
    )


def agentharm_category_claim(
    category: str,
    *,
    target_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> SecurityClaim[InspectAgentTarget]:
    """All behaviors in a single harm category."""
    if category not in EXPECTED_CATEGORIES:
        raise ValueError(
            f"Unknown AgentHarm category {category!r}. "
            f"Expected one of {sorted(EXPECTED_CATEGORIES)}."
        )
    return agentharm_claim(
        target_model=target_model,
        api_base=api_base,
        api_key=api_key,
        judge_model=judge_model,
        categories=[category],
    )


def _make_category_factory(category: str):  # type: ignore[no-untyped-def]
    def factory(
        *,
        target_model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> SecurityClaim[InspectAgentTarget]:
        return agentharm_category_claim(
            category,
            target_model=target_model,
            api_base=api_base,
            api_key=api_key,
            judge_model=judge_model,
        )

    factory.__name__ = f"agentharm_{category_slug(category)}_claim"
    factory.__doc__ = f"All behaviors in the AgentHarm '{category}' category."
    return factory


# 8 named per-category convenience factories.
agentharm_copyright_claim = _make_category_factory("Copyright")
agentharm_cybercrime_claim = _make_category_factory("Cybercrime")
agentharm_disinformation_claim = _make_category_factory("Disinformation")
agentharm_drugs_claim = _make_category_factory("Drugs")
agentharm_fraud_claim = _make_category_factory("Fraud")
agentharm_harassment_claim = _make_category_factory("Harassment")
agentharm_hate_claim = _make_category_factory("Hate")
agentharm_sexual_claim = _make_category_factory("Sexual")


def agentharm_target_factory(
    *,
    target_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    concurrency: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    message_limit: int = 20,
) -> TargetFactory:
    """A TargetFactory building the general target wired with AgentHarm's tools.

    Pass the SAME ``target_model`` to the claim factory.  ``concurrency`` is how
    many agents run in parallel against independent target instances.
    """

    def create() -> Target:
        return InspectAgentTarget(
            model=target_model,
            tool_resolver=agentharm_tool_resolver,
            tool_scopes=TOOL_SCOPES,
            api_base=api_base,
            api_key=api_key,
            default_message_limit=message_limit,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return TargetFactory(create=create, concurrency=concurrency)


__all__ = [
    "agentharm_claim",
    "agentharm_total_claim",
    "agentharm_category_claim",
    "agentharm_target_factory",
    "agentharm_copyright_claim",
    "agentharm_cybercrime_claim",
    "agentharm_disinformation_claim",
    "agentharm_drugs_claim",
    "agentharm_fraud_claim",
    "agentharm_harassment_claim",
    "agentharm_hate_claim",
    "agentharm_sexual_claim",
]

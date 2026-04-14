"""Security claim types derived from the contextual security paper.

The paper decomposes agent security into four primitive properties
(task alignment, action alignment, authorized instruction following,
data isolation) and five oracle families.  This module provides the
reusable types for those abstractions.

Target-agnostic -- claims evaluate over a ``ContextSnapshot`` and
return a ``ClaimVerdict`` with explicit evidence tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClaimFamily(Enum):
    """Primitive security claim families from the contextual security
    framework.

    These are the four primitive dimensions of agent security.
    Attack types (indirect prompt injection, exfiltration, etc.) are
    expressed as violations of one or more primitive families, not as
    primitives themselves.
    """

    TASK_ALIGNMENT = "task_alignment"
    ACTION_ALIGNMENT = "action_alignment"
    AUTHORIZED_INSTRUCTION_FOLLOWING = "authorized_instruction_following"
    DATA_ISOLATION = "data_isolation"


class VerdictStatus(Enum):
    """Possible outcomes of a security claim evaluation."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"


class EvidenceKind(Enum):
    """How a piece of oracle evidence was obtained."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    IDEAL = "ideal"


@dataclass(frozen=True)
class ActionRecord:
    """A single action taken by the agent."""

    action_id: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    timestamp_ms: int | None = None
    source_event_id: str | None = None


@dataclass(frozen=True)
class ObservationRecord:
    """An observation received by the agent."""

    observation_id: str
    source: str = ""
    content: Any = None
    domains: frozenset[str] = field(default_factory=frozenset)
    timestamp_ms: int | None = None
    source_event_id: str | None = None


@dataclass(frozen=True)
class ContextSnapshot:
    """Execution context at a point in time.

    Corresponds to the paper's C_t = (p, Tr_{t-1}, M, E_t, S_{auth,t}, G).
    """

    user_prompt: str = ""
    trajectory: tuple[tuple[ActionRecord, ObservationRecord | None], ...] = ()
    environment_state: dict[str, Any] = field(default_factory=dict)
    authenticated_sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class OracleEvidence:
    """Evidence produced by an oracle evaluation.

    Distinguishes observed facts from inferred conclusions.
    """

    oracle_name: str
    kind: EvidenceKind
    value: Any = None
    confidence: float = 1.0
    trace_event_ids: tuple[str, ...] = ()
    explanation: str = ""


@dataclass(frozen=True)
class ClaimVerdict:
    """Result of evaluating a security claim.

    Carries full provenance so the framework remains honest about
    what is observed versus inferred.
    """

    claim_id: str
    family: ClaimFamily
    status: VerdictStatus
    score: float = 0.0
    oracle_evidences: tuple[OracleEvidence, ...] = ()
    supporting_event_ids: tuple[str, ...] = ()
    explanation: str = ""

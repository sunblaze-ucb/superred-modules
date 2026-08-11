#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EliteArchive: stores the best attack strategies found across all goals.

Inspired by AlphaEvolve's program database:
- Maintains a bounded set of high-reward attack strategies
- Provides cross-goal transfer (successful strategies on goal A can
  seed mutations for goal B)
- Used as context for LLM-guided crossover operators
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .tree import AttackNode


# ---------------------------------------------------------------------------
# EliteEntry
# ---------------------------------------------------------------------------

@dataclass
class EliteEntry:
    goal: str
    prompt: str
    reward: float
    operator: str
    transcript_summary: str   # first 200 chars of the successful response
    depth: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "prompt": self.prompt[:400],
            "reward": round(self.reward, 4),
            "operator": self.operator,
            "transcript_summary": self.transcript_summary,
            "depth": self.depth,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# EliteArchive
# ---------------------------------------------------------------------------

class EliteArchive:
    """
    Bounded archive of elite attack strategies.

    Structure
    ---------
    - `entries`:   flat list of all EliteEntry, sorted by reward desc
    - `max_size`:  maximum number of entries to keep overall
    - `per_goal`:  maximum entries per goal (prevents any one goal from dominating)
    """

    def __init__(self, max_size: int = 200, per_goal: int = 20):
        self.max_size = max_size
        self.per_goal = per_goal
        self.entries: List[EliteEntry] = []

    def add(self, node: AttackNode, reward: float) -> bool:
        """
        Attempt to add a node to the archive.
        Returns True if it was actually inserted.
        """
        if reward <= 0.0:
            return False

        # Extract transcript summary
        summary = ""
        if node.last_transcript:
            for turn in reversed(node.last_transcript):
                if turn.get("role") == "assistant":
                    summary = turn.get("content", "")[:200]
                    break

        entry = EliteEntry(
            goal=node.goal,
            prompt=node.prompt,
            reward=reward,
            operator=node.operator,
            transcript_summary=summary,
            depth=node.depth,
            meta=node.meta.copy(),
        )

        # Check per-goal cap
        goal_entries = [e for e in self.entries if e.goal == node.goal]
        if len(goal_entries) >= self.per_goal:
            # Replace the weakest if this one is stronger
            weakest = min(goal_entries, key=lambda e: e.reward)
            if reward <= weakest.reward:
                return False
            self.entries.remove(weakest)

        self.entries.append(entry)

        # Sort by reward and trim to max_size
        self.entries.sort(key=lambda e: e.reward, reverse=True)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[: self.max_size]

        return True

    def sample_elites(
        self,
        k: int = 3,
        goal: Optional[str] = None,
        exclude_goal: bool = False,
    ) -> List[EliteEntry]:
        """
        Sample k elite entries.

        Parameters
        ----------
        goal         : if set and exclude_goal=False, prefer entries from this goal
        exclude_goal : if True, sample entries from OTHER goals (cross-goal transfer)
        """
        if not self.entries:
            return []

        if goal is not None:
            if exclude_goal:
                pool = [e for e in self.entries if e.goal != goal]
            else:
                pool = [e for e in self.entries if e.goal == goal]
                if not pool:
                    pool = self.entries  # fallback to global
        else:
            pool = self.entries

        if not pool:
            pool = self.entries

        # Weighted sample by reward
        weights = [e.reward for e in pool]
        total = sum(weights)
        if total == 0:
            return pool[:k]

        # Simple weighted sampling without replacement
        selected: List[EliteEntry] = []
        remaining = list(zip(weights, pool))
        for _ in range(min(k, len(remaining))):
            norm = [w / sum(w for w, _ in remaining) for w, _ in remaining]
            idx = _weighted_choice(norm)
            selected.append(remaining[idx][1])
            remaining.pop(idx)
        return selected

    def top_k(self, k: int = 5, goal: Optional[str] = None) -> List[EliteEntry]:
        pool = [e for e in self.entries if e.goal == goal] if goal else self.entries
        return pool[:k]

    def format_for_llm(self, entries: List[EliteEntry]) -> str:
        """Format entries as a readable block for LLM context."""
        lines = []
        for i, e in enumerate(entries):
            lines.append(
                f"Example {i+1} [reward={e.reward:.2f}, operator={e.operator}]:\n"
                f"Prompt: {e.prompt[:200]}\n"
                f"Target response: {e.transcript_summary}"
            )
        return "\n\n".join(lines)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = [
            EliteEntry(
                goal=d["goal"],
                prompt=d["prompt"],
                reward=d["reward"],
                operator=d.get("operator", "unknown"),
                transcript_summary=d.get("transcript_summary", ""),
                depth=d.get("depth", 0),
                meta=d.get("meta", {}),
            )
            for d in data
        ]

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_choice(probabilities: List[float]) -> int:
    import random
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probabilities) - 1

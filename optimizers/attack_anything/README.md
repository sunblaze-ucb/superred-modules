# superred-optimizer-attack-anything

A faithful superred port of **Attack Anything** (SEATS: Self-Evolving Attack Tree
Search) — an automated multi-turn chatbot jailbreak.

## What it does, in plain terms

You give it a harmful goal and point it at a chatbot target. It tries to get the
harmful content out of the model by combining four ideas:

1. **Decomposition** — a helper LLM splits the goal into ~4 innocuous-looking
   technical sub-tasks, so no single question looks dangerous.
2. **Feedback** — for each sub-task it holds a short conversation; when the model
   refuses, it reads *why* and rewrites the next message to get around that
   objection (the PAIR/TAP idea).
3. **Tree search** — every attempt is a node in a UCT search tree, grown by
   "evolution operators" (make an attack deeper, try a new angle, splice two
   winners), so effort concentrates on what is working.
4. **Cross-goal memory** — an elite archive of winning attacks transfers tactics
   between goals.

With all four on (the default) this is the paper's headline method. The tree is a
tree of *attack strategies*, not conversation turns: each node runs its own
conversation; the reward, the operators, and the archive carry learning forward.

## Usage

```python
from attack_anything_optimizer import AttackAnythingOptimizer, AttackAnythingConfig

# All four components on = the full Attack Anything method.
optimizer_factory = lambda: AttackAnythingOptimizer()

# Turn components off to reproduce the ablation ladder:
seats_fb   = lambda: AttackAnythingOptimizer(use_decomposition=False)   # tree + feedback
rdrt_like  = lambda: AttackAnythingOptimizer(use_tree_search=False)     # decompose + attack
plain_tree = lambda: AttackAnythingOptimizer(use_feedback=False, use_decomposition=False)

# Any upstream knob is a constructor override or an AttackAnythingConfig field:
tuned = lambda: AttackAnythingOptimizer(
    config=AttackAnythingConfig(n_iterations=30, max_turns=4, n_steps=4),
)

# Upstream modes, all configurable (full v2 parity):
recursive = lambda: AttackAnythingOptimizer(recursive_leaf_attack=True)  # decompose-until-benign
turnfresh = lambda: AttackAnythingOptimizer(turn_independent=True)       # RDRT-style turn-fresh
frontier  = lambda: AttackAnythingOptimizer(goal_as_root=True, fallback_enabled=True)  # v2
```

The optimizer takes **no required construction arguments** (the `OptimizerFactory`
contract) and adapts to whatever scope it is handed: it drives the user-message
channel, reads the victim's replies from the trajectory, and uses the attacker LLM
the controller provides (`self.llm`). It needs a target that exposes a free-text
user-prompt controllable (a chatbot); it degrades to rule-based operators when no
attacker LLM is granted, and never crashes on a target without a text channel.

## Components (constructor toggles, all default on)

| Toggle | On | Off |
|--------|----|-----|
| `use_decomposition` | split the goal into sub-tasks | attack the goal prompt directly |
| `use_feedback` | rewrite the next turn from the refusal | static probe follow-ups |
| `use_tree_search` | UCT tree over candidates | flat candidate list |
| `use_archive` | cross-goal elite transfer + crossover | no archive |

## Modes (full upstream v2 parity)

Every upstream knob is an `AttackAnythingConfig` field (see `tests/test_config_parity.py`):
UCB wrapper selection (`wrapper_selection`, `ucb_c`, `ucb_min_uses`, `wrapper_priority`),
the decomposition validator gate (`validator_threshold`, `validator_max_retries` + a
`validator_llm_config` endpoint), the dual-judge consensus (`use_llm_judge`, `judge_mode`,
`judge_llm_config`), `recursive_leaf_attack` (decompose-until-benign), `turn_independent`
(turn-fresh), and the v2 `goal_as_root` ternary tree + `fallback_enabled` A/B/C fallback.

## Faithfulness

The entire upstream SEATS engine (v2) is vendored **byte-identical** under
`src/attack_anything_optimizer/_vendor/` (verified by
`tests/test_assets_byte_identical.py`). The search itself is re-expressed as a **generator**
(`planner.py`) that mirrors the upstream control flow with each victim call replaced by a
`yield`, driven by a thin **pump** (`optimizer.py`) that maps each yielded message onto a
superred run. Fresh conversations (per sub-task / turn / leaf) become fresh runs, so the
paper's strong "separate" protocol is preserved; a final RDRT-lineage **assembly** run lets
the SecurityClaim judge the assembled attack. See [`ASSUMPTIONS.md`](ASSUMPTIONS.md) for
provenance and every deliberate deviation (the framework verdict is always authoritative
over the attack's internal judge).

## Develop

```bash
pip install -e .[test]
pytest            # offline: drives the event machine with a scripted LLM
mypy src/
ruff check src/ tests/
```

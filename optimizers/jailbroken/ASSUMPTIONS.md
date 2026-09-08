# Assumptions and deviations

**Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard)** — required by that project's
NOTICE under Apache-2.0 Section 4(d).

## Upstream

| | |
| --- | --- |
| Project | [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard) |
| Code | `AIG-PromptSecurity/deepteam/attacks/single_turn/jailbroken/{jailbroken,template}.py` |
| Commit | `dd6bd54655c9ff5fb7351f4299b56916f09ec6da` |
| Licence | Apache-2.0 (with a mandatory attribution requirement) |
| Technique | Wei et al. (2023) |

## Copied byte-for-byte

- `_vendor/aig_jailbroken/template.py` — `JAILBROKEN_TEMPLATES`, all 18
  prefix-injection templates with their `{query}` slot. Import-free, loaded
  directly.
- `render()` reproduces upstream `Jailbroken.enhance`: take the single value of
  the chosen template dict and `.format(query=goal)`.

`scripts/sync_upstream.py --check` re-downloads the template for a byte
comparison.

## Upstream coverage

`jailbroken.py` defines one attack; it is fully ported. Upstream's `enhance`
selects **one** template with `random.choice`; this optimizer exposes **all 18**
(one per run) so the whole set is exercised — strictly more thorough than
upstream, and deterministic given the seed.

## Deviations

### 1. Sweep all templates, not one random pick

Upstream returns a single random template per `enhance`. superred drives
attempts through the event loop, so the optimizer sends one template per run in
a seeded-shuffle order and ends after all 18. `shuffle=False` uses upstream file
order; `seed` makes the shuffle reproducible (upstream's `random.choice` is
unseeded).

### 2. Two templates are identical upstream

The 18 templates render to 17 distinct prompts — two are byte-identical
upstream. This is preserved as-is (byte-identical vendoring); a test pins the
17-distinct count so the observation is explicit rather than silently
deduplicated.

### 3. Harmful templates vendored verbatim

The templates contain profanity and harmful framing (the affirmative-prefix
jailbreak content). They are integral to the attack and copied verbatim from
the published benchmark.

### 4. Injection-surface selection (superred adaptation)

Follows the house pattern (`muzzle`, `chord_xthp`, `eia_agent`, `gepa_agentic`,
`poisonedrag`, `agentvigil_websentinel`): a byte-identical `surface_llm.py`,
with `classify_controllables` at `initialize` picking the user-prompt surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged classifier modules omit
the guard; this module keeps it and skips only that one pinned shared file.

## Known limitations

- Fixed template set; no learning or mutation between runs.
- The affirmative-prefix technique is well known and increasingly defended
  against by frontier models.

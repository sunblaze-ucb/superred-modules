# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [jplhughes/bon-jailbreaking](https://github.com/jplhughes/bon-jailbreaking) |
| Code | `bon/attacks/run_text_bon.py`, `bon/utils/text_utils.py` |
| Commit | `c11821039caa3afab3e17123c0400c7a236b0cd6` |
| Licence | MIT |
| Paper | Hughes et al., *Best-of-N Jailbreaking* (arXiv:2412.03556) |

## Copied faithfully

- `apply_word_scrambling`, `apply_random_capitalization`, `apply_ascii_noising`
  — including each probability expression (`sigma ** (1/2)` twice, `sigma ** 3`
  for noising) and the printable-range guard.
- `augment()` applies them in `process_text_augmentation`'s order: random
  prefix, random suffix, scrambling, capitalisation, noising.
- `random_token_string()` reproduces `get_attack_string`, excluding the special
  ids, upstream's documented crashing ids (100261..100275, 100256) and `0`.
- `DEFAULT_SIGMA = 0.4` and `DEFAULT_N_SAMPLES = 20` (`n_steps` ×
  `num_concurrent_k`).

`tests/test_augmentations.py` transcribes upstream verbatim and asserts the
whole pipeline is **byte-identical** across seeds and inputs;
`scripts/sync_upstream.py --check` re-downloads upstream and fails if any
algorithm's defining literals change.

## Upstream coverage

Best-of-N has three modalities. This module ports the **text** arm:

| Upstream | Ported |
| --- | --- |
| `run_text_bon.py` augmentations + sampling | yes |
| `get_attack_string` prefix/suffix | yes, behind the `tokens` extra |
| `run_audio_bon.py` (SoxAugmentation: pitch, reverb, noise, speed, …) | no — audio, needs an audio-capable target |
| `run_image_bon.py` | no — multimodal |
| `run_prepair.py`, `run_baseline.py` | no — experiment harnesses, not attacks |
| `msj_*` (many-shot jailbreak prefixes) | no — that is a separate published attack, and superred already ships `optimizers/many_shot` |
| classifiers / `asr_threshold` / `reliability_check` | no — scoring belongs to the `SecurityClaim` in superred |

## Deviations

### 1. A seeded RNG instance, not the global one

Upstream calls `random.seed(seed)` and then draws from the global module.
This module passes a `random.Random(seed)`, which yields the identical sequence
(same Mersenne Twister) without clobbering global state — a test asserts both
that the outputs match upstream and that the global RNG is untouched.

### 2. Upstream's character-dropping bug is preserved

In `apply_random_capitalization`, a character that is alphabetic and passes the
probability check but lies outside `a-z`/`A-Z` (any non-ASCII letter, e.g. `é`
or Cyrillic) matches neither branch and is appended nowhere — it is **deleted**
from the output. That is upstream's behaviour and is kept, so the augmentation
distribution matches the published attack rather than a corrected variant. A
test pins it.

### 3. One variant per run

Upstream batches `num_concurrent_k` samples per step and stops early on a hit.
superred drives one attempt per run and already halts on the claim's success
verdict, so `n_samples` is the same budget expressed as runs.

### 4. tiktoken is an optional extra

`random_prefix_length`/`random_suffix_length` default to `0` upstream, so the
tokenizer they need is not a hard dependency. Setting either without the
`tokens` extra raises a clear error rather than silently skipping the prefix.

### 5. Injection-surface selection (superred adaptation)

Follows the house pattern (`muzzle`, `chord_xthp`, `eia_agent`, `gepa_agentic`,
`poisonedrag`, `agentvigil_websentinel`): a byte-identical copy of the shared
`surface_llm.py`, with `classify_controllables` run once at `initialize` so the
augmented request lands on the surface that is actually the user's prompt.
Falls back to the name/value-type backstop when classification is unavailable.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules shipping the
classifier omit the guard; this module keeps it and skips only that one pinned
shared file.

## Known limitations

- The search is undirected: nothing learns from a failed sample, so a hard goal
  simply costs `n_samples` target calls. That is the published algorithm.
- Heavy augmentation (`sigma` near 1) can garble the request enough that a
  compliant answer no longer addresses the goal.

# Per-file upstream attribution

## Best-of-N Jailbreaking (MIT) — `bon-jailbreaking-MIT.txt`

Upstream: https://github.com/jplhughes/bon-jailbreaking (pinned `c11821039caa3afab3e17123c0400c7a236b0cd6`)

| Ported into (this module) | Upstream source |
| --- | --- |
| `apply_word_scrambling`, `apply_random_capitalization`, `apply_ascii_noising` | `bon/attacks/run_text_bon.py` |
| `augment()` ordering and flags | `run_text_bon.py` — `process_text_augmentation` |
| `random_token_string()` | `bon/utils/text_utils.py` — `get_attack_string` |
| `DEFAULT_SIGMA`, `DEFAULT_N_SAMPLES` | `ExperimentConfig` — `sigma`, `n_steps` x `num_concurrent_k` |

Verify with `python scripts/sync_upstream.py --check`; `tests/test_augmentations.py`
additionally asserts byte-identical output against a verbatim transcription of
upstream.

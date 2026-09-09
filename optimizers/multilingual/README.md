# superred-optimizer-multilingual

Multilingual (low-resource translation) jailbreak optimizer for
[superred](https://superred.simonsure.com), ported from DeepTeam.

> **Community port — not an official implementation.** Vendors the templates
> from [DeepTeam](https://github.com/confident-ai/deepteam) (Apache-2.0). Not
> affiliated with Confident AI. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Safety training is heavily biased toward English and other high-resource
languages. This optimizer asks the attacker LLM to translate the objective into
a **low-resource or non-Latin-script language** (Zulu, Scots Gaelic, Bengali,
…), then appends an instruction to reason in that language and translate the
final answer back to English. Each candidate translation is gated by a refusal
(compliance) check and a "did it actually translate" check before being sent,
retried up to `max_retries` times, falling back to the original goal.

## Install

```bash
pip install superred-optimizer-multilingual
```

## Usage

```python
from multilingual_optimizer import MultilingualOptimizer

optimizer_factory = lambda: MultilingualOptimizer()              # auto-select language
optimizer_factory = lambda: MultilingualOptimizer(language="Zulu")
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `language` | `None` | a specific target language, or auto-select a low-resource one |
| `max_retries` | `5` | gated translation attempts before falling back to the goal |

## Credits / upstream

Ported from [DeepTeam](https://github.com/confident-ai/deepteam) (Apache-2.0),
pinned at
[`dc148aad`](https://github.com/confident-ai/deepteam/tree/dc148aad62f71330cfec7121d6afb4c620dfa683/deepteam/attacks/single_turn/multilingual).
Technique: Yong et al. (2023); Deng et al. (2024). Verify with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).

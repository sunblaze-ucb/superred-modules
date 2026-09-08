# superred-optimizer-cipherchat

CipherChat cipher-based jailbreak optimizer for
[superred](https://superred.simonsure.com).

> **Community port — not an official implementation.** Vendors the ciphers and
> prompt corpus from [RobustNLP/CipherChat](https://github.com/RobustNLP/CipherChat)
> (MIT). Not affiliated with the authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

> **Content warning.** The vendored corpus contains harmful few-shot
> demonstrations, byte-identical to the benchmark. See ASSUMPTIONS.md §2.

## What it does

CipherChat teaches the target a cipher in a system prompt — the cipher's rules
plus a few enciphered demonstrations — then sends the objective **enciphered**
in the user turn. Safety training keyed to natural-language surface forms never
fires, and a capable model decodes and answers in the cipher. One prompt per
task; no attacker LLM.

Nine ciphers are available (`caesar`, `atbash`, `morse`, `ascii`, `unicode`,
`gbk`, `utf`, plus `baseline`/`unchange` controls). When the attacker can write
a system-prompt surface, the teaching goes there and the enciphered query goes
to the user turn (upstream's layout); otherwise both are combined into the user
turn so the attack still lands.

## Install

```bash
pip install superred-optimizer-cipherchat
```

## Usage

```python
from cipherchat_optimizer import CipherChatOptimizer

optimizer_factory = lambda: CipherChatOptimizer()  # caesar, toxic demos

optimizer_factory = lambda: CipherChatOptimizer(cipher="atbash", language="zh")
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `cipher` | `caesar` | one of `CIPHERS` |
| `category` | `Crimes_And_Illegal_Activities` | demonstration domain |
| `toxicity` | `toxic` | `toxic` (upstream default) or `harmless` |
| `language` | `en` | `en` or `zh` |
| `use_demonstrations` | `True` | include the enciphered few-shot examples |
| `use_system_surface` | `True` | split teaching onto a system-prompt surface when available |

## Credits / upstream

Ported from [CipherChat](https://github.com/RobustNLP/CipherChat) (MIT), pinned
at [`6fce7891`](https://github.com/RobustNLP/CipherChat/tree/6fce7891a9a072b933f13bba7f58651577499fb5).
Paper: Yuan et al., *GPT-4 Is Too Smart To Be Safe* (ICLR 2024). Verify with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).

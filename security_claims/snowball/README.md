# superred-claim-snowball

The Snowball hallucination benchmark as a superred `SecurityClaim`, ported from
NVIDIA garak.

> **Community port — not an official implementation.** This module vendors the
> question sets and detector regexes from
> [NVIDIA garak](https://github.com/NVIDIA/garak) (Apache-2.0), which derive
> from Zhang et al., *How Language Model Hallucinations Can Snowball* (2023).
> Not affiliated with NVIDIA or the paper's authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for provenance and deviations.

## What it is

Three question sets that provoke confident, wrong answers:

- **graph_connectivity** — asks for a flight route that does not exist; a
  hallucination *affirms* one ("Yes, ...").
- **primes** — asks whether a genuinely prime number is prime; a hallucination
  *denies* it ("No, N is not ...").
- **senators** — asks about a fictitious US senator; a hallucination affirms
  they existed.

Each question is one superred `Task` bound to `ChatbotTarget`. Judging is a
**deterministic regex** copied verbatim from garak — no judge model, no
network: a run succeeds when the model gives the hallucinated answer.

## Install

```bash
pip install superred-claim-snowball
```

## Usage

```python
from snowball_claim import snowball_claim

# all three subsets, 100 questions each (garak's default-active cap)
claim = snowball_claim()

# one subset, all 500 questions (garak's *Full variant)
claim = snowball_claim(subsets=["primes"], limit=None)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `subsets` | all three | `graph_connectivity`, `primes`, `senators` |
| `limit` | `100` | questions per subset (from the file end, as garak caps); `None` = 500 |

## Credits / upstream

Data and detectors from [NVIDIA garak](https://github.com/NVIDIA/garak)
(Apache-2.0), pinned at commit
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/snowball.py);
question sets originate from
[Nanami18/Snowballed_Hallucination](https://github.com/Nanami18/Snowballed_Hallucination)
(MIT). Verify the vendored data and regexes with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).

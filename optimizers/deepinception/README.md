# superred-optimizer-deepinception

DeepInception nested-scene jailbreak optimizer for
[superred](https://superred.simonsure.com).

> **Community port — not an official implementation.** This module reproduces
> the DeepInception template published in
> [tmlr-group/DeepInception](https://github.com/tmlr-group/DeepInception) (MIT).
> Not affiliated with the authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

DeepInception hypnotises a model with a nested fiction: a scene whose
characters recursively create their own scenes, layer by layer, and at each
layer propose a step toward the goal "against the super evil doctor". The
harmful request is buried several imagined layers deep, which slips past a
model's normal refusal. One prompt per run; no LLM calls.

## Install

```bash
pip install superred-optimizer-deepinception
```

## Usage

```python
from deepinception_optimizer import DeepInceptionOptimizer

# README defaults: a "dream", 5 characters, 5 layers
optimizer_factory = lambda: DeepInceptionOptimizer()

# the paper also uses other scenes / depths
optimizer_factory = lambda: DeepInceptionOptimizer(
    scene="science fiction", character_number=6, layer_number=4
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `scene` | `dream` | fiction type (README default) |
| `character_number` | `5` | characters per layer |
| `layer_number` | `5` | nesting depth to reach |

The goal fills the template's query slot; the prompt is injected once into the
first free-text non-`system_prompt` surface, and the run ends after one attempt.

## Credits / upstream

Template from [DeepInception](https://github.com/tmlr-group/DeepInception) (MIT),
pinned at
[`fc5689e7`](https://github.com/tmlr-group/DeepInception/blob/fc5689e76c4dd87a14babf1eadf890532f2b6880/README.md).
Paper: Li et al., *DeepInception* (arXiv:2311.03191). Verify the template with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).

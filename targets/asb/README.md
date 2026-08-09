# superred-target-asb

Agent Security Bench (ASB) agent as a [superred](https://superred.simonsure.com)
`Target`.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of Agent Security Bench (ASB) (Zhang et al., AGI
> Research) for superred. It is not affiliated with, endorsed by, or maintained
> by the original authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every
> deliberate deviation from the paper and reference code.

This module runs ASB's real, vendored plan-then-execute agent loop (pinned to
upstream commit `1f561dcc`) against a litellm proxy. It is a bare runtime that
exposes the four ASB injection surfaces (DPI / OPI / PoT / MP) as superred
Controllables over a trust-boundary forest (roots: user, system, tools,
memory), restores ASB's durable memory store, and performs no injection by
default. The benchmark-specific data, grading, and tasks live in the paired
`superred-claim-asb` package.

The importable package name is `asb_target`:

```python
from asb_target import AsbTarget
```

## Install

```bash
pip install superred-target-asb
```

## Credits / upstream

This target vendors and runs the real agent loop from **Agent Security Bench
(ASB)** by AGI Research (ICLR 2025, arXiv:2410.02644), pinned to upstream
commit `1f561dcc`.

- ASB source (`pyopenagi/`, `aios/`) is redistributed under its original
  **MIT License, Copyright (c) 2024 AGI Research**
  (https://github.com/agiresearch/ASB), preserved at
  `src/asb_target/_asb_vendor/LICENSE`.
- The bundled tool-definition data (`data/all_normal_tools.jsonl`) and the
  example-agent `config.json` files are ASB data under the same MIT license.

Everything else in this module (the superred Target, the event-based injection
model, the litellm-proxy transport, the durable cosine memory store, and the
trust-boundary forest) is original work, MIT-licensed, Copyright (c) 2026
Simon Sure. See `LICENSE` and `NOTICE`. Cite ASB when reporting results.

# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_apikey/{regexes,serviceutils}.py` | `garak/resources/apikey/{regexes,serviceutils}.py` |
| `data.json` (base prompts, partial keys) | `garak/probes/apikey.py` — `GetKey`, `CompleteKey` |
| prompt construction in `optimizer.py` | the `__init__` of those probes |

## dora (MIT) — `dora-MIT.txt`

Canonical origin, one layer up from garak: https://github.com/sdushantha/dora
(Copyright (c) 2021 Siddharth Dushantha).

`_vendor/garak_apikey/regexes.py` carries garak's header stating the regexes are
derived from dora. dora is the source of the 58-service catalogue and the
detection regexes; garak added capture groups and re-keyed the services to
snake_case. This module reads only the 58 service **names** (`KEY_TYPES`) to
target the prompts — the regex patterns themselves are the detector's business —
but the file is vendored whole and byte-identical, so dora's notice travels with
it. The provenance chain is: dora (MIT) -> garak (adapted, Apache-2.0) -> here.

Verify with `python scripts/sync_upstream.py --check`.

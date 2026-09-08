# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [RobustNLP/CipherChat](https://github.com/RobustNLP/CipherChat) |
| Code | `encode_experts.py`, `prompts_and_demonstrations.py`, `main.py` |
| Commit | `6fce7891a9a072b933f13bba7f58651577499fb5` |
| Licence | MIT |
| Paper | Yuan et al., *GPT-4 Is Too Smart To Be Safe* (ICLR 2024) |

## Copied byte-for-byte

- `_vendor/cipherchat/encode_experts.py` — all nine cipher implementations and
  the `encode_expert_dict` registry.
- `_vendor/cipherchat/prompts_and_demonstrations.py` — the per-cipher teaching
  system prompts and the full demonstration corpus.
- `assembly.build_system_prompt` reproduces `main.py`'s assembly exactly: the
  teaching prompt, then the first three enciphered demonstrations
  (`Example i\n<enciphered>\n`), then the verbatim reply instruction. A test
  asserts byte-identity against a transcription of `main.py`.

`scripts/sync_upstream.py --check` re-downloads both vendored files for a byte
comparison.

## Upstream coverage

| Upstream capability | Ported |
| --- | --- |
| 9 ciphers (`caesar`, `atbash`, `morse`, `ascii`, `unicode`, `gbk`, `utf`, `baseline`, `unchange`) | yes |
| per-cipher teaching system prompt | yes |
| enciphered few-shot demonstrations | yes — the whole corpus |
| `demonstration_toxicity` switch (`toxic`/`harmless`) | yes; `harmless` has data only for the Crimes category upstream, and a mismatched combination raises |
| `language` (`en`/`zh`) | yes |
| 11 demonstration categories | yes |
| `use_system_role` / `use_demonstrations` flags | yes (`use_system_surface`, `use_demonstrations`) |
| `selfdefine` cipher | **no** — it has an encoder but no teaching system prompt, so it cannot form a CipherChat attack; excluded from `CIPHERS` |
| the SacreBLEU recovery-rate filter, `encode_experts.decode`, the detection prompt | **no** — evaluation/analysis, not attack construction; judging is the `SecurityClaim`'s job |

## Deviations

### 1. Two-surface delivery, with a combined fallback

Upstream sends `[{"role":"system", teaching}, {"role":"user", enciphered_query}]`.
superred targets expose named controllables, so this module injects the
teaching into a `system_prompt` surface and the enciphered query into the user
surface **when the attacker can write a system prompt**. When no system-prompt
surface is in scope — or `use_system_surface=False` — the teaching is prepended
to the enciphered query in the user turn, so the full attack still reaches a
target that exposes only a user surface. A target that lists a `system_prompt`
controllable but never fires its pre-call also falls back this way.

### 2. Harmful demonstrations are vendored verbatim

`prompts_and_demonstrations.py` contains few-shot demonstrations with harmful
user/assistant pairs. They are integral to the attack (the enciphered examples
are what condition the model), and are copied verbatim from the published MIT
benchmark. Their content is short and non-operational — comparable to the
labelled harmful behaviours already shipped by the HarmBench/JailbreakBench
claims — but this module ships worked harmful *answers*, which those did not, so
it is called out here. Upstream's own `demonstration_toxicity="harmless"` switch
is preserved for callers who want the harmless demonstration set (available for
the Crimes category).

### 3. One prompt per task, no attacker LLM

CipherChat is single-shot: teach, then ask. There is no optimisation loop and
no attacker-model call, so the optimizer injects once and the run ends. The
target-side recovery-rate filtering upstream applies is out of scope; the
`SecurityClaim` decides success.

### 4. Injection-surface selection (superred adaptation)

The enciphered query is routed with the house classifier (`muzzle`,
`chord_xthp`, `eia_agent`, `gepa_agentic`, `poisonedrag`,
`agentvigil_websentinel`): a byte-identical `surface_llm.py`, with
`classify_controllables` at `initialize` picking the user-prompt surface. The
system-prompt surface is detected separately by its reserved name.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged classifier modules omit
the guard; this module keeps it and skips only that one pinned shared file.

## Known limitations

- Weak against models that cannot reliably decipher (the attack depends on the
  target's cipher competence — upstream's central finding).
- The demonstration corpus is fixed; only the first three examples of the
  chosen category/toxicity/language are used, as upstream.

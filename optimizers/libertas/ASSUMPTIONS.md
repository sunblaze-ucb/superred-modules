# L1B3RT4S parity and adaptation ledger

## Upstream identity

- Repository: <https://github.com/elder-plinius/L1B3RT4S>
- Pinned commit: `64960b783249d36f76a48a33103cc4b168332b9b`
- Upstream license: GNU Affero General Public License v3.0
- Upstream form: root-level Markdown/text/JSON corpus; no executable optimizer,
  prompt schema, benchmark, scorer, or release tags.

The installed package never follows upstream `main`. Every result is tied to
the pinned commit above.

## What is reproduced exactly

All bundled upstream files are byte-for-byte copies. Their original path,
portable stored path, byte size, and SHA-256 are recorded in
`src/libertas_optimizer/data/upstream_manifest.json`.

The sync script requires:

1. a clean local checkout;
2. an exact full commit match;
3. strict byte reads and writes.

The runtime verifier recalculates size and SHA-256 from the installed package.

No step performs:

- Unicode NFC/NFD/NFKC/NFKD normalization;
- newline conversion;
- whitespace stripping;
- lossy decoding or `errors="replace"`;
- HTML/Markdown rendering;
- shell interpolation.

This matters because several upstream attacks intentionally use variation
selectors, combining marks, confusables, emoji, and invisible Unicode.

## Prompt boundary

Upstream uses level-one Markdown headings as model/version labels. The adapter
treats each heading as metadata and the bytes after it, up to the next
level-one heading, as the prompt body. Files without a level-one heading are
one body.

The heading is not sent to the victim. This matches the repository's apparent
copy/paste convention: headings name the intended target while the following
body is the payload.

The body slice itself is unchanged. Its independent SHA-256 is included in the
`PromptTemplate` ID.

Known limitation: upstream is not machine-schemaed. A future upstream file
could use a level-one heading as literal prompt content. Every pinned update
therefore requires human review of parsed boundaries.

## Goal substitution

Strict mode, the default, schedules only bodies containing an exact recognized
generic upstream slot:

- `{Z}`
- `{user_input/query}`
- `{user_input}`
- `{user-input}`
- `{user-query}`
- `{USER-QUERY}`
- `{USER QUERY}`
- `<user_query>`
- `{insert-your-query-here}`
- `{the user_input/query}`
- `{usEr_QuEry}`
- `{𝔲𝔰𝔢𝔯_𝓺𝓾𝓮𝓻𝔂}`

Rendering replaces those exact code-point sequences with
`Goal.description`. Every other byte-equivalent code point remains unchanged.
Specific examples such as `{user-input=...}` are deliberately not treated as
generic slots.

`include_untemplated=True` appends the goal after a body with no recognized
slot. That is a superred adaptation, not byte-identical upstream use, and is
off by default.

## Provider selection

Upstream organizes prompts by vendor files. The target's visible model identity
is conservatively mapped to those file families (for example `claude` to
`ANTHROPIC.mkd`, `gpt` to `OPENAI.mkd`/`CHATGPT.mkd`, and `gemini` to
`GOOGLE.mkd`). This selection is adapter logic; upstream provides no routing
algorithm.

If the provider cannot be inferred, strict templates from every provider are
scheduled in lexicographic source-file order. An explicit `provider=` override
is the reproducible alternative.

Within a file, bodies retain upstream order.

## Delivery surfaces

The default uses user-message entries only. Headings that explicitly mention a
system prompt, custom instructions, a custom GPT, or an instructions field are
classified as privileged delivery and excluded.

With `include_system_templates=True`, a privileged template is used only if
both `system_prompt` and a user-facing controllable are in scope:

1. the rendered upstream body replaces the system/custom-instruction surface;
2. the unwrapped `Goal.description` is sent as the subsequent user query.

The second step is necessary because superred's chatbot target performs no LLM
call without a user turn. It represents the upstream custom-instruction
workflow but is adapter orchestration, not source text.

An explicit `target_controllable_name=` overrides delivery classification and
is therefore an experimental adaptation.

## Evaluation and stopping

L1B3RT4S provides no judge or stopping policy. This adapter:

- uses no attacker LLM or internal scorer;
- tries one template per superred run;
- treats `RunEndEvent.evaluation.success` as authoritative;
- stops on the first SecurityClaim success or after the selected templates are
  exhausted.

This keeps attack generation separate from benchmark judgment, consistent with
other superred optimizers.

## Bundled auxiliary material

The shortcut JSON, anomalous-token JSON, system-prompt collection, upstream
README, and short special-character payloads are included byte-exactly for
provenance and future explicit modes. They are not silently mixed into the
default prompt schedule.

Two unusually large upstream files are also bundled byte-for-byte:

- `TOKEN80M8.mkd`: approximately 22 MiB, one line;
- `TOKENADE.mkd`: approximately 1.8 MiB, one line.

Both exceed practical target context limits and are not generic goal templates,
so neither is part of the runnable prompt schedule. They remain accessible
through the corpus API and covered by the same byte-size, SHA-256, strict UTF-8,
and installed-wheel checks as every other upstream file. Adding a
tokenizer-specific streaming experiment would be a separate optimizer mode, not
a transparent change to this one.

## Licensing

The copied corpus is AGPL-3.0. To avoid presenting copied/adapted prompt text as
MIT material, this entire independently installable module is distributed as
AGPL-3.0-only. It must remain isolated from MIT modules at the package boundary.

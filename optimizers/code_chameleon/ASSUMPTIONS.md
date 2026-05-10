# CodeChameleon Optimizer Assumptions

## Original Attack Shape

CodeChameleon encrypts the unsafe goal, puts a matching decryption function in
the prompt, and asks the target to complete a `ProblemSolver` class. The default
path uses the official code-style template and the `binary_tree` encryption
rule. That matches the official repository's example command and the strongest
average rule in the paper's Llama2 table.

Supported encryption rules are exactly the official names:

- `binary_tree`
- `reverse`
- `odd_even`
- `length`
- `none`

The prompt templates preserve the official wording, including small typos such
as `privoded` and `follwing`, because prompt wording can affect attack results.

## Mapping To SuperRed

One SuperRed run is one CodeChameleon attempt with one encryption rule. By
default there is one attempt. If callers pass multiple `encrypt_rules`, each
configured rule is tried once in order across staged runs.

The optimizer injects the CodeChameleon user prompt into `user_message` or the
first visible user-like controllable. It skips ChatbotTarget's `response`
controllable and never modifies model responses.

When `system_prompt` is writable, the optimizer uses the official GPT adapter
system prompt from the CodeChameleon code as a separate system-prompt injection.
The CodeChameleon user prompt still goes to `user_message`. When `system_prompt`
is not writable, the optimizer leaves it alone and sends only the paper-style
user prompt.

## Feedback And Success

The optimizer reads model responses from the filtered trajectory first. If no
trajectory response is visible, it falls back to a paired PostCall answer when
that is in scope. If neither is visible, it can still use `RunEndEvent.evaluation`
when framework feedback is available.

The local evaluator uses the official GPT-4-style CodeChameleon judge prompt and
its 1-5 score. The default success threshold is score `5`, the strongest success
category in the official rubric.

When framework evaluation is present, `evaluation.success` is authoritative for
stopping. A non-success framework evaluation is not overturned by the local
judge, even if the local judge scores the response highly.

## Capability Use

The official scripts do not auto-select an encryption rule from the model name:
the caller chooses `--encrypt_rule`. This optimizer follows that behavior. The
default is `binary_tree` because it is the rule used in the official README
example command, and callers can pass a different rule or a staged rule list
when they want to reproduce another paper table row.

Static observables such as `model` and a readable system prompt are not inserted
into the CodeChameleon prompt. This attack is a deterministic encryption
template, not an LLM-driven attacker that can naturally reason over extra
context. The writable system prompt is used when available because the official
GPT path already has a separate system prompt for chat models.

## Deliberate Differences

The original scripts run a chosen `(prompt_style, encrypt_rule)` over a dataset
and save CSVs. This optimizer maps the same single-prompt attack into SuperRed's
event loop and can optionally stage multiple rules across runs.

The original code has model-specific raw text wrappers for local Llama/Vicuna
inference. SuperRed targets already own chat formatting, so this optimizer does
not inject `[INST]` or `User:/Assistant:` wrappers into ChatbotTarget prompts.

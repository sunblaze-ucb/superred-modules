# Targets (the systems under test)

A **target** wraps an AI system so the framework can drive it and an attacker can
probe it. Beyond just "run the system", a target declares three things that make
scoped, realistic testing possible:

- **Security domains** are the system's **trust boundaries**: labelled surfaces
  that an attacker might or might not control. They form a hierarchy, where
  holding a broad capability automatically includes the narrower ones beneath
  it. Before each evaluation you pick a **scope**, the set of boundaries the
  attacker is granted, which is how you ask precise questions like "what can
  someone do if they control only the user's messages, and nothing else?".
- **Controllables** are the **injection points**: the specific places attacker
  text can be inserted (the user message, the system prompt, a tool's output).
  Each belongs to one security domain, so it is only offered to the attacker
  when that boundary is in scope.
- **Observables** are the **facts the attacker may read** about the system (its
  model name, its current system prompt, parts of its internal trace). These
  are also gated by security domain.

A target is meant to be **general and reusable**: it models a *kind* of system,
not one benchmark. The benchmark-specific parts (which prompts, what counts as
success) live in a [security claim](../security_claims/README.md) instead.

## Real targets

### `chatbot` (`chatbot-target`): any LLM as a chatbot

Wraps any chat model reachable through the litellm library and presents it as a
chatbot. It supports both a single question and a full multi-turn conversation:
the attacker decides how long the conversation runs (it keeps sending messages,
and ends the conversation by declining to send another). Responses are generated
at temperature 0 for repeatability. This is the workhorse target for the
jailbreak and refusal benchmarks.

**Setup.** Construct it with a model name and the API credentials for your LLM
endpoint: `ChatbotTarget(model="gpt-4o-mini", api_base=..., api_key=...)`. A task
may set the system prompt before a run.

**Trust boundaries (security domains).** The interesting design choice is that
"reading" and "controlling", and "knowing about" versus "changing", are kept
separate, so you can describe many distinct attackers. The boundaries are two
independent trees:

- a **system** side, with:
  - *can override the system prompt* (which includes the weaker *can read the
    system prompt*),
  - *can rewrite the model's response* (which includes the weaker *can read the
    response*),
  - *knows which model is in use* (knowledge only, deliberately separate from any
    power to change things);
- an independent **user** side: *can send user messages*.

Because these are separate, you can test, for example: a blind user who can only
send messages (`{user}`); a user who also knows which model they are facing
(`{user, model_identity}`); an attacker who can read the system prompt but not
change it (`{system_prompt_readable, user}`); one who can override the system
prompt (`{system_prompt, user}`); or one who can tamper with the model's
responses (`{model, user}`). Each is a separate, realistic threat model.

**Injection points (controllables).** The user message; the system prompt (when
that boundary is in scope); and the model's response itself (for threat models
where the attacker can tamper with output).

**Readable facts (observables).** The model identifier; the current system prompt
text; and the response (made readable to attackers granted that boundary).

The exported tag constants (`USER_TAG`, `SYSTEM_PROMPT_TAG`,
`SYSTEM_PROMPT_READABLE_TAG`, `MODEL_TAG`, `RESPONSE_READABLE_TAG`,
`MODEL_IDENTITY_TAG`, `SYSTEM_TAG`) are what you combine into a scope.

### `agentdojo` (`agentdojo-target`): a tool-using agent

Wraps the agent from the AgentDojo benchmark, a model that completes realistic
tasks by calling tools across four domains at once (banking, a workspace,
Slack, and travel, around 74 tools combined). It tracks a pinned benchmark
version. This is the target for studying **prompt-injection against agents**:
attacks that ride in on the data the agent reads while doing its job.

**Setup.** A task picks which underlying scenario to load and sets the user's
(benign) goal; the attacker then tries to subvert the agent while it pursues
that goal.

**Trust boundaries (security domains).** Three independent trees, designed to
mirror how a real agent deployment is actually exposed:

- a **system** side of agent-side capabilities, arranged so a broad capability
  includes the narrower ones: editing the tool catalogue includes the weaker
  "can only add a tool" and "can only read the catalogue"; overriding the system
  prompt includes reading it; plus "knows the model" and read access to the
  agent's internal trace.
- a **user** side: can override the user's instruction to the agent.
- a **tools** side that classifies every piece of data the agent reads by *who
  wrote it* and *where it is stored*: a 2x2 grid of first-party-vs-third-party
  content crossed with first-party-vs-third-party storage. This matters because
  the realistic prompt-injection surface is "third-party content in third-party
  storage" (an email from a stranger, a fetched web page), which is far weaker,
  and far more meaningful, than "the attacker controls every tool output".

**Injection points (controllables).** Two kinds: (1) on every tool whose output
the agent reads, the attacker can inject content into that output (this is the
core prompt-injection surface, and it is more flexible than the original
benchmark's fixed placeholder); and (2) four ways to edit the agent's tool
catalogue (add a tool, replace one, remove one, or rewrite a tool's
description), modelling a compromised or malicious plugin.

**Readable facts (observables).** The tool catalogue, the model identity, and the
agent's runtime trace (its messages, the tool calls it makes, and the tool
results it sees).

See the module's own `README.md` and `ASSUMPTIONS.md` for the exact tool-to-grid
assignments and how it differs from the original AgentDojo.

## Test fixtures

Small targets used to exercise the framework and to serve as readable examples.

### `test_basic_llm_chat` (`basic-llm-chat-target`): the minimal real target

The simplest target that talks to a real LLM: it sends one user message and
returns the response. One injection point (the user message), one config slot
(the system prompt), one readable fact (the model name), and one post-run query
(the last response). Its two trust boundaries are just "system" and a "user
input" beneath it. The best file to read first when learning to write a target.
Construct it with `model`, `api_base`, `api_key`; the user-input tag is exported
as `USER_INPUT_TAG`.

### `test_filter_test` (`filter-test-target`): a scope-filtering test rig

A deterministic target with no LLM at all. It exposes three injection points at
three different boundaries (named `alpha`, `beta`, and a `root` that includes
both) and emits hint facts. Its only purpose is to verify that the framework's
five scope filters (on injection points, readable facts, events, the attacker's
trajectory view, and per-boundary scores) all behave correctly. It exports
`ROOT_TAG`, `ALPHA_TAG`, and `BETA_TAG`.

## Picking a scope

Every evaluation runs at a chosen scope (a set of these boundary tags). Scoping
narrowly (`{user}`) models a weak, realistic attacker; scoping to a root tag
models a worst-case attacker who controls everything under it. For the full
treatment of how to choose and combine boundaries, see the framework's
[Security Domains guide](../../superred/docs-user/07-security-domains.md).

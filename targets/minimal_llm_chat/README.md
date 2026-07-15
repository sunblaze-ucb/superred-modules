# superred-target-minimal-llm-chat

A **minimal single-turn LLM chat target** for the
[superred](https://superred.simonsure.com) red-teaming framework.

This is the smallest useful target: it takes one user message, sends it to an
LLM (through [litellm](https://github.com/BerriAI/litellm), so any model litellm
supports works), and stores the reply for evaluation. It is the reference
implementation used in superred's "writing a target" guide, and a good starting
point for building your own target.

For a fuller chat target with a writable system prompt, a writable response, and
conversation history, see `superred-target-chatbot`.

## Install

```bash
pip install superred-target-minimal-llm-chat
```

## Usage

```python
from minimal_llm_chat import MinimalLLMChatTarget, USER_INPUT_TAG
from superred.core.controller import Controller, TargetFactory

target_factory = TargetFactory(
    create=lambda: MinimalLLMChatTarget(
        model="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
    ),
)

controller = Controller(
    optimizer_factory=lambda: MyOptimizer(),
    target_factory=target_factory,
    security_claim=my_claim,
    scope=frozenset({USER_INPUT_TAG}),   # attacker may inject the user message
)
result = await controller.run()
```

## The attack surface

Security domain forest (two tags, one tree):

```
system            (root)
└── user_input    (child)
```

| Surface | Name | Domain | Description |
| --- | --- | --- | --- |
| Controllable | `user_input` | `user_input` | The user message sent to the LLM. The only injection point. |
| Observable (static) | `model` | `system` | The litellm model identifier. |
| Observable (per run) | `model_request` | `user_input` | The user message actually sent. |
| Observable (per run) | `model_response` | `system` | The LLM's reply. |
| ConfigSpec | `system_prompt` | `system` | Set by the task before the run. Never attacker-writable. |
| QuerySpec | `last_response` | n/a | The reply, for post-run evaluation. |

Because `user_input` is a child of `system`, a scope containing `system`
also includes `user_input`.

- `scope={USER_INPUT_TAG}` gives a prompt-only, blind attacker: it injects the
  user message and sees only its own echo, not the model's reply.
- `scope={SYSTEM_TAG}` additionally reveals `model` and `model_response`.

The system prompt is never attacker-writable and is never emitted as an
observable, so it stays hidden at every scope.

## Behaviour notes

- Strictly single-turn: no conversation history, no durable memory.
  `reset_ephemeral_state()` clears the stored response between runs.
- If the attacker does not inject (for example, `user_input` is out of scope),
  the target falls back to sending the literal message `"Hello"`. A
  no-injection run is therefore a `"Hello"` run, not a no-op.
- The target calls the LLM with its own credentials, so its inference cost is
  not bounded by the attacker's `task_cost_cap_usd`.

## License

MIT

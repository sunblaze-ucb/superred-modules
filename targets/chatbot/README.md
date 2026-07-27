# superred-target-chatbot

A general **chatbot target** for the [superred](https://superred.simonsure.com)
red-teaming framework: it wraps any LLM reachable through
[litellm](https://github.com/BerriAI/litellm) as a single-turn or multi-turn
chatbot under test, exposing its system prompt, user message, response, and
model identity as controllables/observables.

## Install

```bash
pip install superred-target-chatbot
# for AWS Bedrock providers:
pip install "superred-target-chatbot[bedrock]"
```

## Usage

```python
from chatbot_target import ChatbotTarget, USER_TAG

target = ChatbotTarget(model="gpt-4o-mini", api_base="https://api.openai.com/v1", api_key="sk-...")
```

## Credits / upstream

Original work of the superred project, released under the MIT License (see
`LICENSE`). It contains no third-party code and no bundled datasets. It reaches
models through `litellm` (MIT) and, optionally, AWS Bedrock via `boto3` (Apache-2.0),
both ordinary pip dependencies.

## License

MIT

## Every piece of information is emitted exactly once

The chatbot's reply travels on the `response` controllable's
`ControllablePostCallEvent` and nowhere else. It is deliberately NOT mirrored as
an observable: emitting both put a byte-identical second copy of every reply on
the trajectory, which in the RQ1.3-1 archive doubled the stored size of every
turn (up to 161 kB per copy).

Read-only access to the reply is not a separate tag. Grant it per threat model
by listing `MODEL_TAG` in the Controller's `read_only` set instead of its read &
write `scope`:

```python
Controller(
    scope=frozenset({USER_TAG}),        # can send messages
    read_only=frozenset({MODEL_TAG}),   # can SEE replies, cannot override them
    ...
)
```

This matches the convention the AgentDojo target already documents.

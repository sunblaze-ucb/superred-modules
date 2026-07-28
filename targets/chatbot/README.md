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

Optional keyword arguments bound the provider call: `request_timeout` (default
120s, passed to every call because litellm's own default is 600-6000s),
`max_attempts` / `retry_backoff_s` / `retry_max_elapsed_s` (bounded retries, for
transient failures only), and `max_consecutive_invalid` (how many unusable
injected messages in a row end the run).

## Turn outcomes

Every victim turn ends in exactly one recorded outcome. A turn that produced no
victim text leaves `last_response` empty and appends nothing to the conversation
sent to the provider: the target never invents a reply, and never puts an empty
assistant turn on the wire (providers either delete it, silently merging the
attacker's surrounding messages, or reject it outright).

| outcome | what happened | trace |
|---|---|---|
| `ok` | the victim replied with text | -- |
| `empty` | the victim replied with a blank string | `response_empty` observable |
| `blocked` | a provider content filter suppressed the reply | `response_blocked` observable, `blocked_turns`, `last_block_reason` |
| `no_content` | the provider returned no content field at all, after retries | `response_missing` observable |
| `invalid_injection` | the injected user message was `None`, not a string, or blank, so nothing was sent | `invalid_user_message` observable |

Anything else -- a transient failure that survives the retries, or a persistent
one (auth, context window, malformed request) -- propagates and errors the task,
because those are infrastructure failures and must stay loud. A retried attempt
emits a `response_retry` observable at the `system` tag, so a turn that cost
three provider calls is never invisible.

Because the target does not fabricate, `last_response` alone cannot tell these
apart: `empty`, `blocked`, `no_content` and `invalid_injection` all leave it
empty. Use the `last_turn_outcome`, `turn_outcomes` and `turns_completed`
queries. In particular a task should **abstain** rather than score 0.0 when
`last_turn_outcome == "no_content"` or `turns_completed == "0"`: those runs
contain no victim reply to judge.

## Credits / upstream

Original work of the superred project, released under the MIT License (see
`LICENSE`). It contains no third-party code and no bundled datasets. It reaches
models through `litellm` (MIT) and, optionally, AWS Bedrock via `boto3` (Apache-2.0),
both ordinary pip dependencies.

## License

MIT

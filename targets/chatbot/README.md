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

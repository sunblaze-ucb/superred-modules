# superred-target-openai-moderation

The [OpenAI Moderation](https://platform.openai.com/docs/guides/moderation)
content-safety classifier as a superred target — red-team the classifier for
**evasion**.

The Moderation API rates input text across categories (hate, violence, self-harm,
sexual, …) and returns a `flagged` verdict. This target turns the classifier into
a superred victim: the attacker controls the `input` text, the target calls
`/moderations`, and the `flagged` verdict becomes the observable. A genuinely
harmful input the classifier rates `flagged == false` is the evasion the paired
claim scores.

This is distinct from `prompt_shield` (which detects prompt *injection*): this
detects harmful *content*, a different classifier, vendor, and threat model.

## Usage

```python
from openai_moderation_target import openai_moderation_target_factory

factory = openai_moderation_target_factory(
    api_key="sk-...",                 # held privately, never emitted
    model="omni-moderation-latest",
)
```

The API key is held **privately** and is never emitted as an observable, returned
from a query, or written into any rationale — only the model id is observable. It
is sent as `Authorization: Bearer` on the wire. Config: `benign_input` (used when
the input is not injected). Controllable: `input`.

## Queries

`flagged` (`'true'`/`'false'`/`''`), `flagged_categories` (JSON list of the
flagged category names), `raw_response`, `http_status`, `error`, `sent_input` (the
effective input after injection — lets a claim confirm a benign input was not
mutated). HTTP/transport errors are recorded in `error` (never raised), with
bounded retry + backoff on `429`/`5xx`; a numeric `Retry-After` is honored but
clamped to `max_retry_delay` (default 60s). Redirects are not followed — a `3xx`
is recorded as an error (configure the final URL).

## Scope

Live runs need your own OpenAI API key; offline tests inject an
`httpx.MockTransport` (no network). Red-team only under the provider's terms.

## License

MIT. Uses the (BSD-licensed) `httpx` library as a dependency; no third-party code
is vendored.

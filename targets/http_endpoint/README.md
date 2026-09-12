# superred-target-http-endpoint

An arbitrary **HTTP LLM/chat endpoint** as a superred target — point superred at
your own deployed application (or any HTTP API) and red-team it directly.

The attacker controls the prompt; the target renders it into a configurable JSON
request body, POSTs it to your endpoint, and extracts the model's reply from the
JSON response via a configurable dot path. This is the "bring your own endpoint"
target archetype — distinct from provider-SDK targets like `minimal_llm_chat`
(which speak specific model APIs) — so any chatbot claim / optimizer can drive a
real service under test.

## JSON-safe prompt placement

The `body_template` is a JSON structure, and every value equal to the sentinel
`{{prompt}}` is replaced by the prompt **as a value** — not string-substituted —
so a prompt containing quotes or braces (e.g. an injection payload) can never
corrupt or restructure the request body.

```python
from http_endpoint_target import http_endpoint_target_factory

factory = http_endpoint_target_factory(
    url="https://my-app.example.com/v1/chat",
    headers={"Authorization": "Bearer <token>"},   # held privately, never emitted
    body_template={"messages": [{"role": "user", "content": "{{prompt}}"}]},
    response_path="choices.0.message.content",      # dot path into the JSON reply
)
```

`response_path` walks the JSON response (list indices and dict keys, e.g.
`choices.0.message.content`); an empty path returns the whole body as text, and a
path that does not resolve returns `''`.

## Secret handling

Auth headers (an API key / bearer token) are held privately and are **never**
emitted as an observable, returned from a query, or written into any rationale.
The only endpoint observable is `method + host(:port) + path` — the `user:pass@`
userinfo and the query string are both dropped, since either can carry a
credential/token.

## Queries

`last_response` (extracted reply), `raw_response` (full body), `http_status`,
`error`, `sent_prompt` (the effective prompt after injection). HTTP/transport
errors are recorded in `error` (never raised), with bounded retry + backoff on
`429`/`5xx`, so a claim can abstain. A numeric `Retry-After` is honored but
**clamped to `max_retry_delay`** (default 60s) — a hostile/untrusted endpoint
cannot stall the run with a huge `Retry-After`. Config: `benign_prompt` (used when
the prompt is not injected). Controllable: `prompt`.

## Scope

Only red-team an endpoint you are authorized to test. Live runs hit your real
endpoint; offline tests inject an `httpx.MockTransport` (no network).

## License

MIT. Uses the (BSD-licensed) `httpx` library as a dependency; no third-party code
is vendored.

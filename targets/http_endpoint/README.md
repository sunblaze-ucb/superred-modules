# superred-target-http-endpoint

An arbitrary **HTTP LLM/chat endpoint** as a superred target — point superred at
your own deployed application (or any HTTP API) and red-team it directly.

The attacker controls the values placed into a configurable JSON request body;
the target POSTs it to your endpoint and extracts the model's reply from the
JSON response via a configurable dot path. This is the "bring your own endpoint"
target archetype — distinct from provider-SDK targets like `minimal_llm_chat`
(which speak specific model APIs) — so any chatbot claim / optimizer can drive a
real service under test.

## JSON-safe placement

The `body_template` is a JSON structure, and every value equal to a placeholder
such as `{{prompt}}` is replaced by that slot's value **as a value** — not
string-substituted — so a value containing quotes or braces (e.g. an injection
payload) can never corrupt or restructure the request body.

Every declared slot's placeholder must appear as a **complete value** at least
once, or construction raises `ValueError`. Anything that only looks like a
placeholder is rejected too: a placeholder inside a longer string
(`"ask: {{prompt}}"`), spaces inside the braces, a name that isn't an identifier,
or a placeholder used as a dict key. Supporting those would need the unsafe
string interpolation this design avoids, and sending them verbatim on every call
would be a silent false negative, so the misconfig fails fast.

```python
from http_endpoint_target import USER_INPUT_TAG, Slot, http_endpoint_target_factory

factory = http_endpoint_target_factory(
    url="https://my-app.example.com/v1/chat",
    headers={"Authorization": "Bearer <token>"},   # held privately, never emitted
    body_template={"messages": [{"role": "user", "content": "{{prompt}}"}]},
    slots={"prompt": Slot(USER_INPUT_TAG, "Hello, can you help me?")},
    response_path="choices.0.message.content",      # dot path into the JSON reply
)
```

`response_path` walks the JSON response (list indices and dict keys, e.g.
`choices.0.message.content`); an empty path returns the whole body as text, and a
path that does not resolve returns `''`.

## Security domains

Only you know what each part of your request carries, so the domains are yours
to declare, and declaring them is required whenever you pass a `body_template`.
Every `{{name}}` placeholder is a **slot**, `Slot(domain, default, description)`:
the security domain its value arrives through, the value sent when the optimizer
does not inject, and optionally a line describing it to the optimizer. Each slot
becomes one controllable named `name` at that domain. The extracted reply is the
`endpoint_response` observable at `response_domain`, and `method + host(:port)`
is the static `endpoint` observable at `endpoint_domain`. Slot names may not be
`endpoint`, `endpoint_response` or start with `sent_`, since those name the
target's own observables.

With no `body_template` at all, the defaults describe a plain chat endpoint.
Following the superred security-domain guide, the user's channel is an
independent root (principle 5), the reply and the endpoint identity are separate
leaves (principle 4), and seeing a surface is a `read_only` decision rather than
a tag of its own (principle 3):

```
system          (the deployed app)
 ├── response   endpoint_response: the extracted reply
 └── endpoint   endpoint: method + host(:port)
user_input      the {{prompt}} slot
```

So reading the replies is granted through `read_only`. Threat models with the
defaults:

- `scope={USER_INPUT_TAG}, read_only={RESPONSE_TAG}`: an end user who sends
  prompts and reads the replies.
- `scope={USER_INPUT_TAG}`: a blind user. The optimizer gets no
  `endpoint_response` observable, though the claim's evaluation still reaches it,
  and a judge's rationale may quote the reply.
- `scope={USER_INPUT_TAG}, read_only={SYSTEM_TAG}`: also knows which host it is
  attacking.

A RAG app that sends a retrieved document next to the user's question declares
both slots, with the document channel on a root of its own:

```python
from http_endpoint_target import (
    RESPONSE_TAG, USER_INPUT_TAG, Slot, http_endpoint_target_factory,
)
from superred.core.types.security_domain import SecurityDomainTag

DOCUMENT_TAG = SecurityDomainTag("retrieved_document")   # build once, reuse

factory = http_endpoint_target_factory(
    url="https://my-rag.example.com/v1/answer",
    body_template={"question": "{{prompt}}", "context": ["{{document}}"]},
    slots={
        "prompt": Slot(USER_INPUT_TAG, "What are your opening hours?"),
        "document": Slot(
            DOCUMENT_TAG, "We are open 9-5 on weekdays.", "A document the app retrieves."
        ),
    },
    response_path="answer",
)
# Indirect prompt injection: plant the document, read the answers.
# Controller(..., scope=frozenset({DOCUMENT_TAG}), read_only=frozenset({RESPONSE_TAG}))
```

Tags match by identity, so give the Controller the same tag objects you gave the
target. Construction therefore rejects a new tag object that equals one of the
exported tags (a fresh `SecurityDomainTag("user_input")`, say, which would print
and compare like `USER_INPUT_TAG` yet match no scope built from it), and two
different tag objects that share a name. It also rejects a template placeholder
that no slot declares, since it would be sent literally with no security domain
behind it.

## Secret handling

Auth headers (an API key / bearer token) are held privately and are **never**
emitted as an observable, returned from a query, or written into any rationale.
The only endpoint observable is `method + host(:port)` — the `user:pass@` userinfo,
the URL path, and the query string are all dropped, since any of them can carry a
credential/token (a malformed URL can smear a password across the path or query).
The host is emitted only after being validated as a bare hostname / IP literal;
anything unexpected is redacted to `(unparsable url)`.

## Queries

`last_response` (extracted reply), `raw_response` (full body), `http_status`,
`error`, `sent_prompt` (the effective `{{prompt}}` value after injection) and
`sent_inputs` (JSON object of every slot's effective value). HTTP/transport
errors are recorded in `error` (never raised), with bounded retry + backoff on
`429`/`5xx`, so a claim can abstain. A numeric `Retry-After` is honored but
**clamped to `max_retry_delay`** (default 60s) — a hostile/untrusted endpoint
cannot stall the run with a huge `Retry-After`. The response is also read under a
**total-time cap** (`max_response_time`, default 60s — httpx's per-op `timeout`
does not bound a slow byte-trickle) and a streamed **body-size cap**
(`max_response_bytes`, default 1 MB); responses are requested uncompressed
(`Accept-Encoding: identity`) and a compressed response is refused rather than
decompressed, so a slow, huge-body, or compression-bomb endpoint can neither
stall the run nor exhaust memory. Redirects are **not** followed —
a `3xx` (HTTP→HTTPS, trailing-slash, SSO/auth redirect) is recorded as an error,
so configure the final URL directly. Config: `benign_<slot>` per slot
(`benign_prompt` with the defaults) sets the value sent when that slot is not
injected. Controllables: one per slot (`prompt` with the defaults). Observables:
`endpoint`, `endpoint_response`, and `sent_<slot>` for each slot's value, at the
slot's domain.

## Scope

Only red-team an endpoint you are authorized to test. Live runs hit your real
endpoint; offline tests inject an `httpx.MockTransport` (no network).

## License

MIT. Uses the (BSD-licensed) `httpx` library as a dependency; no third-party code
is vendored.

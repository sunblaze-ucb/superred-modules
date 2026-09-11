# superred-target-prompt-shield

[Azure AI Content Safety **Prompt Shields**](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
as a superred target — so an optimizer can red-team the *detector* itself.

Prompt Shields is a synchronous detector that classifies a user prompt (direct
injection) and/or grounding documents (indirect injection) as attack /
not-attack. This target makes the detector a superred victim: the attacker
controls the `user_prompt` and/or `document` text, the target calls
`text:shieldPrompt`, and the detector's `attackDetected` verdict is the
observable. A genuine attack the detector rates `attackDetected == false` is an
**evasion** — the security failure the paired
[`superred-claim-prompt-shield`](../../security_claims/prompt_shield) scores.

## Usage

```python
from prompt_shield_target import prompt_shield_target_factory

factory = prompt_shield_target_factory(
    endpoint="https://<resource>.cognitiveservices.azure.com",  # your resource
    api_key="<key>",           # sent as Ocp-Apim-Subscription-Key; never logged
    api_version="2024-09-01",
    concurrency=1,             # F0 (free) tier is rate-limited; raise on standard tier
)
```

Drive it with any optimizer. The `goal_passthrough` optimizer submits an attack
verbatim (the un-obfuscated detection baseline); a jailbreak/mutation optimizer
measures optimizer-assisted evasion. Set the target `channel` config to
`user_prompt` (default) or `document` to choose the injection surface.

## REST contract

```
POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01
Ocp-Apim-Subscription-Key: <key>
Content-Type: application/json
{"userPrompt": "...", "documents": ["..."]}
->
{"userPromptAnalysis": {"attackDetected": bool},
 "documentsAnalysis": [{"attackDetected": bool}]}
```

Queries: `attack_detected` (overall, `'true'`/`'false'`/`''`),
`user_prompt_attack_detected`, `document_attack_detected`, `raw_response`,
`http_status`, `error`, `channel`. An HTTP error leaves `attack_detected` empty
so the claim can abstain rather than score a non-measurement.

## Testing & credentials

Offline tests inject an `httpx.MockTransport` — no network, no key. Live runs
need your **own** Azure AI Content Safety resource + key (there is an F0 free
tier). The API key is held privately and never emitted as an observable, returned
from a query, or written into a rationale. Like other credential-gated targets,
live runs are not reproducible in CI.

## Terms of use

Red-teaming **your own** Content Safety resource is permitted under Microsoft's
[Penetration Testing Rules of Engagement](https://www.microsoft.com/en-us/msrc/pentest-rules-of-engagement),
which explicitly cover *"testing your tenant's security monitoring and detection
systems."* Microsoft's Product Terms permit evaluating/testing generative-AI
services to the extent of those Rules. Constraints: use only a resource you own,
respect the tier's rate limits (no denial-of-service), and note that publishing
**comparative benchmark** results of the service may be separately restricted —
check the Product Terms before publishing numbers. Running the evaluation for
your own purposes is fine.

## License

MIT. This module calls Azure Prompt Shields; no Microsoft code or data is
vendored.

# DecodingTrust-Agent (DTAP) port

This is the superred port of **DecodingTrust-Agent (DTAP)**, an agent
red-teaming benchmark. DTAP ships, per task, a Dockerized environment (one or
more MCP tool servers backed by stateful services) plus an adversarial objective
and a *verifiable environment-state judge* that inspects the live service state
after a run to decide whether the safety property was violated. It spans many
domains (travel, customer service, filesystem, and so on) under two threat
models: "direct" (the user prompt is adversarial) and "indirect" (a third party
injects through the environment). The port wires that benchmark into superred so
that any optimizer can attack a DTAP victim agent under a precisely scoped threat
model, and the outcome is judged by DTAP's own judge run out of band.

Upstream: `AI-secure/DecodingTrust-Agent`, commit `e0323a52`, Apache-2.0.

## The four packages

```
                          dtap-scaffold  (frozen, agent-agnostic base)
                          forest / controllables / observables / specs
                          Docker+MCP-proxy+injection lifecycle / dataset / judge
                            |                        |                       |
            +---------------+----------+   +---------+----------+   (imported by)
            |                          |   |                    |        |
   dtap-claudecode-target     dtap-openclaw-target        security-claim-dtap
   (Claude Code agent)        (OpenClaw agent)            (DTAP-BENCH: one Task
   four agent-specific hooks  four agent-specific hooks   per per-task config.yaml,
                                                          target-agnostic, judge)
```

| Package | Dir | Import name | What it is |
|---|---|---|---|
| `dtap-scaffold` | `targets/dtap_scaffold` | `dtap_scaffold` | The **frozen, agent-agnostic base**. Owns the whole superred `Target` lifecycle: the security-domain forest, the DTAP injection controllables, the emit-once observables, config/query specs, the Docker env stack + host MCP proxy + env injector collaborators, the dataset loader, and the byte-faithful out-of-band judge runner. Both targets subclass `dtap_scaffold.agent_base.DtapAgentTarget`. |
| `dtap-claudecode-target` | `targets/dtap_claudecode` | `dtap_claudecode_target` | The **Claude Code** victim agent (the `claude_agent_sdk` Python SDK driving the `@anthropic-ai/claude-code` CLI). Adds only the four agent-specific hooks. An "assistant" in the project taxonomy (container-isolated, full native tools + DTAP env tools over MCP). |
| `dtap-openclaw-target` | `targets/dtap_openclaw` | `dtap_openclaw_target` | The **OpenClaw** victim agent (the OpenClaw CLI). Adds only the four agent-specific hooks. An "agent" in the taxonomy (MCP tool-caller with native `exec`/`fs`). |
| `security-claim-dtap` | `security_claims/dtap` | `security_claim_dtap` | The **DTAP-BENCH security claim**: one `DtapTask` per per-task `config.yaml`, target-agnostic (binds to the base `Target`, raises `NotApplicable` for non-DTAP targets). Hierarchical factories by domain / threat model / risk category, plus convenience target factories that lazily import either agent. |

### How they fit

- A `DtapTask` (from the claim) configures whichever DTAP target via the base's
  config slots, the optimizer attacks through the in-scope DTAP controllables,
  the target runs one agent episode in Docker, and the task's `evaluate()` calls
  DTAP's judge (out of band) on the live environment state.
- The **claim depends only on `superred` + `dtap-scaffold`**. The two target
  packages (which carry the heavier Docker / agent-SDK deps) are lazily imported
  inside the claim's `dtap_claudecode_target_factory` /
  `dtap_openclaw_target_factory`, so you install only the target you run.
- The **attack content and the injection vector are the optimizer's and the
  Controller scope's concern, not the claim's**: there is no bundled attacker.
  DTAP's four vectors map to controllables (user prompt, tool description,
  environment content, skill) plus superred-afforded surfaces the Controller scope
  selects per experiment: the system prompt, env-tool return tampering, and the
  `host` trust boundary (the machine the agent runs on) split into
  `host_filesystem` (place/edit/delete files before the run) and
  `host_code_execution` (an interactive code-execution foothold). The agent's own
  native OS tool calls stay observable-only.

## Sourcing: images, SDKs, dataset

These are the external artifacts the port pulls in. None are built or fetched on
a host without Docker / network; the offline test suites mock every one of them.

### Dataset (HuggingFace)

The per-task trees (`config.yaml` / `setup.sh` / `judge.py` / `metadata/`) are
**not vendored** (large, dataset-licensed). They resolve via
`dtap_scaffold.dataset.resolve_dataset_root`:

1. an explicit `dataset_root=` argument, else
2. the `DTAP_DATASET_ROOT` environment variable, else
3. `./dataset`.

With `download=True` the requested domains are fetched from the HuggingFace
dataset repo **`AI-Secure/DecodingTrust-Agent-Platform`** (`repo_type="dataset"`,
needs the `dtap-scaffold[dataset]` extra, i.e. `huggingface_hub`). For
reproducible, fully offline runs, prefer pinning `DTAP_DATASET_ROOT` to a local
checkout. DTAP ships 14 domains; this port covers the **11 text-only** ones
(`dtap_scaffold.text_domains.TEXT_ONLY_DOMAINS`); `browser`, `macos`, and
`windows` are vision/GUI driven and out of scope (the enumerator skips them).

### Env-server images (Docker Hub) and the DTAP SDK

The per-task **environment** containers (the MCP backends) pull from Docker Hub
under **`decodingtrustagent/*`**; there are no local image builds for the
text domains. The env/MCP/compose/judge configuration (`dt_arena/config`:
`mcp.yaml`, `env.yaml`, `injection_mcp.yaml`, the per-env `docker-compose`s, and
the judge helpers) is supplied by the pinned upstream package
**`decodingtrust-agent-sdk==0.2.12`** (the `dtap-scaffold[sdk]` extra,
lazy-imported by `dtap_scaffold.docker.env_registry` and the judge runner).

### Agent runtime images

Each victim agent runs one episode per `docker run` in its own image:

| Agent | Image tag (`DEFAULT_IMAGE`) | Built from | Pins |
|---|---|---|---|
| Claude Code | `dtap-claudecode:latest` | `targets/dtap_claudecode/docker/Dockerfile` | `python:3.13-slim` + Node 20 + `@anthropic-ai/claude-code` (Node CLI) + `claude-agent-sdk` (Python). Pin the two SDK versions when productionizing. |
| OpenClaw | `decodingtrustagent/dtap-openclaw:openclaw-2026.4` | `targets/dtap_openclaw/docker/Dockerfile` | `node:24-bookworm-slim` + `openclaw@2026.4` (the tag pins the npm version; keep in sync with `driver.DEFAULT_IMAGE`). |

Build commands:

```bash
# Claude Code agent image (build context is the package source dir, which holds driver.py)
docker build -f targets/dtap_claudecode/docker/Dockerfile \
    -t dtap-claudecode:latest targets/dtap_claudecode/src/dtap_claudecode_target

# OpenClaw agent image
docker build -t decodingtrustagent/dtap-openclaw:openclaw-2026.4 \
    targets/dtap_openclaw/docker/
```

The agents run their **own** inference through their own provider client
(`ANTHROPIC_BASE_URL` for Claude Code; the LiteLLM provider wiring for OpenClaw),
pointed at the experiment's LiteLLM proxy. That token spend is out of band and
is **not** charged to the optimizer's budget.

### Pinned-dependency summary

| Dependency | Where | Purpose |
|---|---|---|
| `huggingface_hub>=0.20` | `dtap-scaffold[dataset]` (host) | Auto-download the dataset domains |
| `decodingtrust-agent-sdk==0.2.12` | `dtap-scaffold[sdk]` (host) | Env/MCP/compose config + judge harness |
| `claude-agent-sdk` (Python) + `@anthropic-ai/claude-code` (Node) | inside `dtap-claudecode:latest` | The Claude Code agent loop |
| `openclaw@2026.4` (Node) | inside `decodingtrustagent/dtap-openclaw:...` | The OpenClaw agent loop |

## Install (offline-capable, dev)

From the repo root (the shared venv lives at the workspace root). The framework
plus the scaffold and one target are the minimum to run anything; install the
heavier extras only where Docker / network is available.

```bash
source ../.venv/bin/activate                 # the shared Python 3.13 venv
pip install -e ../superred                    # framework first
pip install -e targets/dtap_scaffold          # the frozen base
pip install -e targets/dtap_claudecode        # and/or targets/dtap_openclaw
pip install -e security_claims/dtap           # the claim
pip install -e optimizers/goal_passthrough    # an optimizer (baseline)

# Optional, only where used:
pip install -e "targets/dtap_scaffold[dataset]"   # HF auto-download
pip install -e "targets/dtap_scaffold[sdk]"       # env/judge harness (Docker hosts)
```

## Tests

### Offline (no Docker, no network, no LLM) -- runs now

Every Docker / subprocess / HTTP / SDK / LLM boundary is mocked, and every test
that genuinely needs Docker or a live model is marked and skipped. The packages'
`pyproject.toml` already set `addopts = -m 'not docker and not live'`, so a plain
`pytest` excludes the gated tests. After the editable installs above:

```bash
python -m pytest targets/dtap_scaffold/tests -q
python -m pytest targets/dtap_claudecode/tests -q
python -m pytest targets/dtap_openclaw/tests -q
python -m pytest security_claims/dtap/tests -q
```

Without installing, run any package against the shared venv with the scaffold on
`PYTHONPATH` (the dependency every other package shares):

```bash
PYTHONPATH="security_claims/dtap/src:targets/dtap_scaffold/src" \
    ../.venv/bin/python -m pytest security_claims/dtap/tests -q -p no:cacheprovider
```

Some faithfulness tests (the golden-hash / goal byte-identity checks) need the
dataset; they `skipif` it is absent, so set `DTAP_DATASET_ROOT` to run them.

### Gated: Docker + live model -- runs where Docker exists

These build/pull the images and call the live proxy, so they only run on a host
with a Docker daemon and LiteLLM credentials. They are skipped otherwise.

```bash
# Claude Code container + live-model end-to-end
LITELLM_API_KEY=... LITELLM_API_BASE=... \
    python -m pytest targets/dtap_claudecode/tests -m "docker and live"

# OpenClaw container end-to-end
LITELLM_API_KEY=... LITELLM_API_BASE=... \
    python -m pytest targets/dtap_openclaw/tests -m "docker and live"
```

## Run end to end

An end-to-end run wires the DTAP claim (`security_claim_dtap`), a DTAP agent
target (e.g. the Claude Code target), and an optimizer (e.g. the `goal_passthrough` baseline)
into a framework `Controller`. It needs a running Docker daemon, the dataset
at `DTAP_DATASET_ROOT`, and `LITELLM_API_KEY` / `LITELLM_API_BASE` in the
environment. Keep the per-task budget small for a first smoke run.

## Faithfulness

Each package carries an `ASSUMPTIONS.md` recording the upstream source and every
deliberate deviation (transcript schema, the env-tool emit-once split, the
native-tool menus, the single-host-proxy MCP convention, the judge transport,
and the one-config-to-one-Task mapping). The judge predicates (`eval_attack` for
malicious, `eval_task` for benign) and the adversarial goals are upstream's own,
run byte-faithfully out of band.

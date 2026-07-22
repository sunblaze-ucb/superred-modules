# superred-target-dtap-scaffold

Shared scaffolding for the DecodingTrust-Agent (DTAP) superred targets: the
agent-agnostic `DtapAgentTarget` base, the security-domain forest, the
controllables/observables, the pre-run/post-run specs, the text-only domain
allowlist, the env/MCP/Docker lifecycle, the host MCP proxy, the env-injection
bridge, and the byte-faithful judge runner.

The two concrete DTAP targets (Claude Code, OpenClaw) subclass `DtapAgentTarget`
and implement only a handful of agent-specific hooks. The DTAP task/goal dataset
lives in the separate `security-claim-dtap` package.

## Credits / upstream

This package is original work (MIT, Copyright (c) 2026 Simon Sure). It contains
no vendored third-party source code and no bundled benchmark dataset.

It is a faithful superred port of, and a runtime wrapper around, the
**DecodingTrust-Agent (DTap)** red-teaming platform for AI agents
(https://github.com/AI-secure/DecodingTrust-Agent, Apache-2.0). DTAP's
environment-pool orchestration is reimplemented here (no code copied); the
optional `sdk` extra installs the upstream `decodingtrust-agent-sdk` (pinned
`==0.2.12`, Apache-2.0), which this scaffold lazy-imports at runtime. The
`decodingtrust-agent-sdk` is not redistributed as part of this package.

The bundled `data/tool_trees.json` is an original per-service authorization-tree
mapping authored for this project; it references DTAP tool names but is not a
copy of any DTAP dataset. The DTAP task/goal dataset lives in the separate
`security-claim-dtap` package.

Please cite the DecodingTrust-Agent (DTap) platform when reporting results
produced with this target.

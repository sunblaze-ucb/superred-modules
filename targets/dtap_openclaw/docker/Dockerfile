# OpenClaw DTAP agent runtime image.
#
# Runs ONE episode per `docker run`: the host bind-mounts a per-episode state
# directory at /state (containing openclaw.json, AGENTS.md, skills, task.json),
# and the entrypoint drives the turns with the OpenClaw CLI (see run_turns.mjs).
# Native exec/fs tools are ENABLED in this port (upstream DTAP disabled them), so
# they execute against this disposable container filesystem -- never the host.
#
# Build + push (the tag pins the OpenClaw npm version; keep it in sync with
# driver.DEFAULT_IMAGE):
#   docker build -t dtap-openclaw:openclaw-2026.6.10 .
FROM node:24-bookworm-slim

# Pin the OpenClaw CLI version (keep in sync with driver.DEFAULT_IMAGE's tag).
ARG OPENCLAW_VERSION=2026.6.10
RUN npm install -g "openclaw@${OPENCLAW_VERSION}" \
    && npm cache clean --force

# Entrypoint that reads /state/task.json and runs the turns.
COPY run_turns.mjs /usr/local/bin/run_turns.mjs
RUN chmod +x /usr/local/bin/run_turns.mjs

# State (config, workspace/AGENTS.md, skills, traces) is bind-mounted here; HOME
# points at it so OpenClaw's per-profile config dir lands under the bound volume.
ENV HOME=/state
WORKDIR /state

ENTRYPOINT ["node", "/usr/local/bin/run_turns.mjs"]

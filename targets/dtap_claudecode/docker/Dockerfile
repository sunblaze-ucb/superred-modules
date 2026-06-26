# Claude Code DTAP agent image.
#
# Runs driver.py, which drives claude_agent_sdk's ClaudeSDKClient (the Python
# Claude Agent SDK) against the host MCP proxy. The SDK in turn spawns the
# @anthropic-ai/claude-code Node CLI, so the image needs Node + the CLI + the
# Python SDK. The agent reaches the host MCP proxy via host.docker.internal
# (the container is started with --add-host host.docker.internal:host-gateway).
#
# This image is NOT built offline (no Docker / Node on the host); it is built and
# run later in the environment that has Docker. Pin versions when productionizing.
FROM python:3.13-slim

# Node 20 (the Claude Code CLI requires a current Node runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# The Claude Code CLI (spawned by the SDK) + the Python Claude Agent SDK.
RUN npm install -g @anthropic-ai/claude-code \
    && pip install --no-cache-dir claude-agent-sdk

# The single-file, dependency-light driver. It is self-contained (stdlib +
# claude_agent_sdk only) and does not import the rest of the host package.
COPY driver.py /opt/dtap/driver.py

# The per-instance host dir is mounted here at runtime (task.json in, transcript
# + result out). ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL are
# passed as -e at `docker run`.
WORKDIR /dtap
ENV DTAP_TASK_FILE=/dtap/task.json
# permission_mode="bypassPermissions" -> the CLI's --dangerously-skip-permissions,
# which refuses to run as root unless it knows it is sandboxed. The container is
# the isolation boundary (the host also passes -e IS_SANDBOX=1 belt-and-suspenders).
ENV IS_SANDBOX=1
ENTRYPOINT ["python", "/opt/dtap/driver.py"]

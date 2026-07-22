"""Real container + live-model end-to-end (SKIPPED unless Docker + LLM creds
are available). Runs the full DtapClaudeCodeTarget against the built agent image
and a litellm-proxied model, with fake env collaborators so no DTAP env stack is
required."""

from __future__ import annotations

import json
import os
import shutil

import pytest

pytestmark = [pytest.mark.docker, pytest.mark.live]

_DOCKER = shutil.which("docker") is not None
_CREDS = bool(os.environ.get("LITELLM_API_KEY") and os.environ.get("LITELLM_API_BASE"))
_IMAGE = os.environ.get("DTAP_CLAUDECODE_IMAGE", "dtap-claudecode:latest")


@pytest.mark.skipif(not _DOCKER, reason="requires a Docker daemon + CLI")
@pytest.mark.skipif(not _CREDS, reason="requires LITELLM_API_KEY / LITELLM_API_BASE")
async def test_real_container_episode():
    # Imported here so collection never fails when superred extras differ.
    from dtap_claudecode_target import DtapClaudeCodeTarget

    target = DtapClaudeCodeTarget(
        model=os.environ.get("DTAP_MODEL", "claude-opus-4-8"),
        api_base=os.environ["LITELLM_API_BASE"],
        api_key=os.environ["LITELLM_API_KEY"],
        image=_IMAGE,
        max_turns=4,
    )
    # No MCP servers / env injection -> the agent runs with native tools only,
    # exercising the real container + SDK + transcript -> converter path.
    target.set_config("active_mcp_servers", json.dumps([]))
    target.set_config("env_injection_config", json.dumps({}))
    target.set_config("system_prompt", "You are a helpful assistant. Answer concisely.")
    target.set_config("user_prompt", json.dumps(["Reply with exactly the word PONG."]))
    target.set_config("task_dir", "/tmp/dtap-e2e/1")
    target.set_config("native_tools_policy", "enabled")

    events = []

    def emit(evt):
        events.append(evt)

    async def send_event(evt):
        from superred.core.types.events import ControllableNoInjection

        return ControllableNoInjection(event=evt, controllable=getattr(evt, "controllable", None))

    try:
        await target.run(emit, send_event)
        final = target.query("final_response")
        assert isinstance(final, str)
        assert final  # the model produced something
    finally:
        await target.teardown()

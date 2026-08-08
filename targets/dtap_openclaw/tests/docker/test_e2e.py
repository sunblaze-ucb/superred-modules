"""End-to-end OpenClaw episode against a REAL Docker daemon + LiteLLM proxy.

SKIPPED offline: requires (1) a Docker daemon + the built OpenClaw image and
(2) LiteLLM proxy credentials. Runs later in an environment that has both. The
offline suite (``test_target`` / ``test_driver`` / ``test_trajectory``) covers
every seam this exercises end to end, with the Docker/agent boundary mocked.
"""

from __future__ import annotations

import os
import shutil

import pytest
from dtap_scaffold.types import AgentLaunchSpec

from dtap_openclaw_target import driver
from dtap_openclaw_target.target import OS_FILESYSTEM_DISALLOWED_TOOLS

pytestmark = [pytest.mark.docker, pytest.mark.live]

_HAVE_DOCKER = shutil.which("docker") is not None
_HAVE_CREDS = bool(os.getenv("LITELLM_API_KEY") and os.getenv("LITELLM_API_BASE"))


@pytest.mark.skipif(not _HAVE_DOCKER, reason="needs a Docker daemon + CLI")
@pytest.mark.skipif(not _HAVE_CREDS, reason="needs LITELLM_API_KEY/BASE")
def test_real_openclaw_container_runs_and_emits_trace(tmp_path) -> None:
    """Run a no-MCP single-turn episode in the real image and read the trace back."""
    from dtap_openclaw_target import trajectory

    spec = AgentLaunchSpec(
        model=os.environ.get("DTAP_TEST_MODEL", "openai/gpt-4o-mini"),
        api_base=os.environ["LITELLM_API_BASE"],
        api_key=os.environ["LITELLM_API_KEY"],
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
        instructions=("Say the word READY and nothing else.",),
        proxy_url="http://host.docker.internal:9999/mcp",
        mcp_server_names=(),
        native_tool_deny=OS_FILESYSTEM_DISALLOWED_TOOLS,  # no tools needed for this smoke
        output_dir=str(tmp_path),
    )

    image = os.environ.get("DTAP_OPENCLAW_IMAGE", driver.DEFAULT_IMAGE)
    out_dir = driver.run_openclaw_container(spec, image=image, timeout=600.0)

    artifact = trajectory.convert(out_dir, mcp_servers=())
    assert artifact.final_response  # the model produced some final text

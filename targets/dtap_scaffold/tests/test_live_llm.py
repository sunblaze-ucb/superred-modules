"""Live LLM smoke through the LiteLLM proxy.

This exercises the exact LLM-config path the DTAP port depends on: the agent
targets run their OWN inference and the OOB judges call an LLM, both routed (by
model id + api_base + api_key) to the user's LiteLLM proxy. Docker is not needed
for this test, so it is the live LLM check runnable without a Docker daemon.

Gated: needs LITELLM_API_KEY / LITELLM_API_BASE (from the environment or the
sibling superred-experiments/.env). Run with: pytest -m live -o addopts="".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

# The scaffold's default OOB-judge model (dtap_scaffold.judge_runner), which the
# user's proxy is known to serve. The agent backbone model is a separate
# construction concern, also routed through the same proxy.
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-2024-05-13"


def _creds() -> tuple[str | None, str | None]:
    key = os.environ.get("LITELLM_API_KEY")
    base = os.environ.get("LITELLM_API_BASE")
    if not (key and base):
        for parent in Path(__file__).resolve().parents:
            env = parent / "superred-experiments" / ".env"
            if env.is_file():
                for line in env.read_text().splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
                break
    return os.environ.get("LITELLM_API_KEY"), os.environ.get("LITELLM_API_BASE")


def test_proxy_completion_for_judge_model():
    """The default judge model answers a tiny prompt via the proxy (cost-capped)."""
    key, base = _creds()
    if not (key and base):
        pytest.skip("no LITELLM_API_KEY / LITELLM_API_BASE for live tests")
    litellm = pytest.importorskip("litellm")

    response = litellm.completion(
        model=DEFAULT_JUDGE_MODEL,
        api_base=base,
        api_key=key,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=5,
        temperature=0,
    )
    text = (response.choices[0].message.content or "").strip()
    assert text, "expected a non-empty completion from the LiteLLM proxy"

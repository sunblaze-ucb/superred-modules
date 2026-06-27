"""Grounded OpenClaw gateway configuration.

OpenClaw is configured from ``<stateDir>/openclaw.json`` (the gateway reads
its state dir from ``OPENCLAW_STATE_DIR`` / ``OPENCLAW_CONFIG_DIR``). The
schema used here is taken from the real OpenClaw source:

- Model selection + per-provider overrides live under ``models.providers.*``
  (``baseUrl``, ``apiKey``, ``api``, ``request.allowPrivateNetwork``, ``models``)
  with ``models.mode: "replace"`` to fully control the provider list. See
  ``src/config/types.models.ts`` (``MODEL_APIS``) and the local-mode config
  fixture in ``src/tui/tui-pty-local.e2e.test.ts``.
- The active model is selected via ``agents.defaults.model.primary`` (and the
  ``main`` agent entry), with ``agents.defaults.models.<id>.agentRuntime.id``.
- Tool restriction is ``tools.profile`` (config, not a runtime RPC).
- External plugins/extensions are loaded from ``<stateDir>/extensions/<name>/``
  and gated by ``plugins.allow`` (see ``src/security/audit-plugins-trust.ts``).

This replaces the previous (ungrounded) approach of passing provider/extension
settings via ``OPENCLAW_PROVIDER_URL`` / ``OPENCLAW_EXTENSIONS_DIR`` env vars,
which are not OpenClaw config keys.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

# The injection plugin's id, used both as the extensions/ subdirectory name and
# in ``plugins.allow``. Must match the plugin's package.json ``name``.
DEFAULT_PLUGIN_NAME = "superred-injection"

# Provider API adapter for an OpenAI-compatible ``/v1/chat/completions`` upstream
# (which is what the superred LLM proxy speaks). One of the values in
# ``MODEL_APIS`` (src/config/types.models.ts).
DEFAULT_PROVIDER_API = "openai-completions"


def _split_model_id(model_id: str) -> tuple[str, str]:
    """Split ``"<provider>/<model>"`` into ``(provider, model)``.

    OpenClaw model ids are provider-qualified (README: ``agent.model:
    "<provider>/<model-id>"``). If no provider prefix is present, fall back to a
    synthetic ``"superred"`` provider so the override still applies.
    """
    if "/" in model_id:
        provider, model = model_id.split("/", 1)
        return provider or "superred", model
    return "superred", model_id


def build_gateway_config(
    *,
    model_id: str = "",
    provider_base_url: str = "",
    provider_api_key: str = "",
    provider_api: str = DEFAULT_PROVIDER_API,
    tool_policy: str = "",
    workspace_dir: str | None = None,
    plugin_names: list[str] | None = None,
    allow_private_network: bool = True,
) -> dict[str, Any]:
    """Build the ``openclaw.json`` config dict.

    Only includes blocks that are actually configured, so a minimal call
    (no model/provider) produces a minimal, valid config.

    Args:
        model_id: Provider-qualified model id (``"<provider>/<model>"``).
        provider_base_url: Base URL the gateway should call for model requests.
            Point this at the superred LLM proxy to record/inject model calls.
            ``/v1`` is appended if not already present (OpenClaw calls
            ``<baseUrl>/chat/completions``).
        provider_api_key: API key for the provider override.
        provider_api: One of OpenClaw's ``MODEL_APIS`` adapter ids.
        tool_policy: ``tools.profile`` name (e.g. ``"minimal"``).
        workspace_dir: ``agents.defaults.workspace`` path.
        plugin_names: Plugin ids to allow (``plugins.allow``); the injection
            plugin must be listed for the gateway to load it.
        allow_private_network: Set ``request.allowPrivateNetwork`` on the
            provider so the gateway may call a loopback / ``host.docker.internal``
            proxy (blocked by default as SSRF protection).
    """
    config: dict[str, Any] = {}

    if plugin_names:
        config["plugins"] = {"enabled": True, "allow": list(plugin_names)}

    agents_defaults: dict[str, Any] = {}
    if workspace_dir:
        agents_defaults["workspace"] = workspace_dir

    if model_id and provider_base_url:
        provider, model = _split_model_id(model_id)
        base_url = provider_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        provider_entry: dict[str, Any] = {
            "baseUrl": base_url,
            "api": provider_api,
            "models": [{"id": model}],
        }
        if provider_api_key:
            provider_entry["apiKey"] = provider_api_key
        if allow_private_network:
            provider_entry["request"] = {"allowPrivateNetwork": True}

        config["models"] = {
            "mode": "replace",
            "providers": {provider: provider_entry},
        }
        agents_defaults["model"] = {"primary": model_id}
        agents_defaults["models"] = {model_id: {"agentRuntime": {"id": "openclaw"}}}

    if agents_defaults:
        config["agents"] = {
            "defaults": agents_defaults,
            "list": [{"id": "main", "default": True}],
        }
        if "model" in agents_defaults:
            config["agents"]["list"][0]["model"] = agents_defaults["model"]

    if tool_policy:
        config["tools"] = {"profile": tool_policy}

    return config


def materialize_state_dir(
    state_dir: Path,
    *,
    config: dict[str, Any],
    plugin_src: Path | None = None,
    plugin_name: str = DEFAULT_PLUGIN_NAME,
) -> None:
    """Write ``openclaw.json`` and install the injection plugin under ``state_dir``.

    Creates::

        <state_dir>/openclaw.json
        <state_dir>/workspace/
        <state_dir>/extensions/<plugin_name>/   (copied from plugin_src)

    The gateway loads extensions from ``<stateDir>/extensions`` and reads config
    from ``<stateDir>/openclaw.json`` (when ``OPENCLAW_STATE_DIR`` /
    ``OPENCLAW_CONFIG_DIR`` point here).
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "workspace").mkdir(parents=True, exist_ok=True)

    (state_dir / "openclaw.json").write_text(json.dumps(config, indent=2))

    if plugin_src is not None and plugin_src.exists():
        ext_dir = state_dir / "extensions" / plugin_name
        if ext_dir.exists():
            shutil.rmtree(ext_dir)
        shutil.copytree(plugin_src, ext_dir)

"""Offline tests for the DTAP env registry (parses the vendored config YAMLs).

These read the three real config YAMLs from the upstream clone (no Docker, no
network). They skip when the clone is absent. The clone root is taken from
``$DT_ROOT`` or the conventional ``tmp/DecodingTrust-Agent`` checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dtap_scaffold.docker import env_registry
from dtap_scaffold.docker.env_registry import EnvRegistry, EnvRegistryError

_CLONE_ROOT = Path(
    os.environ.get("DT_ROOT") or "/Users/simonsure/research/superred/tmp/DecodingTrust-Agent"
)
_CONFIG_DIR = _CLONE_ROOT / "dt_arena" / "config"

pytestmark = pytest.mark.skipif(
    not _CONFIG_DIR.is_dir(),
    reason=f"DTAP clone config not found at {_CONFIG_DIR}",
)


@pytest.fixture
def reg() -> EnvRegistry:
    return EnvRegistry(_CONFIG_DIR)


# --------------------------- mcp.yaml: server -> environment ---------------


def test_server_environment_string_and_list(reg: EnvRegistry) -> None:
    # `environment` may be a single string ...
    assert reg.server_environments("travel-suite") == ["travel"]
    assert reg.server_environments("OS-filesystem") == ["os-filesystem"]
    # ... or a list (browser fronts two GUI environments).
    assert reg.server_environments("browser") == ["ecommerce", "custom-website"]


def test_server_lookup_is_case_insensitive(reg: EnvRegistry) -> None:
    assert reg.server_environments("os-filesystem") == reg.server_environments("OS-filesystem")
    assert reg.mcp_server("TELECOM") is reg.mcp_server("telecom")


def test_unknown_server_raises(reg: EnvRegistry) -> None:
    assert reg.mcp_server("does-not-exist") is None
    with pytest.raises(EnvRegistryError):
        reg.server_environments("does-not-exist")


# --------------------------- env.yaml: compose + ports + reset -------------


def test_compose_file_resolves_under_sdk_root(reg: EnvRegistry) -> None:
    path = reg.compose_file("travel")
    assert path.is_absolute()
    assert str(path).endswith("dt_arena/envs/travel/docker-compose.yml")


def test_env_ports(reg: EnvRegistry) -> None:
    ports = reg.env_ports("travel")
    assert "TRAVEL_PORT" in ports
    assert ports["TRAVEL_PORT"]["default"] == 10300
    assert ports["TRAVEL_PORT"]["container_port"] == 10300


def test_env_metadata(reg: EnvRegistry) -> None:
    # default health timeout vs an env that overrides it.
    assert reg.health_timeout("travel") == 120
    assert reg.health_timeout("bigquery") == 180
    assert reg.disable_reuse("bigquery") is True
    assert reg.disable_reuse("travel") is False
    # per-env reset-script timeout: explicit override vs upstream's 60s default.
    assert reg.reset_script_timeout("terminal") == 180
    assert reg.reset_script_timeout("travel") == 60


def test_health_timeout_override(reg: EnvRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    # `calendar` has no env.yaml health_timeout, so our override for its unsatisfiable
    # (curl-less) healthcheck applies instead of the 120s default.
    assert reg.health_timeout("calendar") == 20
    # ... but an explicit env.yaml value still wins over an override.
    monkeypatch.setitem(env_registry._HEALTH_TIMEOUT_OVERRIDES, "bigquery", 5)
    assert reg.health_timeout("bigquery") == 180


# --------------------------- text-only validation -------------------------


def test_text_only_rejects_gui_servers(reg: EnvRegistry) -> None:
    for gui_server in ("browser", "macos-os", "windows-os"):
        with pytest.raises(ValueError):
            reg.require_text_only([gui_server])


def test_text_only_accepts_text_servers(reg: EnvRegistry) -> None:
    # No exception for text-only servers.
    reg.require_text_only(["travel-suite", "OS-filesystem", "telecom", "salesforce"])


def test_domain_for_server(reg: EnvRegistry) -> None:
    assert reg.domain_for_server("browser") == "browser"
    assert reg.domain_for_server("macos-os") == "macos"
    assert reg.domain_for_server("windows-os") == "windows"
    assert reg.domain_for_server("travel-suite") is None  # text-only -> no excluded domain


# --------------------------- mcp port-key selection -----------------------


def test_mcp_port_key_selection(reg: EnvRegistry) -> None:
    # Default PORT when no MCP-specific port var exists.
    assert env_registry.mcp_port_key(reg.mcp_server("travel-suite"), "mcp") == "PORT"
    # A dedicated *_MCP_PORT wins.
    assert env_registry.mcp_port_key(reg.mcp_server("telecom"), "mcp") == "TELECOM_MCP_PORT"
    assert env_registry.mcp_port_key(reg.mcp_server("HospitalClient"), "mcp") == "HOSPITAL_MCP_PORT"


# --------------------------- injection_mcp.yaml ---------------------------


def test_required_injection_servers(reg: EnvRegistry) -> None:
    required = reg.required_injection_servers({"travel-injection": "all", "bogus-injection": "all"})
    assert set(required) == {"travel-injection"}  # unknown name dropped
    assert required["travel-injection"]["name"] == "travel-injection"


# --------------------------- config-dir resolution ------------------------


def _simulate_no_installed_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the installed-SDK (``importlib.resources``) branch to miss.

    ``resolve_config_dir`` prefers the installed ``dt_arena`` package over the
    ``DT_ROOT`` clone, so when decodingtrust-agent-sdk IS installed in the dev
    venv these resolution-order tests would otherwise observe the wheel's config
    dir. Patching ``importlib.resources.files`` to raise makes the DT_ROOT /
    not-found branches deterministic whether or not the SDK is installed.
    """

    def _no_files(_name: str):
        raise ModuleNotFoundError("simulated: dt_arena not installed")

    monkeypatch.setattr(env_registry.importlib.resources, "files", _no_files)


def test_resolve_config_dir_explicit() -> None:
    assert env_registry.resolve_config_dir(_CONFIG_DIR) == _CONFIG_DIR


def test_resolve_config_dir_via_dt_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTAP_ROOT", raising=False)
    monkeypatch.setenv("DT_ROOT", str(_CLONE_ROOT))
    _simulate_no_installed_sdk(monkeypatch)  # else the installed wheel's config wins
    assert env_registry.resolve_config_dir() == _CONFIG_DIR


def test_resolve_config_dir_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTAP_ROOT", raising=False)
    monkeypatch.delenv("DT_ROOT", raising=False)
    _simulate_no_installed_sdk(monkeypatch)  # neither env var nor installed SDK
    with pytest.raises(EnvRegistryError):
        env_registry.resolve_config_dir()


def test_load_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DT_ROOT", str(_CLONE_ROOT))
    assert env_registry.load() is env_registry.load()

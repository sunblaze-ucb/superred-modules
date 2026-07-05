"""Shared helpers for OpenClaw target pytest modules."""

from test_support.cli import lan_ip, openclaw_cli_ready
from test_support.docker import docker_daemon_ready, docker_image, docker_target
from test_support.gemini import (
    DEFAULT_GEMINI_BASE,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PROVIDER_TIMEOUT_S,
    docker_gemini_target,
    gemini_api_key,
    gemini_target,
    provider_base_url,
    provider_model,
)
from test_support.send_event import (
    assert_tool_injection_persisted_on_next_run,
    injecting_send_event,
    passthrough_send_event,
)
from test_support.stubs import (
    container_stub_llm_server,
    container_stub_tool_calling_llm_server,
    echo_stub_upstream_for_host_proxy,
    local_web_page_server,
    loopback_recording_stub_llm_server,
    loopback_stub_llm_server,
    loopback_stub_tool_calling_llm_server,
    loopback_stub_upstream_for_host_proxy,
)

__all__ = [
    "DEFAULT_GEMINI_BASE",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_PROVIDER_TIMEOUT_S",
    "assert_tool_injection_persisted_on_next_run",
    "container_stub_llm_server",
    "container_stub_tool_calling_llm_server",
    "docker_daemon_ready",
    "docker_image",
    "docker_target",
    "docker_gemini_target",
    "echo_stub_upstream_for_host_proxy",
    "gemini_api_key",
    "gemini_target",
    "injecting_send_event",
    "lan_ip",
    "local_web_page_server",
    "loopback_recording_stub_llm_server",
    "loopback_stub_llm_server",
    "loopback_stub_tool_calling_llm_server",
    "loopback_stub_upstream_for_host_proxy",
    "openclaw_cli_ready",
    "passthrough_send_event",
    "provider_base_url",
    "provider_model",
]

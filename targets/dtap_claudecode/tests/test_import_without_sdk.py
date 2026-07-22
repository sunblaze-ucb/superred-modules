"""The package (incl. driver.py) must import WITHOUT claude_agent_sdk installed,
because the SDK lives only inside the agent Docker image."""

from __future__ import annotations

import importlib
import importlib.util


def test_claude_agent_sdk_is_not_a_host_dependency():
    # Meaningful only when the SDK is absent (the shared dev venv). If it happens
    # to be installed, the import-success assertions below still hold.
    if importlib.util.find_spec("claude_agent_sdk") is not None:
        import pytest

        pytest.skip("claude_agent_sdk is installed in this environment")


def test_package_imports_without_sdk():
    pkg = importlib.import_module("dtap_claudecode_target")
    assert hasattr(pkg, "DtapClaudeCodeTarget")
    assert hasattr(pkg, "convert")


def test_driver_module_imports_without_sdk():
    # driver.py defers the claude_agent_sdk import into run_episode, so the module
    # itself imports fine and its pure helpers are usable.
    drv = importlib.import_module("dtap_claudecode_target.driver")
    assert callable(drv._serialize_message)
    assert callable(drv._build_options)
    assert drv.PROXY_SERVER_NAME == "dtap_proxy"

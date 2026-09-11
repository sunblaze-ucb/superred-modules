"""Tests for the SafeClawArena target contract and pure runtime helpers.

The Docker execution path (build/provision/run/capture) needs Docker + the
platform image and is not exercised here (mirrors the dtap_openclaw target)."""

from __future__ import annotations

import json

import pytest

from safeclawarena_target import (
    SafeClawArenaTarget,
    safeclawarena_target_factory,
)
from safeclawarena_target.runtime import (
    PLATFORMS,
    file_check_targets,
    http_route_targets,
    platform_config,
)


def test_security_domain_forest() -> None:
    tgt = SafeClawArenaTarget()
    dom = tgt.security_domain
    names = {t.name for t in dom._tags.values()}
    assert names == {
        "system", "skills", "memory", "config",
        "user", "external_data", "sim_google", "file_content", "tools",
    }
    assert {r.name for r in dom.roots} == {"system", "user", "external_data", "tools"}


def test_config_and_query_specs() -> None:
    tgt = SafeClawArenaTarget()
    assert {c.name for c in tgt.config_specs} == {"task", "platform"}
    assert {q.name for q in tgt.query_specs} == {"post_state"}


def test_controllables_and_observables() -> None:
    tgt = SafeClawArenaTarget(platform="nemoclaw")
    assert {c.name for c in tgt.get_controllables()} == {"user_message", "external_data"}
    obs = {o.observable.name: o.content for o in tgt.get_observables()}
    assert obs["platform"] == "nemoclaw"
    assert "model_identity" in obs and "task_metadata" in obs


def test_config_roundtrip_and_query_default() -> None:
    tgt = SafeClawArenaTarget()
    tgt.set_config("platform", "nemoclaw")
    tgt.set_config("task", json.dumps({"metadata": {"task_id": "ssi-1.1-001", "dimension": "SSI"}}))
    assert tgt._platform == "nemoclaw"
    assert tgt._task["metadata"]["task_id"] == "ssi-1.1-001"
    # post_state is empty JSON until a run captures it
    assert json.loads(tgt.query("post_state")) == {}
    # task metadata now surfaces on the observable
    obs = {o.observable.name: o.content for o in tgt.get_observables()}
    assert json.loads(obs["task_metadata"])["task_id"] == "ssi-1.1-001"


def test_platform_config_pure() -> None:
    # SecLaw is excluded: upstream Dockerfile.seclaw COPYs a seclaw/ source absent
    # from the repo, so it is not buildable from the canonical artifact.
    assert set(PLATFORMS) == {"openclaw", "nemoclaw"}
    assert platform_config("openclaw")["workspace"] == "/root/.openclaw/workspace"
    with pytest.raises(ValueError):
        platform_config("seclaw")
    with pytest.raises(ValueError):
        platform_config("bogus")


def test_check_target_extraction_pure() -> None:
    task = {
        "evaluation": {
            "success_condition": {"checks": [
                {"type": "file_contains", "target": "/a"},
                {"type": "taint_detected_in_file", "target": "/b"},
                {"type": "http_route_exists", "target": "/hook"},
                {"type": "output_contains", "pattern": "x"},  # not a file/route
            ]},
            "safe_condition": {"checks": [{"type": "file_created", "target": "/c"}]},
        }
    }
    assert file_check_targets(task) == ["/a", "/b", "/c"]
    assert http_route_targets(task) == ["/hook"]


def test_factory_creates_fresh_targets() -> None:
    fac = safeclawarena_target_factory(platform="openclaw")
    a, b = fac.create(), fac.create()
    assert isinstance(a, SafeClawArenaTarget) and a is not b
    assert fac.concurrency == 1

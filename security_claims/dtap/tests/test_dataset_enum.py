"""Offline tests for the dataset loader/parser, run against a local dataset root.

Uses ``$DTAP_DATASET_ROOT`` if set, else the local DTAP clone; skipped if neither
is present. No network / Docker / LLM.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from security_claim_dtap.dataset import (
    TaskConfig,
    iter_task_config_paths,
    parse_task_config,
    resolve_dataset_root,
)

_CLONE = Path("/Users/simonsure/research/superred/tmp/DecodingTrust-Agent/dataset")
_ROOT = Path(os.environ.get("DTAP_DATASET_ROOT") or _CLONE)

pytestmark = pytest.mark.skipif(
    not _ROOT.is_dir(), reason="no local DTAP dataset root (set DTAP_DATASET_ROOT)"
)


def test_resolve_dataset_root_explicit():
    assert resolve_dataset_root(root=_ROOT) == _ROOT.resolve()


def test_enumerate_travel_text_only():
    paths = list(iter_task_config_paths(_ROOT, domains=["travel"]))
    assert paths, "expected at least one travel config.yaml"
    rootp = _ROOT.resolve()
    assert all(p.relative_to(rootp).parts[0] == "travel" for p in paths)
    assert all(p.name == "config.yaml" for p in paths)


def test_browser_excluded_as_non_text():
    # Even if explicitly requested, the excluded GUI domain yields nothing.
    assert list(iter_task_config_paths(_ROOT, domains=["browser"])) == []


def test_parse_benign_travel_task():
    benign = list(iter_task_config_paths(_ROOT, domains=["travel"], types=["benign"]))
    assert benign, "expected travel benign tasks"
    tc = parse_task_config(benign[0], root=_ROOT)
    assert isinstance(tc, TaskConfig)
    assert tc.domain == "travel"
    assert tc.type == "benign"
    assert tc.threat_model is None
    assert tc.task_dir.is_dir()
    assert tc.servers, "benign task should mount at least one MCP server"
    assert tc.instructions, "benign task should carry a task_instruction"
    assert tc.malicious_goal is None


def test_parse_malicious_travel_task():
    mal = list(iter_task_config_paths(_ROOT, domains=["travel"], types=["malicious"]))
    assert mal, "expected travel malicious tasks"
    tc = parse_task_config(mal[0], root=_ROOT)
    assert tc.domain == "travel"
    assert tc.type == "malicious"
    assert tc.threat_model in {"direct", "indirect"}
    assert tc.malicious_goal, "malicious task must carry a byte-identical malicious_goal"


def test_threat_model_filter():
    indirect = list(
        iter_task_config_paths(
            _ROOT, domains=["travel"], types=["malicious"], threat_models=["indirect"]
        )
    )
    for p in indirect:
        tc = parse_task_config(p, root=_ROOT)
        assert tc.threat_model == "indirect"

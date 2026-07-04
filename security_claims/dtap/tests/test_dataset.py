"""Golden-hash byte-faithfulness + Goal byte-equality against the real dataset."""

from __future__ import annotations

import pytest
import yaml
from conftest import dataset_root, requires_dataset
from dtap_scaffold.dataset import iter_task_config_paths, parse_task_config

from security_claim_dtap.dataset import (
    GOLDEN_HASHES_PATH,
    build_golden_hashes,
    hash_task,
    load_golden_hashes,
)
from security_claim_dtap.task import DtapTask

# ---------------------------------------------------------------------------
# hash_task
# ---------------------------------------------------------------------------


def test_hash_task_empty_dir_is_deterministic(tmp_path) -> None:
    # No config.yaml / judge.py -> hashes empty bytes, but stable + non-empty digest.
    h1 = hash_task(tmp_path)
    h2 = hash_task(tmp_path)
    assert h1 == h2
    assert len(h1) == 64


def test_hash_task_changes_with_goal_and_judge(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text("Attack:\n  malicious_goal: do harm\n")
    base = hash_task(tmp_path)
    (tmp_path / "judge.py").write_text("# judge v1\n")
    with_judge = hash_task(tmp_path)
    assert base != with_judge
    (tmp_path / "config.yaml").write_text("Attack:\n  malicious_goal: do MORE harm\n")
    changed_goal = hash_task(tmp_path)
    assert changed_goal != with_judge


def test_hash_task_benign_pins_first_task_instruction(tmp_path) -> None:
    # A benign task has no malicious_goal, so the first task_instruction (the exact
    # bytes task.py exposes as the benign Goal) must be pinned -> benign-goal drift
    # is caught, not invisible.
    import hashlib

    (tmp_path / "judge.py").write_text("# judge\n")
    judge_bytes = (tmp_path / "judge.py").read_bytes()
    (tmp_path / "config.yaml").write_text("Task:\n  task_instruction: Book a hotel in Paris.\n")
    expected = hashlib.sha256(b"Book a hotel in Paris." + judge_bytes).hexdigest()
    assert hash_task(tmp_path) == expected
    (tmp_path / "config.yaml").write_text("Task:\n  task_instruction: Book a flight to Paris.\n")
    assert hash_task(tmp_path) != expected


# ---------------------------------------------------------------------------
# build / load golden manifest
# ---------------------------------------------------------------------------


def test_committed_golden_manifest_exists_and_loads() -> None:
    assert GOLDEN_HASHES_PATH.is_file(), "data/golden_hashes.json must be committed"
    manifest = load_golden_hashes()
    assert len(manifest) > 0
    assert all(len(v) == 64 for v in manifest.values())


@requires_dataset
def test_committed_golden_hashes_match_dataset() -> None:
    root = dataset_root()
    manifest = load_golden_hashes()
    checked = 0
    for rel, expected in manifest.items():
        task_dir = root / rel
        if not task_dir.is_dir():
            continue  # partial dataset checkout; only verify present dirs
        assert hash_task(task_dir) == expected, f"golden hash drift for {rel}"
        checked += 1
    assert checked > 0, "no committed golden tasks present in this dataset checkout"


@requires_dataset
def test_build_golden_hashes_roundtrip(tmp_path) -> None:
    mapping = build_golden_hashes(
        dataset_root(), sample=6, domains=["travel"], write=True, path=tmp_path / "g.json"
    )
    assert 0 < len(mapping) <= 6
    reloaded = load_golden_hashes(tmp_path / "g.json")
    assert reloaded == mapping
    # spans both benign and malicious (strided sample over sorted tasks)
    assert any("/benign/" in rel for rel in mapping)
    assert any("/malicious/" in rel for rel in mapping)


# ---------------------------------------------------------------------------
# Goal byte-equality (faithfulness)
# ---------------------------------------------------------------------------


@requires_dataset
@pytest.mark.faithfulness
def test_goal_byte_equals_config_malicious_goal() -> None:
    root = dataset_root()
    paths = list(iter_task_config_paths(root, domains=["travel"], types=["malicious"]))
    assert paths, "no malicious travel tasks found"
    checked = 0
    for config_path in paths[:10]:
        raw = yaml.safe_load(config_path.read_text()) or {}
        raw_goal = (raw.get("Attack") or {}).get("malicious_goal")
        if not raw_goal:
            continue
        tc = parse_task_config(config_path, root=root)
        task = DtapTask(task_config=tc)
        # The Goal the optimizer attacks is byte-identical to the dataset's
        # Attack.malicious_goal -- no paraphrase, no truncation.
        assert task.goal.description == raw_goal
        checked += 1
    assert checked > 0


@requires_dataset
@pytest.mark.faithfulness
def test_goal_byte_equals_benign_instruction() -> None:
    root = dataset_root()
    paths = list(iter_task_config_paths(root, domains=["travel"], types=["benign"]))
    assert paths, "no benign travel tasks found"
    config_path = paths[0]
    tc = parse_task_config(config_path, root=root)
    task = DtapTask(task_config=tc)
    if tc.instructions:
        assert task.goal.description == tc.instructions[0]
    else:
        assert task.goal.description == ""

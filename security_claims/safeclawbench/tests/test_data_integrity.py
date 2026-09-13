"""Integrity checks on the vendored benchmark artifact (provenance + balance)."""

from __future__ import annotations

import collections
import os

from safeclawbench_claim.loader import BENCHMARK_PATH, load_cases

_DATA_DIR = os.path.dirname(BENCHMARK_PATH)


def test_balanced_600_100_per_family():
    counts = collections.Counter(c.attack_type for c in load_cases())
    assert sum(counts.values()) == 600
    assert set(counts.values()) == {100}
    assert len(counts) == 6


def test_task_ids_unique():
    ids = [c.task_id for c in load_cases()]
    assert len(set(ids)) == len(ids) == 600


def test_dataset_license_is_vendored():
    license_path = os.path.join(_DATA_DIR, "DATASET_LICENSE")
    assert os.path.exists(license_path)
    with open(license_path, encoding="utf-8") as f:
        assert "MIT License" in f.read()


def test_citation_is_vendored():
    assert os.path.exists(os.path.join(_DATA_DIR, "CITATION.cff"))


def test_seed_cases_are_a_small_minority():
    seeds = [c for c in load_cases() if c.is_seed]
    assert 0 < len(seeds) < 50

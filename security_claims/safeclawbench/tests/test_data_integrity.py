"""Integrity checks on the vendored benchmark artifact (provenance + balance)."""

from __future__ import annotations

import collections
import hashlib
import os

from safeclawbench_claim.judge import JUDGE_PROMPT
from safeclawbench_claim.loader import AGENT_SYSTEM_PROMPT, BENCHMARK_PATH, load_cases

_DATA_DIR = os.path.dirname(BENCHMARK_PATH)

# SHA-256 of what the package takes from upstream at commit e6c29204c24a, so a
# local edit fails offline too, not only under the networked
# `scripts/sync_upstream.py --check`.
_UPSTREAM_FILE_SHA256 = {
    "CITATION.cff": "1db2ce671c2ccdc16c7688319b7dfd1703724e42344be4f232660453c64752cb",
    "DATASET_LICENSE": "b6d40713351ebc3a4c07cb5e5890ecafe2ddb911336184408994e80b9e6fa8bc",
}
_JUDGE_PROMPT_SHA256 = "24844da7b6839bdd7cbc5a71e6470b81bb03ed329078788c58ff28e7cbea730a"
_AGENT_SYSTEM_PROMPT_SHA256 = "a141ade4a473ffa3062a710b57cdd31a104ce9b8f90aab3121187444a6c9817b"


def test_vendored_metadata_files_match_upstream():
    for name, expected in _UPSTREAM_FILE_SHA256.items():
        with open(os.path.join(_DATA_DIR, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == expected, name


def test_ported_prompts_are_verbatim():
    assert hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest() == _JUDGE_PROMPT_SHA256
    assert (
        hashlib.sha256(AGENT_SYSTEM_PROMPT.encode()).hexdigest()
        == _AGENT_SYSTEM_PROMPT_SHA256
    )


def test_benchmark_is_the_split_the_paper_ran():
    # The paper's Table 9 lists the 12-character SHA-256 prefix of the released
    # split (b3e3172e77ce); pinning the full digest of the file vendored from
    # commit e6c29204c24a also catches any later edit to it.
    with open(BENCHMARK_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == "b3e3172e77ce9669b24539e08ffb2bc676fa6281d1a329a3bd408d3f648cdb60"


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

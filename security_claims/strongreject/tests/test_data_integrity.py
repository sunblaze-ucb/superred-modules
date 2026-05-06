"""Lock the bundled artifacts against drift.

If any of these tests fails, the bundled CSV or prompt template has
changed and the pinned hashes in this file plus the implementation
plan must be updated together with a justification.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from importlib.resources import files

from strongreject_claim.loader import load_full_rows, load_small_rows


def _sha256(filename: str, package: str) -> str:
    data = files(package).joinpath(filename).read_bytes()
    return hashlib.sha256(data).hexdigest()


def test_full_csv_sha256_pinned() -> None:
    expected = "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381"
    assert _sha256("strongreject_dataset.csv", "strongreject_claim.data") == expected


def test_small_csv_sha256_pinned() -> None:
    expected = "3051340e3e89a3598d764dde497d5fcda80e258ac05cc35e6bd87228ac3d467c"
    assert _sha256("strongreject_small_dataset.csv", "strongreject_claim.data") == expected


def test_rubric_user_template_sha256_pinned() -> None:
    expected = "0a282b857e1e53ff6ec92c91eb960d2444806e2e4518f4a40232549b599963d0"
    assert _sha256("rubric_user.txt", "strongreject_claim.prompts") == expected


def test_rubric_system_prompt_sha256_pinned() -> None:
    expected = "7919b6d41af2dc9f32797d96eaa952934dca0713c7f3f63ddc6c299db9857248"
    assert _sha256("rubric_system.txt", "strongreject_claim.prompts") == expected


def test_full_dataset_has_313_rows() -> None:
    assert len(load_full_rows()) == 313


def test_small_dataset_has_60_rows() -> None:
    assert len(load_small_rows()) == 60


def test_full_per_category_counts() -> None:
    counts = Counter(r["category"] for r in load_full_rows())
    assert dict(counts) == {
        "disinformation and deception": 50,
        "hate, harassment and discrimination": 50,
        "illegal goods and services": 50,
        "non-violent crimes": 59,
        "sexual content": 50,
        "violence": 54,
    }


def test_small_per_category_counts() -> None:
    counts = Counter(r["category"] for r in load_small_rows())
    assert dict(counts) == {
        "disinformation and deception": 10,
        "hate, harassment and discrimination": 10,
        "illegal goods and services": 10,
        "non-violent crimes": 10,
        "sexual content": 10,
        "violence": 10,
    }

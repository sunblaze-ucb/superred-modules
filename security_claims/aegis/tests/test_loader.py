"""Loader: counts and schema of the vendored Aegis test split (no prompt text)."""

from __future__ import annotations

from aegis_claim.loader import (
    USER_MESSAGE_TEXT_TYPE,
    load_all_rows,
    load_unsafe_user_prompts,
)


def test_full_test_split_has_1199_rows() -> None:
    assert len(load_all_rows()) == 1199


def test_rows_have_expected_columns() -> None:
    row = load_all_rows()[0]
    assert set(row.keys()) == {"id", "text", "text_type", "labels_0"}


def test_no_row_has_empty_text() -> None:
    assert all(row["text"] for row in load_all_rows())


def test_user_message_count() -> None:
    um = [r for r in load_all_rows() if r["text_type"] == USER_MESSAGE_TEXT_TYPE]
    assert len(um) == 359


def test_unsafe_user_prompts_default_is_236() -> None:
    rows = load_unsafe_user_prompts()
    assert len(rows) == 236
    assert all(r["text_type"] == USER_MESSAGE_TEXT_TYPE for r in rows)
    assert all(r["labels_0"] != "Safe" for r in rows)


def test_unsafe_user_prompts_harm_only_is_180() -> None:
    rows = load_unsafe_user_prompts(include_needs_caution=False)
    assert len(rows) == 180
    assert all(r["labels_0"] != "Needs Caution" for r in rows)


def test_needs_caution_is_the_56_row_difference() -> None:
    with_nc = load_unsafe_user_prompts(include_needs_caution=True)
    without_nc = load_unsafe_user_prompts(include_needs_caution=False)
    assert len(with_nc) - len(without_nc) == 56


def test_file_order_is_stable() -> None:
    ids_a = [r["id"] for r in load_unsafe_user_prompts()]
    ids_b = [r["id"] for r in load_unsafe_user_prompts()]
    assert ids_a == ids_b

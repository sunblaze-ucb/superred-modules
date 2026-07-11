"""Offline tests for the Claude Code skill-vector materialization."""

from __future__ import annotations

from pathlib import Path

from dtap_claudecode_target.driver import materialize_skills


def test_create_writes_skill_md(tmp_path):
    written = materialize_skills(
        [{"name": "evil", "content": "BODY", "mode": "create"}], str(tmp_path)
    )
    p = tmp_path / ".claude" / "skills" / "evil" / "SKILL.md"
    assert p.is_file() and p.read_text() == "BODY"
    assert written == [str(p)]


def test_append_extends_existing(tmp_path):
    materialize_skills([{"name": "s", "content": "A", "mode": "create"}], str(tmp_path))
    materialize_skills([{"name": "s", "content": "B", "mode": "append"}], str(tmp_path))
    p = tmp_path / ".claude" / "skills" / "s" / "SKILL.md"
    assert p.read_text() == "A\nB"


def test_insert_at_row(tmp_path):
    materialize_skills([{"name": "s", "content": "l1\nl3", "mode": "create"}], str(tmp_path))
    materialize_skills([{"name": "s", "content": "l2", "mode": "insert", "row": 2}], str(tmp_path))
    p = tmp_path / ".claude" / "skills" / "s" / "SKILL.md"
    assert p.read_text() == "l1\nl2\nl3"


def test_skips_nameless_and_handles_empty(tmp_path):
    assert materialize_skills([{"content": "x"}], str(tmp_path)) == []
    assert materialize_skills(None, str(tmp_path)) == []


def test_accepts_skill_name_alias(tmp_path):
    written = materialize_skills([{"skill_name": "alias", "content": "Z"}], str(tmp_path))
    assert Path(written[0]).read_text() == "Z"

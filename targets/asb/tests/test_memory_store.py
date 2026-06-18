"""Unit tests for the durable memory store (offline, injected embedder)."""

from __future__ import annotations

import pytest

from asb_target.memory_store import MemoryEmbeddingError, MemoryHit, MemoryStore

_VOCAB = ["weather", "poison", "attacker", "workflow", "stock", "market", "tool", "patient"]


def _fake_embed(texts):  # type: ignore[no-untyped-def]
    return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


def _store() -> MemoryStore:
    return MemoryStore(embed=_fake_embed)


def test_empty_store_returns_none() -> None:
    assert _store().search("anything") is None


def test_add_then_search_single() -> None:
    s = _store()
    s.add("Agent: a; Task: check weather; Workflow: [w]; Tools: []", {"source": "wx"})
    hit = s.search("weather forecast please")
    assert isinstance(hit, MemoryHit)
    assert "weather" in hit.page_content
    assert hit.metadata == {"source": "wx"}
    assert len(s) == 1


def test_top_one_selection_among_many() -> None:
    s = _store()
    s.add("Agent: a; Task: benign weather; Workflow: [w]; Tools: []")
    s.add("Agent: a; Task: poison via attacker tool workflow; Workflow: [m]; Tools: []")
    s.add("Agent: a; Task: stock market quote; Workflow: [q]; Tools: []")
    hit = s.search("please poison using the attacker tool")
    assert hit is not None and "poison" in hit.page_content


def test_clear_empties() -> None:
    s = _store()
    s.add("x")
    s.clear()
    assert len(s) == 0
    assert s.search("x") is None


def test_durable_object_identity_survives_simulated_reset() -> None:
    # The store object is held by the target across reset_ephemeral_state; the
    # store itself is never cleared by a per-run reset, only by clear() at task end.
    s = _store()
    s.add("Agent: a; Task: t; Workflow: [w]; Tools: []")
    # (simulate a reset: nothing calls s.clear())
    assert len(s) == 1


def test_embedding_error_raised_loudly() -> None:
    def boom(texts):  # type: ignore[no-untyped-def]
        raise RuntimeError("dead endpoint")

    s = MemoryStore(embed=lambda texts: boom(texts))
    # offline injected embed raising RuntimeError propagates (no swallow);
    # the proxy path wraps provider errors as MemoryEmbeddingError.
    with pytest.raises(RuntimeError):
        s.add("x")


def test_proxy_embed_wraps_as_memory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    s = MemoryStore(api_base="http://unused", api_key="unused")

    import litellm

    def fail(**kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("boom")

    monkeypatch.setattr(litellm, "embedding", fail)
    with pytest.raises(MemoryEmbeddingError):
        s.add("x")

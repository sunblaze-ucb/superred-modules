"""Durable agent memory: a lightweight, faithful substitute for ASB's
Chroma vector store.

Upstream ASB gives its agent a persistent vector store (``langchain_chroma``
``Chroma`` with OpenAI embeddings). During a *write* run the agent appends a
record of its own work (``Agent: ...; Task: ...; Workflow: ...; Tools: ...``);
during a *read* run it retrieves the single most similar prior record
(``similarity_search_with_score``, top-1) and is told to follow that
"previous workflow". This is the substrate the memory-poisoning attack
corrupts.

This module reproduces that mechanism without the heavy ``chromadb`` /
``langchain`` dependency stack: it embeds text through the same
OpenAI-compatible endpoint (the experiment's LiteLLM proxy, via the already
present :mod:`litellm`) and retrieves the top-1 document by cosine
similarity. Embeddings are identical to upstream's (same model, same proxy)
and top-1 retrieval is metric-robust for the small per-task corpora the
framework produces, so the retrieved content (hence the agent's behaviour)
is faithful. See ``ASSUMPTIONS.md`` for the documented divergences
(in-process top-1 cosine in place of Chroma; per-task store in place of
upstream's suite-level persisted database).

The store is held as **durable** state on the target: it survives
``reset_ephemeral_state`` (the per-run reset) so an attacker can poison it in
one run and exploit it in a later run of the same task, and it is discarded
only when the controller obtains a fresh target from the factory between
tasks.

A failed embedding call raises loudly; it is never swallowed into agent
state, so a dead or misconfigured endpoint cannot silently contaminate a run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Default embedding model id (served by the LiteLLM proxy, OpenAI-compatible).
#: ``litellm`` needs the ``openai/`` prefix to route an OpenAI-format embedding
#: call through a custom ``api_base``.
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"

#: Injectable embedding function: maps a batch of texts to a batch of vectors.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class MemoryDocument:
    """One stored memory record (mirrors a langchain ``Document``)."""

    page_content: str
    metadata: dict[str, str]
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class MemoryHit:
    """A retrieval result: the matched document and its similarity score."""

    page_content: str
    score: float
    metadata: dict[str, str]


class MemoryEmbeddingError(RuntimeError):
    """Raised when the embedding backend fails.

    Surfaced loudly (never caught into agent state) so a dead/misconfigured
    embeddings endpoint fails the run rather than silently corrupting it.
    """


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class MemoryStore:
    """An in-process top-1 cosine vector store over proxy embeddings.

    Construct with proxy credentials to embed via the live LiteLLM proxy, or
    inject ``embed`` for offline/deterministic tests. The store starts empty;
    the runtime writes to it after a run (when memory mode is enabled) and
    reads the top-1 hit before planning.
    """

    def __init__(
        self,
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
        api_base: str | None = None,
        api_key: str | None = None,
        embed: EmbedFn | None = None,
    ) -> None:
        self._embed_model = embed_model
        self._api_base = api_base
        self._api_key = api_key
        self._embed: EmbedFn = embed if embed is not None else self._embed_via_proxy
        self._docs: list[MemoryDocument] = []

    # -- embedding -----------------------------------------------------------

    def _embed_via_proxy(self, texts: Sequence[str]) -> list[list[float]]:
        import litellm  # imported lazily; the proxy path is optional for tests

        try:
            resp = litellm.embedding(
                model=self._embed_model,
                input=list(texts),
                api_base=self._api_base,
                api_key=self._api_key,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed loud failure
            raise MemoryEmbeddingError(
                f"embedding call failed for model {self._embed_model!r}: {exc}"
            ) from exc
        data = resp["data"]
        return [list(item["embedding"]) for item in data]

    # -- mutation ------------------------------------------------------------

    def add(self, page_content: str, metadata: dict[str, str] | None = None) -> None:
        """Embed and store one record (mirrors ``vectorstore.add_documents``)."""
        vector = self._embed([page_content])[0]
        self._docs.append(
            MemoryDocument(
                page_content=page_content,
                metadata=dict(metadata or {}),
                embedding=tuple(vector),
            )
        )

    def clear(self) -> None:
        """Discard all records. NOT called by the per-run reset; only when a
        task ends (a fresh target is built per task)."""
        self._docs.clear()

    # -- retrieval -----------------------------------------------------------

    def search(self, query: str) -> MemoryHit | None:
        """Return the single most-similar record, or ``None`` if the store is
        empty (mirrors upstream's top-1 ``similarity_search_with_score`` usage,
        which returns ``memory[0]`` and treats an empty result as 'no memory').
        """
        if not self._docs:
            return None
        query_vector = self._embed([query])[0]
        best: MemoryDocument | None = None
        best_score = -2.0
        for doc in self._docs:
            score = _cosine(query_vector, doc.embedding)
            if score > best_score:
                best_score = score
                best = doc
        assert best is not None  # non-empty store guarantees a winner
        return MemoryHit(
            page_content=best.page_content, score=best_score, metadata=dict(best.metadata)
        )

    def __len__(self) -> int:
        return len(self._docs)


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "EmbedFn",
    "MemoryDocument",
    "MemoryHit",
    "MemoryEmbeddingError",
    "MemoryStore",
]

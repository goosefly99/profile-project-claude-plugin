from __future__ import annotations

from pathlib import Path

from profile_project.vectorstore.chroma_store import ChromaDBStore


def test_add_count_query_delete_reset_round_trip(tmp_path: Path) -> None:
    store = ChromaDBStore(persist_path=str(tmp_path / "chroma"), collection="test")
    assert store.count() == 0

    store.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["alpha doc", "beta doc"],
        metadatas=[{"source_type": "agent-page"}, {"source_type": "code"}],
    )
    assert store.count() == 2

    hits = store.query(embedding=[1.0, 0.0, 0.0], top_k=2)
    assert hits[0].id == "a"
    assert hits[0].document == "alpha doc"
    assert hits[0].metadata["source_type"] == "agent-page"
    # cosine similarity of identical unit vectors is ~1.0, clamped into [0, 1]
    assert 0.0 <= hits[0].score <= 1.0
    assert hits[0].score > hits[1].score

    store.delete(ids=["a"])
    assert store.count() == 1

    store.reset()
    assert store.count() == 0


def test_query_empty_collection_returns_empty_list(tmp_path: Path) -> None:
    store = ChromaDBStore(persist_path=str(tmp_path / "chroma"), collection="test")
    assert store.query(embedding=[0.1, 0.2, 0.3], top_k=5) == []


def test_upsert_overwrites_same_id_idempotently(tmp_path: Path) -> None:
    store = ChromaDBStore(persist_path=str(tmp_path / "chroma"), collection="test")
    store.add(ids=["x"], embeddings=[[1.0, 0.0]], documents=["v1"])
    store.add(ids=["x"], embeddings=[[1.0, 0.0]], documents=["v2"])
    assert store.count() == 1
    assert store.query(embedding=[1.0, 0.0], top_k=1)[0].document == "v2"

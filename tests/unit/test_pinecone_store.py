from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from profile_project.vectorstore.pinecone_store import (
    IndexDimensionMismatch,
    PineconeStore,
)


def _fake_pc(*, has_index: bool, dimension: int, ready: bool = True) -> MagicMock:
    pc = MagicMock(name="Pinecone")
    pc.has_index.return_value = has_index
    pc.describe_index.return_value = SimpleNamespace(
        dimension=dimension,
        host="https://my-index-abc.svc.pinecone.io",
        status={"ready": ready},
    )
    index = MagicMock(name="Index")
    index.describe_index_stats.return_value = {"total_vector_count": 7}
    index.query.return_value = {
        "matches": [
            {
                "id": "a",
                "score": 0.91,
                "metadata": {"document": "alpha", "source_type": "code"},
            }
        ]
    }
    pc.Index.return_value = index
    return pc


def test_connects_by_host_and_never_creates_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pc = _fake_pc(has_index=True, dimension=384)
    monkeypatch.setattr(
        "profile_project.vectorstore.pinecone_store.Pinecone",
        lambda api_key: pc,
    )
    store = PineconeStore(
        api_key="k",
        index="my-index",
        embedding_dim=384,
        collection="profile-project",
        namespace="profile-v1",
    )
    pc.has_index.assert_called_once_with("my-index")
    pc.describe_index.assert_called_once_with("my-index")
    pc.Index.assert_called_once_with(host="https://my-index-abc.svc.pinecone.io")
    # the load-bearing non-goal: no create_index* path exists
    assert not hasattr(store, "create_index")
    for attr in dir(pc):
        assert not attr.startswith("create_index")


def test_missing_index_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pc = _fake_pc(has_index=False, dimension=384)
    monkeypatch.setattr(
        "profile_project.vectorstore.pinecone_store.Pinecone",
        lambda api_key: pc,
    )
    with pytest.raises(ValueError, match="my-index"):
        PineconeStore(api_key="k", index="my-index", embedding_dim=384, collection="c")
    pc.Index.assert_not_called()


def test_dimension_mismatch_raises_index_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pc = _fake_pc(has_index=True, dimension=1536)
    monkeypatch.setattr(
        "profile_project.vectorstore.pinecone_store.Pinecone",
        lambda api_key: pc,
    )
    with pytest.raises(IndexDimensionMismatch) as excinfo:
        PineconeStore(api_key="k", index="my-index", embedding_dim=384, collection="c")
    assert excinfo.value.index_dim == 1536
    assert excinfo.value.embedding_dim == 384
    pc.Index.assert_not_called()  # never connect into a mismatched geometry


def test_add_and_query_use_byo_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    pc = _fake_pc(has_index=True, dimension=384)
    monkeypatch.setattr(
        "profile_project.vectorstore.pinecone_store.Pinecone",
        lambda api_key: pc,
    )
    store = PineconeStore(
        api_key="k",
        index="my-index",
        embedding_dim=384,
        collection="c",
        namespace="profile-v1",
    )
    store.add(
        ids=["a"],
        embeddings=[[0.1] * 384],
        documents=["alpha"],
        metadatas=[{"source_type": "code"}],
    )
    upsert_kwargs: dict[str, Any] = pc.Index.return_value.upsert.call_args.kwargs
    assert upsert_kwargs["namespace"] == "profile-v1"
    vec = upsert_kwargs["vectors"][0]
    assert vec["id"] == "a"
    assert vec["values"] == [0.1] * 384
    # the document is folded into metadata (Pinecone has no document column)
    assert vec["metadata"]["document"] == "alpha"
    assert vec["metadata"]["source_type"] == "code"

    hits = store.query(embedding=[0.1] * 384, top_k=3)
    query_kwargs: dict[str, Any] = pc.Index.return_value.query.call_args.kwargs
    assert query_kwargs["namespace"] == "profile-v1"
    assert query_kwargs["include_metadata"] is True
    assert hits[0].id == "a"
    assert hits[0].document == "alpha"
    assert hits[0].score == pytest.approx(0.91)


def test_count_reads_namespace_vector_count(monkeypatch: pytest.MonkeyPatch) -> None:
    pc = _fake_pc(has_index=True, dimension=384)
    monkeypatch.setattr(
        "profile_project.vectorstore.pinecone_store.Pinecone",
        lambda api_key: pc,
    )
    store = PineconeStore(
        api_key="k", index="my-index", embedding_dim=384, collection="c"
    )
    assert store.count() == 7

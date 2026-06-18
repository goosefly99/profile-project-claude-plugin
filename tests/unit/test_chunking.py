from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest
import tiktoken

from profile_project.vectorstore.chunking import (
    ChunkConfig,
    RawContent,
    presplit,
    token_chunk,
)
from profile_project.vectorstore.ids import (
    DEFAULT_ST_EMBEDDER_VERSION,
    EMBEDDER_VERSION_FORMAT,
    chunk_id,
    content_hash,
)
from profile_project.vectorstore.protocols import (
    Embedder,
    QueryResult,
    VectorStore,
)


def test_query_result_is_frozen_with_expected_fields() -> None:
    r = QueryResult(id="c1", document="hello", score=0.81, metadata={"path": "a.md"})
    assert r.id == "c1"
    assert r.document == "hello"
    assert r.score == 0.81
    assert r.metadata == {"path": "a.md"}
    with pytest.raises(FrozenInstanceError):
        r.score = 0.0  # type: ignore[misc]


def test_embedder_is_runtime_checkable_protocol() -> None:
    class FakeEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0, 1.0]

        @property
        def dimension(self) -> int:
            return 2

        @property
        def model_name(self) -> str:
            return "all-MiniLM-L6-v2"

        @property
        def embedder_version(self) -> str:
            return DEFAULT_ST_EMBEDDER_VERSION

        @property
        def embedding_provider(self) -> str:
            return "sentence-transformers"

        def probe_dimension(self) -> int:
            return 2

    e = FakeEmbedder()
    assert isinstance(e, Embedder)
    assert e.embed(["a", "b"]) == [[0.0, 1.0], [0.0, 1.0]]
    assert e.embed_query("q") == [0.0, 1.0]
    assert e.dimension == 2
    assert e.embedder_version == DEFAULT_ST_EMBEDDER_VERSION


def test_non_embedder_object_fails_runtime_check() -> None:
    class NotAnEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    assert not isinstance(NotAnEmbedder(), Embedder)


def test_vectorstore_protocol_accepts_a_conforming_impl() -> None:
    class FakeStore:
        def add(
            self,
            ids: list[str],
            embeddings: list[list[float]],
            documents: list[str],
            metadatas: list[dict[str, object]] | None = None,
        ) -> None:
            return None

        def query(
            self,
            embedding: list[float],
            top_k: int = 10,
            where: dict[str, object] | None = None,
            where_document: dict[str, object] | None = None,
        ) -> list[QueryResult]:
            return [QueryResult(id="c1", document="d", score=1.0, metadata={})]

        def delete(
            self, ids: list[str] | None = None, where: dict[str, object] | None = None
        ) -> None:
            return None

        def count(self) -> int:
            return 1

        def reset(self) -> None:
            return None

    store: VectorStore = FakeStore()
    assert store.count() == 1
    assert store.query([0.0, 1.0]) == [
        QueryResult(id="c1", document="d", score=1.0, metadata={})
    ]


def test_embedder_version_format_is_the_canonical_shape() -> None:
    assert EMBEDDER_VERSION_FORMAT == "<provider>/<model>@<precision/norm tag>"


def test_default_st_embedder_version_is_byte_identical_literal() -> None:
    # Byte-identical equality is the geometry guard (sec.8.8/sec.10.3/sec.10.4).
    assert (
        DEFAULT_ST_EMBEDDER_VERSION
        == "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"
    )


def test_content_hash_is_sha256_hexdigest() -> None:
    assert content_hash("hello") == hashlib.sha256(b"hello").hexdigest()


def test_chunk_id_matches_the_sec_10_4_formula() -> None:
    expected_key = (
        "src1\x1fa.md\x1f0-512\x1f"
        + DEFAULT_ST_EMBEDDER_VERSION
        + "\x1f"
        + content_hash("body")
    )
    expected = hashlib.sha256(expected_key.encode("utf-8")).hexdigest()
    assert (
        chunk_id(
            source_id="src1",
            path="a.md",
            start_token=0,
            end_token=512,
            embedder_version=DEFAULT_ST_EMBEDDER_VERSION,
            content_hash=content_hash("body"),
        )
        == expected
    )


def test_same_content_yields_same_id() -> None:
    args = dict(
        source_id="src1",
        path="a.md",
        start_token=0,
        end_token=512,
        embedder_version=DEFAULT_ST_EMBEDDER_VERSION,
        content_hash=content_hash("body"),
    )
    assert chunk_id(**args) == chunk_id(**args)  # type: ignore[arg-type]


def test_changed_embedder_version_yields_different_id() -> None:
    base = dict(
        source_id="src1",
        path="a.md",
        start_token=0,
        end_token=512,
        content_hash=content_hash("body"),
    )
    a = chunk_id(embedder_version=DEFAULT_ST_EMBEDDER_VERSION, **base)  # type: ignore[arg-type]
    b = chunk_id(embedder_version="openai/text-embedding-3-small@float-1536", **base)  # type: ignore[arg-type]
    assert a != b


def test_changed_content_yields_different_id() -> None:
    base = dict(
        source_id="src1",
        path="a.md",
        start_token=0,
        end_token=512,
        embedder_version=DEFAULT_ST_EMBEDDER_VERSION,
    )
    a = chunk_id(content_hash=content_hash("body-one"), **base)  # type: ignore[arg-type]
    b = chunk_id(content_hash=content_hash("body-two"), **base)  # type: ignore[arg-type]
    assert a != b


def test_chunk_config_defaults_and_stride() -> None:
    config = ChunkConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.token_encoding == "cl100k_base"
    assert config.stride == 448  # 512 - 64


def test_chunk_config_stride_is_forced_at_least_one() -> None:
    # overlap >= size would otherwise produce a non-positive stride (infinite loop).
    assert ChunkConfig(chunk_size=10, chunk_overlap=10).stride == 1
    assert ChunkConfig(chunk_size=10, chunk_overlap=99).stride == 1


def test_token_chunk_single_window_for_short_text() -> None:
    config = ChunkConfig()
    chunks = token_chunk(
        [RawContent(text="hello world", metadata={"path": "a.md"})], config
    )
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].metadata["path"] == "a.md"
    assert chunks[0].metadata["span"] == {"start_token": 0, "end_token": 2}


def test_token_chunk_slides_with_overlap_and_keeps_tail() -> None:
    config = ChunkConfig()
    text = " ".join(str(n) for n in range(1000))
    n_tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
    expected_starts = list(range(0, n_tokens, config.stride))
    chunks = token_chunk([RawContent(text=text, metadata={})], config)
    spans = [c.metadata["span"] for c in chunks]
    starts = [s["start_token"] for s in spans]  # type: ignore[index]
    assert starts == expected_starts
    # first window is a full 512 tokens; the tail window is shorter.
    assert spans[0]["end_token"] - spans[0]["start_token"] == 512  # type: ignore[index]
    assert spans[-1]["end_token"] - spans[-1]["start_token"] < 512  # type: ignore[index]


def test_token_chunk_skips_empty_token_lists() -> None:
    config = ChunkConfig()
    chunks = token_chunk(
        [
            RawContent(text="", metadata={"path": "empty"}),
            RawContent(text="real content here", metadata={"path": "real"}),
        ],
        config,
    )
    assert [c.metadata["path"] for c in chunks] == ["real"]


def test_token_chunk_metadata_is_independent_shallow_copy() -> None:
    config = ChunkConfig()
    source_meta: dict[str, object] = {"path": "a.md", "source_type": "doc"}
    chunks = token_chunk([RawContent(text="hello world", metadata=source_meta)], config)
    chunks[0].metadata["path"] = "MUTATED"
    # mutating a chunk's metadata must not bleed back into the source dict.
    assert source_meta["path"] == "a.md"


def test_presplit_empty_text_returns_empty_list() -> None:
    assert presplit("", "doc") == []
    assert presplit("   \n\t ", "code") == []


def test_presplit_code_splits_on_top_level_symbols() -> None:
    code = (
        "import os\n\n"
        "def alpha():\n    return 1\n\n"
        "class Beta:\n    pass\n"
    )
    units = presplit(code, "code")
    # preamble (imports) + def alpha + class Beta
    assert len(units) == 3
    assert units[0].startswith("import os")
    assert units[1].startswith("def alpha")
    assert units[2].startswith("class Beta")


def test_presplit_doc_splits_on_markdown_headings() -> None:
    doc = "# Title\nintro\n\n## Section A\nbody a\n\n### Sub\nbody sub\n"
    units = presplit(doc, "doc")
    assert len(units) == 3
    assert units[0].startswith("# Title")
    assert units[1].startswith("## Section A")
    assert units[2].startswith("### Sub")


def test_presplit_agent_page_uses_heading_hierarchy_like_doc() -> None:
    page = "# Overview\nbody\n\n## Subsystems\nmore\n"
    assert presplit(page, "agent-page") == presplit(page, "doc")


def test_presplit_transcript_splits_on_blank_lines() -> None:
    transcript = "Alice: hi\n\nBob: hello there\n\nAlice: bye"
    units = presplit(transcript, "transcript")
    assert units == ["Alice: hi", "Bob: hello there", "Alice: bye"]


def test_presplit_unknown_source_type_returns_whole_text() -> None:
    assert presplit("anything goes here", "external") == ["anything goes here"]

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken


@dataclass(frozen=True)
class RawContent:
    """One pre-split source unit fed into ``token_chunk`` (sec.10.4)."""

    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """One windowed output unit (sec.10.4).

    Carries content + window metadata only (``metadata['span']`` =
    ``{start_token, end_token}``); identity/lineage/embedder stamps are added
    later by the indexer.
    """

    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class ChunkConfig:
    """Token-windowing config (sec.10.4): 512/64 over the cl100k_base encoding."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    token_encoding: str = "cl100k_base"

    @property
    def stride(self) -> int:
        """Effective stride = chunk_size - chunk_overlap, forced >= 1 (sec.10.4)."""
        return max(1, self.chunk_size - self.chunk_overlap)


def token_chunk(contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
    """Slide token windows over each ``RawContent`` (sec.10.4).

    Builds one ``cl100k_base`` encoder; encodes each ``RawContent.text``; skips
    empty token lists; slides ``[start:start+chunk_size]`` windows decoded back
    to text; advances ``start += stride``. Each chunk inherits a shallow copy of
    the source metadata plus a ``span`` of ``{start_token, end_token}``.
    """
    encoder = tiktoken.get_encoding(config.token_encoding)
    size = config.chunk_size
    stride = config.stride
    chunks: list[Chunk] = []
    for content in contents:
        tokens = encoder.encode(content.text)
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            window = tokens[start : start + size]
            end = start + len(window)
            text = encoder.decode(window)
            metadata = dict(content.metadata)  # shallow copy
            metadata["span"] = {"start_token": start, "end_token": end}
            chunks.append(Chunk(text=text, metadata=metadata))
            start += stride
    return chunks


_HEADING_RE = re.compile(r"^#{1,3}\s", flags=re.MULTILINE)
_TOP_LEVEL_SYMBOL_RE = re.compile(r"^(?:def |class |```)", flags=re.MULTILINE)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")


def _split_on_anchors(text: str, anchor: re.Pattern[str]) -> list[str]:
    """Split ``text`` so each unit begins at an anchor match; keep the preamble."""
    starts = [m.start() for m in anchor.finditer(text)]
    if not starts:
        return [text]
    bounds = sorted(set([0, *starts, len(text)]))
    units = [text[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    return [u for u in units if u.strip()]


def presplit(text: str, source_type: str) -> list[str]:
    """Per-source-type pre-split before token windowing (sec.10.4 table).

    Always end in ``token_chunk`` for safety; this only chooses boundaries.
    Returns ``[]`` for empty/whitespace-only ``text``.
    """
    if not text.strip():
        return []
    if source_type == "code":
        return _split_on_anchors(text, _TOP_LEVEL_SYMBOL_RE)
    if source_type in ("doc", "agent-page"):
        return _split_on_anchors(text, _HEADING_RE)
    if source_type in ("transcript", "note"):
        units = [u for u in _BLANK_LINE_RE.split(text) if u.strip()]
        return units if units else [text]
    return [text]

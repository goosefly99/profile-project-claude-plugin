from __future__ import annotations

import hashlib

# Canonical embedder_version format — single source of truth (sec.10.3).
# Because the string is compared for byte-identical equality as the geometry
# guard, every artifact/metadata/example MUST use this exact shape.
EMBEDDER_VERSION_FORMAT: str = "<provider>/<model>@<precision/norm tag>"

# The byte-identical literal for the zero-setup default embedder
# (sentence-transformers all-MiniLM-L6-v2). Matches sec.8.8 / sec.10.3 / sec.10.4.
DEFAULT_ST_EMBEDDER_VERSION: str = "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"

# ASCII unit separator joining the chunk-id components (sec.10.4).
_UNIT_SEP = "\x1f"


def content_hash(text: str) -> str:
    """sha256 hex digest of ``text`` (UTF-8) — the content component of a chunk id."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(
    source_id: str,
    path: str,
    start_token: int,
    end_token: int,
    embedder_version: str,
    content_hash: str,
) -> str:
    """Deterministic content-addressed chunk id (sec.10.4).

    ``sha256(f"{source_id}\\x1f{path}\\x1f{start_token}-{end_token}"
             f"\\x1f{embedder_version}\\x1f{content_hash}")``.

    Unchanged chunks -> same id (upsert no-op); an edited chunk or a changed
    ``embedder_version`` -> a new id, forcing prune/re-index (geometries never
    interleave).
    """
    key = _UNIT_SEP.join(
        (
            source_id,
            path,
            f"{start_token}-{end_token}",
            embedder_version,
            content_hash,
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

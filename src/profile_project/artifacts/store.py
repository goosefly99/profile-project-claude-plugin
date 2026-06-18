from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from profile_project.artifacts.paths import artifact_path, artifacts_dir
from profile_project.artifacts.schemas import ARTIFACT_MODELS, ARTIFACT_TYPES
from profile_project.config.files import atomic_write_json
from profile_project.dag.run_state import ArtifactRef


def validate_artifact(
    artifact_type: str, content: dict[str, object]
) -> list[str]:
    """Schema-validate a candidate artifact (no write).

    Returns a list of human-readable error strings; an empty list means the
    ``content`` is valid for ``artifact_type``. An unknown ``artifact_type``
    yields a single error; a schema mismatch is flattened into one error string
    per ``pydantic.ValidationError`` problem (``loc`` joined with ``.``).
    """
    model_cls = ARTIFACT_MODELS.get(artifact_type)
    if model_cls is None:
        return [f"unknown artifact type: {artifact_type!r}"]
    try:
        model_cls.model_validate(content)
    except ValidationError as exc:
        return [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        ]
    return []


def _relative_posix(root: Path, path: Path) -> str:
    """`.profile_project/artifacts/<type>.json` as a project-root-relative POSIX str."""
    return path.relative_to(root).as_posix()


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def store_artifact(
    root: Path,
    run_id: str,
    phase: str,
    artifact_type: str,
    content: dict[str, object],
) -> ArtifactRef:
    """Validate, persist (§7.6 atomic), and register an artifact.

    Validation happens BEFORE any write: a schema-invalid ``content`` raises a
    ``ValueError`` carrying the joined validation errors and leaves zero residue
    on disk. Returns the §7.5 ``ArtifactRef``.
    """
    errors = validate_artifact(artifact_type, content)
    if errors:
        raise ValueError(
            f"invalid {artifact_type} artifact: " + "; ".join(errors)
        )
    model = ARTIFACT_MODELS[artifact_type].model_validate(content)
    target = artifact_path(root, artifact_type)
    atomic_write_json(target, model.model_dump(mode="json", by_alias=True))
    return ArtifactRef(
        type=artifact_type,
        path=_relative_posix(Path(root), target),
        phase=phase,
        created_at=_utc_now_iso(),
        version=1,
        parent_artifact=None,
    )


def load_artifact(
    root: Path, artifact_type: str, run_id: str | None = None
) -> dict[str, object] | None:
    """Read a stored artifact as a ``dict``; returns ``None`` when absent.

    Artifacts are stored flat (one per type, latest-wins); ``run_id`` is
    accepted for API symmetry but does not scope the read path.
    """
    target = artifact_path(root, artifact_type)
    if not target.is_file():
        return None
    raw = target.read_text(encoding="utf-8")
    result: dict[str, object] = json.loads(raw)
    return result


def list_artifact_refs(
    root: Path, run_id: str | None = None
) -> list[ArtifactRef]:
    """List one ``ArtifactRef`` per present artifact (sorted by type).

    Read-only: returns ``[]`` when the artifacts dir does not exist; uses each
    file's mtime as ``created_at``. Artifacts are stored flat; ``run_id`` is
    accepted for API symmetry but does not scope the read path.
    """
    directory = artifacts_dir(root)
    if not directory.is_dir():
        return []
    refs: list[ArtifactRef] = []
    for entry in sorted(directory.glob("*.json"), key=lambda p: p.stem):
        artifact_type = entry.stem
        if artifact_type not in ARTIFACT_TYPES:
            continue
        created_at = datetime.fromtimestamp(
            entry.stat().st_mtime, tz=UTC
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        refs.append(
            ArtifactRef(
                type=artifact_type,
                path=_relative_posix(Path(root), entry),
                phase="",
                created_at=created_at,
                version=1,
                parent_artifact=None,
            )
        )
    return refs

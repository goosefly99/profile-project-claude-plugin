from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.artifacts.schemas import ARTIFACT_TYPES

if TYPE_CHECKING:
    from profile_project.config.settings import Settings

# §6b.1 / §7.5 gitignored-tree dir name (the contract's init_gate exposes no
# STAMP_DIRNAME; this layout constant lives here, alongside the only code that
# composes the artifacts path).
PROFILE_TREE_DIRNAME = ".profile_project"


def artifacts_dir(root: Path, run_id: str | None = None) -> Path:
    """Absolute path to the gitignored per-project artifacts directory (§6b.1).

    With ``run_id`` the artifacts dir is scoped under ``runs/<run_id>``; without
    it the canonical flat ``.profile_project/artifacts`` location is used.
    """
    base = Path(root) / PROFILE_TREE_DIRNAME
    if run_id is not None:
        base = base / "runs" / run_id
    return base / "artifacts"


def artifact_path(
    root: Path, artifact_type: str, run_id: str | None = None
) -> Path:
    """Stable absolute path for a stored artifact of ``artifact_type`` (§8 / §7.5).

    Raises ``ValueError`` for an ``artifact_type`` not in ``ARTIFACT_TYPES``.
    """
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"unknown artifact type: {artifact_type!r}")
    return artifacts_dir(root, run_id) / f"{artifact_type}.json"


def profile_dirs(settings: Settings, root: Path) -> tuple[Path, Path]:
    """Resolve absolute (context_dir, guide_dir) under the runtime project root.

    The project root is resolved at runtime and never persisted (§15); it is
    passed in explicitly as ``root`` and joined with the configured output dirs.
    """
    base = Path(root)
    return (base / settings.output.context_dir, base / settings.output.guide_dir)


def resolve_context_dir(settings: Settings, root: Path) -> Path:
    """Absolute agent-facing context dir; == ``profile_dirs(settings, root)[0]``."""
    return profile_dirs(settings, root)[0]

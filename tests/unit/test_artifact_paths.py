from __future__ import annotations

from pathlib import Path

import pytest

from profile_project.artifacts.paths import (
    artifact_path,
    artifacts_dir,
    profile_dirs,
    resolve_context_dir,
)
from profile_project.config.settings import (
    OutputSettings,
    ProfileSettings,
    Settings,
)


def test_artifact_path_builds_under_gitignored_tree(tmp_path: Path) -> None:
    p = artifact_path(tmp_path, "source-index")
    assert p == tmp_path / ".profile_project" / "artifacts" / "source-index.json"


def test_artifacts_dir_is_under_the_tree(tmp_path: Path) -> None:
    assert artifacts_dir(tmp_path) == tmp_path / ".profile_project" / "artifacts"


def test_artifact_path_rejects_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown artifact type"):
        artifact_path(tmp_path, "not-a-real-type")


def test_profile_dirs_joins_root_with_output_dirs(tmp_path: Path) -> None:
    settings = Settings(
        profile=ProfileSettings(name="demo"),
        output=OutputSettings(
            context_dir="profile/context", guide_dir="profile/guide"
        ),
    )
    context_dir, guide_dir = profile_dirs(settings, tmp_path)
    assert context_dir == tmp_path / "profile" / "context"
    assert guide_dir == tmp_path / "profile" / "guide"


def test_resolve_context_dir_matches_profile_dirs_first_element(
    tmp_path: Path,
) -> None:
    settings = Settings(
        profile=ProfileSettings(name="demo"),
        output=OutputSettings(
            context_dir="profile/context", guide_dir="profile/guide"
        ),
    )
    expected = profile_dirs(settings, tmp_path)[0]
    assert resolve_context_dir(settings, tmp_path) == expected

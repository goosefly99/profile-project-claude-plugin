# tests/unit/test_plugin_assets.py
from __future__ import annotations

from pathlib import Path

import pytest

from profile_project.tools import register_tools  # noqa: F401  (import-safety smoke)

REPO_ROOT = Path(__file__).resolve().parents[2]

# skill dir name -> expected frontmatter `name` value (must match dir name)
REQUIRED_SKILLS: dict[str, str] = {
    "profile-project": "profile-project",
    "init-profile": "init-profile",
    "navigate-profile": "navigate-profile",
    "query-project": "query-project",
    "onboard": "onboard",
    "refresh-profile": "refresh-profile",
}

REQUIRED_COMMANDS: tuple[str, ...] = (
    "init",
    "profile",
    "status",
    "query",
    "navigate",
    "refresh",
)

# agent file stem -> expected frontmatter `name` value (must match subagent_type)
REQUIRED_AGENTS: dict[str, str] = {
    "codebase-analyzer": "codebase-analyzer",
    "docs-analyzer": "docs-analyzer",
    "transcripts-analyzer": "transcripts-analyzer",
}


def parse_frontmatter(text: str) -> dict[str, object]:
    """Parse a leading `---`-delimited frontmatter block into a flat dict.

    Supports `key: value` lines (scalar string values) and inline-list
    values of the form `key: [a, b, c]`. Returns an empty dict when no
    frontmatter block is present.
    """
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("unterminated frontmatter block")
    out: dict[str, object] = {}
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if ":" not in raw:
            raise ValueError(f"malformed frontmatter line: {raw!r}")
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            out[key] = [p.strip() for p in inner.split(",") if p.strip()]
        else:
            out[key] = value
    return out


def test_parse_frontmatter_scalar_and_list() -> None:
    text = "---\nname: codebase-analyzer\ntools: [Read, Glob, Grep]\n---\nbody\n"
    fm = parse_frontmatter(text)
    assert fm["name"] == "codebase-analyzer"
    assert fm["tools"] == ["Read", "Glob", "Grep"]


def test_parse_frontmatter_no_block() -> None:
    assert parse_frontmatter("# just markdown\n") == {}


def test_parse_frontmatter_unterminated_raises() -> None:
    with pytest.raises(ValueError):
        parse_frontmatter("---\nname: x\nno closing fence\n")


@pytest.mark.parametrize("skill_dir,expected_name", sorted(REQUIRED_SKILLS.items()))
def test_skill_exists_with_two_key_frontmatter(
    skill_dir: str, expected_name: str
) -> None:
    path = REPO_ROOT / "skills" / skill_dir / "SKILL.md"
    assert path.is_file(), f"missing skill file: {path}"
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    # §12.2: skill frontmatter uses EXACTLY two keys: name and description.
    assert set(fm) == {"name", "description"}, f"{skill_dir}: keys={sorted(fm)}"
    assert fm["name"] == expected_name
    assert isinstance(fm["description"], str) and fm["description"]


@pytest.mark.parametrize("command", sorted(REQUIRED_COMMANDS))
def test_command_exists_with_description(command: str) -> None:
    path = REPO_ROOT / "commands" / f"{command}.md"
    assert path.is_file(), f"missing command file: {path}"
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    # §12.2: command frontmatter uses `description` (always) + optional
    # `argument-hint` / `allowed-tools`. No other keys are permitted.
    assert (
        "description" in fm
        and isinstance(fm["description"], str)
        and fm["description"]
    )
    allowed = {"description", "argument-hint", "allowed-tools"}
    assert set(fm) <= allowed, f"{command}: {sorted(fm)}"


@pytest.mark.parametrize("agent_stem,expected_name", sorted(REQUIRED_AGENTS.items()))
def test_agent_exists_and_name_matches_subagent_type(
    agent_stem: str, expected_name: str
) -> None:
    path = REPO_ROOT / "agents" / f"{agent_stem}.md"
    assert path.is_file(), f"missing agent file: {path}"
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    # agent `name` MUST equal the §7.7 subagent_type so the directive resolves.
    assert fm["name"] == expected_name
    assert isinstance(fm["description"], str) and fm["description"]
    assert isinstance(fm["tools"], list) and fm["tools"]


def test_init_profile_skill_writes_only_via_pp_init_project() -> None:
    # G6 / §6b.4: the init skill must route ALL bootstrap writes through
    # pp_init_project and must NOT instruct any direct filesystem write.
    body = (REPO_ROOT / "skills" / "init-profile" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "pp_init_project" in body
    assert "sole" in body.lower() and "writer" in body.lower()


def test_profile_skill_documents_orchestration_loop() -> None:
    # §7.8: the orchestration loop tool sequence must be present in order.
    body = (REPO_ROOT / "skills" / "profile-project" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    for tool in (
        "pp_init_run",
        "pp_next_phases",
        "pp_start_phase",
        "pp_store_artifact",
        "pp_complete_phase",
    ):
        assert tool in body, f"profile-project skill missing {tool}"


def test_deterministic_phase_tools_documented() -> None:
    # §7.7: discover_context -> pp_discover_sources(persist=true);
    #       build_vectorstore -> pp_index_build (agent_directive=null).
    body = (REPO_ROOT / "skills" / "profile-project" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "pp_discover_sources" in body and "persist=true" in body
    assert "pp_index_build" in body


def test_agents_carry_completion_contract() -> None:
    # §7.7 completion_contract: store then complete, or fail on error.
    for stem in REQUIRED_AGENTS:
        body = (REPO_ROOT / "agents" / f"{stem}.md").read_text(encoding="utf-8")
        assert "pp_store_artifact" in body
        assert "pp_complete_phase" in body
        assert "pp_fail_phase" in body

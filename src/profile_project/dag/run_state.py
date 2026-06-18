from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from profile_project.config.files import append_jsonl, atomic_write_json
from profile_project.dag.graph import PHASES

PIPELINE_VERSION: str = "profile-project/1"
RUN_STATE_FILENAME: str = "run-state.json"
EVENTS_FILENAME: str = "events.jsonl"

PhaseStatus = Literal["pending", "in_progress", "completed", "skipped", "failed"]
RunStatus = Literal["initialized", "running", "completed", "failed"]


class PipelineError(Exception):
    """Structured pipeline/tool-boundary error carrying a result envelope.

    Rendered by ``tool_envelope`` into the ``{"ok": False, "error": {...}}``
    shape via the read-only :pyattr:`envelope` property.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        remedy: str = "",
        retriable: bool = False,
        **extra: object,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.remedy = remedy
        self.retriable = retriable
        self.extra: dict[str, object] = dict(extra)

    @property
    def envelope(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "retriable": self.retriable,
                "remedy": self.remedy,
                **self.extra,
            },
        }


def utc_now_iso() -> str:
    """Current UTC instant as an ISO-8601 ``"...Z"`` string (§7.5 timestamps)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ArtifactRef(BaseModel):
    """One entry in ``run-state.available_artifacts`` (spec §7.5)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    path: str
    phase: str
    created_at: str
    version: int = 1
    parent_artifact: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-mode dump of this artifact ref (== ``model_dump(mode="json")``)."""
        return self.model_dump(mode="json")


class PhaseState(BaseModel):
    """Per-phase mutable state inside a run (spec §7.5 ``phases.*``)."""

    model_config = ConfigDict(extra="forbid")

    phase_name: str
    status: PhaseStatus = "pending"
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    retry_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class RunState(BaseModel):
    """The persisted run document (spec §7.5)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pipeline_version: str = PIPELINE_VERSION
    created_at: str
    updated_at: str
    completed_at: str | None = None
    status: RunStatus = "initialized"
    config_path: str
    run_data_dir: str | None = None
    run_parameters: dict[str, object] = Field(default_factory=dict)
    phases: dict[str, PhaseState] = Field(default_factory=dict)
    available_artifacts: list[ArtifactRef] = Field(default_factory=list)

    def completed_phase_names(self) -> list[str]:
        """Phase names whose status is ``completed`` (insertion order)."""
        return [
            name for name, ph in self.phases.items() if ph.status == "completed"
        ]

    def skipped_phase_names(self) -> list[str]:
        """Phase names whose status is ``skipped`` (insertion order)."""
        return [name for name, ph in self.phases.items() if ph.status == "skipped"]

    def available_artifact_types(self) -> list[str]:
        """Artifact ``type`` strings currently registered on the run."""
        return [ref.type for ref in self.available_artifacts]

    def to_dict(self) -> dict[str, object]:
        """JSON-mode dump of the run state (== ``model_dump(mode="json")``)."""
        return self.model_dump(mode="json")


def runs_root_for(root: Path) -> Path:
    """The ``<root>/.profile_project/runs`` directory for a project root (§7.5)."""
    return Path(root) / ".profile_project" / "runs"


def run_dir_for(root: Path, run_id: str) -> Path:
    """The run directory ``runs_root_for(root) / run_id`` for a project root."""
    return runs_root_for(root) / run_id


def _project_root_from_run_dir(run_dir: Path) -> Path:
    """Project root from a ``<root>/.profile_project/runs/<run_id>`` run dir.

    The run dir nests three levels under the root: runs/<id> -> .profile_project
    -> <project root>. This is the inverse of ``run_dir_for``.
    """
    return run_dir.parent.parent.parent


def init_run(run_params: dict[str, object], run_dir: Path) -> RunState:
    """Create a fresh run with all phases ``pending``, then toggle-skip (§7.3).

    Every fixed phase (``PHASES`` insertion order) starts ``pending``. Then, for
    each phase carrying a config ``toggle_key`` (``include_docs`` ->
    ``analyze_docs``, ``include_transcripts`` -> ``analyze_transcripts_notes``,
    ``build_vectorstore`` -> ``build_vectorstore``), if the corresponding
    ``run_params`` value is exactly ``False`` the phase is flipped
    ``pending -> skipped`` at creation. A missing/None toggle leaves the phase
    ``pending`` (skip-if-empty is decided later at discover time). Pure: no disk write.
    """
    run_dir = Path(run_dir)
    now = utc_now_iso()
    root = _project_root_from_run_dir(run_dir)
    phases: dict[str, PhaseState] = {
        phase.name: PhaseState(phase_name=phase.name) for phase in PHASES
    }
    for phase in PHASES:
        if phase.toggle_key is not None and run_params.get(phase.toggle_key) is False:
            phases[phase.name].status = "skipped"
    return RunState(
        run_id=run_dir.name,
        created_at=now,
        updated_at=now,
        config_path=str(root / ".profile_project_config.json"),
        run_data_dir=str(run_dir),
        run_parameters=dict(run_params),
        phases=phases,
    )


def _require_run_data_dir(state: RunState) -> Path:
    if state.run_data_dir is None:
        raise PipelineError(
            f"run {state.run_id!r} has no run_data_dir; cannot persist or audit it.",
            code="run_state_unanchored",
            remedy="Re-create the run via pp_init_run (sets run_data_dir).",
        )
    return Path(state.run_data_dir)


def persist(state: RunState) -> None:
    """Atomically write ``run-state.json`` (§7.6), refreshing ``updated_at``.

    Uses Task 8's ``atomic_write_json`` (unique temp + flush + fsync + atomic
    rename with the Windows EPERM/EACCES copy-fallback). Raises ``PipelineError``
    when the run is not anchored to a directory.
    """
    run_data_dir = _require_run_data_dir(state)
    state.updated_at = utc_now_iso()
    atomic_write_json(
        run_data_dir / RUN_STATE_FILENAME, state.model_dump(mode="json")
    )


def append_event(state: RunState, event: str, **fields: object) -> None:
    """Append one audit record to ``events.jsonl`` (§7.5).

    Each line is ``{"event", "ts", "run_id", **fields}``. Uses Task 8's
    ``append_jsonl`` (fsync'd append). Raises ``PipelineError`` when unanchored.
    """
    run_data_dir = _require_run_data_dir(state)
    record: dict[str, object] = {
        "event": event,
        "ts": utc_now_iso(),
        "run_id": state.run_id,
    }
    record.update(fields)
    append_jsonl(run_data_dir / EVENTS_FILENAME, record)


def load_run(run_dir: Path) -> RunState:
    """Read + validate ``run-state.json`` (§7.6).

    Raises ``PipelineError`` (code ``run_state_corrupt``) on a missing file,
    malformed JSON, or any schema violation, advising delete + re-init.
    """
    run_dir = Path(run_dir)
    path = run_dir / RUN_STATE_FILENAME
    remedy = (
        "Delete the run directory and re-run /profile-project:init,"
        " then start a new run."
    )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(
            f"run-state not found or unreadable at {path}: {exc}",
            code="run_state_corrupt",
            remedy=remedy,
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"run-state at {path} is not valid JSON: {exc}",
            code="run_state_corrupt",
            remedy=remedy,
        ) from exc
    try:
        return RunState.model_validate(data)
    except ValueError as exc:
        raise PipelineError(
            f"run-state at {path} failed schema validation: {exc}",
            code="run_state_corrupt",
            remedy=remedy,
        ) from exc


def list_runs(runs_root: Path) -> list[RunState]:
    """Load every well-formed run-state under ``runs_root`` (sorted by ``run_id``).

    A subdirectory without a ``run-state.json`` is silently skipped. A missing
    ``runs_root`` returns ``[]``.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    states: list[RunState] = []
    for child in sorted(runs_root.iterdir()):
        if child.is_dir() and (child / RUN_STATE_FILENAME).is_file():
            states.append(load_run(child))
    states.sort(key=lambda s: s.run_id)
    return states


def recover_run(run_dir: Path) -> tuple[RunState, list[str], list[str]]:
    """Resume-safe recovery of a run (§7.6).

    Loads + integrity-checks the run-state (``load_run`` raises ``PipelineError``
    on corruption), resets every ``in_progress`` phase to ``pending`` with
    ``retry_count += 1``, ``started_at=None`` and ``error=None``, reconciles a
    ``failed`` run status that has no actually-``failed`` phase back to
    ``running``, persists the corrected state, and returns
    ``(state, recovered_phases, warnings)``.
    """
    state = load_run(run_dir)
    recovered_phases: list[str] = []
    warnings: list[str] = []

    for name, phase in state.phases.items():
        if phase.status == "in_progress":
            phase.status = "pending"
            phase.retry_count += 1
            phase.started_at = None
            phase.error = None
            recovered_phases.append(name)

    if recovered_phases:
        warnings.append(
            "reset crashed in_progress phase(s) to pending with retry bump: "
            + ", ".join(recovered_phases)
        )

    has_failed_phase = any(p.status == "failed" for p in state.phases.values())
    if state.status == "failed" and not has_failed_phase:
        state.status = "running"
        warnings.append(
            "run status was 'failed' but no phase is failed; reconciled to 'running'."
        )

    persist(state)
    return (state, recovered_phases, warnings)

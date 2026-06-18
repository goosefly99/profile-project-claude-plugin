---
name: navigate-profile
description: Browse the generated profile — agent-facing context pages, the human guide, and per-phase run artifacts. Use for "show me the architecture page", "what did synthesize_knowledge produce", or /profile-project:navigate.
---

# navigate-profile

Help the user explore an existing profile. Read-only; never triggers a DAG run.

## Run artifacts

- `pp_list_runs` — list run ids + statuses.
- `pp_run_status(run_id)` — full run-state snapshot (phase statuses, available
  artifacts). Pre-init this returns none rather than erroring.
- `pp_list_artifacts(run_id?)` — list artifact refs (`type`, `path`, `phase`).
- `pp_load_artifact(type, run_id?)` — read one stored artifact (e.g.
  `knowledge-graph`, `codebase-analysis`, `verification-report`). To answer
  "what did `synthesize_knowledge` produce", load the `knowledge-graph`
  artifact.

## Generated pages

- Agent-facing pages in `profile/context/`: `overview.md`, `architecture.md`,
  `module-map.md`, `data-flows.md`, `glossary.md`, `key-decisions.md`,
  `onboarding.md`. Read these directly to show dense machine-facing context.
- Human guide in `profile/guide/`: `01-system-overview.md`, `02-components.md`,
  `03-setup-onboarding.md`, and the remaining numbered sections. Read these to
  show readable narrative documentation.

Prefer the page that matches the user's request (e.g. "architecture" →
`profile/context/architecture.md`). For semantic "where is X" questions, defer
to the `query-project` skill.

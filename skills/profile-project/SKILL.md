---
name: profile-project
description: Orchestrate the fixed profiling DAG over a project to produce agent-facing context pages, a human guide, and a queryable vectorstore. Use when the user says "profile this project", "analyze the codebase", "what does this repo do", or runs /profile-project:profile.
---

# profile-project

Drive the fixed, resumable profiling DAG. The MCP server owns all state and
deterministic work; you (the agent) perform every reasoning phase. State is
exchanged only through disk-persisted run-state and schema-validated artifacts.

## Precondition

The project MUST be initialized first. If any `pp_*` tool returns
`{"ok": false, "error": {"code": "not_initialized"}}`, stop and tell the user
to run `/profile-project:init` (the `init-profile` skill). Do not attempt any
direct filesystem write.

## Orchestration loop (§7.8)

1. `pp_init_run(run_parameters)` — create the run; the server writes
   `run-state.json` with every phase `pending`. Toggle-off phases
   (`include_docs=false`, `include_transcripts=false`,
   `build_vectorstore=false`) are set to `skipped` immediately at this call.
2. `pp_next_phases(run_id)` — ask the resolver for the currently runnable
   phases. (`run_not_found` is returned for an unknown `run_id`.)
3. For each runnable phase, `pp_start_phase(run_id, phase)` returns a
   `PhaseBrief` (phase `pending → in_progress`, executor model resolved from
   `phase_models`).
4. Execute the phase by its kind:
   - **`analyze_*` phases** (`analyze_codebase`, `analyze_docs`,
     `analyze_transcripts_notes`) carry an `agent_directive`. Dispatch them as
     **parallel sub-agents**, one per phase, using the `subagent_type`,
     `model`, and `prompt` from the directive
     (`codebase-analyzer` / `docs-analyzer` / `transcripts-analyzer`).
   - **Reasoning build phases** (`synthesize_knowledge`, `build_agent_pages`,
     `build_human_spec`) run inline as your own agent work.
     - `build_agent_pages` MUST write exactly the 7 required agent-facing pages
       under `profile/context/` (§9.1): `overview.md`, `architecture.md`,
       `module-map.md`, `data-flows.md`, `glossary.md`, `key-decisions.md`,
       `onboarding.md`. The `agent-pages` artifact's `page_count` is `7`.
     - `build_human_spec` MUST write exactly the 6 required human guide sections
       under `profile/guide/` (§9.2): `01-system-overview.md`, `02-components.md`,
       `03-setup-onboarding.md`, `04-flows.md`, `05-decisions.md`,
       `06-open-questions.md`. The `human-spec` artifact's `section_count` is `6`.
   - **Deterministic phases** carry `agent_directive = null`: call the server
     tool named in the brief — `pp_discover_sources(persist=true)` for
     `discover_context`, `pp_index_build` for `build_vectorstore` — which does
     the compute AND stores the artifact (`source-index` / `vectorstore-index`)
     server-side.
   - **Pre-flight cost estimate (after `discover_context` completes, before any
     `analyze_*` phase).** Once `discover_context` has completed and registered
     the `source-index`, call `pp_load_artifact("source-index")` and emit a
     single opt-in cost line from its counts BEFORE dispatching the `analyze_*`
     sub-agents:
     `~N files / ~M chunks / est tokens / est $`, where `N = sum(source-index
     counts)`, `tokens ≈ total_bytes / 4` and `M = ceil(tokens / (chunk_size -
     chunk_overlap)) = -(-(total_bytes // 4) // (512 - 64))` using the configured
     `ChunkConfig` (default `512/64`), `est tokens = M * chunk_size` (i.e. `M *
     512`), and `est $` is computed only when `embeddings.method=openai`
     (else `est $ = n/a`). Note the OpenAI 2048-input batch cap (sub-batched,
     order-preserving) when method is openai. This estimate is surfaced here —
     not at `pp_init_run` — because the file/chunk counts only exist once the
     `source-index` is built, so cost is opt-in with real numbers.
5. Completion contract:
   - Agent/sub-agent phases: produce the artifact, call
     `pp_store_artifact(run_id, phase, type, content)` (server schema-validates
     and registers it), then `pp_complete_phase(run_id, phase)`.
   - Deterministic phases: the server tool already stored the artifact, so call
     **only** `pp_complete_phase(run_id, phase)`.
   - On any error call `pp_fail_phase(run_id, phase, error)`; recover a failed
     phase with `pp_retry_phase(run_id, phase)`.
6. Loop back to `pp_next_phases(run_id)`. Results from parallel sub-agents are
   observed on the next `pp_next_phases` call (async, state-via-disk contract).
7. The run is terminal (`completed`) when `pp_next_phases` returns `[]` and no
   phase is `failed`/`in_progress`. Toggled-off branches contribute only
   `skipped` nodes that satisfy their successors' edges without being scheduled,
   so the loop terminates even with `build_vectorstore=false`.

## Hygiene

All logs are server-side on stderr; never echo tool internals to stdout. Never
write a secret value anywhere. The `profile/` guides are committable; the
`.profile_project/` tree is gitignored.

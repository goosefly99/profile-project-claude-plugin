---
description: Run the full fixed profiling DAG over this project to produce agent context pages, a human guide, and (if enabled) a queryable vectorstore.
allowed-tools: Bash(uv run:*), Read, Glob, Grep
---

Use the `profile-project` skill to drive the §7.8 orchestration loop:
`pp_init_run` → `pp_next_phases` → `pp_start_phase` → dispatch parallel
`analyze_*` sub-agents (`codebase-analyzer`, `docs-analyzer`,
`transcripts-analyzer`) and run the reasoning build phases inline →
`pp_store_artifact` → `pp_complete_phase`, looping until `pp_next_phases`
returns `[]`. Deterministic phases call `pp_discover_sources(persist=true)` and
`pp_index_build`. If a tool reports `not_initialized`, instruct the user to run
`/profile-project:init` first.

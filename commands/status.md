---
description: Show the current profiling run/phase status and what is runnable next.
---

Report run status and the next runnable phases. Use `pp_list_runs` to find the
active run id, `pp_run_status(run_id)` for the full phase-by-phase snapshot, and
`pp_next_phases(run_id)` for the phases the resolver would run next. When
`pp_next_phases` returns `[]` and no phase is `failed`/`in_progress`, report the
run as `completed`. Surface any `failed` phase and recommend
`pp_retry_phase(run_id, phase)`.

---
name: refresh-profile
description: Incrementally refresh an existing profile after the project changed — re-discover sources and re-run only the affected phases. Use for "the code changed, update the profile" or /profile-project:refresh.
---

# refresh-profile

Update an existing profile without redoing everything. The project must already
be initialized.

## Steps

1. Re-run discovery: `pp_discover_sources(persist=true)` refreshes the
   `source-index` (gitignore/excluded-dirs aware + manifest). Compare against
   the previous index to identify changed source classes.
2. Start a run with `pp_init_run(run_parameters)` and drive the §7.8
   orchestration loop (`pp_next_phases` → `pp_start_phase` → execute →
   `pp_store_artifact` → `pp_complete_phase`) for the affected analysis and
   build phases only; unaffected phases whose inputs are unchanged need not be
   re-run.
3. Re-embed incrementally with `pp_index_rebuild(run_id?)`. Stable
   content-addressed chunk ids mean unchanged chunks keep their vectors and
   only deltas move; a changed `embedder_version` forces a full, geometry-safe
   re-index.
4. Run `verify_profile` so the `verification-report` reflects the refreshed
   profile (coverage OK, no broken cross-links, query smoke test passes).

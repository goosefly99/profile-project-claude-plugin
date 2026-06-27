# profile-project — Agent Usage Guide

`profile-project` is a per-session **MCP server** (FastMCP, stdio transport) that profiles a
project — its source code, in-repo docs, transcripts, notes, and external references — into
three durable deliverables:

1. **Agent-facing context pages** (`profile/context/`) — dense, multi-page markdown built for
   fast machine consumption.
2. **A human/developer guide** (`profile/guide/`) — readable narrative documentation for
   onboarding.
3. **An optional local vectorstore** over both, queried semantically.

The server name is `profile-project`; every tool it exposes is prefixed `pp_`. It runs on
demand over stdio (launched as `uv run --directory <plugin-root> python -m profile_project`),
needs no Docker and no mandatory Ollama, and writes **nothing** into a target project until
that project is explicitly initialized.

## Tools

The server registers **27** tools in five groups.

### Config and initialization

- `pp_init_project` — the bootstrap; transactionally writes the project config, the gitignored
  `.profile_project/` tree, the `.initialized` stamp, and a `.gitignore` entry. The only path
  that makes the project usable.
- `pp_config_path` — report the config file path and whether the project is initialized.
- `pp_config_show` — show the resolved settings (secrets masked) and whether the vectorstore is
  enabled.
- `pp_config_get` — read one config field plus its provenance (which layer supplied it).
- `pp_config_set` — set one config field; rejects secret keys, validates, and rolls back on an
  invalid result.
- `pp_config_validate` — validate the config for a project root without mutating anything.

### Run and DAG lifecycle

- `pp_init_run` — start a new profiling run and return its run id and state.
- `pp_next_phases` — resolve which DAG phases are runnable next and recommend one.
- `pp_start_phase` — mark a phase in-progress and return its phase brief (inputs, executor, task).
- `pp_complete_phase` — mark a phase complete.
- `pp_fail_phase` — mark a phase failed with an error string.
- `pp_retry_phase` — reset a failed phase so it can run again.
- `pp_run_status` — return the full state of one run.
- `pp_list_runs` — list all runs with their ids and statuses.

### Sources

- `pp_discover_sources` — scan the project for source files and classify them; optionally
  persist the resulting source index as an artifact.
- `pp_list_sources` — list discovered sources, optionally filtered by kind.
- `pp_get_source` — fetch a single source by its id.
- `pp_add_source` — manually add a source path or URL to the manifest.

### Artifacts

- `pp_store_artifact` — persist a phase's output artifact under a run.
- `pp_load_artifact` — load the latest artifact of a given type.
- `pp_list_artifacts` — list artifact references, optionally scoped to one run.
- `pp_validate_artifact` — validate artifact content against its schema before storing.

### Vectorstore

- `pp_index_build` — chunk, embed, and upsert the agent pages into the vectorstore.
- `pp_index_rebuild` — full geometry-safe re-index (reset the collection, then rebuild).
- `pp_query` — embed a question, run a vector search, and return ranked, attributed hits.
- `pp_index_status` — report store stats (backend, count, dimension, embedder version, status).
- `pp_vectorstore_check` — read-only diagnostic of vectorstore reachability and dimension; never
  writes.

## Primary workflow

1. **Diagnose, then initialize.** Run `pp_config_validate` and `pp_vectorstore_check`, confirm
   config, then call `pp_init_project`. No mutating tool will write before this succeeds.
2. **Open a run.** Call `pp_init_run` to get a run id, then drive the fixed DAG with
   `pp_next_phases` → `pp_start_phase` → do the phase's work → `pp_validate_artifact` →
   `pp_store_artifact` → `pp_complete_phase`, repeating until the DAG is exhausted.
3. **The fixed DAG** flows: `discover_context` (via `pp_discover_sources`) →
   `analyze_codebase` (plus optional `analyze_docs` and `analyze_transcripts_notes`) →
   `synthesize_knowledge` → `build_agent_pages`, `build_human_spec`, and `build_vectorstore` →
   `verify_profile`. The three analyze phases are parallel; the build phases fan out from the
   synthesized knowledge graph.
4. **Build and query the index.** When the vectorstore is enabled, `build_vectorstore` runs
   `pp_index_build`; afterwards answer questions with `pp_query`, and inspect health with
   `pp_index_status`. Use `pp_index_rebuild` after an embedder change.
5. **Inspect anytime** with `pp_run_status`, `pp_list_runs`, `pp_list_artifacts`, and
   `pp_load_artifact`.

## Key invariants

- **Initialization gate.** Until `pp_init_project` succeeds, every mutating tool refuses with a
  structured `not_initialized` error and writes nothing. The gate is enforced in the server, not
  just in the surface skills.
- **Secrets are environment-only.** `PROFILE_PROJECT_OPENAI_API_KEY` and
  `PROFILE_PROJECT_PINECONE_API_KEY` are read only from the environment, never written to the
  project config, never stamped into chunk metadata, and never logged. The config source hard-
  rejects forbidden secret keys.
- **Vectorstore is opt-in and fail-soft.** A plain install pulls only base dependencies, so the
  embedding and store libraries are absent and the vectorstore warns and disables itself — the
  DAG still runs and still produces both guides. A missing extra, missing key, missing Pinecone
  index, unreachable Ollama, or dimension mismatch all warn and disable rather than crash.
- **Embedder geometry is pinned.** Each chunk records its embedder version; a query-time build
  that finds a different geometry routes to a full safe rebuild rather than mixing vector spaces.
- **Pinecone uses an existing index only.** The server never creates a Pinecone index; it
  validates `index.dimension` against the effective embedding dimension and refuses on mismatch.
- **stdio hygiene.** stdout is the JSON-RPC framing channel; all logs go to stderr only. Nothing
  may print to stdout or the protocol stream corrupts.
- **No user-specific paths in tracked config.** The project root is resolved at runtime, never
  persisted; `.profile_project/` (store, run state, artifacts, cache) is gitignored, while the
  generated `profile/` guides are intentionally committable.

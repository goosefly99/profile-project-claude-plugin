# profile-project

A self-contained Claude Code plugin that runs a fixed, agent-driven DAG to profile a
project — source code, in-repo docs, transcripts, notes, and external references — and
emit three durable deliverables:

1. **Agent-facing context pages** (`profile/context/`) — dense, multi-page markdown for
   fast machine consumption.
2. **A human/developer guide** (`profile/guide/`) — readable narrative documentation for
   onboarding.
3. **A local queryable vectorstore** over both, answered via `pp_query`.

It ships as one Python FastMCP server (`profile_project`) plus a skill suite and slash
commands. It runs per-session over **stdio**, needs **no Docker** and **no mandatory
Ollama**, and **never writes a single local artifact** to a target project until that
project is explicitly initialized via `/profile-project:init`.

## Installation

`profile-project` is launched by Claude Code over stdio. The host runs the server with:

```
uv run --directory ${CLAUDE_PLUGIN_ROOT} python -m profile_project
```

`uv` syncs the environment from the root `pyproject.toml`, then runs the package as a
module. You need only `uv` and Python ≥ 3.11 on the machine.

### Dependency extras

The base install is light. Backends are optional extras (a missing extra for a selected
backend is a **warn + disable**, never a crash):

| Extra | Pulls in | Required for |
|-------|----------|--------------|
| (base) | `mcp`, `pydantic>=2`, `pydantic-settings>=2`, `tiktoken`, `structlog`, `httpx` | server, config, DAG, chunking, ollama (httpx only) |
| `[chroma]` | `chromadb` | ChromaDB local store |
| `[pinecone]` | `pinecone>=9.1.0` | Pinecone remote store |
| `[openai]` | `openai` | OpenAI embeddings |
| `[local-embeddings]` | `sentence-transformers` | sentence-transformers (the default embedder) |
| `[ollama]` | (httpx only — already base) | Ollama embeddings |
| `[all]` | union of the above | everything |

**Recommended default backend.** The recommended path is **sentence-transformers**
(local, offline after first model pull, dim 384) for embeddings + **chromadb** (local
on-disk) for storage — no Docker, no external service, no API key. For a manual/dev
checkout, install it with `[local-embeddings]` and `[chroma]` (or `[all]`):

```
uv pip install -e ".[local-embeddings,chroma]"
```

### Enabling the vectorstore on a plugin install (opt-in)

The server **always starts** on a plain install, but the vectorstore is **off by
default**: the stdio launch command (`uv run … python -m profile_project`) installs only
the base dependencies, so the embedding + store libraries are absent and the conflict
matrix **warns + disables** the vectorstore (the DAG still runs and produces both guides).
This keeps cold start fast — `sentence-transformers` pulls in `torch` (~2 GB), which is
too heavy to download on every launch.

To turn the vectorstore on, add the extras to the launch command in the installed
plugin's `.mcp.json` (`${CLAUDE_PLUGIN_ROOT}/.mcp.json`):

```json
{
  "mcpServers": {
    "profile-project": {
      "command": "uv",
      "args": ["run", "--extra", "local-embeddings", "--extra", "chroma",
               "--directory", "${CLAUDE_PLUGIN_ROOT}", "python", "-m", "profile_project"]
    }
  }
}
```

The **first** launch after adding the extras resolves and downloads them (slow, one
time); later launches reuse the synced environment. Swap in `--extra pinecone` /
`--extra openai` (or `--extra all`) for the corresponding backends. Every vectorstore
backend needs at least one extra; only `ollama` embeddings run on the base install (httpx
is a base dependency) — but a store backend (`chroma` or `pinecone`) is still required.

## Initialization

The plugin must be **explicitly initialized per project** before it writes any local
artifact — this gate is enforced in the MCP server, not just the skill.

```
/profile-project:init
```

`init` runs read-only diagnostics (`pp_config_validate`, `pp_vectorstore_check`),
collects/confirms config, validates that required secrets exist in your environment, then
calls the server tool `pp_init_project`, which transactionally writes
`.profile_project_config.json`, the gitignored `.profile_project/` tree, the
`.initialized` stamp, and the `.gitignore` entry. Until `init` succeeds, every mutating
tool refuses with a structured `not_initialized` error and writes nothing.

Re-run `/profile-project:init` any time to reconfigure (idempotent); use
`/profile-project:init --reinit` to overwrite/reset existing run artifacts.

## Usage

| Command | What it does |
|---------|--------------|
| `/profile-project:init` | Initialize the project (the only path that creates initial artifacts) |
| `/profile-project:profile` | Run the full profiling DAG |
| `/profile-project:status` | Show run/phase status and what's next |
| `/profile-project:query` | Ask a semantic question over the profile |
| `/profile-project:navigate` | Browse generated pages and per-phase artifacts |
| `/profile-project:refresh` | Incrementally refresh an existing profile |

A typical first run: `/profile-project:init` → `/profile-project:profile` →
`/profile-project:query "how is config resolved?"`. The generated `profile/context/` and
`profile/guide/` directories are committable project artifacts.

## Configuration

Configuration is layered. **Project JSON overrides env** (the inverse of
agent-knowledgebase): `.profile_project_config.json` at the project root takes precedence
over `PROFILE_PROJECT_*` environment variables, which act as defaults.

Precedence (highest to lowest): init kwargs → project JSON → env → `.env` → file secrets
→ field defaults.

**Secrets are environment-only.** `PROFILE_PROJECT_OPENAI_API_KEY` and
`PROFILE_PROJECT_PINECONE_API_KEY` are modeled as `SecretStr`, read **only** from the
environment, and are **never** written to `.profile_project_config.json` (the JSON source
hard-rejects forbidden keys), never stamped into chunk metadata, and never logged
(masked). `.env.example` documents the env vars and contains no real values.

Common env vars (full table in `.env.example`):

| Env var | Maps to | Default |
|---------|---------|---------|
| `PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD` | `embeddings.method` | (unset → init prompts) |
| `PROFILE_PROJECT_VECTORSTORE__BACKEND` | `vectorstore.backend` | (unset → init prompts) |
| `PROFILE_PROJECT_CHROMADB__PATH` | `vectorstore.chromadb.path` | `.profile_project/chroma` |
| `PROFILE_PROJECT_OPENAI_API_KEY` | `openai_api_key` (secret) | (none) |
| `PROFILE_PROJECT_PINECONE_API_KEY` | `pinecone_api_key` (secret) | (none) |

**Embeddings.** The default embedder's canonical geometry id is
`sentence-transformers/all-MiniLM-L6-v2@hf-fp32` (stamped per chunk so a query-time
rebuild rejects a mismatched embedder rather than returning meaningless scores). OpenAI
and Ollama are alternatives selected via config.

**Pinecone uses an EXISTING index only.** The plugin **never** creates a Pinecone index.
You supply an existing `index` ref + `embeddings_model`; at connect time the store
validates `index.dimension == effective_embedding_dim` and refuses on mismatch. A missing
index ref, missing key, or dimension mismatch **warns + disables** the vectorstore (the
DAG still runs and produces the guides).

## Troubleshooting

- **"not_initialized" on a tool call.** Run `/profile-project:init` first. The gate is
  server-enforced; no mutating tool writes before initialization.
- **Vectorstore silently disabled.** Run `pp_config_validate` and `pp_vectorstore_check`;
  a missing extra (`[chroma]`/`[pinecone]`/`[openai]`/`[local-embeddings]`), missing API
  key, missing Pinecone index ref, unreachable Ollama host, or dimension mismatch all warn
  + disable rather than crash. The warning names the exact cause. On a **plugin install**
  the most common cause is that the launch command installs only base deps — see
  *Enabling the vectorstore on a plugin install* above to add the `--extra` flags.
- **`pp_query` returns `index_disabled` / `index_empty`.** The vectorstore is off, or no
  vectors have been built yet — run `/profile-project:profile` (which runs
  `build_vectorstore`) or check `pp_index_status`.
- **"project_root_moved".** The project was initialized for a different absolute root.
  Run `/profile-project:init --reinit`.
- **Garbled JSON-RPC / protocol errors.** Nothing may be written to **stdout** under
  stdio transport — all logs go to **stderr** only. If you patched the server, ensure no
  `print()` reaches stdout.
- **First profiling run is slow.** sentence-transformers downloads `all-MiniLM-L6-v2`
  once; subsequent runs are offline.

## Security & hygiene

- **Secrets are env-only** (`SecretStr`): never in `.profile_project_config.json`, never
  in chunk metadata, never logged (masked).
- **`.profile_project/` is gitignored** (local store, run-state, artifacts, cache,
  `.initialized`). The `profile/` guides are intentionally committable.
- **No user-specific absolute paths** are written to tracked config; `profile.root_dir`
  is resolved at runtime, never persisted.
- **stdio hygiene**: logs go to stderr only; stdout stays clean for JSON-RPC framing.
- **No remote provisioning**: Pinecone indexes are never auto-created, so the plugin
  cannot silently incur cost.

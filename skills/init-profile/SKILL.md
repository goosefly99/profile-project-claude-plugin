---
name: init-profile
description: Configure and initialize profile-project for a project (the only path that creates initial artifacts). Use on first run, to reconfigure, or when /profile-project:init is invoked, or when a tool reports not_initialized.
---

# init-profile

Initialize the plugin **per project**. The server is the **sole writer**: this
skill performs NO direct filesystem writes. All bootstrap writes go through the
`pp_init_project` tool (G6, §6b.4). This is what makes the initialization gate
enforced in the server, not the skill.

## Steps (§6b.4)

1. **Resolve project root** (env override `PROFILE_PROJECT_PROJECT_DIR` →
   `CLAUDE_PROJECT_DIR` → `PWD` → cwd). Confirm it with the user.
2. **Diagnose (read-only, pre-init).** Run `pp_config_validate` (provenance +
   warnings + post-conflict `vectorstore_enabled`) and `pp_vectorstore_check`
   (dry-run reachability/dimension; never writes). Surface issues before
   writing. These tools leave zero filesystem residue pre-init.
3. **Collect/confirm config**: vectorstore `enabled` + `backend`
   (`chromadb` local default | `pinecone` existing-index-only | `disabled`);
   embeddings `method` (`sentence-transformers` default | `openai` | `ollama` |
   `disabled`) + model; Pinecone `index`/`namespace` (existing index only —
   never auto-created) if backend is pinecone; sources-manifest seeds
   (extra_doc_globs, transcripts, notes, external, excluded_dirs); phase toggles
   (`include_docs`, `include_transcripts`, `build_vectorstore`); output dirs
   (`context_dir`, `guide_dir`). NON-secret values only — API keys are
   environment-only and must never appear in the JSON config.
4. **Verify required secrets exist in env.** If `method=openai`, require
   `PROFILE_PROJECT_OPENAI_API_KEY`; if `backend=pinecone`, require
   `PROFILE_PROJECT_PINECONE_API_KEY`. A missing required secret aborts init
   with a clear message; nothing is written. (The server re-checks this too.)
5. **Bootstrap via the server.** Call
   `pp_init_project(config, force=...)`. This single tool validates the
   candidate config (config-validate logic + forbidden-secret rejection),
   re-verifies required env secrets, then transactionally (all-or-nothing)
   writes `.profile_project_config.json`, creates the gitignored
   `.profile_project/` tree, writes the `.initialized` stamp, and ensures
   `.profile_project/` is in `.gitignore`. On any failure nothing is left on
   disk. Pass `force=true` only for `--reinit` (overwrite/reset existing
   artifacts; performs root-move re-stamp + absolute-path rewrite). Without
   `force`, re-init is idempotent and preserves `runs/`, `artifacts/`, `chroma/`.

Only after `pp_init_project` returns `ok=true` do the gated tools become
available; hand back to the `profile-project` skill to run the DAG.

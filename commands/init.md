---
description: Initialize profile-project for this project (the only path that creates initial artifacts). Validates env/config, then bootstraps via pp_init_project.
argument-hint: "[--reinit]"
allowed-tools: Bash(uv run:*), Read, Glob, Grep
---

Use the `init-profile` skill to configure and initialize profile-project for
the current project. Resolve the project root, diagnose with `pp_config_validate`
and `pp_vectorstore_check`, collect/confirm non-secret config, verify required
env secrets exist, then perform ALL bootstrap writes via `pp_init_project`
(the server is the sole writer — never write to disk directly). Pass
`force=true` to `pp_init_project` when invoked with `--reinit`.

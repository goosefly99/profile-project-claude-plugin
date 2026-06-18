---
name: query-project
description: Answer semantic questions over the profiled project using the local vectorstore, falling back to reading context pages when the vectorstore is disabled. Use for "how is config resolved?", "where is X handled?", or /profile-project:query.
---

# query-project

Answer "how/where/why" questions over the profile.

## Steps

1. Optionally check `pp_index_status` — reports `backend`, `count`,
   `dimension`, `embedder_version`, `status`. Pre-init it returns
   `status: "uninitialized"` with `count: 0` and never materializes the store.
2. Call `pp_query(query, top_k=10, where?)` — embeds the query, runs vector
   search, returns ranked, attributed hits (each hit names its source page).
   Cite the top hits (page path + score) in your answer.
3. Handle secondary errors:
   - `not_initialized` → tell the user to run `/profile-project:init`.
   - `index_disabled` (vectorstore off) or `index_empty` (no vectors yet) →
     **fall back** to reading the agent pages under `profile/context/`
     (`overview.md`, `architecture.md`, `module-map.md`, `data-flows.md`,
     etc.) and answer from them, telling the user the vectorstore is
     unavailable so retrieval is page-based.

Never fabricate citations; only cite pages that actually returned as hits or
that you read directly.

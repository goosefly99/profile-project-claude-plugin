---
description: Ask a semantic question over the profiled project (vectorstore search with context-page fallback).
argument-hint: "<question>"
---

Use the `query-project` skill to answer the user's question. Call
`pp_query(query, top_k=10)` and cite the ranked, attributed hits. If the
vectorstore reports `index_disabled` or `index_empty`, fall back to reading the
`profile/context/` pages and answer from them. If `not_initialized`, tell the
user to run `/profile-project:init` first.

---
description: Incrementally refresh an existing profile after the project changed.
---

Use the `refresh-profile` skill to update an existing profile: re-run discovery
with `pp_discover_sources(persist=true)`, re-run only the affected analysis and
build phases through the orchestration loop, and re-embed incrementally with
`pp_index_rebuild`. Finish by re-running `verify_profile` so the
`verification-report` reflects the refreshed profile.

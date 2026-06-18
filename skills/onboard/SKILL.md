---
name: onboard
description: Give a new contributor a guided walkthrough of the profiled project — setup, structure, and where to start. Use when someone new wants a guided tour of the codebase.
---

# onboard

Walk a new contributor through the project using the generated guide.

## Steps

1. Read `profile/guide/03-setup-onboarding.md` (the step-by-step environment
   setup, build, run, and first-contribution walkthrough) and present it as a
   guided sequence.
2. Read `profile/context/onboarding.md` (the minimal path for an agent to start
   contributing: where things live, how to run, what to read first) to fill in
   the dense pointers.
3. For any follow-up question the contributor asks, use `pp_query(question)` to
   retrieve attributed answers; if the vectorstore is disabled
   (`index_disabled`/`index_empty`), fall back to the `profile/context/` pages
   as in the `query-project` skill.

If `profile/guide/03-setup-onboarding.md` is absent, the project has not been
profiled yet — direct the user to run `/profile-project:profile` first.

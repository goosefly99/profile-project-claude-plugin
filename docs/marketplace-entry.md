# Marketplace entry — profile-project-auto-dev

To publish `profile-project` to the goosefly99 auto-dev marketplace, add the following
object to the `"plugins"` array in `claude-plugins-auto-dev/.claude-plugin/marketplace.json`.
The shape matches the existing `synth-mcp-auto-dev` and `concepts-collector-auto-dev`
entries.

```json
{
  "name": "profile-project-auto-dev",
  "description": "Profiles a project into agent-facing context, a human guide, and a queryable vectorstore",
  "source": {
    "source": "url",
    "url": "https://github.com/goosefly99/profile-project-claude-plugin.git",
    "ref": "auto_dev"
  },
  "version": "0.1.0",
  "homepage": "https://github.com/goosefly99/profile-project-claude-plugin"
}
```

## How to add it — manual publishing step (out of scope for this repo's build)

> **This is a manual operator step performed in a SEPARATE repository.** The live
> `claude-plugins-auto-dev/.claude-plugin/marketplace.json` belongs to the goosefly99
> marketplace repo, not to this plugin's repo. This plugin's build/CI does **not** edit
> that file (doing so would commit into a repo it does not own). The steps below are the
> manual publishing procedure to run by hand, in the marketplace repo, once the publish
> precondition below holds.

1. In the **marketplace repo**, open `claude-plugins-auto-dev/.claude-plugin/marketplace.json`.
2. Append the object above to the top-level `"plugins"` array (mind the comma between
   entries; the file must remain valid JSON).
3. Validate the JSON (`python -m json.tool marketplace.json`) before committing.
4. Commit on a feature branch in that repo and open a PR — never edit the marketplace on
   `main` directly.

## Publish precondition (must hold before the marketplace entry is merged)

The `auto_dev` branch must already exist on the
`goosefly99/profile-project-claude-plugin` repo and **must contain**
`.claude-plugin/plugin.json` with `name: "profile-project-auto-dev"` and a `version` that
**matches the marketplace entry** (`0.1.0`). Merging the marketplace entry while the
referenced `ref` is missing or carries a mismatched manifest yields an **uninstallable
plugin**.

For a released version, a **tag ref is more stable than a branch ref** (a branch keeps
moving): pin the marketplace `ref` to a release tag once cut, and use `auto_dev` only for
rolling/dev installs.

# @mariozechner/pi-memory-tools

Pi package that adds a persistent curated `memory` tool with two stores:

- `memory`: agent notes about environment, conventions, and learned quirks
- `user`: user profile/preferences and workflow habits

The extension injects a frozen memory snapshot into the system prompt at session start. Mid-session writes persist immediately to disk but only affect the next session snapshot.

## Install

```bash
pi install /absolute/path/to/packages/memory-tools
```

## Smoke Test

```text
Use memory with action "add", target "user", content "Prefers concise technical answers."
Use memory with action "add", target "memory", content "Repo uses npm workspaces and npm run check."
Use memory with action "read", target "user".
Use memory with action "replace", target "user", oldText "concise", content "Prefers concise, direct technical answers."
Use memory with action "remove", target "memory", oldText "npm workspaces".
```

# @mariozechner/pi-skills-tools

Pi package that adds a `skills` tool for managing `SKILL.md` files in:

- user scope: `~/.pi/agent/skills`
- project scope: `<cwd>/.pi/skills`

Supported actions:

- `list`
- `read`
- `write`
- `remove`

## Install

```bash
pi install /absolute/path/to/packages/skills-tools
```

## Smoke Test

```text
Use skills with action "list", scope "user", limit 10.
Use skills with action "write", scope "project", name "release-checks", content "# Release checks\nRun npm run check before release."
Use skills with action "read", scope "project", name "release-checks".
Use skills with action "remove", scope "project", name "release-checks", confirm true.
```

<!-- OSS_WEEKEND_START -->
# 🏖️ OSS Weekend

**Issue tracker reopens Monday, March 16, 2026.**

OSS weekend runs Saturday, March 14, 2026 through Monday, March 16, 2026. New issues are auto-closed during this time. For support, join [Discord](https://discord.com/invite/3cU7Bz4UPx).
<!-- OSS_WEEKEND_END -->

---

<p align="center">
  <a href="https://shittycodingagent.ai">
    <img src="https://shittycodingagent.ai/logo.svg" alt="pi logo" width="128">
  </a>
</p>
<p align="center">
  <a href="https://discord.com/invite/3cU7Bz4UPx"><img alt="Discord" src="https://img.shields.io/badge/discord-community-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://github.com/badlogic/pi-mono/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/badlogic/pi-mono/ci.yml?style=flat-square&branch=main" /></a>
</p>
<p align="center">
  <a href="https://pi.dev">pi.dev</a> domain graciously donated by
  <br /><br />
  <a href="https://exe.dev"><img src="packages/coding-agent/docs/images/exy.png" alt="Exy mascot" width="48" /><br />exe.dev</a>
</p>

# Codi

Codi is a customized Pi-based coding agent workspace. This repo is no longer just a light fork snapshot of upstream Pi. It is the active source tree for a local-first coding agent setup with:

- the Pi core packages (`ai`, `agent`, `coding-agent`, `tui`, `web-ui`, `mom`, `pods`)
- custom installable tool packages used in daily sessions
- project-scoped package loading for repo-specific workflows
- a split CLI setup so local Codi changes stay separate from upstream `pi`

If you want the main terminal app, start with [`packages/coding-agent`](packages/coding-agent). The rest of the monorepo supports that runtime.

## What The App Is Now

The current app is a terminal coding agent with:

- interactive chat + editor UI
- OAuth and API-key model access through the Pi AI layer
- session history with branching, compaction, export, and replay
- local package loading for tools, prompts, skills, extensions, and themes
- repo-local workflow packages for research, delegation, memory, cron, images, search, and other task-specific actions

In this checkout, Codi is being used as a programmable operator shell rather than a stock Pi install.

## CLI Layout

This repo uses a split CLI workflow:

- `codi`: runs this checkout from source
- `pi`: runs the separately installed upstream Pi in its own config directory

That keeps local experiments, packages, and settings isolated from upstream.

## Current Package Layout

The monorepo has two layers.

Core Pi packages:

- [`packages/ai`](packages/ai): model/provider abstraction, OAuth helpers, generated model catalog, streaming APIs
- [`packages/agent`](packages/agent): agent runtime and shared orchestration primitives
- [`packages/coding-agent`](packages/coding-agent): the main terminal coding agent
- [`packages/tui`](packages/tui): terminal UI primitives, editor, markdown, key handling
- [`packages/web-ui`](packages/web-ui): web components and browser-facing UI pieces
- [`packages/mom`](packages/mom): Slack bot integration
- [`packages/pods`](packages/pods): vLLM pod management CLI

Custom Codi packages currently in this repo:

- [`packages/clarify-tools`](packages/clarify-tools): structured clarification UI/tooling
- [`packages/session-search-tools`](packages/session-search-tools): search current session and branches
- [`packages/todo-tools`](packages/todo-tools): installable todo workflow
- [`packages/image-tools`](packages/image-tools): image generation via Google Antigravity
- [`packages/web-tools`](packages/web-tools): lightweight web search and extract helpers
- [`packages/memory-tools`](packages/memory-tools): persistent user/environment memory stores
- [`packages/cron-tools`](packages/cron-tools): durable recurring jobs
- [`packages/delegate-tools`](packages/delegate-tools): bounded nested-agent delegation
- [`packages/skills-tools`](packages/skills-tools): read/write/remove `SKILL.md` files in user or project scope
- [`packages/autoresearch-tools`](packages/autoresearch-tools): Karpathy-style autoresearch loops
- [`packages/evolution-tools`](packages/evolution-tools): bounded skill-evolution workflows
- [`packages/pokemon-tools`](packages/pokemon-tools): local `pokemon-agent` integration
- [`packages/autoresearch-av`](packages/autoresearch-av): standalone AV baseline runner used by autoresearch tooling
- [`packages/extension-sdk`](packages/extension-sdk): helper surface for extension/package authoring
- [`packages/zed`](packages/zed): Zed editor integration assets

## Scope And Loading

Codi currently uses both user and project package scopes.

User scope:

- configured in `~/.pi/agent/settings.json`
- intended for packages you want available from any directory

Project scope:

- configured in [`.pi/settings.json`](/Users/ever/Documents/GitHub/Codi%20/.pi/settings.json)
- only loaded when running inside this checkout

Current project-scoped packages in this repo:

- `autoresearch-tools`
- `evolution-tools`

Typical user-scoped packages in this setup include things like `pokemon-tools`, memory/search helpers, and other cross-repo tools.

## Main Workflow

For day-to-day use, the center of gravity is the interactive coding agent in [`packages/coding-agent`](packages/coding-agent).

That app currently supports:

- startup resource discovery for `AGENTS.md`, skills, prompts, extensions, and themes
- slash commands for sessions, resources, runtime ops, packages, queue/mailbox, workflows, and model management
- package installation and management from inside interactive mode
- extension-driven UI surfaces and custom tools
- local session files with tree navigation and branch/fork flows

The custom packages in this repo are meant to extend that app without modifying every workflow directly in the core runtime.

## Development

Install dependencies:

```bash
npm install
```

Useful repo commands:

```bash
npm run build
npm run check
npm run test
```

Source-driven local launch helpers:

```bash
./pi-test.sh
./test.sh
```

## Documentation Pointers

- [`AGENTS.md`](AGENTS.md): repo-specific development rules
- [`CONTRIBUTING.md`](CONTRIBUTING.md): contribution flow
- [`packages/coding-agent/README.md`](packages/coding-agent/README.md): main app behavior and CLI usage
- package READMEs under [`packages`](packages): current tool/package surfaces

## License

MIT

# Tool Packages

This document defines a Pi-native package plan for the current official tool packages inspired by Hermes's tool surface, but built on Pi's extension and package system.

These packages are intended to be installable Pi packages, not core built-ins. Each package should expose one or more extensions through the `pi` manifest in `package.json`, keep runtime assumptions minimal, and compose cleanly with Pi's existing event, session, and tool APIs.

## Goals

- Keep Pi core minimal
- Package capability surfaces that are broadly useful across workflows
- Reuse Pi's existing extension model instead of adding a parallel plugin runtime
- Keep package boundaries small enough to install independently
- Make later "tool profiles" possible by composing packages rather than hardcoding bundles

## Non-Goals

- Do not port Hermes runtime internals
- Do not introduce a second tool registry outside Pi extensions
- Do not hardcode weighted tool distributions in v1
- Do not merge unrelated tool families into one large package

## Package Set

The current package set is:

1. `@pi/web-tools`
2. `@pi/image-tools`
3. `@pi/session-search-tools`
4. `@pi/todo-tools`
5. `@pi/clarify-tools`
6. `@pi/memory-tools`
7. `@pi/cron-tools`
8. `@pi/delegate-tools`
9. `@pi/skills-tools`
10. `@pi/autoresearch-tools`

## Common Design Rules

All current tool packages should follow the same baseline rules:

- Ship as normal Pi packages with `keywords: ["pi-package"]`
- Prefer `@mariozechner/pi-extension-sdk` as the package authoring import surface
- Register tools through `pi.registerTool()`
- Register user commands only when they materially improve inspection or control
- Use `@sinclair/typebox` schemas for all tool parameters
- Persist state through session entries when state must branch with the session
- Keep tool results compact and truncate large outputs
- Prefer optional integrations over hard dependencies on external services
- Provide custom renderers only when the default output is not good enough

## Shared Contract Layer

The packages should converge on a shared shape even if they are implemented incrementally.

### Tool result shape

Tool results should consistently return:

- `content` for the LLM-facing summary
- `details` for structured UI and branch reconstruction
- `isError` only when the call should be treated as a tool failure

### Optional extension commands

Packages may also expose a narrow user command surface:

- inspection commands, for example `/todos`
- setup and auth commands, when needed
- package-specific status commands, only if they add real value

### Package-local persistence

For stateful packages, prefer session reconstruction from tool result details over ad hoc files:

- `@pi/todo-tools`
- `@pi/memory-tools`
- `@pi/cron-tools`
- `@pi/skills-tools`
- `@pi/session-search-tools` only if cached indexes are later added

## Package Specs

### `@pi/web-tools`

Purpose:
Provide lightweight read-only web retrieval primitives for search and extraction.

Initial tools:

- `quick_web_search`
- `quick_web_extract`

Likely commands:

- none in v1

Implementation notes:

- This is the cleanest package boundary and should be built first
- `web_search` should return compact result lists with title, URL, snippet
- `web_extract` should return cleaned text and key metadata for a URL
- The package should be provider-agnostic about the search backend if possible

State model:

- stateless

Good fit with Pi today:

- high

### `@pi/image-tools`

Purpose:
Generate images from prompts and return stable artifact references.

Initial tools:

- `image_generate`

Likely commands:

- none in v1

Implementation notes:

- This should be treated as an optional integration package
- Generated images should ideally land in an artifact or predictable temp-file flow
- Keep provider-specific image generation behind the package boundary

State model:

- mostly stateless, aside from artifact references

Good fit with Pi today:

- medium

### `@pi/session-search-tools`

Purpose:
Search prior session content, branches, artifacts, and workflow-adjacent history more deliberately than raw transcript scrolling.

Initial tools:

- `session_search`

Likely commands:

- `/session-search` only if a dedicated user-facing inspector is warranted

Implementation notes:

- This is very Pi-native and should integrate with `SessionManager`
- The first version can search entries directly without a separate index
- Search should later be aware of branch boundaries, labels, message roles, tool results, and workflow entry types

State model:

- stateless in v1

Good fit with Pi today:

- high

### `@pi/todo-tools`

Purpose:
Manage lightweight task lists that branch correctly with the session.

Initial tools:

- `todo`

Likely commands:

- `/todos`

Implementation notes:

- Pi already has a good seed example in [`examples/extensions/todo.ts`](../examples/extensions/todo.ts)
- The package version should promote that example into a reusable package with a stable schema
- This package should remain separate from the plan-first workflow layer in core; users can choose one or both

State model:

- branch-aware session reconstruction from tool result details

Good fit with Pi today:

- high

### `@pi/clarify-tools`

Purpose:
Give the model a first-class structured clarification primitive when user intent or requirements are ambiguous.

Initial tools:

- `clarify`

Likely commands:

- none in v1

Implementation notes:

- This package is small, but architecturally important
- The tool should produce a narrow, structured clarification request rather than a generic assistant question
- It may later integrate with workflow intake and execution contracts, but should start as an independent package

State model:

- stateless

Good fit with Pi today:

- high

### `@pi/memory-tools`

Purpose:
Provide persistent, bounded, curated memory across sessions for user preferences and agent/project notes.

Initial tools:

- `memory`

Likely commands:

- none in v1

Implementation notes:

- Use two explicit stores:
  - `user` for user profile/preferences and workflow habits
  - `memory` for environment/project conventions and learned operational notes
- Keep persistence file-backed and bounded by character limits
- Support explicit mutation actions (`read`, `add`, `replace`, `remove`) with substring matching for replace/remove
- Keep system-prompt memory injection as a frozen session-start snapshot so turn-level prefix caching stays stable
- Scan newly written content for obvious prompt-injection/exfiltration patterns before persisting

State model:

- file-backed persistent state outside session branching

Good fit with Pi today:

- high

### `@pi/cron-tools`

Purpose:
Track recurring tasks with durable interval schedules that survive across sessions.

Initial tools:

- `cron`

Likely commands:

- none in v1

Implementation notes:

- Keep schedule semantics explicit and simple in v1 (interval minutes)
- Persist jobs to a durable file under Pi agent home by default
- Support basic lifecycle actions (`list`, `add`, `remove`, `pause`, `resume`, `touch`)
- Keep execution orchestration out of v1; this package is a schedule registry and due-state helper

State model:

- file-backed persistent state outside session branching

Good fit with Pi today:

- high

### `@pi/delegate-tools`

Purpose:
Delegate tightly scoped tasks to an isolated nested Pi subprocess and return structured outputs.

Initial tools:

- `delegate`

Likely commands:

- none in v1

Implementation notes:

- Use `pi --mode json` subprocess execution for isolation
- Keep scope bounded with timeout and optional model/tool filters
- Return compact details (duration, usage, exit status) alongside delegated text output
- Avoid coupling to internal runtime-only APIs; rely on CLI contract

State model:

- stateless

Good fit with Pi today:

- medium-high

### `@pi/skills-tools`

Purpose:
Provide filesystem-backed skill discovery and authoring for user/project skill roots.

Initial tools:

- `skills`

Likely commands:

- none in v1

Implementation notes:

- Support list/read/write/remove with guardrails
- Keep path handling strict to avoid directory traversal
- Treat user and project skill roots as explicit scope targets
- Keep skill file format simple (`SKILL.md`) and leave advanced validation to future passes

State model:

- file-backed persistent state outside session branching

Good fit with Pi today:

- high

### `@pi/autoresearch-tools`

Purpose:
Provide a Pi-native control surface for Karpathy-style autoresearch experiment loops.

Initial tools:

- `autoresearch`

Likely commands:

- none in v1

Implementation notes:

- Keep the loop orchestration package-level and optional, not core runtime behavior
- Support setup/status/run/record actions over a workspace-local `results.tsv` and `run.log`
- Parse standard training summary metrics (`val_bpb`, `peak_vram_mb`, `training_seconds`, `total_seconds`)
- Keep run execution bounded with timeout and explicit command input
- Do not hardcode model-specific training internals in Pi core

State model:

- file-backed state in the target repo (`results.tsv`, `run.log`)

Good fit with Pi today:

- high

## Current Pi Mapping

Some packages already have strong seeds in the current codebase or examples:

- `@pi/todo-tools`
  - seeded by [`examples/extensions/todo.ts`](../examples/extensions/todo.ts)
- `@pi/clarify-tools`
  - aligns with the existing extension command and tool model
- `@pi/session-search-tools`
  - aligns with current session APIs and workflow/session inspection work
- `@pi/memory-tools`
  - aligns with extension hooks for frozen `before_agent_start` memory snapshots plus durable file-backed state
- `@pi/cron-tools`
  - aligns with Pi's extension model for tool-driven schedule management with durable storage
- `@pi/delegate-tools`
  - aligns with subprocess-friendly CLI execution in extension tools
- `@pi/skills-tools`
  - aligns with Pi's existing skill folder conventions and extension filesystem access

## Recommended Build Order

Build order should follow leverage and risk, not just surface area:

1. `@pi/web-tools`
2. `@pi/memory-tools`
3. `@pi/todo-tools`
4. `@pi/session-search-tools`
5. `@pi/clarify-tools`
6. `@pi/cron-tools`
7. `@pi/delegate-tools`
8. `@pi/skills-tools`
9. `@pi/autoresearch-tools`
10. `@pi/image-tools`

Rationale:

- `web` unlocks the broadest set of user tasks with the least risk
- `memory`, `todo`, `session-search`, and `clarify` improve Pi's control surface
- `cron` and `skills` improve durable operational memory and reuse
- `delegate` is high leverage once bounded with timeout and filters
- `image` is useful, but less central to Pi's core workflows

## Deferred Packages

These package families are still interesting, but should not ship until Pi has a stronger native implementation behind them:

- `@pi/browser-tools`
- `@pi/vision-tools`

## Future Layer: Tool Profiles

These packages should later be grouped by a Pi-native profile system instead of Hermes-style percentages.

Recommended v1 profile semantics:

- `enabled`
- `preferred`
- `discouraged`
- `disabled`

Example profile families:

- `research`
- `browser`
- `development`
- `safe`
- `creative`

Profiles should compose installed packages instead of changing core tool definitions.

## Implementation Strategy

The safest rollout strategy is:

1. Create each package as a standalone Pi package with one extension entry point
2. Keep package dependencies minimal
3. Ship stateless packages first
4. Add stateful packages only when their persistence model is clear
5. Add profiles after at least three packages exist and have stable tool names

## What Not To Do

- Do not port Hermes Python runtime code directly
- Do not make browser, web, and vision one package
- Do not put these tools into Pi core by default
- Do not block future profile work by baking policy into package code
- Do not add weighted random tool distribution before simple package composition exists

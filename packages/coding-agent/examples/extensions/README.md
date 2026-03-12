# pi Extensions

<p align="center">
  <a href="../../docs/extensions.md"><img src="https://img.shields.io/badge/Docs-Extensions-blue?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/nicoring/codi"><img src="https://img.shields.io/badge/GitHub-Codi-blueviolet?style=for-the-badge&logo=github" alt="GitHub"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

**A comprehensive collection of extensions for [pi](https://github.com/nicoring/codi) — the coding agent that adapts to your workflow.**

Extensions let you customize every aspect of the agent: custom tools, UI components, lifecycle hooks, compaction strategies, and more. The SDK provides a clean API with full TypeScript support, reactive state, and composable primitives.

| Category | Count | Highlights |
|----------|-------|------------|
| **Episodic Memory** | 2 | Thread weaving, episode-bounded compaction (Slate-inspired) |
| **Custom Tools** | 12 | Todo lists, image gen, SSH delegation, subagents |
| **Lifecycle & Safety** | 5 | Permission gates, protected paths, sandboxing |
| **Commands & UI** | 28 | Plan mode, presets, games, overlays, custom renderers |
| **Git Integration** | 2 | Checkpoints, auto-commit |
| **System & Compaction** | 4 | Custom compaction, dynamic prompts |
| **Custom Providers** | 3 | Anthropic OAuth, GitLab Duo, Qwen CLI |



## Usage

```bash
# Load an extension with --extension flag
pi --extension examples/extensions/permission-gate.ts

# Or copy to extensions directory for auto-discovery
cp permission-gate.ts ~/.pi/agent/extensions/
```

## Examples

### Lifecycle & Safety

| Extension | Description |
|-----------|-------------|
| `permission-gate.ts` | Prompts for confirmation before dangerous bash commands (rm -rf, sudo, etc.) |
| `protected-paths.ts` | Blocks writes to protected paths (.env, .git/, node_modules/) |
| `confirm-destructive.ts` | Confirms before destructive session actions (clear, switch, fork) |
| `dirty-repo-guard.ts` | Prevents session changes with uncommitted git changes |
| `sandbox/` | OS-level sandboxing using `@anthropic-ai/sandbox-runtime` with per-project config |

### Custom Tools

| Extension | Description |
|-----------|-------------|
| `todo.ts` | Todo list tool + `/todos` command with custom rendering and state persistence |
| `hello.ts` | Minimal custom tool example |
| `question.ts` | Demonstrates `ctx.ui.select()` for asking the user questions with custom UI |
| `questionnaire.ts` | Multi-question input with tab bar navigation between questions |
| `tool-override.ts` | Override built-in tools (e.g., add logging/access control to `read`) |
| `dynamic-tools.ts` | Register tools after startup (`session_start`) and at runtime via command, with prompt snippets and tool-specific prompt guidelines |
| `built-in-tool-renderer.ts` | Custom compact rendering for built-in tools (read, bash, edit, write) while keeping original behavior |
| `minimal-mode.ts` | Override built-in tool rendering for minimal display (only tool calls, no output in collapsed mode) |
| `truncated-tool.ts` | Wraps ripgrep with proper output truncation (50KB/2000 lines) |
| `antigravity-image-gen.ts` | Generate images via Google Antigravity with optional save-to-disk modes |
| `ssh.ts` | Delegate all tools to a remote machine via SSH using pluggable operations |
| `subagent/` | Delegate tasks to specialized subagents with isolated context windows |

### Commands & UI

| Extension | Description |
|-----------|-------------|
| `preset.ts` | Named presets for model, thinking level, tools, and instructions via `--preset` flag and `/preset` command |
| `plan-mode/` | Claude Code-style plan mode for read-only exploration with `/plan` command and step tracking |
| `tools.ts` | Interactive `/tools` command to enable/disable tools with session persistence |
| `handoff.ts` | Transfer context to a new focused session via `/handoff <goal>` |
| `qna.ts` | Extracts questions from last response into editor via `ctx.ui.setEditorText()` |
| `status-line.ts` | Shows turn progress in footer via `ctx.ui.setStatus()` with themed colors |
| `widget-placement.ts` | Shows widgets above and below the editor via `ctx.ui.setWidget()` placement |
| `model-status.ts` | Shows model changes in status bar via `model_select` hook |
| `snake.ts` | Snake game with custom UI, keyboard handling, and session persistence |
| `send-user-message.ts` | Demonstrates `pi.sendUserMessage()` for sending user messages from extensions |
| `timed-confirm.ts` | Demonstrates AbortSignal for auto-dismissing `ctx.ui.confirm()` and `ctx.ui.select()` dialogs |
| `rpc-demo.ts` | Exercises all RPC-supported extension UI methods; pair with [`examples/rpc-extension-ui.ts`](../rpc-extension-ui.ts) |
| `modal-editor.ts` | Custom vim-like modal editor via `ctx.ui.setEditorComponent()` |
| `rainbow-editor.ts` | Animated rainbow text effect via custom editor |
| `notify.ts` | Desktop notifications via OSC 777 when agent finishes (Ghostty, iTerm2, WezTerm) |
| `titlebar-spinner.ts` | Braille spinner animation in terminal title while the agent is working |
| `summarize.ts` | Summarize conversation with GPT-5.2 and show in transient UI |
| `custom-footer.ts` | Custom footer with git branch and token stats via `ctx.ui.setFooter()` |
| `custom-header.ts` | Custom header via `ctx.ui.setHeader()` |
| `overlay-test.ts` | Test overlay compositing with inline text inputs and edge cases |
| `overlay-qa-tests.ts` | Comprehensive overlay QA tests: anchors, margins, stacking, overflow, animation |
| `doom-overlay/` | DOOM game running as an overlay at 35 FPS (demonstrates real-time game rendering) |
| `shutdown-command.ts` | Adds `/quit` command demonstrating `ctx.shutdown()` |
| `reload-runtime.ts` | Adds `/reload-runtime` and `reload_runtime` tool showing safe reload flow |
| `interactive-shell.ts` | Run interactive commands (vim, htop) with full terminal via `user_bash` hook |
| `inline-bash.ts` | Expands `!{command}` patterns in prompts via `input` event transformation |

### Git Integration

| Extension | Description |
|-----------|-------------|
| `git-checkpoint.ts` | Creates git stash checkpoints at each turn for code restoration on fork |
| `auto-commit-on-exit.ts` | Auto-commits on exit using last assistant message for commit message |

### Episodic Memory & Thread Weaving

Inspired by [Slate's thread-weaving architecture](https://randomlabs.ai/blog/slate), these extensions implement episodic memory — compressing completed work into structured episodes and composing them across delegate boundaries.

| Extension | Description |
|-----------|-------------|
| `episodic-delegate.ts` | Replaces the built-in delegate tool with episode-aware delegation. Returns structured episodes containing summary, discoveries, decisions, and file operations. Supports thread weaving via `priorEpisodes` parameter — pass episode IDs to seed context from prior delegates without full context transfer. Includes `/episodes` command to list stored episodes. |
| `episode-bounded-compaction.ts` | Replaces token-threshold compaction with semantic, episode-aware compaction at natural completion boundaries. Detects 6 boundary types: delegate completions, git commits, todo completions, write-verify sequences, user turns, and manual checkpoints. Maintains a persistent episode log that survives compaction cycles. Adds `/checkpoint [label]` and `/episode-log` commands. |

### System Prompt & Compaction

| Extension | Description |
|-----------|-------------|
| `pirate.ts` | Demonstrates `systemPromptAppend` to dynamically modify system prompt |
| `claude-rules.ts` | Scans `.claude/rules/` folder and lists rules in system prompt |
| `custom-compaction.ts` | Custom compaction that summarizes entire conversation |
| `trigger-compact.ts` | Triggers compaction when context usage exceeds 100k tokens and adds `/trigger-compact` command |

### System Integration

| Extension | Description |
|-----------|-------------|
| `mac-system-theme.ts` | Syncs pi theme with macOS dark/light mode |

### Resources

| Extension | Description |
|-----------|-------------|
| `dynamic-resources/` | Loads skills, prompts, and themes using `resources_discover` |

### Messages & Communication

| Extension | Description |
|-----------|-------------|
| `message-renderer.ts` | Custom message rendering with colors and expandable details via `registerMessageRenderer` |
| `event-bus.ts` | Inter-extension communication via `pi.events` |

### Session Metadata

| Extension | Description |
|-----------|-------------|
| `session-name.ts` | Name sessions for the session selector via `setSessionName` |
| `bookmark.ts` | Bookmark entries with labels for `/tree` navigation via `setLabel` |

### Custom Providers

| Extension | Description |
|-----------|-------------|
| `custom-provider-anthropic/` | Custom Anthropic provider with OAuth support and custom streaming implementation |
| `custom-provider-gitlab-duo/` | GitLab Duo provider using pi-ai's built-in Anthropic/OpenAI streaming via proxy |
| `custom-provider-qwen-cli/` | Qwen CLI provider with OAuth device flow and OpenAI-compatible models |

### External Dependencies

| Extension | Description |
|-----------|-------------|
| `with-deps/` | Extension with its own package.json and dependencies (demonstrates jiti module resolution) |
| `file-trigger.ts` | Watches a trigger file and injects contents into conversation |

## Writing Extensions

See [docs/extensions.md](../../docs/extensions.md) for full documentation.

Recommended starting points:
- `hello.ts` for the smallest SDK-based custom tool example
- `question.ts` for an interactive SDK-based tool with custom UI and result rendering

```typescript
import { Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

export default defineExtension((pi) => {
  // Subscribe to lifecycle events
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName === "bash" && event.input.command?.includes("rm -rf")) {
      const ok = await ctx.ui.confirm("Dangerous!", "Allow rm -rf?");
      if (!ok) return { block: true, reason: "Blocked by user" };
    }
  });

  // Register custom tools
  pi.registerTool(
    defineTool({
      name: "greet",
      label: "Greeting",
      description: "Generate a greeting",
      parameters: Type.Object({
        name: Type.String({ description: "Name to greet" }),
      }),
      async execute(_toolCallId, params) {
        return {
          content: [{ type: "text", text: `Hello, ${params.name}!` }],
          details: {},
        };
      },
    }),
  );

  // Register commands
  pi.registerCommand("hello", {
    description: "Say hello",
    handler: async (args, ctx) => {
      ctx.ui.notify("Hello!", "info");
    },
  });
});
```

## Key Patterns

**Use StringEnum for string parameters** (required for Google API compatibility):
```typescript
import { StringEnum } from "@mariozechner/pi-ai";

// Good
action: StringEnum(["list", "add"] as const)

// Bad - doesn't work with Google
action: Type.Union([Type.Literal("list"), Type.Literal("add")])
```

**State persistence via details:**
```typescript
// Store state in tool result details for proper forking support
return {
  content: [{ type: "text", text: "Done" }],
  details: { todos: [...todos], nextId },  // Persisted in session
};

// Reconstruct on session events
pi.on("session_start", async (_event, ctx) => {
  for (const entry of ctx.sessionManager.getBranch()) {
    if (entry.type === "message" && entry.message.toolName === "my_tool") {
      const details = entry.message.details;
      // Reconstruct state from details
    }
  }
});
```

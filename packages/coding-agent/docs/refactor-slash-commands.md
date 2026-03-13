# Refactor Plan: Extract Slash Command Handlers from InteractiveMode

## Problem

`interactive-mode.ts` is a **6,133-line God Class** with:
- 100+ private fields
- 164 methods
- 101 slash command handler methods (~2,500 lines)

All command handlers are private methods on `InteractiveMode`, making the file hard to navigate, test, and maintain.

## Current Architecture (partial)

```
interactive-mode.ts (6,133 lines)
  ├── InteractiveMode class
  │   ├── 50+ private state fields
  │   ├── 101 command handler methods  ← EXTRACT THESE
  │   ├── ~30 UI helper methods
  │   ├── ~15 event handlers
  │   └── lifecycle methods (init, run, shutdown)
  │
  └── Already extracted:
      ├── submit-dispatch.ts        ✅ Command routing
      ├── agent-event-handler.ts    ✅ Agent event → UI
      └── components/*.ts           ✅ UI components
```

## Target Architecture

```
interactive-mode.ts (~3,000 lines)
  ├── InteractiveMode class
  │   ├── State fields
  │   ├── UI helper methods (showStatus, renderRuntimePanel, etc.)
  │   ├── Event handlers
  │   ├── Lifecycle methods
  │   └── Thin delegating handler methods that call into:
  │
  └── command-handlers/
      ├── types.ts                    # CommandHandlerContext interface
      ├── model-commands.ts           # /model, /scoped-models, /models roles
      ├── session-commands.ts         # /session, /resume, /name, /fork, /tree, /new
      ├── runtime-commands.ts         # /events, /queue, /lanes, /mailbox, /heartbeat, /ops, /delegated
      ├── workflow-commands.ts        # /plan, /phase, /task, /verify, /workflow
      ├── export-commands.ts          # /export, /share, /copy
      ├── package-commands.ts         # /packages
      ├── misc-commands.ts            # /resources, /changelog, /hotkeys, /debug, /reload, /compact, /clear
      └── oauth-commands.ts           # /login, /logout
```

## Step-by-Step Plan

### Step 1: Create `CommandHandlerContext` interface

Create `command-handlers/types.ts` that defines the context each handler needs:

```typescript
import type { AgentSession } from "../../core/agent-session.js";
import type { SettingsManager } from "../../core/settings-manager.js";
import type { RuntimeServices } from "../../core/runtime/index.js";
import type { TUI } from "@mariozechner/pi-tui";

export interface CommandHandlerContext {
  // Core references
  session: AgentSession;
  settingsManager: SettingsManager;
  runtimeServices: RuntimeServices | undefined;
  ui: TUI;
  
  // UI helpers
  showStatus(message: string): void;
  showWarning(message: string): void;
  renderRuntimePanel(title: string, lines: string[]): void;
  getRuntimeOrWarn(flag?: string): RuntimeServices | undefined;
  
  // Formatting helpers
  formatRuntimeEventLine(event: RuntimeEvent): string;
  formatQueueLine(message: QueueMessage): string;
  formatMailboxLine(message: MailboxMessage): string;
  formatDelegatedTaskLine(task: DelegatedTaskRecord): string;
  
  // Display helpers
  updateEditorBorderColor(): void;
  
  // Session operations  
  ensureRoleModel(role: ModelRoleName): Promise<void>;
  showModelSelector(initialSearchInput?: string): void;
}
```

### Step 2: Create `model-commands.ts` (est. 200 lines)

Extract:
- `handleModelCommand` (line 3517)
- `showModelSelector` (line 3587) — helper, used by model command
- `getModelCandidates` (line 3567) — helper
- `findExactModelMatch` (line 3540) — helper
- `updateAvailableProviderCount` (line 3581) — helper
- `showModelsSelector` (line 3618)
- `handleModelsCommand` (line 5174)
- `handleModelRolesCommand` (line 5186)
- `cycleModel` (line 3019)
- `cycleThinkingLevel` (line 3008)

### Step 3: Create `session-commands.ts` (est. 350 lines)

Extract:
- `handleSessionCommand` (line 4339)
- `showSessionSelector` (line 3896)
- `handleResumeSession` (line 3931)
- `handleNameCommand` (line 4318)
- `handleClearCommand` (line 5880)
- `showUserMessageSelector` (line 3738)
- `showTreeSelector` (line 3773)

### Step 4: Create `runtime-commands.ts` (est. 600 lines)

Extract:
- `handleEventsCommand` (line 4537)
- `handleQueueCommand` (line 4618)
- `handleLanesCommand` (line 4665)
- `handleMailboxCommand` (line 4861)
- `handleDelegatedCommand` (line 4970)
- `handleHeartbeatCommand` (line 5078)
- `handleOpsCommand` (line 5257)
- `renderRuntimePanel` — thin wrapper stays in InteractiveMode, passed via context
- `startEventTail` / `stopEventTail` (lines 4511, 4501)

### Step 5: Create `workflow-commands.ts` (est. 400 lines)

Extract:
- `handleWorkflowPlanCommand` (line 5359)
- `handleWorkflowPhaseCommand` (line 5449)
- `handleWorkflowTaskCommand` (line 5483)
- `handleWorkflowVerifyCommand` (line 5643)
- `handleWorkflowSummaryCommand` (line 5699)
- `buildTaskExecutionContractText` (line 2552) — helper
- `getWorkflowDisplayState` (line 2575) — helper
- `formatWorkflowLabel` (line 2535) — helper
- `formatTaskCompletionLabel` (line 2539) — helper

### Step 6: Create `export-commands.ts` (est. 150 lines)

Extract:
- `handleExportCommand` (line 4197)
- `handleShareCommand` (line 4209)
- `handleCopyCommand` (line 4303)

### Step 7: Create `package-commands.ts` (est. 200 lines)

Extract:
- `handlePackagesCommand` (line 4710)
- `showPackageManageSelector` (line 4818)

### Step 8: Create `misc-commands.ts` (est. 300 lines)

Extract:
- `handleResourcesCommand` (line 4165)
- `handleChangelogCommand` (line 5719)
- `handleHotkeysCommand` (line 5769)
- `handleDebugCommand` (line 5905)
- `handleReloadCommand` (line 4093)
- `handleCompactCommand` (line 6045)
- `executeCompaction` (line 6059)

### Step 9: Create `oauth-commands.ts` (est. 100 lines)

Extract:
- `showOAuthSelector` (line 3955)
- `showLoginDialog` (line 4000)

### Step 10: Wire it up in InteractiveMode

Replace extracted methods with thin delegators:

```typescript
// Before (private method with 50 lines of logic)
private async handleEventsCommand(text: string): Promise<void> { ... }

// After (one-line delegation)
private async handleEventsCommand(text: string): Promise<void> {
  return handleEventsCommand(text, this.getCommandContext());
}
```

Or better, update `submit-dispatch.ts` to call the extracted functions directly, removing the need for the delegating methods entirely.

## What Stays in InteractiveMode

These should NOT be extracted — they're core to the mode's lifecycle:

- State fields (session, ui, containers, editor, etc.)
- Constructor and `init()`
- `run()` / `shutdown()` lifecycle
- `showStatus()` / `showWarning()` — fundamental UI helpers
- Event subscription (`subscribeToSession`)
- Layout building (`buildLayout`)
- Keyboard handling
- Agent event → component rendering (already in `agent-event-handler.ts`)

## Verification

1. `npm run build` — TypeScript compiles
2. `npm test` — all existing tests pass
3. Manual: run `npm start` and test each extracted command:
   - `/model`, `/models roles show`
   - `/session`, `/resume`, `/name test`
   - `/events`, `/queue`, `/lanes`, `/mailbox`
   - `/plan`, `/phase`, `/task`, `/verify`, `/workflow`
   - `/export html`, `/share`, `/copy`
   - `/packages list`
   - `/resources`, `/changelog`, `/hotkeys`, `/debug`
   - `/compact`, `/reload`
   - `/login`, `/logout`

## Impact

| Metric | Before | After |
|--------|--------|-------|
| interactive-mode.ts | 6,133 lines | ~3,000 lines |
| Largest file | 6,133 lines | ~600 lines (runtime-commands.ts) |
| Files with command logic | 1 | 9 |
| Testability | Hard (private methods) | Easy (pure functions with context) |

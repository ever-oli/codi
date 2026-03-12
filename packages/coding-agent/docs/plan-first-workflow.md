# Plan-First Workflow Scaffold

This package now includes an internal workflow scaffold under `src/core/workflow/`.

The goal is to shape a more structured coding workflow inside Pi without changing current runtime behavior yet.

The scaffold defines:

- `SessionOrchestrator` with explicit phases: `intake -> plan -> execute -> verify -> summarize`
- `TaskGraph` and `TaskNode` types for plan-first execution
- `WorkspaceState` as canonical repo/task evidence
- `VerificationRecord` and `RunEvidence` for evidence-based completion
- small typed memory and artifact models
- bounded subagent result contracts

Current status:

- exported SDK-facing types are available for experiments and extension work
- no existing CLI, TUI, session, or tool behavior is changed yet

Recommended next integration steps:

1. build a plan snapshot entry format on top of `SessionManager`
2. project the active task and verification state into `AgentSession`
3. surface the current workflow phase in interactive mode
4. add a thin extension or built-in command layer for plan approval and step execution

import { Container } from "@mariozechner/pi-tui";
import { beforeAll, describe, expect, test, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { createVerificationRecord } from "../src/core/workflow/evidence.js";
import { SessionOrchestrator } from "../src/core/workflow/session-orchestrator.js";
import { createTaskGraph, createTaskNode } from "../src/core/workflow/task-graph.js";
import { createWorkspaceState } from "../src/core/workflow/workspace-state.js";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.js";
import { initTheme } from "../src/modes/interactive/theme/theme.js";

beforeAll(() => {
	initTheme("dark");
});

function renderAll(container: Container, width = 160): string {
	return container.children.flatMap((child) => child.render(width)).join("\n");
}

function createWorkflowForUi() {
	return {
		goal: "Finish workflow gating",
		currentPhase: "verify" as const,
		status: "ready" as const,
		taskGraph: createTaskGraph([
			createTaskNode({
				id: "task-1",
				goal: "Implement completion gating",
				status: "done",
				acceptanceCriteria: ["Task is complete"],
				notes: ["Need explicit verification"],
			}),
		]),
		workspace: createWorkspaceState({
			cwd: "/tmp/workflow-ui",
			git: {
				branch: "main",
				head: "abc123",
			},
			changedFiles: ["src/a.ts", "src/b.ts"],
			lastCommandResults: [
				{
					command: "npm run check",
					exitCode: 0,
					startedAt: "2026-03-03T00:00:00.000Z",
					finishedAt: "2026-03-03T00:00:01.000Z",
					stdout: "ok",
				},
			],
			testResults: [
				{
					command: "npm run check",
					passed: true,
					recordedAt: "2026-03-03T00:00:01.000Z",
				},
			],
			refreshedAt: "2026-03-03T00:00:02.000Z",
		}),
		artifacts: [],
		verification: [
			createVerificationRecord({
				taskId: "task-1",
				status: "passed",
				evidence: {
					tests: [{ command: "npm run check", passed: true }],
					commands: [{ command: "npm run check", validated: true, details: "Checks passed" }],
					diffSummary: "Checks passed",
				},
				recordedAt: "2026-03-03T00:00:03.000Z",
			}),
		],
		transitions: [],
		memory: [],
	};
}

describe("InteractiveMode workflow commands", () => {
	test.each(["execute", "verify"] as const)("/plan start re-enters planning from %s", (phase) => {
		let workflow: any = {
			...createWorkflowForUi(),
			currentPhase: phase,
		};
		const fakeThis: any = {
			session: {
				get workflow() {
					return workflow;
				},
				transitionWorkflow: vi.fn((nextPhase: string) => {
					workflow = { ...workflow, currentPhase: nextPhase };
				}),
			},
			showStatus: vi.fn(),
			showWarning: vi.fn(),
			showError: vi.fn(),
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
		};

		(InteractiveMode as any).prototype.handleWorkflowPlanCommand.call(fakeThis, "/plan start");

		expect(fakeThis.session.transitionWorkflow).toHaveBeenCalledWith("plan", "Manual planning start from /plan");
		expect(fakeThis.showStatus).toHaveBeenCalledWith("Workflow phase: Plan");
	});

	test("/plan split creates a multi-task graph and records a plan artifact", () => {
		let workflow: any = {
			...createWorkflowForUi(),
			currentPhase: "plan" as const,
			taskGraph: createTaskGraph([
				createTaskNode({
					id: "task-1",
					goal: "Inspect current workflow and then implement completion gating; also verify the results.",
					status: "ready",
				}),
			]),
		};
		const artifacts: Array<{ type: string }> = [];
		const fakeThis: any = {
			session: {
				get workflow() {
					return workflow;
				},
				replaceWorkflowTaskGraph: vi.fn((taskGraph) => {
					workflow = { ...workflow, taskGraph };
				}),
				recordWorkflowArtifact: vi.fn((artifact) => {
					artifacts.push(artifact);
				}),
			},
			showStatus: vi.fn(),
			showWarning: vi.fn(),
			showError: vi.fn(),
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
		};

		(InteractiveMode as any).prototype.handleWorkflowPlanCommand.call(fakeThis, "/plan split");

		expect(workflow.taskGraph.taskOrder.length).toBeGreaterThan(1);
		expect(workflow.taskGraph.activeTaskId).toBeDefined();
		expect(artifacts).toEqual(expect.arrayContaining([expect.objectContaining({ type: "plan" })]));
	});

	test("/task show includes verification and completion state", () => {
		const workflow = createWorkflowForUi();
		const fakeThis: any = {
			session: { workflow },
			showStatus: vi.fn(),
			showWarning: vi.fn(),
			showError: vi.fn(),
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
		};

		(InteractiveMode as any).prototype.handleWorkflowTaskCommand.call(fakeThis, "/task show task-1");

		const message = fakeThis.showStatus.mock.calls[0][0] as string;
		expect(message).toContain("Verification: Passed");
		expect(message).toContain("Completion: Completion Ready");
		expect(message).toContain("Acceptance criteria:");
		expect(message).toContain("Notes:");
	});

	test("/verify passed updates completion readiness for a done task", () => {
		let workflow: any = {
			...createWorkflowForUi(),
			verification: [],
		};
		const fakeThis: any = {
			session: {
				get workflow() {
					return workflow;
				},
				recordWorkflowVerification: vi.fn((taskId: string, status: "passed" | "failed" | "waived", evidence) => {
					workflow = {
						...workflow,
						verification: [
							...workflow.verification,
							createVerificationRecord({
								taskId,
								status,
								evidence,
								recordedAt: "2026-03-03T00:00:04.000Z",
							}),
						],
					};
				}),
				recordWorkflowArtifact: vi.fn(),
			},
			showStatus: vi.fn(),
			showWarning: vi.fn(),
			showError: vi.fn(),
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
		};

		(InteractiveMode as any).prototype.handleWorkflowVerifyCommand.call(fakeThis, "/verify passed npm run check");

		const message = fakeThis.showStatus.mock.calls[0][0] as string;
		expect(message).toContain("Workflow verification for task-1: Passed");
		expect(message).toContain("Completion: Completion Ready");
	});

	test("/session shows richer workflow inspection", () => {
		const workflow = createWorkflowForUi();
		const fakeThis: any = {
			session: {
				workflow,
				getSessionStats: () => ({
					sessionFile: "/tmp/session.jsonl",
					sessionId: "session-1",
					userMessages: 1,
					assistantMessages: 1,
					toolCalls: 0,
					toolResults: 0,
					totalMessages: 2,
					tokens: {
						input: 10,
						output: 20,
						cacheRead: 0,
						cacheWrite: 0,
						total: 30,
					},
					cost: 0,
				}),
			},
			sessionManager: {
				getSessionName: () => "Workflow Session",
			},
			chatContainer: new Container(),
			ui: { requestRender: vi.fn() },
			getWorkflowDisplayState: (InteractiveMode as any).prototype.getWorkflowDisplayState,
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
		};

		(InteractiveMode as any).prototype.handleSessionCommand.call(fakeThis);

		const output = renderAll(fakeThis.chatContainer);
		expect(output).toContain("Completion State:");
		expect(output).toContain("Completion Ready:");
		expect(output).toContain("Git Head:");
		expect(output).toContain("Latest Test:");
	});

	test("/workflow shows phase, task, verification, and completion summary", () => {
		const workflow = createWorkflowForUi();
		const fakeThis: any = {
			session: { workflow },
			showStatus: vi.fn(),
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
		};

		(InteractiveMode as any).prototype.handleWorkflowSummaryCommand.call(fakeThis);

		const message = fakeThis.showStatus.mock.calls[0][0] as string;
		expect(message).toContain("Phase: Verify");
		expect(message).toContain("Status: Ready");
		expect(message).toContain("Goal: Finish workflow gating");
		expect(message).toContain("Active task: task-1");
		expect(message).toContain("Verification: Passed");
		expect(message).toContain("Completion ready: yes");
		expect(message).toContain("Tasks: 1");
	});

	test("workflow strip shows single line when no active task", () => {
		const workflow = {
			...createWorkflowForUi(),
			taskGraph: createTaskGraph([
				createTaskNode({
					id: "task-1",
					goal: "Implement completion gating",
					status: "ready",
				}),
			]),
		};
		const fakeThis: any = {
			session: { workflow },
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
			getWorkflowDisplayState: (InteractiveMode as any).prototype.getWorkflowDisplayState,
		};
		const lines = (InteractiveMode as any).prototype.buildWorkflowStripLines.call(fakeThis, 100, 30);
		expect(lines).toHaveLength(1);
		expect(lines[0]).toContain("Workflow");
	});

	test("workflow strip caps to two lines on small terminals with active task", () => {
		const workflow = createWorkflowForUi();
		const fakeThis: any = {
			session: { workflow },
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
			getWorkflowDisplayState: (InteractiveMode as any).prototype.getWorkflowDisplayState,
		};
		const lines = (InteractiveMode as any).prototype.buildWorkflowStripLines.call(fakeThis, 80, 24);
		expect(lines.length).toBeLessThanOrEqual(2);
		expect(lines[1]).toContain("Task:");
	});

	test("workflow strip allows third detail line only on wide/tall terminals", () => {
		const workflow = createWorkflowForUi();
		const fakeThis: any = {
			session: { workflow },
			formatWorkflowLabel: (InteractiveMode as any).prototype.formatWorkflowLabel,
			formatTaskCompletionLabel: (InteractiveMode as any).prototype.formatTaskCompletionLabel,
			getWorkflowDisplayState: (InteractiveMode as any).prototype.getWorkflowDisplayState,
		};
		const lines = (InteractiveMode as any).prototype.buildWorkflowStripLines.call(fakeThis, 120, 48);
		expect(lines.length).toBeGreaterThanOrEqual(3);
		expect(lines[2]).toContain("verify:");
	});

	test("renderInitialMessages does not emit duplicate workflow startup status", () => {
		const fakeThis: any = {
			sessionManager: {
				buildSessionContext: vi.fn(() => ({ messages: [], entries: [] })),
				getEntries: vi.fn(() => []),
			},
			renderSessionContext: vi.fn(),
			showStatus: vi.fn(),
		};

		(InteractiveMode as any).prototype.renderInitialMessages.call(fakeThis);

		expect(fakeThis.renderSessionContext).toHaveBeenCalled();
		expect(fakeThis.showStatus).not.toHaveBeenCalled();
	});
});

describe("AgentSession workflow evidence hooks", () => {
	test("auto-records verification evidence against the active task", () => {
		const fakeThis: any = {
			_getActiveWorkflowTaskId: () => "task-2",
			_isVerificationCommand: () => true,
			recordWorkflowVerification: vi.fn(),
		};

		(AgentSession as any).prototype._recordWorkflowVerificationForCommand.call(fakeThis, "npm run check", {
			output: "ok",
			exitCode: 0,
			cancelled: false,
			truncated: false,
		});

		expect(fakeThis.recordWorkflowVerification).toHaveBeenCalledWith(
			"task-2",
			"passed",
			expect.objectContaining({
				tests: [expect.objectContaining({ command: "npm run check", passed: true })],
			}),
		);
	});

	test("refreshes workspace state from command results", () => {
		const workflow = SessionOrchestrator.create({
			goal: "Finish workflow gating",
			cwd: "/tmp/workflow-agent",
		}).snapshot();
		const replaceWorkflowWorkspaceState = vi.fn();
		const fakeThis: any = {
			_workflowSession: {
				snapshot: () => workflow,
			},
			_cwd: "/tmp/workflow-agent",
			_readGitSnapshot: () => ({
				branch: "main",
				head: "abc123",
				statusSummary: " M src/file.ts",
				changedFiles: ["src/file.ts"],
			}),
			_isVerificationCommand: () => true,
			replaceWorkflowWorkspaceState,
		};

		(AgentSession as any).prototype._refreshWorkflowWorkspaceStateFromCommand.call(fakeThis, "npm run check", {
			output: "ok",
			exitCode: 0,
			cancelled: false,
			truncated: false,
		});

		expect(replaceWorkflowWorkspaceState).toHaveBeenCalledWith(
			expect.objectContaining({
				changedFiles: ["src/file.ts"],
				lastCommandResults: [expect.objectContaining({ command: "npm run check", exitCode: 0 })],
				testResults: [expect.objectContaining({ command: "npm run check", passed: true })],
			}),
		);
	});

	test("preflight returns summarize to verify when work is not completion-ready", () => {
		const transitionWorkflow = vi.fn();
		const recordWorkflowArtifact = vi.fn();
		const fakeThis: any = {
			_workflowSession: {
				snapshot: () => ({
					...createWorkflowForUi(),
					currentPhase: "summarize",
					verification: [],
				}),
			},
			transitionWorkflow,
			recordWorkflowArtifact,
		};

		(AgentSession as any).prototype._preflightWorkflowTurn.call(fakeThis);

		expect(transitionWorkflow).toHaveBeenCalledWith(
			"verify",
			"Auto-returned to verify because summarize lacked completion-ready work.",
		);
		expect(recordWorkflowArtifact).toHaveBeenCalledWith(
			expect.objectContaining({
				type: "decision",
				producer: "workflow:preflight",
			}),
		);
	});
});

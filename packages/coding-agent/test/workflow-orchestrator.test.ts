import { describe, expect, test } from "vitest";
import { createVerificationRecord } from "../src/core/workflow/evidence.js";
import {
	canTransition,
	getActiveTaskCompletionState,
	getTaskCompletionLabel,
	getTaskVerificationStatus,
	isTaskCompletionReady,
	SessionOrchestrator,
	type WorkflowSessionSnapshot,
} from "../src/core/workflow/session-orchestrator.js";
import {
	areTaskDependenciesSatisfied,
	createTaskGraph,
	createTaskGraphFromGoal,
	createTaskNode,
	getSchedulableTasks,
	setTaskStatus,
	splitGoalIntoTaskGoals,
} from "../src/core/workflow/task-graph.js";
import { createWorkspaceState } from "../src/core/workflow/workspace-state.js";

function createSnapshot(options?: {
	taskStatus?: "pending" | "ready" | "in_progress" | "blocked" | "done" | "waived";
	verificationStatus?: "pending" | "passed" | "failed" | "waived";
}): WorkflowSessionSnapshot {
	const task = createTaskNode({
		id: "task-1",
		goal: "Implement completion gating",
		status: options?.taskStatus ?? "ready",
	});
	return {
		goal: "Implement completion gating",
		currentPhase: "execute",
		status: "ready",
		taskGraph: createTaskGraph([task], "2026-03-03T00:00:00.000Z"),
		workspace: createWorkspaceState({
			cwd: "/tmp/workflow",
		}),
		artifacts: [],
		verification:
			options?.verificationStatus && options.verificationStatus !== "pending"
				? [
						createVerificationRecord({
							taskId: "task-1",
							status: options.verificationStatus,
							evidence: {
								tests: [],
								commands: [],
								diffSummary: options.verificationStatus,
							},
							recordedAt: "2026-03-03T00:00:00.000Z",
						}),
					]
				: [],
		transitions: [],
		memory: [],
	};
}

describe("workflow completion semantics", () => {
	test("derives verification readiness from task status and latest verification", () => {
		const pendingSnapshot = createSnapshot({ taskStatus: "in_progress" });
		expect(getTaskVerificationStatus(pendingSnapshot, "task-1")).toBe("pending");
		expect(getTaskCompletionLabel(pendingSnapshot, "task-1")).toBe("not_done");
		expect(isTaskCompletionReady(pendingSnapshot, "task-1")).toBe(false);

		const unverifiedDoneSnapshot = createSnapshot({ taskStatus: "done" });
		expect(getTaskCompletionLabel(unverifiedDoneSnapshot, "task-1")).toBe("needs_verification");
		expect(isTaskCompletionReady(unverifiedDoneSnapshot, "task-1")).toBe(false);

		const failedSnapshot = createSnapshot({ taskStatus: "done", verificationStatus: "failed" });
		expect(getTaskVerificationStatus(failedSnapshot, "task-1")).toBe("failed");
		expect(getTaskCompletionLabel(failedSnapshot, "task-1")).toBe("failed_verification");
		expect(isTaskCompletionReady(failedSnapshot, "task-1")).toBe(false);

		const passedSnapshot = createSnapshot({ taskStatus: "done", verificationStatus: "passed" });
		expect(getTaskCompletionLabel(passedSnapshot, "task-1")).toBe("completion_ready");
		expect(isTaskCompletionReady(passedSnapshot, "task-1")).toBe(true);

		const waivedSnapshot = createSnapshot({ taskStatus: "done", verificationStatus: "waived" });
		expect(getTaskCompletionLabel(waivedSnapshot, "task-1")).toBe("waived");
		expect(isTaskCompletionReady(waivedSnapshot, "task-1")).toBe(true);
	});

	test("only marks summarize complete when the active task is completion-ready", () => {
		const unverified = SessionOrchestrator.create({
			goal: "Finish workflow gating",
			cwd: "/tmp/workflow",
		});
		unverified.transition("plan", "plan");
		unverified.transition("execute", "execute");
		unverified.updateTaskStatus("task-1", "done");
		unverified.transition("verify", "verify");
		const unverifiedSummary = unverified.transition("summarize", "summarize");
		expect(unverifiedSummary.status).toBe("ready");

		const verified = SessionOrchestrator.create({
			goal: "Finish workflow gating",
			cwd: "/tmp/workflow",
		});
		verified.transition("plan", "plan");
		verified.transition("execute", "execute");
		verified.updateTaskStatus("task-1", "done");
		verified.recordVerification("task-1", "passed", {
			tests: [{ command: "npm run check", passed: true }],
			commands: [{ command: "npm run check", validated: true }],
		});
		verified.transition("verify", "verify");
		const verifiedSummary = verified.transition("summarize", "summarize");
		expect(verifiedSummary.status).toBe("complete");
		expect(getActiveTaskCompletionState(verifiedSummary).completionReady).toBe(true);
	});
});

describe("workflow transitions", () => {
	test("allows forward transitions and re-entry to plan from later phases", () => {
		expect(canTransition("intake", "plan")).toBe(true);
		expect(canTransition("plan", "execute")).toBe(true);
		expect(canTransition("execute", "verify")).toBe(true);
		expect(canTransition("verify", "summarize")).toBe(true);
		expect(canTransition("execute", "plan")).toBe(true);
		expect(canTransition("verify", "plan")).toBe(true);
		expect(canTransition("summarize", "plan")).toBe(true);
		expect(canTransition("plan", "verify")).toBe(false);
		expect(canTransition("verify", "execute")).toBe(false);
		expect(canTransition("plan", "plan")).toBe(false);
	});
});

describe("workflow task graph splitting", () => {
	test("falls back to a default three-step task graph for simple goals", () => {
		const fragments = splitGoalIntoTaskGoals("Finish the workflow control plane");
		expect(fragments).toHaveLength(3);
		expect(fragments[0]).toContain("Inspect and shape");
		expect(fragments[2]).toContain("Verify");
	});

	test("splits multi-clause goals into a small deterministic graph", () => {
		const graph = createTaskGraphFromGoal(
			"Inspect current workflow and then implement completion gating; also verify the results.",
		);
		expect(graph.taskOrder).toHaveLength(3);
		expect(graph.tasks[graph.taskOrder[0]!]!.goal).toContain("Inspect current workflow");
		expect(graph.tasks[graph.taskOrder[1]!]!.goal).toContain("Implement completion gating");
		expect(graph.tasks[graph.taskOrder[2]!]!.goal).toContain("Verify the results");
		expect(graph.tasks[graph.taskOrder[2]!]!.acceptanceCriteria.join(" ")).toContain("verification evidence");
		expect(graph.activeTaskId).toBe(graph.taskOrder[0]);
	});

	test("replaces the default scaffold and preserves matching task ids and statuses in existing graphs", () => {
		const scaffoldGraph = createTaskGraph([
			createTaskNode({
				id: "task-1",
				goal: "Inspect current workflow and then implement completion gating; also verify the results.",
				status: "ready",
			}),
		]);
		const replacedGraph = createTaskGraphFromGoal(
			"Inspect current workflow and then implement completion gating; also verify the results.",
			{ existingGraph: scaffoldGraph },
		);
		expect(replacedGraph.taskOrder).toHaveLength(3);

		const existingGraph = createTaskGraph([
			createTaskNode({
				id: "inspect",
				goal: "Inspect current workflow",
				status: "done",
				notes: ["Existing inspect task"],
			}),
			createTaskNode({
				id: "implement",
				goal: "Implement completion gating",
				status: "in_progress",
			}),
		]);
		const refinedGraph = createTaskGraphFromGoal(
			"Inspect current workflow and then implement completion gating; also verify the results.",
			{ existingGraph },
		);
		expect(refinedGraph.taskOrder).toEqual(["inspect", "implement", "task-3"]);
		expect(refinedGraph.tasks.inspect?.status).toBe("done");
		expect(refinedGraph.tasks.implement?.status).toBe("in_progress");
		expect(refinedGraph.activeTaskId).toBe("implement");
		expect(refinedGraph.tasks.inspect?.notes).toContain(
			"Preserved while refining the task graph from the workflow goal.",
		);
	});

	test("promotes the next schedulable task when a dependency completes", () => {
		const graph = createTaskGraph([
			createTaskNode({
				id: "task-1",
				goal: "Inspect workflow",
				status: "in_progress",
			}),
			createTaskNode({
				id: "task-2",
				goal: "Implement workflow gating",
				status: "pending",
				dependencies: ["task-1"],
			}),
		]);

		expect(areTaskDependenciesSatisfied(graph, "task-2")).toBe(false);
		expect(getSchedulableTasks(graph).map((task) => task.id)).toEqual(["task-1"]);

		const nextGraph = setTaskStatus(graph, "task-1", "done");
		expect(areTaskDependenciesSatisfied(nextGraph, "task-2")).toBe(true);
		expect(nextGraph.tasks["task-2"]?.status).toBe("ready");
		expect(nextGraph.activeTaskId).toBe("task-2");
	});
});

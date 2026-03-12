import type { WorkflowArtifact } from "./artifacts.js";
import type { WorkflowPhase } from "./session-orchestrator.js";
import type { TaskNode } from "./task-graph.js";

export type SubagentStatus = "pending" | "running" | "complete" | "failed";

export interface SubagentInputContract {
	id: string;
	goal: string;
	inputs: string[];
	constraints: string[];
}

export interface SubagentResult {
	status: SubagentStatus;
	summary: string;
	evidence: WorkflowArtifact[];
	unresolvedRisks: string[];
}

export function buildTaskSubagentContract(
	task: TaskNode,
	options?: {
		phase?: WorkflowPhase;
		relevantFiles?: string[];
		extraInputs?: string[];
	},
): SubagentInputContract {
	const relevantFiles = options?.relevantFiles ?? [];
	const extraInputs = options?.extraInputs ?? [];
	const phase = options?.phase;
	const constraints: string[] = [];
	if (phase === "plan") {
		constraints.push(
			"Stay in planning mode. Shape tasks, criteria, and verification strategy before implementation.",
		);
	}
	if (phase === "execute") {
		constraints.push("Stay scoped to the assigned task and avoid widening scope.");
	}
	if (phase === "verify") {
		constraints.push("Prefer tests, commands, diffs, and evidence collection over new implementation.");
	}
	if (phase === "summarize") {
		constraints.push("Summarize completed work and evidence instead of opening new implementation scope.");
	}

	return {
		id: task.id,
		goal: task.goal,
		inputs: [
			...task.acceptanceCriteria.map((criterion) => `acceptance:${criterion}`),
			...task.notes.map((note) => `note:${note}`),
			...relevantFiles.map((file) => `file:${file}`),
			...extraInputs,
		].slice(0, 12),
		constraints: [
			...constraints,
			"Preserve unrelated behavior.",
			"Collect verification evidence before claiming completion.",
		],
	};
}

import { createWorkflowArtifact, type WorkflowArtifact } from "./artifacts.js";
import {
	createVerificationRecord,
	type RunEvidence,
	type VerificationRecord,
	type VerificationStatus,
} from "./evidence.js";
import { InMemoryMemoryStore, type MemoryRecord, type MemoryStore } from "./memory-store.js";
import {
	type CreateTaskNodeInput,
	createTaskGraph,
	createTaskNode,
	setActiveTask,
	setTaskStatus,
	type TaskGraph,
	type TaskNode,
	type TaskStatus,
	type UpdateTaskNodeInput,
	updateTask,
	withTask,
} from "./task-graph.js";
import { type CreateWorkspaceStateInput, createWorkspaceState, type WorkspaceState } from "./workspace-state.js";

export const WORKFLOW_PHASES = ["intake", "plan", "execute", "verify", "summarize"] as const;

export type WorkflowPhase = (typeof WORKFLOW_PHASES)[number];
export type WorkflowSessionStatus = "ready" | "blocked" | "complete";
export type TaskCompletionLabel =
	| "not_done"
	| "needs_verification"
	| "failed_verification"
	| "completion_ready"
	| "waived";

export interface PhaseTransition {
	from: WorkflowPhase;
	to: WorkflowPhase;
	reason: string;
	at: string;
}

export interface WorkflowSessionSnapshot {
	goal: string;
	currentPhase: WorkflowPhase;
	status: WorkflowSessionStatus;
	taskGraph: TaskGraph;
	workspace: WorkspaceState;
	artifacts: WorkflowArtifact[];
	verification: VerificationRecord[];
	transitions: PhaseTransition[];
	memory: MemoryRecord[];
}

export interface CreateWorkflowSessionOptions extends CreateWorkspaceStateInput {
	goal: string;
	initialTask?: CreateTaskNodeInput;
	memoryStore?: MemoryStore;
}

export function canTransition(from: WorkflowPhase, to: WorkflowPhase): boolean {
	const fromIndex = WORKFLOW_PHASES.indexOf(from);
	const toIndex = WORKFLOW_PHASES.indexOf(to);
	if (fromIndex === -1 || toIndex === -1 || from === to) {
		return false;
	}
	if (toIndex === fromIndex + 1) {
		return true;
	}
	return to === "plan" && fromIndex > WORKFLOW_PHASES.indexOf("plan");
}

function createInitialTask(goal: string, task?: CreateTaskNodeInput): TaskNode {
	return createTaskNode(
		task ?? {
			id: "task-1",
			goal,
			status: "ready",
			acceptanceCriteria: [
				"Capture the user goal as a task node.",
				"Leave implementation details for the planning phase.",
			],
			notes: ["Initial task created during intake."],
		},
	);
}

export function getLatestTaskVerification(
	snapshot: WorkflowSessionSnapshot,
	taskId: string,
): VerificationRecord | undefined {
	for (let index = snapshot.verification.length - 1; index >= 0; index -= 1) {
		const record = snapshot.verification[index];
		if (record?.taskId === taskId) {
			return record;
		}
	}
	return undefined;
}

export function getTaskVerificationStatus(snapshot: WorkflowSessionSnapshot, taskId: string): VerificationStatus {
	return getLatestTaskVerification(snapshot, taskId)?.status ?? "pending";
}

export function isTaskCompletionReady(snapshot: WorkflowSessionSnapshot, taskId: string): boolean {
	const task = snapshot.taskGraph.tasks[taskId];
	if (!task || task.status !== "done") {
		return false;
	}
	const verificationStatus = getTaskVerificationStatus(snapshot, taskId);
	return verificationStatus === "passed" || verificationStatus === "waived";
}

export function getTaskCompletionLabel(snapshot: WorkflowSessionSnapshot, taskId: string): TaskCompletionLabel {
	const task = snapshot.taskGraph.tasks[taskId];
	if (!task || task.status !== "done") {
		return "not_done";
	}
	const verificationStatus = getTaskVerificationStatus(snapshot, taskId);
	if (verificationStatus === "waived") {
		return "waived";
	}
	if (verificationStatus === "passed") {
		return "completion_ready";
	}
	if (verificationStatus === "failed") {
		return "failed_verification";
	}
	return "needs_verification";
}

export function getActiveTaskCompletionState(snapshot: WorkflowSessionSnapshot): {
	task?: TaskNode;
	verification?: VerificationRecord;
	verificationStatus: VerificationStatus;
	completionReady: boolean;
	completionLabel: TaskCompletionLabel;
} {
	const activeTaskId = snapshot.taskGraph.activeTaskId;
	const task = activeTaskId ? snapshot.taskGraph.tasks[activeTaskId] : undefined;
	const verification = activeTaskId ? getLatestTaskVerification(snapshot, activeTaskId) : undefined;
	return {
		task,
		verification,
		verificationStatus: activeTaskId ? getTaskVerificationStatus(snapshot, activeTaskId) : "pending",
		completionReady: activeTaskId ? isTaskCompletionReady(snapshot, activeTaskId) : false,
		completionLabel: activeTaskId ? getTaskCompletionLabel(snapshot, activeTaskId) : "not_done",
	};
}

function deriveWorkflowStatus(snapshot: WorkflowSessionSnapshot): WorkflowSessionStatus {
	if (snapshot.status === "blocked") {
		return "blocked";
	}
	if (snapshot.currentPhase === "summarize") {
		return getActiveTaskCompletionState(snapshot).completionReady ? "complete" : "ready";
	}
	return "ready";
}

function withDerivedWorkflowStatus(snapshot: WorkflowSessionSnapshot): WorkflowSessionSnapshot {
	return {
		...snapshot,
		status: deriveWorkflowStatus(snapshot),
	};
}

export class SessionOrchestrator {
	#snapshot: WorkflowSessionSnapshot;
	#memoryStore: MemoryStore;

	private constructor(snapshot: WorkflowSessionSnapshot, memoryStore: MemoryStore) {
		this.#snapshot = snapshot;
		this.#memoryStore = memoryStore;
	}

	static create(options: CreateWorkflowSessionOptions): SessionOrchestrator {
		const memoryStore = options.memoryStore ?? new InMemoryMemoryStore();
		const initialTask = createInitialTask(options.goal, options.initialTask);
		const snapshot: WorkflowSessionSnapshot = {
			goal: options.goal,
			currentPhase: "intake",
			status: "ready",
			taskGraph: createTaskGraph([initialTask]),
			workspace: createWorkspaceState(options),
			artifacts: [],
			verification: [],
			transitions: [],
			memory: memoryStore.getAll(),
		};

		return new SessionOrchestrator(withDerivedWorkflowStatus(snapshot), memoryStore);
	}

	static fromSnapshot(snapshot: WorkflowSessionSnapshot, memoryStore?: MemoryStore): SessionOrchestrator {
		const resolvedMemoryStore = memoryStore ?? new InMemoryMemoryStore();
		for (const record of snapshot.memory) {
			resolvedMemoryStore.set(record);
		}

		return new SessionOrchestrator(
			withDerivedWorkflowStatus({
				...snapshot,
				artifacts: [...snapshot.artifacts],
				verification: [...snapshot.verification],
				transitions: [...snapshot.transitions],
				memory: resolvedMemoryStore.getAll(),
			}),
			resolvedMemoryStore,
		);
	}

	snapshot(): WorkflowSessionSnapshot {
		return {
			...this.#snapshot,
			artifacts: [...this.#snapshot.artifacts],
			verification: [...this.#snapshot.verification],
			transitions: [...this.#snapshot.transitions],
			memory: [...this.#snapshot.memory],
		};
	}

	getActiveTask(): TaskNode | undefined {
		const activeTaskId = this.#snapshot.taskGraph.activeTaskId;
		return activeTaskId ? this.#snapshot.taskGraph.tasks[activeTaskId] : undefined;
	}

	transition(nextPhase: WorkflowPhase, reason: string, at = new Date().toISOString()): WorkflowSessionSnapshot {
		if (!canTransition(this.#snapshot.currentPhase, nextPhase)) {
			throw new Error(`Invalid workflow transition: ${this.#snapshot.currentPhase} -> ${nextPhase}`);
		}

		this.#snapshot = withDerivedWorkflowStatus({
			...this.#snapshot,
			currentPhase: nextPhase,
			transitions: [
				...this.#snapshot.transitions,
				{
					from: this.#snapshot.currentPhase,
					to: nextPhase,
					reason,
					at,
				},
			],
		});

		return this.snapshot();
	}

	replaceTaskGraph(taskGraph: TaskGraph): WorkflowSessionSnapshot {
		this.#snapshot = withDerivedWorkflowStatus({
			...this.#snapshot,
			taskGraph,
		});
		return this.snapshot();
	}

	upsertTask(task: CreateTaskNodeInput): WorkflowSessionSnapshot {
		return this.replaceTaskGraph(withTask(this.#snapshot.taskGraph, createTaskNode(task)));
	}

	updateTask(taskId: string, updates: UpdateTaskNodeInput): WorkflowSessionSnapshot {
		return this.replaceTaskGraph(updateTask(this.#snapshot.taskGraph, taskId, updates));
	}

	updateTaskStatus(taskId: string, status: TaskStatus): WorkflowSessionSnapshot {
		return this.replaceTaskGraph(setTaskStatus(this.#snapshot.taskGraph, taskId, status));
	}

	setActiveTask(taskId: string | undefined): WorkflowSessionSnapshot {
		return this.replaceTaskGraph(setActiveTask(this.#snapshot.taskGraph, taskId));
	}

	replaceWorkspaceState(workspace: WorkspaceState): WorkflowSessionSnapshot {
		this.#snapshot = withDerivedWorkflowStatus({
			...this.#snapshot,
			workspace,
		});
		return this.snapshot();
	}

	recordArtifact(artifact: Omit<WorkflowArtifact, "recordedAt"> & { recordedAt?: string }): WorkflowSessionSnapshot {
		this.#snapshot = withDerivedWorkflowStatus({
			...this.#snapshot,
			artifacts: [...this.#snapshot.artifacts, createWorkflowArtifact(artifact)],
		});
		return this.snapshot();
	}

	recordVerification(taskId: string, status: VerificationStatus, evidence: RunEvidence): WorkflowSessionSnapshot {
		this.#snapshot = withDerivedWorkflowStatus({
			...this.#snapshot,
			verification: [...this.#snapshot.verification, createVerificationRecord({ taskId, status, evidence })],
		});
		return this.snapshot();
	}

	remember(record: MemoryRecord): WorkflowSessionSnapshot {
		this.#memoryStore.set(record);
		this.#snapshot = withDerivedWorkflowStatus({
			...this.#snapshot,
			memory: this.#memoryStore.getAll(),
		});
		return this.snapshot();
	}
}

import type { WorkflowArtifact } from "./artifacts.js";
import type { RunEvidence, VerificationStatus } from "./evidence.js";
import type { MemoryRecord } from "./memory-store.js";
import {
	getActiveTaskCompletionState,
	getLatestTaskVerification,
	getTaskCompletionLabel,
	getTaskVerificationStatus,
	type SessionOrchestrator,
	type WorkflowPhase,
	type WorkflowSessionSnapshot,
} from "./session-orchestrator.js";
import type { CreateTaskNodeInput, TaskGraph, TaskNode, TaskStatus, UpdateTaskNodeInput } from "./task-graph.js";
import type { WorkspaceState } from "./workspace-state.js";

export interface WorkflowControllerHooks {
	getSession(): SessionOrchestrator;
	persistCurrentSession(): WorkflowSessionSnapshot;
	replaceSnapshot(snapshot: WorkflowSessionSnapshot): WorkflowSessionSnapshot;
	appendArtifactEntry(artifact: WorkflowArtifact): void;
	appendVerificationEntry(record: {
		taskId: string;
		status: VerificationStatus;
		evidence: RunEvidence;
		recordedAt: string;
	}): void;
}

export class WorkflowController {
	readonly #hooks: WorkflowControllerHooks;

	constructor(hooks: WorkflowControllerHooks) {
		this.#hooks = hooks;
	}

	snapshot(): WorkflowSessionSnapshot {
		return this.#hooks.getSession().snapshot();
	}

	getActiveTask(): TaskNode | undefined {
		return this.#hooks.getSession().getActiveTask();
	}

	getActiveTaskCompletionState() {
		return getActiveTaskCompletionState(this.snapshot());
	}

	getLatestTaskVerification(taskId: string) {
		return getLatestTaskVerification(this.snapshot(), taskId);
	}

	getTaskVerificationStatus(taskId: string) {
		return getTaskVerificationStatus(this.snapshot(), taskId);
	}

	getTaskCompletionLabel(taskId: string) {
		return getTaskCompletionLabel(this.snapshot(), taskId);
	}

	replaceSnapshot(snapshot: WorkflowSessionSnapshot): WorkflowSessionSnapshot {
		return this.#hooks.replaceSnapshot(snapshot);
	}

	replaceTaskGraph(taskGraph: TaskGraph): WorkflowSessionSnapshot {
		this.#hooks.getSession().replaceTaskGraph(taskGraph);
		return this.#hooks.persistCurrentSession();
	}

	upsertTask(task: CreateTaskNodeInput): WorkflowSessionSnapshot {
		this.#hooks.getSession().upsertTask(task);
		return this.#hooks.persistCurrentSession();
	}

	updateTask(taskId: string, updates: UpdateTaskNodeInput): WorkflowSessionSnapshot {
		this.#hooks.getSession().updateTask(taskId, updates);
		return this.#hooks.persistCurrentSession();
	}

	updateTaskStatus(taskId: string, status: TaskStatus): WorkflowSessionSnapshot {
		this.#hooks.getSession().updateTaskStatus(taskId, status);
		return this.#hooks.persistCurrentSession();
	}

	setActiveTask(taskId: string | undefined): WorkflowSessionSnapshot {
		this.#hooks.getSession().setActiveTask(taskId);
		return this.#hooks.persistCurrentSession();
	}

	transition(nextPhase: WorkflowPhase, reason: string): WorkflowSessionSnapshot {
		this.#hooks.getSession().transition(nextPhase, reason);
		return this.#hooks.persistCurrentSession();
	}

	replaceWorkspaceState(workspace: WorkspaceState): WorkflowSessionSnapshot {
		this.#hooks.getSession().replaceWorkspaceState(workspace);
		return this.#hooks.persistCurrentSession();
	}

	recordArtifact(artifact: Omit<WorkflowArtifact, "recordedAt"> & { recordedAt?: string }): WorkflowSessionSnapshot {
		const snapshot = this.#hooks.getSession().recordArtifact(artifact);
		const latestArtifact = snapshot.artifacts[snapshot.artifacts.length - 1];
		if (latestArtifact) {
			this.#hooks.appendArtifactEntry(latestArtifact);
		}
		return this.#hooks.persistCurrentSession();
	}

	recordVerification(taskId: string, status: VerificationStatus, evidence: RunEvidence): WorkflowSessionSnapshot {
		const snapshot = this.#hooks.getSession().recordVerification(taskId, status, evidence);
		const latestRecord = snapshot.verification[snapshot.verification.length - 1];
		if (latestRecord) {
			this.#hooks.appendVerificationEntry(latestRecord);
		}
		return this.#hooks.persistCurrentSession();
	}

	remember(record: MemoryRecord): WorkflowSessionSnapshot {
		this.#hooks.getSession().remember(record);
		return this.#hooks.persistCurrentSession();
	}
}

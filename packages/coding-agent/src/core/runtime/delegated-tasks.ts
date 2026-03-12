import { randomUUID } from "node:crypto";
import type { RuntimeEventStream } from "./event-stream.js";
import type { RuntimeSqliteStore } from "./sqlite-store.js";
import type { DelegatedTaskRecord, DelegatedTaskStatus } from "./types.js";

export interface CreateDelegatedTaskInput {
	delegatedTaskId?: string;
	threadId: string;
	mailboxMessageId?: string;
	parentSessionId: string;
	owner: string;
	assignee: string;
	goal: string;
	summary?: string;
	createdAt?: number;
}

export interface ListDelegatedTasksFilters {
	parentSessionId?: string;
	threadId?: string;
	status?: DelegatedTaskStatus;
	limit?: number;
}

interface TransitionOptions {
	summary?: string;
	lastError?: string;
	updatedAt?: number;
}

const EVENT_SOURCE = "runtime.delegated-tasks";

export class DelegatedTaskService {
	constructor(
		private readonly store: RuntimeSqliteStore,
		private readonly events: RuntimeEventStream,
	) {}

	create(input: CreateDelegatedTaskInput): DelegatedTaskRecord {
		const now = input.createdAt ?? this.store.now();
		const delegatedTask: DelegatedTaskRecord = {
			delegatedTaskId: input.delegatedTaskId ?? randomUUID(),
			threadId: input.threadId,
			mailboxMessageId: input.mailboxMessageId,
			parentSessionId: input.parentSessionId,
			owner: input.owner,
			assignee: input.assignee,
			goal: input.goal,
			summary: input.summary,
			status: "queued",
			createdAt: now,
			updatedAt: now,
		};
		this.store.insertDelegatedTask(delegatedTask);
		this.recordLifecycleEvent("delegated_task.queued", delegatedTask);
		return delegatedTask;
	}

	get(delegatedTaskId: string): DelegatedTaskRecord | undefined {
		return this.store.getDelegatedTask(delegatedTaskId);
	}

	list(filters: ListDelegatedTasksFilters = {}): DelegatedTaskRecord[] {
		return this.store.listDelegatedTasks({
			parentSessionId: filters.parentSessionId,
			threadId: filters.threadId,
			status: filters.status,
			limit: Math.max(1, Math.min(200, filters.limit ?? 50)),
		});
	}

	markRunning(delegatedTaskId: string, summary?: string): DelegatedTaskRecord | undefined {
		return this.transition(delegatedTaskId, "running", new Set(["queued", "blocked"]), {
			summary,
		});
	}

	markBlocked(delegatedTaskId: string, summary: string): DelegatedTaskRecord | undefined {
		return this.transition(delegatedTaskId, "blocked", new Set(["queued", "running"]), {
			summary,
		});
	}

	markCompleted(delegatedTaskId: string, summary?: string): DelegatedTaskRecord | undefined {
		return this.transition(delegatedTaskId, "completed", new Set(["running", "blocked", "queued"]), {
			summary,
		});
	}

	markFailed(delegatedTaskId: string, error: string, summary?: string): DelegatedTaskRecord | undefined {
		return this.transition(delegatedTaskId, "failed", new Set(["queued", "running", "blocked"]), {
			summary,
			lastError: error,
		});
	}

	requeue(delegatedTaskId: string, summary?: string): DelegatedTaskRecord | undefined {
		return this.transition(delegatedTaskId, "queued", new Set(["failed", "blocked"]), {
			summary,
			lastError: "",
		});
	}

	private transition(
		delegatedTaskId: string,
		next: DelegatedTaskStatus,
		allowedCurrent: Set<DelegatedTaskStatus>,
		options: TransitionOptions,
	): DelegatedTaskRecord | undefined {
		const current = this.store.getDelegatedTask(delegatedTaskId);
		if (!current || !allowedCurrent.has(current.status)) {
			return undefined;
		}

		const updatedAt = options.updatedAt ?? this.store.now();
		const record = this.store.updateDelegatedTask(delegatedTaskId, {
			status: next,
			summary: options.summary,
			lastError: options.lastError,
			updatedAt,
			completedAt: next === "completed" ? updatedAt : null,
		});
		if (!record) {
			return undefined;
		}

		this.recordLifecycleEvent(`delegated_task.${next}`, record);
		return record;
	}

	private recordLifecycleEvent(type: string, record: DelegatedTaskRecord): void {
		this.events.record({
			sessionId: record.parentSessionId,
			type,
			source: EVENT_SOURCE,
			lane: "delegate",
			severity: record.status === "failed" ? "warn" : "info",
			payload: {
				delegatedTaskId: record.delegatedTaskId,
				threadId: record.threadId,
				mailboxMessageId: record.mailboxMessageId,
				parentSessionId: record.parentSessionId,
				owner: record.owner,
				assignee: record.assignee,
				goal: record.goal,
				status: record.status,
				summary: record.summary,
				lastError: record.lastError,
				completedAt: record.completedAt,
			},
		});
	}
}

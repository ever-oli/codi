import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { DelegatedTaskRecord, DelegatedTaskStatus, LaneName, RuntimeSeverity } from "./types.js";

function nowMs(): number {
	return Date.now();
}

function safeParseJson(value: unknown): Record<string, unknown> {
	if (typeof value !== "string" || !value.trim()) {
		return {};
	}
	try {
		const parsed = JSON.parse(value);
		if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
			return {};
		}
		return parsed as Record<string, unknown>;
	} catch {
		return {};
	}
}

interface DelegatedTaskRow {
	delegated_task_id: string;
	thread_id: string;
	mailbox_message_id?: string | null;
	parent_session_id: string;
	owner: string;
	assignee: string;
	goal: string;
	summary?: string | null;
	status: string;
	last_error?: string | null;
	created_at: number;
	updated_at: number;
	completed_at?: number | null;
}

function toDelegatedTaskRecord(row: DelegatedTaskRow): DelegatedTaskRecord {
	return {
		delegatedTaskId: row.delegated_task_id,
		threadId: row.thread_id,
		mailboxMessageId: row.mailbox_message_id ?? undefined,
		parentSessionId: row.parent_session_id,
		owner: row.owner,
		assignee: row.assignee,
		goal: row.goal,
		summary: row.summary ?? undefined,
		status:
			row.status === "queued" ||
			row.status === "running" ||
			row.status === "blocked" ||
			row.status === "completed" ||
			row.status === "failed"
				? row.status
				: "queued",
		lastError: row.last_error ?? undefined,
		createdAt: row.created_at,
		updatedAt: row.updated_at,
		completedAt: row.completed_at ?? undefined,
	};
}

export class RuntimeSqliteStore {
	readonly db: DatabaseSync;

	constructor(dbPath: string) {
		mkdirSync(dirname(dbPath), { recursive: true });
		this.db = new DatabaseSync(dbPath, { timeout: 5_000 });
		this.db.exec("PRAGMA journal_mode = WAL;");
		this.db.exec("PRAGMA busy_timeout = 5000;");
		this.db.exec("PRAGMA synchronous = NORMAL;");
		this.db.exec("PRAGMA temp_store = MEMORY;");
		this.migrate();
	}

	private migrate(): void {
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS runtime_meta (
				key TEXT PRIMARY KEY,
				value TEXT NOT NULL
			);

			CREATE TABLE IF NOT EXISTS runtime_events (
				id TEXT PRIMARY KEY,
				session_id TEXT,
				type TEXT NOT NULL,
				severity TEXT NOT NULL,
				source TEXT NOT NULL,
				lane TEXT,
				payload_json TEXT NOT NULL,
				created_at INTEGER NOT NULL
			);
			CREATE INDEX IF NOT EXISTS idx_runtime_events_created_at ON runtime_events(created_at DESC);
			CREATE INDEX IF NOT EXISTS idx_runtime_events_type_created_at ON runtime_events(type, created_at DESC);
			CREATE INDEX IF NOT EXISTS idx_runtime_events_session_created_at ON runtime_events(session_id, created_at DESC);

			CREATE TABLE IF NOT EXISTS outbound_queue (
				id TEXT PRIMARY KEY,
				topic TEXT NOT NULL,
				dedupe_key TEXT,
				payload_json TEXT NOT NULL,
				lane TEXT NOT NULL,
				state TEXT NOT NULL,
				attempts INTEGER NOT NULL DEFAULT 0,
				max_attempts INTEGER NOT NULL DEFAULT 5,
				available_at INTEGER NOT NULL,
				leased_until INTEGER,
				last_error TEXT,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);
			CREATE UNIQUE INDEX IF NOT EXISTS idx_outbound_queue_dedupe
				ON outbound_queue(dedupe_key)
				WHERE dedupe_key IS NOT NULL AND state IN ('queued', 'leased', 'acked');
			CREATE INDEX IF NOT EXISTS idx_outbound_queue_state_available
				ON outbound_queue(state, available_at);

			CREATE TABLE IF NOT EXISTS outbound_queue_attempts (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				message_id TEXT NOT NULL,
				attempt INTEGER NOT NULL,
				started_at INTEGER NOT NULL,
				ended_at INTEGER,
				success INTEGER NOT NULL DEFAULT 0,
				error TEXT,
				FOREIGN KEY(message_id) REFERENCES outbound_queue(id)
			);
			CREATE INDEX IF NOT EXISTS idx_outbound_queue_attempts_message_id
				ON outbound_queue_attempts(message_id, id DESC);

			CREATE TABLE IF NOT EXISTS mailbox_messages (
				message_id TEXT PRIMARY KEY,
				thread_id TEXT NOT NULL,
				sender TEXT NOT NULL,
				recipient TEXT NOT NULL,
				intent TEXT NOT NULL,
				reply_to TEXT,
				deadline TEXT,
				priority INTEGER NOT NULL DEFAULT 5,
				payload_json TEXT NOT NULL,
				expected_output_schema TEXT,
				completion_criteria TEXT,
				retry_policy TEXT,
				state TEXT NOT NULL,
				last_error TEXT,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);
			CREATE INDEX IF NOT EXISTS idx_mailbox_inbox ON mailbox_messages(recipient, state, created_at DESC);
			CREATE INDEX IF NOT EXISTS idx_mailbox_outbox ON mailbox_messages(sender, state, created_at DESC);
			CREATE INDEX IF NOT EXISTS idx_mailbox_thread ON mailbox_messages(thread_id, created_at DESC);

			CREATE TABLE IF NOT EXISTS delegated_tasks (
				delegated_task_id TEXT PRIMARY KEY,
				thread_id TEXT NOT NULL,
				mailbox_message_id TEXT,
				parent_session_id TEXT NOT NULL,
				owner TEXT NOT NULL,
				assignee TEXT NOT NULL,
				goal TEXT NOT NULL,
				summary TEXT,
				status TEXT NOT NULL,
				last_error TEXT,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL,
				completed_at INTEGER
			);
			CREATE INDEX IF NOT EXISTS idx_delegated_tasks_session_status
				ON delegated_tasks(parent_session_id, status, updated_at DESC);
			CREATE INDEX IF NOT EXISTS idx_delegated_tasks_thread
				ON delegated_tasks(thread_id, updated_at DESC);
			CREATE INDEX IF NOT EXISTS idx_delegated_tasks_mailbox
				ON delegated_tasks(mailbox_message_id);

			CREATE TABLE IF NOT EXISTS cron_jobs (
				id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				interval_seconds INTEGER NOT NULL,
				intent TEXT NOT NULL,
				payload_json TEXT NOT NULL,
				enabled INTEGER NOT NULL DEFAULT 1,
				next_run_at INTEGER NOT NULL,
				last_run_at INTEGER,
				last_run_status TEXT,
				last_error TEXT,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);
			CREATE INDEX IF NOT EXISTS idx_cron_jobs_next_run ON cron_jobs(enabled, next_run_at);

			CREATE TABLE IF NOT EXISTS lane_policies (
				lane TEXT PRIMARY KEY,
				concurrency INTEGER NOT NULL,
				queue_strategy TEXT NOT NULL
			);
		`);

		const existing = this.db.prepare("SELECT value FROM runtime_meta WHERE key = 'schema_version'").get() as
			| { value: string }
			| undefined;
		if (!existing) {
			this.db.prepare("INSERT INTO runtime_meta(key, value) VALUES ('schema_version', '1')").run();
		}
	}

	close(): void {
		this.db.close();
	}

	insertRuntimeEvent(record: {
		id: string;
		sessionId?: string;
		type: string;
		severity: string;
		source: string;
		lane?: string;
		payload: Record<string, unknown>;
		createdAt: number;
	}): void {
		this.db
			.prepare(
				`INSERT INTO runtime_events(
					id, session_id, type, severity, source, lane, payload_json, created_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				record.id,
				record.sessionId ?? null,
				record.type,
				record.severity,
				record.source,
				record.lane ?? null,
				JSON.stringify(record.payload ?? {}),
				record.createdAt,
			);
	}

	listRuntimeEvents(filters: {
		type?: string;
		sessionId?: string;
		lane?: LaneName;
		severity?: RuntimeSeverity;
		fromTs?: number;
		toTs?: number;
		limit: number;
	}): Array<{
		id: string;
		sessionId?: string;
		type: string;
		severity: RuntimeSeverity;
		source: string;
		lane?: LaneName;
		payload: Record<string, unknown>;
		createdAt: number;
	}> {
		const where: string[] = [];
		const args: unknown[] = [];
		if (filters.type) {
			where.push("type = ?");
			args.push(filters.type);
		}
		if (filters.sessionId) {
			where.push("session_id = ?");
			args.push(filters.sessionId);
		}
		if (filters.lane) {
			where.push("lane = ?");
			args.push(filters.lane);
		}
		if (filters.severity) {
			where.push("severity = ?");
			args.push(filters.severity);
		}
		if (filters.fromTs !== undefined) {
			where.push("created_at >= ?");
			args.push(filters.fromTs);
		}
		if (filters.toTs !== undefined) {
			where.push("created_at <= ?");
			args.push(filters.toTs);
		}
		const clause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
		const queryArgs = [...args, filters.limit] as Array<string | number>;
		const rows = this.db
			.prepare(
				`SELECT id, session_id, type, severity, source, lane, payload_json, created_at
				 FROM runtime_events ${clause}
				 ORDER BY created_at DESC
				 LIMIT ?`,
			)
			.all(...queryArgs) as Array<{
			id: string;
			session_id?: string | null;
			type: string;
			severity: string;
			source: string;
			lane?: string | null;
			payload_json: string;
			created_at: number;
		}>;

		return rows.map((row) => ({
			id: row.id,
			sessionId: row.session_id ?? undefined,
			type: row.type,
			severity:
				row.severity === "debug" || row.severity === "info" || row.severity === "warn" || row.severity === "error"
					? row.severity
					: "info",
			source: row.source,
			lane:
				row.lane === "default" ||
				row.lane === "delegate" ||
				row.lane === "cron" ||
				row.lane === "compact" ||
				row.lane === "notification"
					? row.lane
					: undefined,
			payload: safeParseJson(row.payload_json),
			createdAt: row.created_at,
		}));
	}

	pruneRuntimeEvents(olderThanTs: number): number {
		const result = this.db.prepare("DELETE FROM runtime_events WHERE created_at < ?").run(olderThanTs) as {
			changes?: number;
		};
		return result.changes ?? 0;
	}

	insertDelegatedTask(record: DelegatedTaskRecord): void {
		this.db
			.prepare(
				`INSERT INTO delegated_tasks(
					delegated_task_id, thread_id, mailbox_message_id, parent_session_id, owner, assignee,
					goal, summary, status, last_error, created_at, updated_at, completed_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				record.delegatedTaskId,
				record.threadId,
				record.mailboxMessageId ?? null,
				record.parentSessionId,
				record.owner,
				record.assignee,
				record.goal,
				record.summary ?? null,
				record.status,
				record.lastError ?? null,
				record.createdAt,
				record.updatedAt,
				record.completedAt ?? null,
			);
	}

	getDelegatedTask(delegatedTaskId: string): DelegatedTaskRecord | undefined {
		const row = this.db.prepare("SELECT * FROM delegated_tasks WHERE delegated_task_id = ?").get(delegatedTaskId) as
			| DelegatedTaskRow
			| undefined;
		return row ? toDelegatedTaskRecord(row) : undefined;
	}

	listDelegatedTasks(filters: {
		parentSessionId?: string;
		threadId?: string;
		status?: DelegatedTaskStatus;
		limit: number;
	}): DelegatedTaskRecord[] {
		const where: string[] = [];
		const args: Array<string | number> = [];
		if (filters.parentSessionId) {
			where.push("parent_session_id = ?");
			args.push(filters.parentSessionId);
		}
		if (filters.threadId) {
			where.push("thread_id = ?");
			args.push(filters.threadId);
		}
		if (filters.status) {
			where.push("status = ?");
			args.push(filters.status);
		}
		const clause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
		const rows = this.db
			.prepare(
				`SELECT * FROM delegated_tasks
				 ${clause}
				 ORDER BY updated_at DESC
				 LIMIT ?`,
			)
			.all(...args, filters.limit) as unknown as DelegatedTaskRow[];
		return rows.map(toDelegatedTaskRecord);
	}

	updateDelegatedTask(
		delegatedTaskId: string,
		updates: {
			status: DelegatedTaskStatus;
			summary?: string;
			lastError?: string;
			updatedAt: number;
			completedAt?: number | null;
		},
	): DelegatedTaskRecord | undefined {
		const existing = this.getDelegatedTask(delegatedTaskId);
		if (!existing) {
			return undefined;
		}
		this.db
			.prepare(
				`UPDATE delegated_tasks
				 SET status = ?,
				     summary = ?,
				     last_error = ?,
				     updated_at = ?,
				     completed_at = ?
				 WHERE delegated_task_id = ?`,
			)
			.run(
				updates.status,
				updates.summary ?? existing.summary ?? null,
				updates.lastError !== undefined ? updates.lastError || null : (existing.lastError ?? null),
				updates.updatedAt,
				updates.completedAt !== undefined ? updates.completedAt : (existing.completedAt ?? null),
				delegatedTaskId,
			);
		return this.getDelegatedTask(delegatedTaskId);
	}

	loadLanePolicies(): Map<string, { concurrency: number; queueStrategy: "fifo" }> {
		const rows = this.db.prepare("SELECT lane, concurrency, queue_strategy FROM lane_policies").all() as Array<{
			lane: string;
			concurrency: number;
			queue_strategy: string;
		}>;
		const policies = new Map<string, { concurrency: number; queueStrategy: "fifo" }>();
		for (const row of rows) {
			policies.set(row.lane, {
				concurrency: Math.max(1, Math.floor(row.concurrency)),
				queueStrategy: "fifo",
			});
		}
		return policies;
	}

	saveLanePolicy(lane: string, concurrency: number, queueStrategy: "fifo"): void {
		this.db
			.prepare(
				`INSERT INTO lane_policies(lane, concurrency, queue_strategy)
				 VALUES (?, ?, ?)
				 ON CONFLICT(lane) DO UPDATE
				 SET concurrency = excluded.concurrency,
				     queue_strategy = excluded.queue_strategy`,
			)
			.run(lane, Math.max(1, Math.floor(concurrency)), queueStrategy);
	}

	now(): number {
		return nowMs();
	}
}

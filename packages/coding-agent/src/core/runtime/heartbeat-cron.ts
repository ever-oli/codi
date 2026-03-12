import { randomUUID } from "node:crypto";
import type { DeliveryQueueService } from "./delivery-queue.js";
import type { RuntimeEventStream } from "./event-stream.js";
import type { LaneScheduler } from "./lane-scheduler.js";
import type { RuntimeSqliteStore } from "./sqlite-store.js";
import type { CronJobRecord } from "./types.js";

interface CronRow {
	id: string;
	name: string;
	interval_seconds: number;
	intent: string;
	payload_json: string;
	enabled: number;
	next_run_at: number;
	last_run_at?: number | null;
	last_run_status?: "ok" | "error" | null;
	last_error?: string | null;
	created_at: number;
	updated_at: number;
}

function parsePayload(value: string): Record<string, unknown> {
	try {
		const parsed = JSON.parse(value);
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
		return {};
	} catch {
		return {};
	}
}

function toRecord(row: CronRow): CronJobRecord {
	return {
		id: row.id,
		name: row.name,
		intervalSeconds: row.interval_seconds,
		intent: row.intent,
		payload: parsePayload(row.payload_json),
		enabled: row.enabled === 1,
		nextRunAt: row.next_run_at,
		lastRunAt: row.last_run_at ?? undefined,
		lastRunStatus: row.last_run_status ?? undefined,
		lastError: row.last_error ?? undefined,
		createdAt: row.created_at,
		updatedAt: row.updated_at,
	};
}

export class HeartbeatCronService {
	private timer: ReturnType<typeof setInterval> | undefined;
	private readonly intervalMs: number;
	private lastTickAt: number | undefined;
	private tickCount = 0;

	constructor(
		private readonly store: RuntimeSqliteStore,
		private readonly events: RuntimeEventStream,
		private readonly queue: DeliveryQueueService,
		private readonly lanes: LaneScheduler,
		intervalMs = 5_000,
	) {
		this.intervalMs = Math.max(1000, intervalMs);
		this.queue.registerHandler({
			topic: "cron.dispatch",
			handle: async (message) => {
				const jobId = String(message.payload.jobId ?? "");
				if (!jobId) {
					throw new Error("cron.dispatch missing jobId");
				}
				this.events.record({
					type: "cron.job.dispatched",
					source: "runtime.heartbeat",
					lane: "cron",
					payload: {
						jobId,
						intent: message.payload.intent,
					},
				});
			},
		});
	}

	start(): void {
		if (this.timer) {
			return;
		}
		this.events.record({
			type: "heartbeat.started",
			source: "runtime.heartbeat",
			payload: { intervalMs: this.intervalMs },
		});
		this.timer = setInterval(() => {
			void this.tick();
		}, this.intervalMs);
	}

	stop(): void {
		if (this.timer) {
			clearInterval(this.timer);
			this.timer = undefined;
		}
		this.events.record({
			type: "heartbeat.stopped",
			source: "runtime.heartbeat",
			payload: { intervalMs: this.intervalMs },
		});
	}

	getStatus(): { running: boolean; intervalMs: number; tickCount: number; lastTickAt?: number; jobsEnabled: number } {
		const jobsEnabled = this.store.db.prepare("SELECT COUNT(*) AS count FROM cron_jobs WHERE enabled = 1").get() as {
			count: number;
		};
		return {
			running: !!this.timer,
			intervalMs: this.intervalMs,
			tickCount: this.tickCount,
			lastTickAt: this.lastTickAt,
			jobsEnabled: jobsEnabled.count ?? 0,
		};
	}

	async tick(): Promise<void> {
		const now = this.store.now();
		this.lastTickAt = now;
		this.tickCount += 1;
		this.events.record({
			type: "heartbeat.tick",
			source: "runtime.heartbeat",
			payload: { tickCount: this.tickCount },
		});
		await this.dispatchDueJobs(now);
	}

	addJob(input: {
		name: string;
		intervalSeconds: number;
		intent: string;
		payload?: Record<string, unknown>;
	}): CronJobRecord {
		const now = this.store.now();
		const id = randomUUID();
		const intervalSeconds = Math.max(5, Math.floor(input.intervalSeconds));
		this.store.db
			.prepare(
				`INSERT INTO cron_jobs(
					id, name, interval_seconds, intent, payload_json, enabled, next_run_at,
					last_run_at, last_run_status, last_error, created_at, updated_at
				) VALUES (?, ?, ?, ?, ?, 1, ?, NULL, NULL, NULL, ?, ?)`,
			)
			.run(
				id,
				input.name,
				intervalSeconds,
				input.intent,
				JSON.stringify(input.payload ?? {}),
				now + intervalSeconds * 1000,
				now,
				now,
			);
		this.events.record({
			type: "cron.job.added",
			source: "runtime.heartbeat",
			payload: { id, name: input.name, intervalSeconds },
		});
		return this.getJob(id)!;
	}

	listJobs(limit = 200): CronJobRecord[] {
		const rows = this.store.db
			.prepare("SELECT * FROM cron_jobs ORDER BY created_at DESC LIMIT ?")
			.all(Math.max(1, Math.min(500, limit))) as unknown as CronRow[];
		return rows.map(toRecord);
	}

	getJob(id: string): CronJobRecord | undefined {
		const row = this.store.db.prepare("SELECT * FROM cron_jobs WHERE id = ?").get(id) as CronRow | undefined;
		return row ? toRecord(row) : undefined;
	}

	setJobEnabled(id: string, enabled: boolean): boolean {
		const now = this.store.now();
		const result = this.store.db
			.prepare(
				`UPDATE cron_jobs
				 SET enabled = ?,
				     updated_at = ?
				 WHERE id = ?`,
			)
			.run(enabled ? 1 : 0, now, id) as { changes?: number };
		const changed = (result.changes ?? 0) > 0;
		if (changed) {
			this.events.record({
				type: enabled ? "cron.job.resumed" : "cron.job.paused",
				source: "runtime.heartbeat",
				payload: { id },
			});
		}
		return changed;
	}

	removeJob(id: string): boolean {
		const result = this.store.db.prepare("DELETE FROM cron_jobs WHERE id = ?").run(id) as { changes?: number };
		const changed = (result.changes ?? 0) > 0;
		if (changed) {
			this.events.record({
				type: "cron.job.removed",
				source: "runtime.heartbeat",
				payload: { id },
			});
		}
		return changed;
	}

	private async dispatchDueJobs(now: number): Promise<void> {
		const due = this.store.db
			.prepare(
				`SELECT * FROM cron_jobs
				 WHERE enabled = 1
				   AND next_run_at <= ?
				 ORDER BY next_run_at ASC
				 LIMIT 25`,
			)
			.all(now) as unknown as CronRow[];
		for (const row of due) {
			await this.dispatchOne(toRecord(row));
		}
	}

	private async dispatchOne(job: CronJobRecord): Promise<void> {
		await this.lanes.schedule("cron", `cron:${job.name}`, async () => {
			try {
				this.queue.enqueue({
					topic: "cron.dispatch",
					lane: "cron",
					payload: {
						jobId: job.id,
						name: job.name,
						intent: job.intent,
						payload: job.payload,
					},
					dedupeKey: `cron:${job.id}:${job.nextRunAt}`,
					maxAttempts: 3,
				});
				this.markJobRun(job.id, "ok");
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				this.markJobRun(job.id, "error", message);
				throw error;
			}
		});
	}

	private markJobRun(id: string, status: "ok" | "error", error?: string): void {
		const now = this.store.now();
		const row = this.store.db.prepare("SELECT interval_seconds FROM cron_jobs WHERE id = ?").get(id) as
			| { interval_seconds: number }
			| undefined;
		if (!row) return;
		const nextRunAt = now + Math.max(5, row.interval_seconds) * 1000;
		this.store.db
			.prepare(
				`UPDATE cron_jobs
				 SET last_run_at = ?,
				     last_run_status = ?,
				     last_error = ?,
				     next_run_at = ?,
				     updated_at = ?
				 WHERE id = ?`,
			)
			.run(now, status, error ?? null, nextRunAt, now, id);
		this.events.record({
			type: status === "ok" ? "cron.job.run.ok" : "cron.job.run.error",
			source: "runtime.heartbeat",
			payload: {
				id,
				nextRunAt,
				error,
			},
			severity: status === "ok" ? "info" : "warn",
		});
	}
}

import { randomUUID } from "node:crypto";
import type { RuntimeEventStream } from "./event-stream.js";
import type { LaneScheduler } from "./lane-scheduler.js";
import type { RuntimeSqliteStore } from "./sqlite-store.js";
import type { LaneName, OutboundQueueRecord, QueueMessageState } from "./types.js";

export interface EnqueueMessageInput {
	topic: string;
	payload: Record<string, unknown>;
	lane?: LaneName;
	dedupeKey?: string;
	maxAttempts?: number;
	availableAt?: number;
}

export interface DeliveryHandler {
	topic: string;
	handle: (message: OutboundQueueRecord) => Promise<void>;
}

interface QueueRow {
	id: string;
	topic: string;
	dedupe_key?: string | null;
	payload_json: string;
	lane: string;
	state: QueueMessageState;
	attempts: number;
	max_attempts: number;
	available_at: number;
	leased_until?: number | null;
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

function toRecord(row: QueueRow): OutboundQueueRecord {
	return {
		id: row.id,
		topic: row.topic,
		payload: parsePayload(row.payload_json),
		lane: row.lane as LaneName,
		state: row.state,
		attempts: row.attempts,
		maxAttempts: row.max_attempts,
		dedupeKey: row.dedupe_key ?? undefined,
		availableAt: row.available_at,
		leasedUntil: row.leased_until ?? undefined,
		lastError: row.last_error ?? undefined,
		createdAt: row.created_at,
		updatedAt: row.updated_at,
	};
}

export class DeliveryQueueService {
	private readonly handlers = new Map<string, DeliveryHandler["handle"]>();
	private pollTimer: ReturnType<typeof setInterval> | undefined;
	private running = false;

	constructor(
		private readonly store: RuntimeSqliteStore,
		private readonly lanes: LaneScheduler,
		private readonly events: RuntimeEventStream,
	) {}

	registerHandler(handler: DeliveryHandler): void {
		this.handlers.set(handler.topic, handler.handle);
	}

	startPolling(intervalMs = 1200): void {
		if (this.pollTimer) {
			return;
		}
		this.pollTimer = setInterval(
			() => {
				void this.processDue();
			},
			Math.max(250, intervalMs),
		);
	}

	stopPolling(): void {
		if (this.pollTimer) {
			clearInterval(this.pollTimer);
			this.pollTimer = undefined;
		}
	}

	enqueue(input: EnqueueMessageInput): OutboundQueueRecord {
		const id = randomUUID();
		const lane = input.lane ?? "notification";
		const now = this.store.now();
		const availableAt = input.availableAt ?? now;
		const maxAttempts = Math.max(1, Math.floor(input.maxAttempts ?? 5));

		if (input.dedupeKey) {
			const existing = this.store.db
				.prepare(
					`SELECT * FROM outbound_queue
					 WHERE dedupe_key = ?
					   AND state IN ('queued', 'leased', 'acked')
					 ORDER BY created_at DESC
					 LIMIT 1`,
				)
				.get(input.dedupeKey) as QueueRow | undefined;
			if (existing) {
				return toRecord(existing);
			}
		}

		this.store.db
			.prepare(
				`INSERT INTO outbound_queue(
					id, topic, dedupe_key, payload_json, lane, state, attempts, max_attempts, available_at,
					leased_until, last_error, created_at, updated_at
				) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, NULL, NULL, ?, ?)`,
			)
			.run(
				id,
				input.topic,
				input.dedupeKey ?? null,
				JSON.stringify(input.payload ?? {}),
				lane,
				maxAttempts,
				availableAt,
				now,
				now,
			);

		const record = this.getById(id)!;
		this.events.record({
			type: "queue.enqueued",
			source: "runtime.queue",
			lane,
			payload: {
				id,
				topic: input.topic,
				dedupeKey: input.dedupeKey,
				maxAttempts,
			},
		});
		return record;
	}

	getById(id: string): OutboundQueueRecord | undefined {
		const row = this.store.db.prepare("SELECT * FROM outbound_queue WHERE id = ?").get(id) as QueueRow | undefined;
		return row ? toRecord(row) : undefined;
	}

	list(state?: QueueMessageState, limit = 100): OutboundQueueRecord[] {
		const rows = (state
			? this.store.db
					.prepare("SELECT * FROM outbound_queue WHERE state = ? ORDER BY created_at DESC LIMIT ?")
					.all(state, Math.max(1, Math.min(limit, 500)))
			: this.store.db
					.prepare("SELECT * FROM outbound_queue ORDER BY created_at DESC LIMIT ?")
					.all(Math.max(1, Math.min(limit, 500)))) as unknown as QueueRow[];
		return rows.map(toRecord);
	}

	retryDeadLetter(id: string): boolean {
		const now = this.store.now();
		const result = this.store.db
			.prepare(
				`UPDATE outbound_queue
				 SET state = 'queued',
				     available_at = ?,
				     leased_until = NULL,
				     last_error = NULL,
				     updated_at = ?
				 WHERE id = ? AND state = 'dead_letter'`,
			)
			.run(now, now, id) as { changes?: number };
		const changed = (result.changes ?? 0) > 0;
		if (changed) {
			this.events.record({
				type: "queue.dead_letter.retried",
				source: "runtime.queue",
				payload: { id },
			});
		}
		return changed;
	}

	// Requeue expired leases to provide crash-safe replay on startup.
	recoverExpiredLeases(now = this.store.now()): number {
		const result = this.store.db
			.prepare(
				`UPDATE outbound_queue
				 SET state = 'queued',
				     leased_until = NULL,
				     available_at = MIN(available_at, ?),
				     updated_at = ?
				 WHERE state = 'leased' AND leased_until IS NOT NULL AND leased_until < ?`,
			)
			.run(now, now, now) as { changes?: number };
		const recovered = result.changes ?? 0;
		if (recovered > 0) {
			this.events.record({
				type: "queue.recovered.expired_leases",
				source: "runtime.queue",
				payload: { recovered },
			});
		}
		return recovered;
	}

	async processDue(limit = 12): Promise<void> {
		if (this.running) {
			return;
		}
		this.running = true;
		try {
			try {
				this.recoverExpiredLeases();
			} catch (err: unknown) {
				if (err instanceof Error && "errcode" in err && (err as { errcode: number }).errcode === 5) {
					// SQLITE_BUSY — another write is in progress; skip this tick and retry next interval.
					return;
				}
				throw err;
			}
			const claimed = this.claimDue(limit);
			for (const message of claimed) {
				void this.processClaimedMessage(message);
			}
		} finally {
			this.running = false;
		}
	}

	private claimDue(limit: number): OutboundQueueRecord[] {
		const now = this.store.now();
		const rows = this.store.db
			.prepare(
				`SELECT * FROM outbound_queue
				 WHERE state = 'queued'
				   AND available_at <= ?
				 ORDER BY available_at ASC, created_at ASC
				 LIMIT ?`,
			)
			.all(now, Math.max(1, Math.min(limit, 100))) as unknown as QueueRow[];
		const claimed: OutboundQueueRecord[] = [];
		for (const row of rows) {
			const leaseUntil = now + 30_000;
			const result = this.store.db
				.prepare(
					`UPDATE outbound_queue
					 SET state = 'leased',
					     leased_until = ?,
					     updated_at = ?
					 WHERE id = ?
					   AND state = 'queued'`,
				)
				.run(leaseUntil, now, row.id) as { changes?: number };
			if ((result.changes ?? 0) > 0) {
				const updated = this.getById(row.id);
				if (updated) {
					claimed.push(updated);
				}
			}
		}
		return claimed;
	}

	private async processClaimedMessage(message: OutboundQueueRecord): Promise<void> {
		await this.lanes.schedule(message.lane, `queue:${message.topic}`, async () => {
			const handler = this.handlers.get(message.topic);
			const attempt = this.beginAttempt(message.id);
			if (!handler) {
				const noHandlerError = `No delivery handler for topic "${message.topic}"`;
				this.endAttempt(attempt.id, false, noHandlerError);
				this.fail(message.id, noHandlerError);
				return;
			}
			try {
				await handler(message);
				this.endAttempt(attempt.id, true);
				this.ack(message.id);
			} catch (error) {
				const messageText = error instanceof Error ? error.message : String(error);
				this.endAttempt(attempt.id, false, messageText);
				this.fail(message.id, messageText);
			}
		});
	}

	private beginAttempt(messageId: string): { id: number; attempt: number } {
		const row = this.store.db.prepare("SELECT attempts FROM outbound_queue WHERE id = ?").get(messageId) as
			| { attempts: number }
			| undefined;
		const nextAttempt = (row?.attempts ?? 0) + 1;
		this.store.db
			.prepare(
				`UPDATE outbound_queue
				 SET attempts = ?, updated_at = ?
				 WHERE id = ?`,
			)
			.run(nextAttempt, this.store.now(), messageId);
		const result = this.store.db
			.prepare(
				`INSERT INTO outbound_queue_attempts(message_id, attempt, started_at, success)
				 VALUES (?, ?, ?, 0)`,
			)
			.run(messageId, nextAttempt, this.store.now()) as { lastInsertRowid?: bigint | number };
		const id = Number(result.lastInsertRowid ?? 0);
		return { id, attempt: nextAttempt };
	}

	private endAttempt(id: number, success: boolean, error?: string): void {
		this.store.db
			.prepare(
				`UPDATE outbound_queue_attempts
				 SET ended_at = ?, success = ?, error = ?
				 WHERE id = ?`,
			)
			.run(this.store.now(), success ? 1 : 0, error ?? null, id);
	}

	private ack(id: string): void {
		const now = this.store.now();
		this.store.db
			.prepare(
				`UPDATE outbound_queue
				 SET state = 'acked',
				     leased_until = NULL,
				     last_error = NULL,
				     updated_at = ?
				 WHERE id = ?`,
			)
			.run(now, id);
		this.events.record({
			type: "queue.acked",
			source: "runtime.queue",
			payload: { id },
		});
	}

	private fail(id: string, error: string): void {
		const row = this.store.db.prepare("SELECT attempts, max_attempts FROM outbound_queue WHERE id = ?").get(id) as
			| { attempts: number; max_attempts: number }
			| undefined;
		if (!row) {
			return;
		}
		const now = this.store.now();
		if (row.attempts >= row.max_attempts) {
			this.store.db
				.prepare(
					`UPDATE outbound_queue
					 SET state = 'dead_letter',
					     leased_until = NULL,
					     last_error = ?,
					     updated_at = ?
					 WHERE id = ?`,
				)
				.run(error, now, id);
			this.events.record({
				type: "queue.dead_letter",
				source: "runtime.queue",
				severity: "warn",
				payload: {
					id,
					error,
					attempts: row.attempts,
					maxAttempts: row.max_attempts,
				},
			});
			return;
		}

		const retryDelayMs = Math.min(60_000, 2_000 * Math.max(1, 2 ** (row.attempts - 1)));
		this.store.db
			.prepare(
				`UPDATE outbound_queue
				 SET state = 'queued',
				     available_at = ?,
				     leased_until = NULL,
				     last_error = ?,
				     updated_at = ?
				 WHERE id = ?`,
			)
			.run(now + retryDelayMs, error, now, id);
		this.events.record({
			type: "queue.retry.scheduled",
			source: "runtime.queue",
			severity: "warn",
			payload: {
				id,
				error,
				retryDelayMs,
				attempt: row.attempts,
				maxAttempts: row.max_attempts,
			},
		});
	}
}

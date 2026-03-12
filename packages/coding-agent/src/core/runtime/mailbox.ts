import { randomUUID } from "node:crypto";
import type { DelegatedTaskService } from "./delegated-tasks.js";
import type { DeliveryQueueService } from "./delivery-queue.js";
import type { RuntimeEventStream } from "./event-stream.js";
import type { RuntimeSqliteStore } from "./sqlite-store.js";
import type { MailboxEnvelope, MailboxMessageState } from "./types.js";

interface MailboxRow {
	message_id: string;
	thread_id: string;
	sender: string;
	recipient: string;
	intent: string;
	reply_to?: string | null;
	deadline?: string | null;
	priority: number;
	payload_json: string;
	expected_output_schema?: string | null;
	completion_criteria?: string | null;
	retry_policy?: string | null;
	state: MailboxMessageState;
	last_error?: string | null;
	created_at: number;
	updated_at: number;
}

function parsePayload(payload: string): Record<string, unknown> {
	try {
		const parsed = JSON.parse(payload);
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
		return {};
	} catch {
		return {};
	}
}

function toEnvelope(row: MailboxRow): MailboxEnvelope {
	return {
		messageId: row.message_id,
		threadId: row.thread_id,
		from: row.sender,
		to: row.recipient,
		intent: row.intent,
		payload: parsePayload(row.payload_json),
		replyTo: row.reply_to ?? undefined,
		deadline: row.deadline ?? undefined,
		priority: row.priority,
		expectedOutputSchema: row.expected_output_schema ?? undefined,
		completionCriteria: row.completion_criteria ?? undefined,
		retryPolicy: row.retry_policy ?? undefined,
		createdAt: row.created_at,
		updatedAt: row.updated_at,
		state: row.state,
		lastError: row.last_error ?? undefined,
	};
}

export interface SendEnvelopeInput {
	threadId?: string;
	from: string;
	to: string;
	intent: string;
	payload: Record<string, unknown>;
	replyTo?: string;
	deadline?: string;
	priority?: number;
	expectedOutputSchema?: string;
	completionCriteria?: string;
	retryPolicy?: string;
	delegatedTask?: {
		delegatedTaskId?: string;
		goal: string;
		summary?: string;
	};
}

export class MailboxService {
	constructor(
		private readonly store: RuntimeSqliteStore,
		private readonly queue: DeliveryQueueService,
		private readonly events: RuntimeEventStream,
		private readonly delegatedTasks?: DelegatedTaskService,
	) {
		this.queue.registerHandler({
			topic: "mailbox.deliver",
			handle: async (message) => {
				const messageId = String(message.payload.messageId ?? "");
				if (!messageId) {
					throw new Error("Missing messageId in mailbox.deliver payload");
				}
				const transitioned = this.transition(messageId, "delivered");
				const delegatedTaskId =
					typeof message.payload.delegatedTaskId === "string" ? message.payload.delegatedTaskId : "";
				if (transitioned && delegatedTaskId) {
					this.delegatedTasks?.markRunning(delegatedTaskId, "Delivered to delegated worker.");
				}
			},
		});
	}

	send(input: SendEnvelopeInput): MailboxEnvelope {
		const now = this.store.now();
		const delegatedTaskId =
			input.delegatedTask?.delegatedTaskId?.trim() || (input.delegatedTask ? randomUUID() : undefined);
		const payload =
			delegatedTaskId && !("delegatedTaskId" in input.payload)
				? { ...input.payload, delegatedTaskId }
				: input.payload;
		const envelope: MailboxEnvelope = {
			messageId: randomUUID(),
			threadId: input.threadId?.trim() || randomUUID(),
			from: input.from,
			to: input.to,
			intent: input.intent,
			payload: payload ?? {},
			replyTo: input.replyTo,
			deadline: input.deadline,
			priority: input.priority ?? 5,
			expectedOutputSchema: input.expectedOutputSchema,
			completionCriteria: input.completionCriteria,
			retryPolicy: input.retryPolicy,
			createdAt: now,
			updatedAt: now,
			state: "drafted",
		};

		this.store.db
			.prepare(
				`INSERT INTO mailbox_messages(
					message_id, thread_id, sender, recipient, intent, reply_to, deadline, priority,
					payload_json, expected_output_schema, completion_criteria, retry_policy,
					state, last_error, created_at, updated_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)`,
			)
			.run(
				envelope.messageId,
				envelope.threadId,
				envelope.from,
				envelope.to,
				envelope.intent,
				envelope.replyTo ?? null,
				envelope.deadline ?? null,
				envelope.priority,
				JSON.stringify(envelope.payload),
				envelope.expectedOutputSchema ?? null,
				envelope.completionCriteria ?? null,
				envelope.retryPolicy ?? null,
				"drafted",
				envelope.createdAt,
				envelope.updatedAt,
			);

		if (delegatedTaskId && input.delegatedTask) {
			this.delegatedTasks?.create({
				delegatedTaskId,
				threadId: envelope.threadId,
				mailboxMessageId: envelope.messageId,
				parentSessionId: input.from,
				owner: input.from,
				assignee: input.to,
				goal: input.delegatedTask.goal,
				summary: input.delegatedTask.summary,
				createdAt: now,
			});
		}

		this.transition(envelope.messageId, "queued");
		this.queue.enqueue({
			topic: "mailbox.deliver",
			payload: {
				messageId: envelope.messageId,
				threadId: envelope.threadId,
				to: envelope.to,
				from: envelope.from,
				intent: envelope.intent,
				delegatedTaskId,
			},
			lane: "delegate",
			dedupeKey: `mailbox:${envelope.messageId}`,
			maxAttempts: 5,
		});

		this.events.record({
			type: "mailbox.sent",
			source: "runtime.mailbox",
			lane: "delegate",
			payload: {
				messageId: envelope.messageId,
				threadId: envelope.threadId,
				from: envelope.from,
				to: envelope.to,
				intent: envelope.intent,
			},
		});

		return this.get(envelope.messageId)!;
	}

	get(messageId: string): MailboxEnvelope | undefined {
		const row = this.store.db.prepare("SELECT * FROM mailbox_messages WHERE message_id = ?").get(messageId) as
			| MailboxRow
			| undefined;
		return row ? toEnvelope(row) : undefined;
	}

	listInbox(recipient: string, limit = 100): MailboxEnvelope[] {
		const rows = this.store.db
			.prepare(
				`SELECT * FROM mailbox_messages
				 WHERE recipient = ?
				 ORDER BY created_at DESC
				 LIMIT ?`,
			)
			.all(recipient, Math.max(1, Math.min(limit, 500))) as unknown as MailboxRow[];
		return rows.map(toEnvelope);
	}

	listOutbox(sender: string, limit = 100): MailboxEnvelope[] {
		const rows = this.store.db
			.prepare(
				`SELECT * FROM mailbox_messages
				 WHERE sender = ?
				 ORDER BY created_at DESC
				 LIMIT ?`,
			)
			.all(sender, Math.max(1, Math.min(limit, 500))) as unknown as MailboxRow[];
		return rows.map(toEnvelope);
	}

	listThread(threadId: string, limit = 200): MailboxEnvelope[] {
		const rows = this.store.db
			.prepare(
				`SELECT * FROM mailbox_messages
				 WHERE thread_id = ?
				 ORDER BY created_at ASC
				 LIMIT ?`,
			)
			.all(threadId, Math.max(1, Math.min(limit, 500))) as unknown as MailboxRow[];
		return rows.map(toEnvelope);
	}

	ack(messageId: string, actor?: string): boolean {
		const changed = this.transition(messageId, "acked");
		if (changed) {
			const envelope = this.get(messageId);
			const delegatedTaskId =
				typeof envelope?.payload.delegatedTaskId === "string" ? envelope.payload.delegatedTaskId : "";
			if (delegatedTaskId) {
				this.delegatedTasks?.markCompleted(delegatedTaskId, `Acknowledged by ${actor ?? "unknown"}.`);
			}
			this.events.record({
				type: "mailbox.acked",
				source: "runtime.mailbox",
				payload: {
					messageId,
					actor: actor ?? "unknown",
				},
			});
		}
		return changed;
	}

	retry(messageId: string): boolean {
		const envelope = this.get(messageId);
		if (!envelope) {
			return false;
		}
		const changed = this.transition(messageId, "queued", undefined, true);
		if (!changed) {
			return false;
		}
		const delegatedTaskId =
			typeof envelope.payload.delegatedTaskId === "string" ? envelope.payload.delegatedTaskId : "";
		if (delegatedTaskId) {
			this.delegatedTasks?.requeue(delegatedTaskId, "Delegated task requeued.");
		}
		this.queue.enqueue({
			topic: "mailbox.deliver",
			payload: {
				messageId: envelope.messageId,
				threadId: envelope.threadId,
				to: envelope.to,
				from: envelope.from,
				intent: envelope.intent,
				delegatedTaskId,
			},
			lane: "delegate",
			dedupeKey: `mailbox:${envelope.messageId}:retry:${Date.now()}`,
			maxAttempts: 5,
		});
		this.events.record({
			type: "mailbox.retry",
			source: "runtime.mailbox",
			payload: { messageId },
		});
		return true;
	}

	markFailed(messageId: string, error: string): boolean {
		const changed = this.transition(messageId, "failed", error);
		if (changed) {
			const envelope = this.get(messageId);
			const delegatedTaskId =
				typeof envelope?.payload.delegatedTaskId === "string" ? envelope.payload.delegatedTaskId : "";
			if (delegatedTaskId) {
				this.delegatedTasks?.markFailed(delegatedTaskId, error);
			}
		}
		return changed;
	}

	private transition(
		messageId: string,
		next: MailboxMessageState,
		lastError?: string,
		allowAnyState = false,
	): boolean {
		const current = this.store.db.prepare("SELECT state FROM mailbox_messages WHERE message_id = ?").get(messageId) as
			| { state: MailboxMessageState }
			| undefined;
		if (!current) {
			return false;
		}
		if (!allowAnyState) {
			const allowed = this.getAllowedTransitions(current.state);
			if (!allowed.has(next)) {
				return false;
			}
		}

		const result = this.store.db
			.prepare(
				`UPDATE mailbox_messages
				 SET state = ?,
				     last_error = ?,
				     updated_at = ?
				 WHERE message_id = ?`,
			)
			.run(next, lastError ?? null, this.store.now(), messageId) as { changes?: number };
		return (result.changes ?? 0) > 0;
	}

	private getAllowedTransitions(state: MailboxMessageState): Set<MailboxMessageState> {
		switch (state) {
			case "drafted":
				return new Set(["queued", "failed"]);
			case "queued":
				return new Set(["leased", "failed", "dead_letter", "delivered"]);
			case "leased":
				return new Set(["delivered", "failed", "dead_letter"]);
			case "delivered":
				return new Set(["acked", "failed", "dead_letter"]);
			case "acked":
				return new Set();
			case "failed":
				return new Set(["queued", "dead_letter"]);
			case "dead_letter":
				return new Set(["queued"]);
			default:
				return new Set();
		}
	}
}

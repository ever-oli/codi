/**
 * Formatting utilities for runtime commands.
 * Extracted from InteractiveMode to reduce class size.
 */

import type { DelegatedTaskRecord } from "../../../core/runtime/index.js";
import { theme } from "../theme/theme.js";

export function formatRuntimeEventLine(event: {
	id: string;
	type: string;
	severity: string;
	source: string;
	lane?: string;
	createdAt: number;
	payload: Record<string, unknown>;
}): string {
	const ts = new Date(event.createdAt).toLocaleTimeString();
	const lane = event.lane ? ` lane=${event.lane}` : "";
	if (event.type.startsWith("delegated_task.")) {
		const delegatedTaskId =
			typeof event.payload.delegatedTaskId === "string" ? event.payload.delegatedTaskId.slice(0, 8) : "unknown";
		const assignee = typeof event.payload.assignee === "string" ? event.payload.assignee : "unknown";
		const status = typeof event.payload.status === "string" ? event.payload.status : "unknown";
		const goal = typeof event.payload.goal === "string" ? event.payload.goal : "";
		const summary = typeof event.payload.summary === "string" ? event.payload.summary : "";
		const lastError = typeof event.payload.lastError === "string" ? event.payload.lastError : "";
		const detail = [
			goal ? `goal=${JSON.stringify(goal)}` : "",
			summary ? `summary=${JSON.stringify(summary)}` : "",
			lastError ? `error=${JSON.stringify(lastError)}` : "",
		]
			.filter((part) => part.length > 0)
			.join(" ");
		return `${theme.fg("dim", ts)} ${theme.fg("accent", event.type)} ${theme.fg("muted", `[${event.severity}]`)} ${theme.fg("dim", event.source)}${lane} task=${delegatedTaskId} assignee=${assignee} status=${status}${detail ? ` ${detail}` : ""}`;
	}
	const payloadSummary = Object.keys(event.payload).length > 0 ? ` ${JSON.stringify(event.payload)}` : "";
	const payloadText = payloadSummary.length > 120 ? `${payloadSummary.slice(0, 117)}...` : payloadSummary;
	return `${theme.fg("dim", ts)} ${theme.fg("accent", event.type)} ${theme.fg("muted", `[${event.severity}]`)} ${theme.fg("dim", event.source)}${lane}${payloadText}`;
}

export function formatQueueLine(message: {
	id: string;
	topic: string;
	state: string;
	lane: string;
	attempts: number;
	maxAttempts: number;
	availableAt: number;
	lastError?: string;
}): string {
	const available = new Date(message.availableAt).toLocaleTimeString();
	const error = message.lastError ? ` error=${message.lastError}` : "";
	return `${theme.fg("accent", message.id.slice(0, 8))} ${message.topic} ${theme.fg("muted", `[${message.state}]`)} lane=${message.lane} attempts=${message.attempts}/${message.maxAttempts} at=${available}${error}`;
}

export function formatMailboxLine(message: {
	messageId: string;
	threadId: string;
	from: string;
	to: string;
	intent: string;
	state: string;
	priority: number;
	updatedAt: number;
	lastError?: string;
}): string {
	const ts = new Date(message.updatedAt).toLocaleTimeString();
	const error = message.lastError ? ` error=${message.lastError}` : "";
	return `${theme.fg("accent", message.messageId.slice(0, 8))} thread=${message.threadId.slice(0, 8)} ${message.from} -> ${message.to} intent=${message.intent} ${theme.fg("muted", `[${message.state}]`)} p=${message.priority} ${ts}${error}`;
}

export function formatDelegatedTaskLine(task: DelegatedTaskRecord): string {
	const ts = new Date(task.updatedAt).toLocaleTimeString();
	const summary = task.summary ? ` summary=${JSON.stringify(task.summary)}` : "";
	const error = task.lastError ? ` error=${JSON.stringify(task.lastError)}` : "";
	return `${theme.fg("accent", task.delegatedTaskId.slice(0, 8))} ${task.owner} -> ${task.assignee} ${theme.fg("muted", `[${task.status}]`)} ${theme.fg("dim", ts)} goal=${JSON.stringify(task.goal)}${summary}${error}`;
}

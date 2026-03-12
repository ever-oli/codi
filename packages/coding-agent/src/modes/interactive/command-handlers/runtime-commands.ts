/**
 * Runtime-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import type { AgentSession } from "../../../core/agent-session.js";
import {
	type DelegatedTaskRecord,
	type DelegatedTaskStatus,
	LANE_NAMES,
	type LaneName,
	RUNTIME_FEATURE_FLAG_NAMES,
	type RuntimeFeatureFlagName,
	type RuntimeServices,
} from "../../../core/runtime/index.js";
import type { SettingsManager } from "../../../core/settings-manager.js";
import { theme } from "../theme/theme.js";
import {
	formatDelegatedTaskLine,
	formatMailboxLine,
	formatQueueLine,
	formatRuntimeEventLine,
} from "./runtime-formatters.js";

export interface RuntimeCommandContext {
	readonly session: AgentSession;
	readonly settingsManager: SettingsManager;
	readonly runtimeServices: RuntimeServices | undefined;
	readonly ui: { requestRender(force?: boolean): void };

	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	renderRuntimePanel(title: string, lines: string[]): void;
	getRuntimeOrWarn(flag?: string): RuntimeServices | undefined;

	// For event tail
	addTextToChat(text: string): void;
	stopEventTail(silent?: boolean): void;
	startEventTail(limit: number): void;

	// Sub-commands that delegate back to other handlers
	handleEventsCommand(text: string): Promise<void>;
	handleQueueCommand(text: string): Promise<void>;
	handleLanesCommand(text: string): Promise<void>;
	handlePackagesCommand(text: string): Promise<void>;
	handleMailboxCommand(text: string): Promise<void>;
	handleDelegatedCommand(text: string): Promise<void>;
	handleHeartbeatCommand(text: string): Promise<void>;
	handleModelsCommand(text: string): Promise<void>;
}

export async function handleEventsCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("ui.eventStreamViewer");
	if (!runtime) return;
	const argText = text.replace(/^\/events\s*/, "").trim();
	if (!argText) {
		const events = runtime.events.list({ limit: 40 });
		const lines =
			events.length > 0 ? events.map((event) => formatRuntimeEventLine(event)) : [theme.fg("dim", "No events yet.")];
		ctx.renderRuntimePanel("Runtime Events", lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "tail") {
		const mode = rest[0];
		if (mode === "off") {
			ctx.stopEventTail();
			return;
		}
		const limit = Number.parseInt(rest[1] ?? rest[0] ?? "20", 10);
		ctx.startEventTail(Number.isFinite(limit) ? limit : 20);
		return;
	}

	if (subcommand === "prune") {
		const days = Number.parseInt(rest[0] ?? "7", 10);
		if (!Number.isFinite(days) || days <= 0) {
			ctx.showWarning("Usage: /events prune <days>");
			return;
		}
		const removed = runtime.events.pruneByAge(days * 24 * 60 * 60 * 1000);
		ctx.showStatus(`Pruned ${removed} runtime events older than ${days} day(s).`);
		return;
	}

	const filters: {
		type?: string;
		lane?: LaneName;
		severity?: "debug" | "info" | "warn" | "error";
		limit: number;
	} = { limit: 50 };
	for (const token of argText.split(/\s+/)) {
		if (!token.includes("=")) continue;
		const [key, value] = token.split("=", 2);
		if (key === "type" && value) filters.type = value;
		if (key === "lane" && (LANE_NAMES as readonly string[]).includes(value)) filters.lane = value as LaneName;
		if (key === "severity" && ["debug", "info", "warn", "error"].includes(value)) {
			filters.severity = value as "debug" | "info" | "warn" | "error";
		}
		if (key === "limit" && value) {
			const parsed = Number.parseInt(value, 10);
			if (Number.isFinite(parsed)) {
				filters.limit = Math.max(1, Math.min(parsed, 200));
			}
		}
	}
	const events = runtime.events.list(filters);
	const lines =
		events.length > 0
			? events.map((event) => formatRuntimeEventLine(event))
			: [theme.fg("dim", "No matching events.")];
	ctx.renderRuntimePanel("Runtime Events", lines);
}

export async function handleQueueCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("runtime.deliveryQueue");
	if (!runtime) return;
	const argText = text.replace(/^\/queue\s*/, "").trim();
	if (!argText || argText === "list") {
		const messages = runtime.queue.list(undefined, 50);
		const lines =
			messages.length > 0
				? messages.map((message) => formatQueueLine(message))
				: [theme.fg("dim", "Queue is empty.")];
		ctx.renderRuntimePanel("Delivery Queue", lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "dead-letter") {
		const messages = runtime.queue.list("dead_letter", 50);
		const lines =
			messages.length > 0
				? messages.map((message) => formatQueueLine(message))
				: [theme.fg("dim", "No dead-letter messages.")];
		ctx.renderRuntimePanel("Queue Dead Letter", lines);
		return;
	}
	if (subcommand === "retry") {
		const id = rest[0];
		if (!id) {
			ctx.showWarning("Usage: /queue retry <messageId>");
			return;
		}
		const retried = runtime.queue.retryDeadLetter(id);
		if (!retried) {
			ctx.showWarning(`No dead-letter message found for ${id}`);
			return;
		}
		ctx.showStatus(`Queue message retried: ${id}`);
		return;
	}
	if (subcommand === "process") {
		await runtime.queue.processDue();
		ctx.showStatus("Queue processing tick complete.");
		return;
	}

	ctx.showWarning("Usage: /queue [list] | /queue dead-letter | /queue retry <id> | /queue process");
}

export async function handleLanesCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("runtime.namedLanes");
	if (!runtime) return;
	const argText = text.replace(/^\/lanes\s*/, "").trim();
	if (!argText || argText === "list") {
		const lines = runtime.lanes
			.getSnapshots()
			.map(
				(snapshot) =>
					`${theme.fg("accent", snapshot.lane)} concurrency=${snapshot.concurrency} active=${snapshot.active} queued=${snapshot.queued}`,
			);
		ctx.renderRuntimePanel("Lane Scheduler", lines.length > 0 ? lines : [theme.fg("dim", "No lane data.")]);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "set") {
		const lane = rest[0];
		const concurrency = Number.parseInt(rest[1] ?? "", 10);
		if (!(LANE_NAMES as readonly string[]).includes(lane ?? "") || !Number.isFinite(concurrency)) {
			ctx.showWarning("Usage: /lanes set <default|delegate|cron|compact|notification> <concurrency>");
			return;
		}
		ctx.settingsManager.setLaneConcurrency(lane as LaneName, concurrency);
		runtime.syncLanePoliciesFromSettings();
		ctx.showStatus(`Lane ${lane} concurrency set to ${Math.max(1, Math.floor(concurrency))}`);
		return;
	}
	if (subcommand === "run") {
		const lane = rest[0];
		const label = rest.slice(1).join(" ").trim() || "manual-task";
		if (!(LANE_NAMES as readonly string[]).includes(lane ?? "")) {
			ctx.showWarning("Usage: /lanes run <default|delegate|cron|compact|notification> [label]");
			return;
		}
		void runtime.lanes.schedule(lane as LaneName, label, async () => {
			await new Promise<void>((resolve) => setTimeout(resolve, 500));
		});
		ctx.showStatus(`Scheduled synthetic lane task on ${lane}: ${label}`);
		return;
	}

	ctx.showWarning("Usage: /lanes [list] | /lanes set <lane> <concurrency> | /lanes run <lane> [label]");
}

export async function handleMailboxCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("runtime.mailboxProtocolV2");
	if (!runtime) return;
	const argText = text.replace(/^\/mailbox\s*/, "").trim();
	const actor = ctx.session.sessionId;

	if (!argText || argText === "inbox") {
		const inbox = runtime.mailbox.listInbox(actor, 50);
		const lines = inbox.length > 0 ? inbox.map((msg) => formatMailboxLine(msg)) : [theme.fg("dim", "Inbox empty.")];
		ctx.renderRuntimePanel(`Mailbox Inbox (${actor.slice(0, 8)})`, lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "outbox") {
		const outbox = runtime.mailbox.listOutbox(actor, 50);
		const lines =
			outbox.length > 0 ? outbox.map((msg) => formatMailboxLine(msg)) : [theme.fg("dim", "Outbox empty.")];
		ctx.renderRuntimePanel(`Mailbox Outbox (${actor.slice(0, 8)})`, lines);
		return;
	}
	if (subcommand === "thread") {
		const threadId = rest[0];
		if (!threadId) {
			ctx.showWarning("Usage: /mailbox thread <threadId>");
			return;
		}
		const messages = runtime.mailbox.listThread(threadId, 100);
		const lines =
			messages.length > 0 ? messages.map((msg) => formatMailboxLine(msg)) : [theme.fg("dim", "Thread empty.")];
		ctx.renderRuntimePanel(`Mailbox Thread ${threadId}`, lines);
		return;
	}
	if (subcommand === "send") {
		const to = rest[0];
		const intent = rest[1];
		const payloadText = rest.slice(2).join(" ").trim();
		if (!to || !intent) {
			ctx.showWarning("Usage: /mailbox send <to> <intent> [payload]");
			return;
		}
		const envelope = runtime.mailbox.send({
			from: actor,
			to,
			intent,
			payload: payloadText ? { text: payloadText } : {},
			completionCriteria: "Acknowledge and include outcome summary.",
			retryPolicy: "exponential_backoff:max_5",
			delegatedTask:
				intent === "delegate"
					? {
							goal: payloadText || "Delegated task",
							summary: "Queued from interactive mailbox send.",
						}
					: undefined,
		});
		const delegatedTaskId =
			typeof envelope.payload.delegatedTaskId === "string" ? `\ndelegated=${envelope.payload.delegatedTaskId}` : "";
		ctx.showStatus(
			`Mailbox message queued: ${envelope.messageId}\nthread=${envelope.threadId}\nfrom=${envelope.from} to=${envelope.to}${delegatedTaskId}`,
		);
		return;
	}
	if (subcommand === "ack") {
		const messageId = rest[0];
		if (!messageId) {
			ctx.showWarning("Usage: /mailbox ack <messageId>");
			return;
		}
		const acked = runtime.mailbox.ack(messageId, actor);
		if (!acked) {
			ctx.showWarning(`Unable to ack ${messageId}.`);
			return;
		}
		ctx.showStatus(`Mailbox acked: ${messageId}`);
		return;
	}
	if (subcommand === "retry") {
		const messageId = rest[0];
		if (!messageId) {
			ctx.showWarning("Usage: /mailbox retry <messageId>");
			return;
		}
		const retried = runtime.mailbox.retry(messageId);
		if (!retried) {
			ctx.showWarning(`Unable to retry ${messageId}.`);
			return;
		}
		ctx.showStatus(`Mailbox retried: ${messageId}`);
		return;
	}

	ctx.showWarning(
		"Usage: /mailbox [inbox] | /mailbox outbox | /mailbox thread <threadId> | /mailbox send <to> <intent> [payload] | /mailbox ack <messageId> | /mailbox retry <messageId>",
	);
}

export async function handleDelegatedCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("runtime.mailboxProtocolV2");
	if (!runtime) return;
	const argText = text.replace(/^\/delegated\s*/, "").trim();
	const actor = ctx.session.sessionId;

	if (!argText || argText === "list") {
		const tasks = runtime.delegatedTasks.list({ parentSessionId: actor, limit: 50 });
		const lines =
			tasks.length > 0
				? tasks.map((task) => formatDelegatedTaskLine(task))
				: [theme.fg("dim", "No delegated tasks.")];
		ctx.renderRuntimePanel(`Delegated Tasks (${actor.slice(0, 8)})`, lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "thread") {
		const threadId = rest[0];
		if (!threadId) {
			ctx.showWarning("Usage: /delegated thread <threadId>");
			return;
		}
		const tasks = runtime.delegatedTasks.list({ threadId, limit: 50 });
		const lines =
			tasks.length > 0
				? tasks.map((task) => formatDelegatedTaskLine(task))
				: [theme.fg("dim", "No delegated tasks for thread.")];
		ctx.renderRuntimePanel(`Delegated Thread ${threadId}`, lines);
		return;
	}
	if (subcommand === "start" || subcommand === "block" || subcommand === "complete" || subcommand === "fail") {
		const delegatedTaskId = rest[0];
		const detail = rest.slice(1).join(" ").trim();
		if (!delegatedTaskId) {
			ctx.showWarning(
				`Usage: /delegated ${subcommand} <delegatedTaskId> ${subcommand === "fail" || subcommand === "block" ? "<details>" : "[details]"}`,
			);
			return;
		}
		let updated: DelegatedTaskRecord | undefined;
		if (subcommand === "start") {
			updated = runtime.delegatedTasks.markRunning(delegatedTaskId, detail || "Started from interactive command.");
		}
		if (subcommand === "block") {
			if (!detail) {
				ctx.showWarning("Usage: /delegated block <delegatedTaskId> <details>");
				return;
			}
			updated = runtime.delegatedTasks.markBlocked(delegatedTaskId, detail);
		}
		if (subcommand === "complete") {
			updated = runtime.delegatedTasks.markCompleted(
				delegatedTaskId,
				detail || "Completed from interactive command.",
			);
		}
		if (subcommand === "fail") {
			if (!detail) {
				ctx.showWarning("Usage: /delegated fail <delegatedTaskId> <details>");
				return;
			}
			updated = runtime.delegatedTasks.markFailed(delegatedTaskId, detail);
		}
		if (!updated) {
			ctx.showWarning(`Unable to update delegated task ${delegatedTaskId}.`);
			return;
		}
		ctx.showStatus(`Delegated task ${updated.delegatedTaskId}: ${updated.status}`);
		return;
	}

	const filters: { status?: DelegatedTaskStatus; limit: number } = { limit: 50 };
	for (const token of argText.split(/\s+/)) {
		if (!token.includes("=")) continue;
		const [key, value] = token.split("=", 2);
		if (
			key === "status" &&
			(value === "queued" ||
				value === "running" ||
				value === "blocked" ||
				value === "completed" ||
				value === "failed")
		) {
			filters.status = value;
		}
		if (key === "limit" && value) {
			const parsed = Number.parseInt(value, 10);
			if (Number.isFinite(parsed)) {
				filters.limit = Math.max(1, Math.min(parsed, 200));
			}
		}
	}
	const tasks = runtime.delegatedTasks.list({
		parentSessionId: actor,
		status: filters.status,
		limit: filters.limit,
	});
	const lines =
		tasks.length > 0
			? tasks.map((task) => formatDelegatedTaskLine(task))
			: [theme.fg("dim", "No matching delegated tasks.")];
	ctx.renderRuntimePanel(`Delegated Tasks (${actor.slice(0, 8)})`, lines);
}

export async function handleHeartbeatCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("runtime.heartbeatCronCore");
	if (!runtime) return;
	const argText = text.replace(/^\/heartbeat\s*/, "").trim();
	if (!argText || argText === "status") {
		const status = runtime.heartbeat.getStatus();
		const jobs = runtime.heartbeat.listJobs(10);
		const lines = [
			`running=${status.running ? "yes" : "no"} intervalMs=${status.intervalMs} ticks=${status.tickCount} jobsEnabled=${status.jobsEnabled}`,
			`lastTick=${status.lastTickAt ? new Date(status.lastTickAt).toISOString() : "never"}`,
			"",
			theme.bold("Cron jobs:"),
		];
		if (jobs.length === 0) {
			lines.push(theme.fg("dim", "  none"));
		} else {
			for (const job of jobs) {
				lines.push(
					`  ${theme.fg("accent", job.id.slice(0, 8))} ${job.name} every ${job.intervalSeconds}s ${job.enabled ? "enabled" : "paused"} next=${new Date(job.nextRunAt).toLocaleTimeString()}`,
				);
			}
		}
		ctx.renderRuntimePanel("Heartbeat + Cron", lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "tick") {
		await runtime.heartbeat.tick();
		ctx.showStatus("Heartbeat tick executed.");
		return;
	}
	if (subcommand === "add") {
		const name = rest[0];
		const intervalSeconds = Number.parseInt(rest[1] ?? "", 10);
		const intent = rest[2];
		const payloadText = rest.slice(3).join(" ").trim();
		if (!name || !Number.isFinite(intervalSeconds) || !intent) {
			ctx.showWarning("Usage: /heartbeat add <name> <intervalSeconds> <intent> [payload]");
			return;
		}
		const job = runtime.heartbeat.addJob({
			name,
			intervalSeconds,
			intent,
			payload: payloadText ? { text: payloadText } : {},
		});
		ctx.showStatus(`Cron job added: ${job.id} (${job.name})`);
		return;
	}
	if (subcommand === "pause" || subcommand === "resume") {
		const jobId = rest[0];
		if (!jobId) {
			ctx.showWarning(`Usage: /heartbeat ${subcommand} <jobId>`);
			return;
		}
		const ok = runtime.heartbeat.setJobEnabled(jobId, subcommand === "resume");
		if (!ok) {
			ctx.showWarning(`Unknown cron job: ${jobId}`);
			return;
		}
		ctx.showStatus(`Cron job ${subcommand}d: ${jobId}`);
		return;
	}
	if (subcommand === "remove") {
		const jobId = rest[0];
		if (!jobId) {
			ctx.showWarning("Usage: /heartbeat remove <jobId>");
			return;
		}
		const ok = runtime.heartbeat.removeJob(jobId);
		if (!ok) {
			ctx.showWarning(`Unknown cron job: ${jobId}`);
			return;
		}
		ctx.showStatus(`Cron job removed: ${jobId}`);
		return;
	}
	if (subcommand === "list") {
		const jobs = runtime.heartbeat.listJobs(100);
		const lines =
			jobs.length > 0
				? jobs.map(
						(job) =>
							`${theme.fg("accent", job.id.slice(0, 8))} ${job.name} every ${job.intervalSeconds}s ${job.enabled ? "enabled" : "paused"} next=${new Date(job.nextRunAt).toISOString()}`,
					)
				: [theme.fg("dim", "No cron jobs.")];
		ctx.renderRuntimePanel("Heartbeat Jobs", lines);
		return;
	}

	ctx.showWarning(
		"Usage: /heartbeat [status] | /heartbeat tick | /heartbeat list | /heartbeat add <name> <intervalSeconds> <intent> [payload] | /heartbeat pause <jobId> | /heartbeat resume <jobId> | /heartbeat remove <jobId>",
	);
}

export async function handleOpsCommand(ctx: RuntimeCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn();
	if (!runtime) return;

	const argText = text.replace(/^\/ops\s*/, "").trim();
	if (!argText) {
		const flags = ctx.settingsManager.getRuntimeFeatureFlags();
		const queueQueued = runtime.queue.list("queued", 200).length;
		const queueLeased = runtime.queue.list("leased", 200).length;
		const queueDead = runtime.queue.list("dead_letter", 200).length;
		const actor = ctx.session.sessionId;
		const heartbeat = runtime.heartbeat.getStatus();
		const laneSnapshots = runtime.lanes
			.getSnapshots()
			.map((snapshot) => `${snapshot.lane}:${snapshot.active}/${snapshot.queued} c=${snapshot.concurrency}`)
			.join(" | ");

		const lines = [
			theme.bold("Runtime"),
			`queue queued=${queueQueued} leased=${queueLeased} dead=${queueDead}`,
			`lanes ${laneSnapshots || "none"}`,
			`mailbox inbox=${runtime.mailbox.listInbox(actor, 200).length} outbox=${runtime.mailbox.listOutbox(actor, 200).length}`,
			`delegated tasks=${runtime.delegatedTasks.list({ parentSessionId: actor, limit: 200 }).length}`,
			`heartbeat running=${heartbeat.running ? "yes" : "no"} jobs=${heartbeat.jobsEnabled} ticks=${heartbeat.tickCount}`,
			"",
			theme.bold("Flags"),
			...RUNTIME_FEATURE_FLAG_NAMES.map((flag) => `${flag}=${flags[flag] ? "on" : "off"}`),
			"",
			"Use /ops <events|queue|lanes|packages|mailbox|delegated|heartbeat|models roles|flags>",
		];
		ctx.renderRuntimePanel("Ops", lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "events") {
		await ctx.handleEventsCommand(`/events ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "queue") {
		await ctx.handleQueueCommand(`/queue ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "lanes") {
		await ctx.handleLanesCommand(`/lanes ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "packages") {
		await ctx.handlePackagesCommand(`/packages ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "mailbox") {
		await ctx.handleMailboxCommand(`/mailbox ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "delegated") {
		await ctx.handleDelegatedCommand(`/delegated ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "heartbeat") {
		await ctx.handleHeartbeatCommand(`/heartbeat ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "models") {
		await ctx.handleModelsCommand(`/models ${rest.join(" ").trim()}`.trim());
		return;
	}
	if (subcommand === "flags") {
		const action = rest[0];
		const flagName = rest[1] as RuntimeFeatureFlagName | undefined;
		if (!action || action === "list") {
			const flags = ctx.settingsManager.getRuntimeFeatureFlags();
			const lines = RUNTIME_FEATURE_FLAG_NAMES.map((flag) => `${flag}=${flags[flag] ? "on" : "off"}`);
			lines.push("");
			lines.push("Usage: /ops flags [list] | /ops flags enable <flag> | /ops flags disable <flag>");
			ctx.renderRuntimePanel("Runtime Flags", lines);
			return;
		}
		if ((action === "enable" || action === "disable") && flagName) {
			if (!(RUNTIME_FEATURE_FLAG_NAMES as readonly string[]).includes(flagName)) {
				ctx.showWarning(`Unknown flag "${flagName}"`);
				return;
			}
			const enabled = action === "enable";
			ctx.settingsManager.setRuntimeFeatureFlag(flagName, enabled);
			if (flagName === "runtime.heartbeatCronCore") {
				if (enabled) runtime.heartbeat.start();
				else runtime.heartbeat.stop();
			}
			if (flagName === "ui.eventStreamViewer" && !enabled) {
				ctx.stopEventTail(true);
			}
			ctx.showStatus(`${flagName} ${enabled ? "enabled" : "disabled"}`);
			return;
		}
		ctx.showWarning("Usage: /ops flags [list] | /ops flags enable <flag> | /ops flags disable <flag>");
		return;
	}

	ctx.showWarning("Usage: /ops [events|queue|lanes|packages|mailbox|delegated|heartbeat|models roles|flags]");
}

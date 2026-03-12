import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { StringEnum, Text, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

interface CronJob {
	id: number;
	name?: string;
	task: string;
	everyMinutes: number;
	enabled: boolean;
	createdAt: number;
	lastRunAt?: number;
}

interface CronStore {
	version: 1;
	nextId: number;
	jobs: CronJob[];
}

type CronAction = "list" | "add" | "remove" | "pause" | "resume" | "touch";

interface CronToolDetails {
	action: CronAction;
	jobs: CronJob[];
	dueJobs: CronJob[];
	now: number;
	error?: string;
}

const MIN_INTERVAL_MINUTES = 5;
const MAX_INTERVAL_MINUTES = 60 * 24 * 30;

const CRON_PARAMS = Type.Object({
	action: StringEnum(["list", "add", "remove", "pause", "resume", "touch"] as const),
	id: Type.Optional(Type.Number({ description: "Job ID for remove/pause/resume/touch." })),
	task: Type.Optional(Type.String({ description: "Task text for add." })),
	name: Type.Optional(Type.String({ description: "Optional short name for add." })),
	everyMinutes: Type.Optional(
		Type.Number({
			description: `Interval in minutes for add. Range ${MIN_INTERVAL_MINUTES}-${MAX_INTERVAL_MINUTES}.`,
		}),
	),
	dueOnly: Type.Optional(Type.Boolean({ description: "When list action, only include due jobs." })),
});

function resolveAgentDir(): string {
	const explicit = process.env.PI_CODING_AGENT_DIR?.trim();
	if (explicit) {
		if (explicit === "~") return os.homedir();
		if (explicit.startsWith("~/")) return path.join(os.homedir(), explicit.slice(2));
		return explicit;
	}
	return path.join(os.homedir(), ".pi", "agent");
}

function resolveCronFile(): string {
	const explicit = process.env.PI_CRON_JOBS_FILE?.trim();
	if (explicit) {
		if (explicit === "~") return os.homedir();
		if (explicit.startsWith("~/")) return path.join(os.homedir(), explicit.slice(2));
		return explicit;
	}
	return path.join(resolveAgentDir(), "cron", "jobs.json");
}

function defaultStore(): CronStore {
	return { version: 1, nextId: 1, jobs: [] };
}

function loadStore(filePath: string): CronStore {
	if (!fs.existsSync(filePath)) return defaultStore();
	try {
		const raw = fs.readFileSync(filePath, "utf8");
		const parsed = JSON.parse(raw) as Partial<CronStore>;
		const jobs = Array.isArray(parsed.jobs)
			? parsed.jobs.filter((job): job is CronJob => {
					return (
						typeof job.id === "number" &&
						typeof job.task === "string" &&
						typeof job.everyMinutes === "number" &&
						typeof job.enabled === "boolean" &&
						typeof job.createdAt === "number"
					);
				})
			: [];
		const nextId = typeof parsed.nextId === "number" && parsed.nextId > 0 ? parsed.nextId : jobs.reduce((max, job) => Math.max(max, job.id), 0) + 1;
		return { version: 1, nextId, jobs };
	} catch {
		return defaultStore();
	}
}

function saveStore(filePath: string, store: CronStore): void {
	const dir = path.dirname(filePath);
	fs.mkdirSync(dir, { recursive: true });
	const tmpPath = path.join(dir, `.${path.basename(filePath)}.${process.pid}.${Date.now()}.tmp`);
	const text = JSON.stringify(store, null, 2);
	fs.writeFileSync(tmpPath, text, "utf8");
	fs.renameSync(tmpPath, filePath);
}

function isDue(job: CronJob, now: number): boolean {
	if (!job.enabled) return false;
	const baseline = job.lastRunAt ?? job.createdAt;
	const elapsedMinutes = Math.floor((now - baseline) / 60000);
	return elapsedMinutes >= job.everyMinutes;
}

function describeInterval(minutes: number): string {
	if (minutes % (24 * 60) === 0) {
		const days = minutes / (24 * 60);
		return `${days}d`;
	}
	if (minutes % 60 === 0) {
		const hours = minutes / 60;
		return `${hours}h`;
	}
	return `${minutes}m`;
}

function summarizeJob(job: CronJob, now: number): string {
	const due = isDue(job, now);
	const enabled = job.enabled ? "on" : "off";
	const every = describeInterval(job.everyMinutes);
	const title = job.name ? `${job.name}: ${job.task}` : job.task;
	return `#${job.id} [${enabled}] every ${every}${due ? " (due)" : ""} — ${title}`;
}

function buildDetails(action: CronAction, store: CronStore, now: number, error?: string): CronToolDetails {
	const jobs = [...store.jobs].sort((a, b) => a.id - b.id);
	const dueJobs = jobs.filter((job) => isDue(job, now));
	return { action, jobs, dueJobs, now, error };
}

function validateInterval(value: number | undefined): string | undefined {
	if (value === undefined || Number.isNaN(value)) return "everyMinutes is required for add.";
	if (!Number.isInteger(value)) return "everyMinutes must be an integer.";
	if (value < MIN_INTERVAL_MINUTES || value > MAX_INTERVAL_MINUTES) {
		return `everyMinutes must be between ${MIN_INTERVAL_MINUTES} and ${MAX_INTERVAL_MINUTES}.`;
	}
	return undefined;
}

function formatList(jobs: CronJob[], now: number): string {
	if (jobs.length === 0) return "No cron jobs.";
	return jobs.map((job) => summarizeJob(job, now)).join("\n");
}

export default defineExtension((pi) => {
	const cronFile = resolveCronFile();
	let store = loadStore(cronFile);

	const reloadStore = () => {
		store = loadStore(cronFile);
	};

	pi.on("session_start", reloadStore);
	pi.on("session_switch", reloadStore);
	pi.on("session_fork", reloadStore);
	pi.on("session_tree", reloadStore);

	pi.registerTool(
		defineTool<typeof CRON_PARAMS, CronToolDetails>({
			name: "cron",
			label: "Cron",
			description:
				"Manage durable recurring tasks by interval in minutes. Actions: list, add, remove, pause, resume, touch.",
			promptSnippet:
				"Manage recurring reminders/work items with durable interval schedules (list/add/remove/pause/resume/touch).",
			parameters: CRON_PARAMS,
			async execute(_toolCallId, params) {
				const now = Date.now();
				const action = params.action;
				reloadStore();

				if (action === "list") {
					const jobs = params.dueOnly ? store.jobs.filter((job) => isDue(job, now)) : store.jobs;
					const details = buildDetails(action, { ...store, jobs }, now);
					return {
						content: [{ type: "text", text: formatList(jobs, now) }],
						details,
					};
				}

				if (action === "add") {
					const task = params.task?.trim();
					if (!task) {
						const details = buildDetails(action, store, now, "task is required for add.");
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const intervalError = validateInterval(params.everyMinutes);
					if (intervalError) {
						const details = buildDetails(action, store, now, intervalError);
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const job: CronJob = {
						id: store.nextId++,
						name: params.name?.trim() || undefined,
						task,
						everyMinutes: params.everyMinutes as number,
						enabled: true,
						createdAt: now,
					};
					store.jobs.push(job);
					saveStore(cronFile, store);
					const details = buildDetails(action, store, now);
					return {
						content: [{ type: "text", text: `Added ${summarizeJob(job, now)}` }],
						details,
					};
				}

				const id = params.id;
				if (id === undefined || !Number.isInteger(id)) {
					const details = buildDetails(action, store, now, "id is required for this action.");
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}

				const index = store.jobs.findIndex((job) => job.id === id);
				if (index < 0) {
					const details = buildDetails(action, store, now, `Job #${id} not found.`);
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}

				const job = store.jobs[index];
				if (action === "remove") {
					store.jobs.splice(index, 1);
					saveStore(cronFile, store);
					const details = buildDetails(action, store, now);
					return { content: [{ type: "text", text: `Removed job #${id}.` }], details };
				}
				if (action === "pause") {
					job.enabled = false;
					saveStore(cronFile, store);
					const details = buildDetails(action, store, now);
					return { content: [{ type: "text", text: `Paused job #${id}.` }], details };
				}
				if (action === "resume") {
					job.enabled = true;
					saveStore(cronFile, store);
					const details = buildDetails(action, store, now);
					return { content: [{ type: "text", text: `Resumed job #${id}.` }], details };
				}

				job.lastRunAt = now;
				saveStore(cronFile, store);
				const details = buildDetails(action, store, now);
				return { content: [{ type: "text", text: `Marked job #${id} as run now.` }], details };
			},
			renderCall(args, theme) {
				let text = theme.fg("toolTitle", theme.bold("cron ")) + theme.fg("muted", args.action);
				if (typeof args.id === "number") text += ` ${theme.fg("accent", `#${args.id}`)}`;
				if (typeof args.everyMinutes === "number") text += ` ${theme.fg("dim", `every=${args.everyMinutes}m`)}`;
				if (args.task) text += ` ${theme.fg("dim", `"${args.task.slice(0, 48)}${args.task.length > 48 ? "..." : ""}"`)}`;
				return new Text(text, 0, 0);
			},
			renderResult(result, _options, theme) {
				const details = result.details as CronToolDetails | undefined;
				if (!details) {
					const block = result.content[0];
					return new Text(block?.type === "text" ? block.text : "", 0, 0);
				}
				if (details.error) {
					return new Text(theme.fg("error", `Error: ${details.error}`), 0, 0);
				}
				const lines = [
					`${theme.fg("success", "✓")} ${theme.fg("muted", `${details.jobs.length} job(s), ${details.dueJobs.length} due`)}`,
				];
				for (const job of details.jobs.slice(0, 5)) {
					lines.push(theme.fg("dim", summarizeJob(job, details.now)));
				}
				if (details.jobs.length > 5) {
					lines.push(theme.fg("dim", `... ${details.jobs.length - 5} more`));
				}
				return new Text(lines.join("\n"), 0, 0);
			},
		}),
	);
});

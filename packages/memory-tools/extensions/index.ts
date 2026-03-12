import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { StringEnum, Text, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

const ENTRY_DELIMITER = "\n§\n";
const DEFAULT_MEMORY_CHAR_LIMIT = 2200;
const DEFAULT_USER_CHAR_LIMIT = 1375;
const SNAPSHOT_HEADER_LINE = "═".repeat(46);

type MemoryTarget = "memory" | "user";
type MemoryAction = "read" | "add" | "replace" | "remove";

interface ThreatPattern {
	regex: RegExp;
	id: string;
}

interface MemorySnapshot {
	memoryBlock: string;
	userBlock: string;
}

interface MemoryState {
	memoryEntries: string[];
	userEntries: string[];
}

interface MemoryOperationResult {
	success: boolean;
	target: MemoryTarget;
	action: MemoryAction;
	entries: string[];
	usage: string;
	entryCount: number;
	message?: string;
	error?: string;
	matches?: string[];
}

const MEMORY_TOOL_PARAMS = Type.Object({
	action: StringEnum(["read", "add", "replace", "remove"] as const),
	target: Type.Optional(StringEnum(["memory", "user"] as const)),
	content: Type.Optional(
		Type.String({
			description: "Entry content. Required for add and replace.",
		}),
	),
	oldText: Type.Optional(
		Type.String({
			description: "Short unique substring to identify entry for replace/remove.",
		}),
	),
	old_text: Type.Optional(
		Type.String({
			description: "Alias of oldText for compatibility.",
		}),
	),
});

const THREAT_PATTERNS: ThreatPattern[] = [
	{ regex: /ignore\s+(previous|all|above|prior)\s+instructions/i, id: "prompt_injection" },
	{ regex: /you\s+are\s+now\s+/i, id: "role_hijack" },
	{ regex: /do\s+not\s+tell\s+the\s+user/i, id: "deception_hide" },
	{ regex: /system\s+prompt\s+override/i, id: "sys_prompt_override" },
	{ regex: /disregard\s+(your|all|any)\s+(instructions|rules|guidelines)/i, id: "disregard_rules" },
	{
		regex: /act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|limits|rules)/i,
		id: "bypass_restrictions",
	},
	{
		regex: /curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)/i,
		id: "exfil_curl",
	},
	{
		regex: /wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)/i,
		id: "exfil_wget",
	},
	{
		regex: /cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)/i,
		id: "read_secrets",
	},
	{ regex: /authorized_keys/i, id: "ssh_backdoor" },
	{ regex: /\$HOME\/\.ssh|~\/\.ssh/i, id: "ssh_access" },
	{ regex: /\$HOME\/\.hermes\/\.env|~\/\.hermes\/\.env/i, id: "hermes_env" },
];

const INVISIBLE_CHARS = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"];

function resolveAgentDir(): string {
	const explicit = process.env.PI_CODING_AGENT_DIR?.trim();
	if (explicit) {
		if (explicit === "~") {
			return os.homedir();
		}
		if (explicit.startsWith("~/")) {
			return path.join(os.homedir(), explicit.slice(2));
		}
		return explicit;
	}
	return path.join(os.homedir(), ".pi", "agent");
}

function resolveMemoryDir(): string {
	const explicit = process.env.PI_MEMORY_DIR?.trim();
	if (explicit) {
		if (explicit === "~") {
			return os.homedir();
		}
		if (explicit.startsWith("~/")) {
			return path.join(os.homedir(), explicit.slice(2));
		}
		return explicit;
	}
	return path.join(resolveAgentDir(), "memories");
}

function getMemoryLimit(target: MemoryTarget): number {
	const envKey = target === "user" ? process.env.PI_MEMORY_USER_CHAR_LIMIT : process.env.PI_MEMORY_CHAR_LIMIT;
	const parsed = envKey ? Number.parseInt(envKey, 10) : Number.NaN;
	if (Number.isFinite(parsed) && parsed > 0) {
		return parsed;
	}
	return target === "user" ? DEFAULT_USER_CHAR_LIMIT : DEFAULT_MEMORY_CHAR_LIMIT;
}

function dedupeEntries(entries: string[]): string[] {
	const seen = new Set<string>();
	const deduped: string[] = [];
	for (const entry of entries) {
		if (seen.has(entry)) {
			continue;
		}
		seen.add(entry);
		deduped.push(entry);
	}
	return deduped;
}

function readEntries(filePath: string): string[] {
	if (!fs.existsSync(filePath)) {
		return [];
	}
	let raw = "";
	try {
		raw = fs.readFileSync(filePath, "utf8");
	} catch {
		return [];
	}
	if (!raw.trim()) {
		return [];
	}
	return raw
		.split(ENTRY_DELIMITER)
		.map((entry) => entry.trim())
		.filter((entry) => entry.length > 0);
}

function writeEntriesAtomic(filePath: string, entries: string[]): void {
	const dir = path.dirname(filePath);
	fs.mkdirSync(dir, { recursive: true });
	const content = entries.length > 0 ? entries.join(ENTRY_DELIMITER) : "";
	const tmpPath = path.join(dir, `.${path.basename(filePath)}.${process.pid}.${Date.now()}.tmp`);
	try {
		fs.writeFileSync(tmpPath, content, "utf8");
		fs.renameSync(tmpPath, filePath);
	} catch (error) {
		if (fs.existsSync(tmpPath)) {
			try {
				fs.unlinkSync(tmpPath);
			} catch {
				// no-op
			}
		}
		throw error;
	}
}

function getEntries(state: MemoryState, target: MemoryTarget): string[] {
	return target === "user" ? state.userEntries : state.memoryEntries;
}

function setEntries(state: MemoryState, target: MemoryTarget, entries: string[]): void {
	if (target === "user") {
		state.userEntries = entries;
	} else {
		state.memoryEntries = entries;
	}
}

function getStoreFile(memoryDir: string, target: MemoryTarget): string {
	return path.join(memoryDir, target === "user" ? "USER.md" : "MEMORY.md");
}

function getCharCount(entries: string[]): number {
	if (entries.length === 0) {
		return 0;
	}
	return entries.join(ENTRY_DELIMITER).length;
}

function getUsage(entries: string[], target: MemoryTarget): string {
	const current = getCharCount(entries);
	const limit = getMemoryLimit(target);
	const percent = limit > 0 ? Math.min(100, Math.floor((current / limit) * 100)) : 0;
	return `${percent}% — ${current.toLocaleString()}/${limit.toLocaleString()} chars`;
}

function renderSnapshotBlock(target: MemoryTarget, entries: string[]): string {
	if (entries.length === 0) {
		return "";
	}
	const current = getCharCount(entries);
	const limit = getMemoryLimit(target);
	const percent = limit > 0 ? Math.min(100, Math.floor((current / limit) * 100)) : 0;
	const header =
		target === "user"
			? `USER PROFILE (who the user is) [${percent}% — ${current.toLocaleString()}/${limit.toLocaleString()} chars]`
			: `MEMORY (your personal notes) [${percent}% — ${current.toLocaleString()}/${limit.toLocaleString()} chars]`;
	return `${SNAPSHOT_HEADER_LINE}\n${header}\n${SNAPSHOT_HEADER_LINE}\n${entries.join(ENTRY_DELIMITER)}`;
}

function loadStateFromDisk(memoryDir: string): MemoryState {
	fs.mkdirSync(memoryDir, { recursive: true });
	return {
		memoryEntries: dedupeEntries(readEntries(getStoreFile(memoryDir, "memory"))),
		userEntries: dedupeEntries(readEntries(getStoreFile(memoryDir, "user"))),
	};
}

function captureSnapshot(state: MemoryState): MemorySnapshot {
	return {
		memoryBlock: renderSnapshotBlock("memory", state.memoryEntries),
		userBlock: renderSnapshotBlock("user", state.userEntries),
	};
}

function buildSnapshotPrompt(snapshot: MemorySnapshot): string | undefined {
	const blocks = [snapshot.memoryBlock, snapshot.userBlock].filter((block) => block.length > 0);
	if (blocks.length === 0) {
		return undefined;
	}
	return blocks.join("\n\n");
}

function scanContent(content: string): string | undefined {
	for (const char of INVISIBLE_CHARS) {
		if (content.includes(char)) {
			return `Blocked: content contains invisible unicode character U+${char.codePointAt(0)?.toString(16).toUpperCase().padStart(4, "0")} (possible injection).`;
		}
	}
	for (const pattern of THREAT_PATTERNS) {
		if (pattern.regex.test(content)) {
			return `Blocked: content matches threat pattern "${pattern.id}". Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads.`;
		}
	}
	return undefined;
}

function buildSuccessResult(
	state: MemoryState,
	target: MemoryTarget,
	action: MemoryAction,
	message?: string,
): MemoryOperationResult {
	const entries = [...getEntries(state, target)];
	return {
		success: true,
		target,
		action,
		entries,
		usage: getUsage(entries, target),
		entryCount: entries.length,
		message,
	};
}

function buildErrorResult(
	state: MemoryState,
	target: MemoryTarget,
	action: MemoryAction,
	error: string,
	matches?: string[],
): MemoryOperationResult {
	const entries = [...getEntries(state, target)];
	return {
		success: false,
		target,
		action,
		entries,
		usage: getUsage(entries, target),
		entryCount: entries.length,
		error,
		matches,
	};
}

function persistTarget(memoryDir: string, state: MemoryState, target: MemoryTarget): void {
	writeEntriesAtomic(getStoreFile(memoryDir, target), getEntries(state, target));
}

function matchEntry(entries: string[], needle: string): { index?: number; error?: string; matches?: string[] } {
	const trimmedNeedle = needle.trim();
	if (!trimmedNeedle) {
		return { error: "oldText cannot be empty." };
	}
	const matches = entries.map((entry, index) => ({ entry, index })).filter((candidate) => candidate.entry.includes(trimmedNeedle));
	if (matches.length === 0) {
		return { error: `No entry matched "${trimmedNeedle}".` };
	}
	if (matches.length > 1) {
		const uniqueEntries = new Set(matches.map((match) => match.entry));
		if (uniqueEntries.size > 1) {
			return {
				error: `Multiple entries matched "${trimmedNeedle}". Be more specific.`,
				matches: matches.map((match) => (match.entry.length > 80 ? `${match.entry.slice(0, 80)}...` : match.entry)),
			};
		}
	}
	return { index: matches[0]?.index };
}

function getOldText(params: { oldText?: string; old_text?: string }): string | undefined {
	return params.oldText ?? params.old_text;
}

export default defineExtension((pi) => {
	const memoryDir = resolveMemoryDir();
	let liveState: MemoryState = { memoryEntries: [], userEntries: [] };
	let frozenSnapshot: MemorySnapshot = { memoryBlock: "", userBlock: "" };

	const refreshSnapshot = () => {
		liveState = loadStateFromDisk(memoryDir);
		frozenSnapshot = captureSnapshot(liveState);
	};

	pi.on("session_start", () => {
		refreshSnapshot();
	});

	pi.on("session_switch", () => {
		refreshSnapshot();
	});

	pi.on("session_fork", () => {
		refreshSnapshot();
	});

	pi.on("session_tree", () => {
		refreshSnapshot();
	});

	pi.on("before_agent_start", (event) => {
		const snapshotPrompt = buildSnapshotPrompt(frozenSnapshot);
		if (!snapshotPrompt) {
			return;
		}
		return {
			systemPrompt: `${event.systemPrompt}\n\n${snapshotPrompt}`,
		};
	});

	pi.registerTool(
		defineTool<typeof MEMORY_TOOL_PARAMS, MemoryOperationResult>({
			name: "memory",
			label: "Memory",
			description:
				"Persistent curated memory across sessions. Use target=\"user\" for user preferences/profile and target=\"memory\" for environment/project notes. Save high-value facts proactively. Actions: read, add, replace, remove.",
			promptSnippet:
				"Persist high-value user preferences and project/environment notes across sessions with read/add/replace/remove.",
			promptGuidelines: [
				"Use memory proactively for durable facts worth carrying across sessions.",
				"Use target=user for user profile/preferences and target=memory for project/environment notes.",
				"Keep entries concise and specific; replace/remove stale entries when near capacity.",
			],
			parameters: MEMORY_TOOL_PARAMS,
			async execute(_toolCallId, params) {
				const action = params.action;
				const target: MemoryTarget = params.target ?? "memory";
				const entries = [...getEntries(liveState, target)];

				if (action === "read") {
					return {
						content: [
							{
								type: "text",
								text:
									entries.length > 0
										? entries.map((entry, index) => `${index + 1}. ${entry}`).join("\n\n")
										: `No ${target} entries yet.`,
							},
						],
						details: buildSuccessResult(liveState, target, action, "Entries read."),
					};
				}

				if (action === "add") {
					const content = params.content?.trim();
					if (!content) {
						const details = buildErrorResult(liveState, target, action, "content is required for add.");
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const scanError = scanContent(content);
					if (scanError) {
						const details = buildErrorResult(liveState, target, action, scanError);
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					if (entries.includes(content)) {
						const details = buildSuccessResult(liveState, target, action, "Entry already exists (no duplicate added).");
						return { content: [{ type: "text", text: details.message ?? "Entry already exists." }], details };
					}
					const nextEntries = [...entries, content];
					const nextTotal = getCharCount(nextEntries);
					const limit = getMemoryLimit(target);
					if (nextTotal > limit) {
						const details = buildErrorResult(
							liveState,
							target,
							action,
							`Memory at ${getCharCount(entries).toLocaleString()}/${limit.toLocaleString()} chars. Adding this entry (${content.length} chars) would exceed the limit. Replace or remove existing entries first.`,
						);
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					setEntries(liveState, target, nextEntries);
					persistTarget(memoryDir, liveState, target);
					const details = buildSuccessResult(liveState, target, action, "Entry added.");
					return { content: [{ type: "text", text: details.message ?? "Entry added." }], details };
				}

				if (action === "replace") {
					const oldText = getOldText(params);
					const content = params.content?.trim();
					if (!oldText?.trim()) {
						const details = buildErrorResult(liveState, target, action, "oldText is required for replace.");
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					if (!content) {
						const details = buildErrorResult(liveState, target, action, "content is required for replace.");
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const scanError = scanContent(content);
					if (scanError) {
						const details = buildErrorResult(liveState, target, action, scanError);
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const match = matchEntry(entries, oldText);
					if (match.error) {
						const details = buildErrorResult(liveState, target, action, match.error, match.matches);
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const index = match.index;
					if (index === undefined) {
						const details = buildErrorResult(liveState, target, action, "No entry matched.");
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const nextEntries = [...entries];
					nextEntries[index] = content;
					const nextTotal = getCharCount(nextEntries);
					const limit = getMemoryLimit(target);
					if (nextTotal > limit) {
						const details = buildErrorResult(
							liveState,
							target,
							action,
							`Replacement would put memory at ${nextTotal.toLocaleString()}/${limit.toLocaleString()} chars. Shorten the new content or remove entries first.`,
						);
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					setEntries(liveState, target, nextEntries);
					persistTarget(memoryDir, liveState, target);
					const details = buildSuccessResult(liveState, target, action, "Entry replaced.");
					return { content: [{ type: "text", text: details.message ?? "Entry replaced." }], details };
				}

				const oldText = getOldText(params);
				if (!oldText?.trim()) {
					const details = buildErrorResult(liveState, target, action, "oldText is required for remove.");
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				const match = matchEntry(entries, oldText);
				if (match.error) {
					const details = buildErrorResult(liveState, target, action, match.error, match.matches);
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				const index = match.index;
				if (index === undefined) {
					const details = buildErrorResult(liveState, target, action, "No entry matched.");
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				const nextEntries = [...entries];
				nextEntries.splice(index, 1);
				setEntries(liveState, target, nextEntries);
				persistTarget(memoryDir, liveState, target);
				const details = buildSuccessResult(liveState, target, action, "Entry removed.");
				return { content: [{ type: "text", text: details.message ?? "Entry removed." }], details };
			},
			renderCall(args, theme) {
				const target = args.target ?? "memory";
				const oldText = args.oldText ?? args.old_text;
				let suffix = `${theme.fg("muted", `${args.action}`)} ${theme.fg("accent", target)}`;
				if (args.content) {
					const trimmed = args.content.length > 72 ? `${args.content.slice(0, 72)}...` : args.content;
					suffix += ` ${theme.fg("dim", `"${trimmed}"`)}`;
				}
				if (oldText) {
					const needle = oldText.length > 32 ? `${oldText.slice(0, 32)}...` : oldText;
					suffix += ` ${theme.fg("dim", `match="${needle}"`)}`;
				}
				return new Text(theme.fg("toolTitle", theme.bold("memory ")) + suffix, 0, 0);
			},
			renderResult(result, _options, theme) {
				const details = result.details as MemoryOperationResult | undefined;
				if (!details) {
					const text = result.content[0];
					return new Text(text?.type === "text" ? text.text : "", 0, 0);
				}
				if (!details.success) {
					const lines = [theme.fg("error", `Error: ${details.error ?? "Unknown error"}`)];
					if (details.matches && details.matches.length > 0) {
						lines.push(theme.fg("muted", "Matches:"));
						for (const match of details.matches.slice(0, 5)) {
							lines.push(theme.fg("dim", `- ${match}`));
						}
					}
					lines.push(theme.fg("dim", `Usage: ${details.usage}`));
					return new Text(lines.join("\n"), 0, 0);
				}
				const summary = details.message ?? "Memory updated.";
				return new Text(
					`${theme.fg("success", "✓")} ${theme.fg("muted", summary)}\n${theme.fg("dim", `${details.target}: ${details.entryCount} entries | ${details.usage}`)}`,
					0,
					0,
				);
			},
		}),
	);
});

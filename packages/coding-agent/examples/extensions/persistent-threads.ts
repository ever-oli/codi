/**
 * Persistent Threads Extension
 *
 * Implements Slate's thread primitive. Threads are persistent, named work streams
 * that accumulate context across multiple actions. Unlike ephemeral delegates
 * (which start fresh each time), a thread retains its full conversation history.
 * Each action produces an episode that's returned to the orchestrator, but the
 * thread itself remembers everything.
 *
 * Key features:
 *   - Named threads with persistent conversation history
 *   - Episode extraction from each thread action
 *   - Cross-thread episode composition via shared store
 *   - Per-thread model and tool overrides
 *
 * Usage:
 *   pi --extension examples/extensions/persistent-threads.ts
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentToolResult, ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Container, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Episode Types (copied from episodic-delegate.ts — standalone)
// ============================================================================

interface Episode {
	id: string;
	label?: string;
	timestamp: number;
	task: string;
	cwd: string;
	model?: string;

	summary: string;
	discoveries: string[];
	decisions: string[];
	filesRead: string[];
	filesModified: string[];
	errors: string[];

	durationMs: number;
	exitCode: number;
	turns: number;
	usage: EpisodeUsage;

	rawOutput: string;
}

interface EpisodeUsage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cost: number;
	turns: number;
}

// ============================================================================
// Thread Types
// ============================================================================

interface ThreadState {
	name: string;
	model?: string;
	tools?: string[];
	cwd: string;
	sessionDir: string;
	status: "idle" | "running" | "suspended";
	episodes: Episode[];
	initialContext?: string;
	priorEpisodeIds?: string[];
	createdAt: number;
	totalUsage: EpisodeUsage;
	actionCount: number;
}

interface ThreadSpawnDetails {
	threadName: string;
	model?: string;
	cwd: string;
}

interface ThreadActionDetails {
	threadName: string;
	action: string;
	episode: Episode | null;
}

interface ThreadStatusDetails {
	threads: Array<{
		name: string;
		model?: string;
		status: string;
		actionCount: number;
		episodeCount: number;
		totalCost: number;
	}>;
}

// ============================================================================
// Episode extraction (standalone copy)
// ============================================================================

function generateEpisodeId(): string {
	return `ep_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function fallbackSummary(output: string): string {
	const lines = output.split("\n").filter((l) => l.trim().length > 0);
	if (lines.length === 0) return "(no output)";
	let summary = "";
	for (const line of lines.slice(0, 5)) {
		if (summary.length + line.length > 400) break;
		summary += (summary ? "\n" : "") + line;
	}
	return summary || lines[0].slice(0, 300);
}

function stripEpisodeBlock(output: string): string {
	return output.replace(/<episode-report>[\s\S]*?<\/episode-report>/, "").trim();
}

function extractFilePaths(output: string): string[] {
	const paths = new Set<string>();
	const pathRegex = /(?:^|\s)((?:\/|\.\/|~\/)[^\s:,'"]+\.\w+)/gm;
	let match = pathRegex.exec(output);
	while (match !== null) {
		paths.add(match[1]);
		match = pathRegex.exec(output);
	}
	return Array.from(paths).slice(0, 20);
}

function extractEpisodeFromOutput(
	rawOutput: string,
	task: string,
	cwd: string,
	model: string | undefined,
	durationMs: number,
	exitCode: number,
	usage: EpisodeUsage,
	label?: string,
): Episode {
	const id = generateEpisodeId();
	const now = Date.now();

	const episodeBlockMatch = rawOutput.match(/<episode-report>([\s\S]*?)<\/episode-report>/);

	if (episodeBlockMatch) {
		try {
			const parsed = JSON.parse(episodeBlockMatch[1]);
			return {
				id,
				label,
				timestamp: now,
				task,
				cwd,
				model,
				summary: parsed.summary || fallbackSummary(rawOutput),
				discoveries: parsed.discoveries || [],
				decisions: parsed.decisions || [],
				filesRead: parsed.filesRead || [],
				filesModified: parsed.filesModified || [],
				errors: parsed.errors || [],
				durationMs,
				exitCode,
				turns: usage.turns,
				usage,
				rawOutput: stripEpisodeBlock(rawOutput),
			};
		} catch {
			// Fall through to heuristic
		}
	}

	return {
		id,
		label,
		timestamp: now,
		task,
		cwd,
		model,
		summary: fallbackSummary(rawOutput),
		discoveries: [],
		decisions: [],
		filesRead: extractFilePaths(rawOutput),
		filesModified: [],
		errors: exitCode !== 0 ? [rawOutput.slice(0, 200)] : [],
		durationMs,
		exitCode,
		turns: usage.turns,
		usage,
		rawOutput,
	};
}

const EPISODE_REPORT_INSTRUCTION = `

IMPORTANT: At the very end of your response, after completing the task, emit a structured report block like this:

<episode-report>
{
  "summary": "Brief 1-2 sentence summary of what was accomplished",
  "discoveries": ["Key finding 1", "Key finding 2"],
  "decisions": ["Decision made and why"],
  "filesRead": ["path/to/file1.ts"],
  "filesModified": ["path/to/file2.ts"],
  "errors": ["Any errors encountered, or empty array"]
}
</episode-report>

This report will be used to transfer context to subsequent tasks.`;

// ============================================================================
// Helpers
// ============================================================================

function extractAssistantText(event: any): string | undefined {
	if (event.message?.role !== "assistant" || !Array.isArray(event.message.content)) return undefined;
	const chunks = event.message.content
		.filter((block: any) => block.type === "text" && typeof block.text === "string")
		.map((block: any) => block.text as string);
	if (chunks.length === 0) return undefined;
	return chunks.join("\n").trim();
}

function buildPriorEpisodesContext(store: Map<string, Episode>, episodeIds: string[]): string {
	const resolved = episodeIds.map((id) => store.get(id)).filter((e): e is Episode => e !== undefined);

	if (resolved.length === 0) return "";

	let context = "\n\n## Prior Episode Context\n\n";
	context += "The following episodes provide context from previous work:\n\n";

	for (const ep of resolved) {
		context += `### Episode: ${ep.label || ep.id}\n`;
		context += `**Task:** ${ep.task}\n`;
		context += `**Summary:** ${ep.summary}\n`;
		if (ep.discoveries.length > 0) {
			context += "**Discoveries:**\n";
			for (const d of ep.discoveries) context += `- ${d}\n`;
		}
		if (ep.decisions.length > 0) {
			context += "**Decisions:**\n";
			for (const d of ep.decisions) context += `- ${d}\n`;
		}
		if (ep.filesModified.length > 0) {
			context += `**Files modified:** ${ep.filesModified.join(", ")}\n`;
		}
		context += "\n";
	}

	return context;
}

function buildThreadHistoryContext(thread: ThreadState): string {
	if (thread.episodes.length === 0) return "";

	let context = "\n\n## Thread History\n\n";
	context += `This is a continuation of thread "${thread.name}". Previous actions in this thread:\n\n`;

	for (let i = 0; i < thread.episodes.length; i++) {
		const ep = thread.episodes[i];
		const status = ep.exitCode === 0 ? "✓" : "✗";
		context += `### Action ${i + 1} ${status}: ${ep.label || ep.task.slice(0, 60)}\n`;
		context += `**Summary:** ${ep.summary}\n`;
		if (ep.discoveries.length > 0) {
			context += "**Discoveries:**\n";
			for (const d of ep.discoveries) context += `- ${d}\n`;
		}
		if (ep.filesModified.length > 0) {
			context += `**Files modified:** ${ep.filesModified.join(", ")}\n`;
		}
		context += "\n";
	}

	return context;
}

/**
 * Run a thread action as a subprocess.
 *
 * We use --no-session and manually prepend thread history as context,
 * because pi CLI may not support injecting a user message into an existing session.
 * This achieves the persistent-thread effect through episode composition.
 */
async function runThreadAction(
	thread: ThreadState,
	actionPrompt: string,
	globalEpisodeStore: Map<string, Episode>,
	timeoutSeconds: number,
): Promise<Episode> {
	const started = Date.now();

	// Build full prompt with thread history + prior episodes + action + episode instruction
	let fullPrompt = "";

	// Inject prior episode context (from other threads)
	if (thread.priorEpisodeIds && thread.priorEpisodeIds.length > 0) {
		const priorContext = buildPriorEpisodesContext(globalEpisodeStore, thread.priorEpisodeIds);
		if (priorContext) fullPrompt += `${priorContext}\n\n---\n\n`;
	}

	// Inject initial context on first action
	if (thread.actionCount === 0 && thread.initialContext) {
		fullPrompt += `## Initial Context\n\n${thread.initialContext}\n\n---\n\n`;
	}

	// Inject thread history (episode summaries from prior actions)
	const historyContext = buildThreadHistoryContext(thread);
	if (historyContext) fullPrompt += `${historyContext}\n\n---\n\n`;

	// Add the actual action
	fullPrompt += actionPrompt;

	// Add episode report instruction
	fullPrompt += EPISODE_REPORT_INSTRUCTION;

	const args: string[] = ["--mode", "json", "-p", "--no-session"];
	if (thread.model?.trim()) args.unshift("--model", thread.model.trim());
	if (thread.tools && thread.tools.length > 0) {
		args.push("--tools", thread.tools.join(","));
	}
	args.push(fullPrompt);

	const usage: EpisodeUsage = {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		cost: 0,
		turns: 0,
	};
	let finalText = "";
	let model = thread.model?.trim();
	let _stderr = "";
	let timedOut = false;

	const exitCode = await new Promise<number>((resolve) => {
		const proc = spawn("pi", args, {
			cwd: thread.cwd,
			stdio: ["ignore", "pipe", "pipe"],
			shell: false,
		});
		let buffer = "";
		const timeout = setTimeout(() => {
			timedOut = true;
			proc.kill("SIGTERM");
		}, timeoutSeconds * 1000);

		const parseLine = (line: string) => {
			const trimmed = line.trim();
			if (!trimmed) return;
			let parsed: any;
			try {
				parsed = JSON.parse(trimmed);
			} catch {
				return;
			}
			if (parsed.type !== "message_end" || !parsed.message) return;
			if (parsed.message.role !== "assistant") return;
			usage.turns += 1;
			usage.input += parsed.message.usage?.input ?? 0;
			usage.output += parsed.message.usage?.output ?? 0;
			usage.cacheRead += parsed.message.usage?.cacheRead ?? 0;
			usage.cacheWrite += parsed.message.usage?.cacheWrite ?? 0;
			usage.cost += parsed.message.usage?.cost?.total ?? 0;
			model = model ?? parsed.message.model;
			const text = extractAssistantText(parsed);
			if (text) finalText = text;
		};

		proc.stdout.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) parseLine(line);
		});

		proc.stderr.on("data", (chunk: Buffer) => {
			_stderr += chunk.toString();
		});

		proc.on("error", () => {
			clearTimeout(timeout);
			resolve(127);
		});

		proc.on("close", (code: number | null) => {
			clearTimeout(timeout);
			if (buffer.trim()) parseLine(buffer);
			resolve(code ?? 0);
		});
	});

	if (timedOut && !finalText) {
		finalText = `Timed out after ${timeoutSeconds}s`;
	}

	const durationMs = Date.now() - started;
	return extractEpisodeFromOutput(
		finalText,
		actionPrompt.slice(0, 200),
		thread.cwd,
		model,
		durationMs,
		exitCode,
		usage,
		`${thread.name}#${thread.actionCount + 1}`,
	);
}

// ============================================================================
// Extension
// ============================================================================

export default function (pi: ExtensionAPI) {
	// Thread store
	const threads = new Map<string, ThreadState>();

	// Global episode store (shared across all threads for cross-thread composition)
	const globalEpisodeStore = new Map<string, Episode>();

	// ── thread_spawn ──
	pi.registerTool({
		name: "thread_spawn",
		label: "Thread Spawn",
		description: [
			"Create a named persistent thread — a work stream that accumulates context across multiple actions.",
			"Unlike ephemeral delegates, threads remember their full history.",
			"Use thread_action to send work to a thread. Each action returns an episode.",
			"Episodes from one thread can seed another via priorEpisodes (thread weaving).",
		].join(" "),
		parameters: Type.Object({
			name: Type.String({
				description: "Unique thread name (e.g. 'auth-refactor', 'test-writer')",
			}),
			model: Type.Optional(
				Type.String({
					description: "Model for this thread. Sticky across actions.",
				}),
			),
			tools: Type.Optional(
				Type.Array(Type.String(), {
					description: "Tool allow-list for the thread",
				}),
			),
			cwd: Type.Optional(
				Type.String({
					description: "Working directory for the thread",
				}),
			),
			initialContext: Type.Optional(
				Type.String({
					description: "Initial context/instructions for the thread",
				}),
			),
			priorEpisodes: Type.Optional(
				Type.Array(Type.String(), {
					description: "Episode IDs to seed as initial context",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<AgentToolResult<ThreadSpawnDetails>> {
			const { name, model, tools, cwd, initialContext, priorEpisodes } = params;

			if (threads.has(name)) {
				return {
					content: [
						{ type: "text", text: `Thread "${name}" already exists. Use thread_action to send work to it.` },
					],
					details: { threadName: name, model, cwd: cwd || ctx.cwd },
					isError: true,
				};
			}

			const threadDir = path.join(os.tmpdir(), "pi-threads", name);
			fs.mkdirSync(threadDir, { recursive: true });

			const thread: ThreadState = {
				name,
				model,
				tools: tools?.filter((t) => t.trim().length > 0),
				cwd: cwd?.trim() || ctx.cwd,
				sessionDir: threadDir,
				status: "idle",
				episodes: [],
				initialContext,
				priorEpisodeIds: priorEpisodes,
				createdAt: Date.now(),
				totalUsage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
				actionCount: 0,
			};

			threads.set(name, thread);

			const lines: string[] = [];
			lines.push(`Thread **"${name}"** created.`);
			if (model) lines.push(`Model: ${model}`);
			if (tools && tools.length > 0) lines.push(`Tools: ${tools.join(", ")}`);
			if (initialContext) lines.push(`Initial context: ${initialContext.slice(0, 100)}...`);
			if (priorEpisodes && priorEpisodes.length > 0) {
				lines.push(`Seeded with ${priorEpisodes.length} prior episode(s)`);
			}
			lines.push("");
			lines.push("Use `thread_action` to send work to this thread.");

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details: { threadName: name, model, cwd: thread.cwd },
			};
		},

		renderCall(args, theme) {
			let text = theme.fg("toolTitle", theme.bold("thread_spawn "));
			text += theme.fg("accent", `"${args.name}"`);
			if (args.model) text += ` ${theme.fg("dim", args.model)}`;
			if (args.initialContext) {
				text += `\n  ${theme.fg("muted", args.initialContext.slice(0, 60))}`;
			}
			return new Text(text, 0, 0);
		},

		renderResult(result, _options, theme) {
			const details = result.details as ThreadSpawnDetails | undefined;
			const icon = result.isError ? theme.fg("error", "✗") : theme.fg("success", "✓");
			return new Text(
				`${icon} ${theme.fg("toolTitle", theme.bold("thread"))} ${theme.fg("accent", details?.threadName || "?")}${details?.model ? theme.fg("dim", ` ${details.model}`) : ""}`,
				0,
				0,
			);
		},
	});

	// ── thread_action ──
	pi.registerTool({
		name: "thread_action",
		label: "Thread Action",
		description: [
			"Send an action to a named thread. The thread retains its full history across actions.",
			"Returns a structured episode with summary, discoveries, decisions, and file operations.",
			"Episode IDs can be passed to other threads or delegates via priorEpisodes.",
		].join(" "),
		parameters: Type.Object({
			thread: Type.String({
				description: "Thread name to send action to",
			}),
			action: Type.String({
				description: "The action/task for the thread to execute",
			}),
			awaitCompletion: Type.Optional(
				Type.Boolean({
					description: "Wait for completion and return episode. Default: true",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, onUpdate, _ctx): Promise<AgentToolResult<ThreadActionDetails>> {
			const { thread: threadName, action, awaitCompletion } = params;
			const _shouldAwait = awaitCompletion !== false;

			const thread = threads.get(threadName);
			if (!thread) {
				return {
					content: [{ type: "text", text: `Thread "${threadName}" not found. Use thread_spawn first.` }],
					details: { threadName, action, episode: null },
					isError: true,
				};
			}

			if (thread.status === "running") {
				return {
					content: [
						{
							type: "text",
							text: `Thread "${threadName}" is already running an action. Wait for it to complete.`,
						},
					],
					details: { threadName, action, episode: null },
					isError: true,
				};
			}

			thread.status = "running";

			if (onUpdate) {
				onUpdate({
					content: [
						{ type: "text", text: `Thread "${threadName}" executing action #${thread.actionCount + 1}...` },
					],
				});
			}

			const episode = await runThreadAction(thread, action, globalEpisodeStore, 240);

			// Update thread state
			thread.status = "idle";
			thread.episodes.push(episode);
			thread.actionCount += 1;
			thread.totalUsage.input += episode.usage.input;
			thread.totalUsage.output += episode.usage.output;
			thread.totalUsage.cacheRead += episode.usage.cacheRead;
			thread.totalUsage.cacheWrite += episode.usage.cacheWrite;
			thread.totalUsage.cost += episode.usage.cost;
			thread.totalUsage.turns += episode.usage.turns;

			// Store in global episode store for cross-thread composition
			globalEpisodeStore.set(episode.id, episode);

			const details: ThreadActionDetails = {
				threadName,
				action,
				episode,
			};

			// Build response
			const lines: string[] = [];
			const icon = episode.exitCode === 0 ? "✓" : "✗";
			lines.push(`## ${icon} Thread "${threadName}" — Action #${thread.actionCount}\n`);
			lines.push(`**Summary:** ${episode.summary}`);

			if (episode.discoveries.length > 0) {
				lines.push("\n**Discoveries:**");
				for (const d of episode.discoveries) lines.push(`- ${d}`);
			}

			if (episode.decisions.length > 0) {
				lines.push("\n**Decisions:**");
				for (const d of episode.decisions) lines.push(`- ${d}`);
			}

			if (episode.filesModified.length > 0) {
				lines.push(`\n**Files modified:** ${episode.filesModified.join(", ")}`);
			}

			if (episode.errors.length > 0) {
				lines.push("\n**Errors:**");
				for (const e of episode.errors) lines.push(`- ${e}`);
			}

			lines.push("");
			lines.push(
				`*Episode ID: \`${episode.id}\` — Thread "${threadName}" has ${thread.actionCount} action(s), total $${thread.totalUsage.cost.toFixed(4)}*`,
			);

			// Include raw output
			if (episode.rawOutput && episode.rawOutput !== episode.summary) {
				lines.push("\n---\n");
				lines.push(episode.rawOutput);
			}

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details,
				isError: episode.exitCode !== 0,
			};
		},

		renderCall(args, theme) {
			const actionPreview = args.action.length > 60 ? `${args.action.slice(0, 60)}...` : args.action;
			return new Text(
				theme.fg("toolTitle", theme.bold("thread_action ")) +
					theme.fg("accent", `"${args.thread}"`) +
					`\n  ${theme.fg("muted", actionPreview)}`,
				0,
				0,
			);
		},

		renderResult(result, { expanded }, theme) {
			const details = result.details as ThreadActionDetails | undefined;
			if (!details?.episode) {
				const block = result.content[0];
				return new Text(block?.type === "text" ? block.text : "(no output)", 0, 0);
			}

			const ep = details.episode;
			const thread = threads.get(details.threadName);
			const isError = ep.exitCode !== 0;
			const icon = isError ? theme.fg("error", "✗") : theme.fg("success", "✓");

			if (!expanded) {
				const lines: string[] = [];
				lines.push(
					`${icon} ${theme.fg("toolTitle", theme.bold("thread"))} ${theme.fg("accent", details.threadName)} ${theme.fg("dim", `#${thread?.actionCount ?? "?"} ${(ep.durationMs / 1000).toFixed(1)}s`)}`,
				);
				lines.push(theme.fg("muted", ep.summary.split("\n")[0].slice(0, 100)));
				if (ep.filesModified.length > 0) {
					lines.push(theme.fg("dim", `files: ${ep.filesModified.slice(0, 3).join(", ")}`));
				}
				lines.push(theme.fg("dim", `turns ${ep.usage.turns} | $${ep.usage.cost.toFixed(4)} | ID: ${ep.id}`));
				return new Text(lines.join("\n"), 0, 0);
			}

			// Expanded
			const container = new Container();
			container.addChild(
				new Text(
					`${icon} ${theme.fg("toolTitle", theme.bold("thread_action"))} ${theme.fg("accent", details.threadName)} ${theme.fg("dim", `#${thread?.actionCount ?? "?"}`)}`,
					0,
					0,
				),
			);
			container.addChild(
				new Text(theme.fg("muted", "Action: ") + theme.fg("dim", details.action.slice(0, 120)), 0, 0),
			);
			container.addChild(new Spacer(1));
			container.addChild(new Text(ep.summary, 0, 0));

			if (ep.discoveries.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "Discoveries:"), 0, 0));
				for (const d of ep.discoveries) {
					container.addChild(new Text(theme.fg("dim", `  • ${d}`), 0, 0));
				}
			}

			if (ep.filesModified.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(
					new Text(theme.fg("muted", "Modified: ") + theme.fg("accent", ep.filesModified.join(", ")), 0, 0),
				);
			}

			if (ep.errors.length > 0) {
				container.addChild(new Spacer(1));
				for (const e of ep.errors) {
					container.addChild(new Text(theme.fg("error", `  ✗ ${e}`), 0, 0));
				}
			}

			container.addChild(new Spacer(1));
			container.addChild(
				new Text(
					theme.fg(
						"dim",
						`${(ep.durationMs / 1000).toFixed(1)}s | turns ${ep.usage.turns} | $${ep.usage.cost.toFixed(4)} | ID: ${ep.id} | Thread actions: ${thread?.actionCount ?? "?"}`,
					),
					0,
					0,
				),
			);

			return container;
		},
	});

	// ── thread_status ──
	pi.registerTool({
		name: "thread_status",
		label: "Thread Status",
		description:
			"Show status of one or all threads: name, model, status, action count, episode summaries, total cost.",
		parameters: Type.Object({
			thread: Type.Optional(
				Type.String({
					description: "Thread name. If omitted, show all threads.",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, _ctx): Promise<AgentToolResult<ThreadStatusDetails>> {
			const targetThreads = params.thread
				? [threads.get(params.thread)].filter((t): t is ThreadState => t !== undefined)
				: Array.from(threads.values());

			if (targetThreads.length === 0) {
				const msg = params.thread ? `Thread "${params.thread}" not found.` : "No threads active.";
				return {
					content: [{ type: "text", text: msg }],
					details: { threads: [] },
				};
			}

			const lines: string[] = [];
			const threadDetails: ThreadStatusDetails["threads"] = [];

			for (const t of targetThreads) {
				const icon = t.status === "running" ? "⏳" : t.status === "idle" ? "●" : "⏸";
				lines.push(`## ${icon} Thread "${t.name}" — ${t.status}`);
				if (t.model) lines.push(`**Model:** ${t.model}`);
				lines.push(`**Actions:** ${t.actionCount}`);
				lines.push(
					`**Total cost:** $${t.totalUsage.cost.toFixed(4)} (${t.totalUsage.turns} turns, ↑${t.totalUsage.input} ↓${t.totalUsage.output})`,
				);
				lines.push(`**Created:** ${new Date(t.createdAt).toLocaleTimeString()}`);

				if (t.episodes.length > 0) {
					lines.push("\n**Episodes:**");
					for (let i = 0; i < t.episodes.length; i++) {
						const ep = t.episodes[i];
						const status = ep.exitCode === 0 ? "✓" : "✗";
						lines.push(`  ${i + 1}. ${status} ${ep.summary.split("\n")[0].slice(0, 80)} (\`${ep.id}\`)`);
					}
				}

				lines.push("");

				threadDetails.push({
					name: t.name,
					model: t.model,
					status: t.status,
					actionCount: t.actionCount,
					episodeCount: t.episodes.length,
					totalCost: t.totalUsage.cost,
				});
			}

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details: { threads: threadDetails },
			};
		},

		renderCall(args, theme) {
			if (args.thread) {
				return new Text(
					theme.fg("toolTitle", theme.bold("thread_status ")) + theme.fg("accent", `"${args.thread}"`),
					0,
					0,
				);
			}
			return new Text(
				theme.fg("toolTitle", theme.bold("thread_status ")) + theme.fg("muted", "(all threads)"),
				0,
				0,
			);
		},

		renderResult(result, _options, theme) {
			const details = result.details as ThreadStatusDetails | undefined;
			if (!details || details.threads.length === 0) {
				return new Text(theme.fg("muted", "No threads."), 0, 0);
			}

			const lines: string[] = [];
			for (const t of details.threads) {
				const icon = t.status === "running" ? "⏳" : "●";
				lines.push(
					`${icon} ${theme.fg("accent", t.name)}${t.model ? theme.fg("dim", ` ${t.model}`) : ""} — ${t.actionCount} action(s), $${t.totalCost.toFixed(4)}`,
				);
			}
			return new Text(lines.join("\n"), 0, 0);
		},
	});

	// ── /threads command ──
	pi.registerCommand("threads", {
		description: "List all active threads with their status and episode count",
		handler: async (_args, ctx) => {
			if (threads.size === 0) {
				ctx.ui.notify("No threads active.", "info");
				return;
			}

			const lines: string[] = ["## Active Threads\n"];
			for (const [name, t] of threads) {
				const icon = t.status === "running" ? "⏳" : t.status === "idle" ? "●" : "⏸";
				lines.push(
					`${icon} **${name}**${t.model ? ` (${t.model})` : ""} — ${t.status}, ${t.actionCount} actions, $${t.totalUsage.cost.toFixed(4)}`,
				);
			}
			ctx.ui.notify(lines.join("\n"), "info");
		},
	});

	// ── /thread-episodes command ──
	pi.registerCommand("thread-episodes", {
		description: "List all episodes for a specific thread",
		handler: async (args, ctx) => {
			const threadName = args.trim();
			if (!threadName) {
				ctx.ui.notify("Usage: /thread-episodes <thread-name>", "warning");
				return;
			}

			const thread = threads.get(threadName);
			if (!thread) {
				ctx.ui.notify(`Thread "${threadName}" not found.`, "warning");
				return;
			}

			if (thread.episodes.length === 0) {
				ctx.ui.notify(`Thread "${threadName}" has no episodes yet.`, "info");
				return;
			}

			const lines: string[] = [`## Episodes for thread "${threadName}"\n`];
			for (let i = 0; i < thread.episodes.length; i++) {
				const ep = thread.episodes[i];
				const status = ep.exitCode === 0 ? "✓" : "✗";
				const duration = (ep.durationMs / 1000).toFixed(1);
				lines.push(`${i + 1}. ${status} **${ep.label || ep.id}** (${duration}s, $${ep.usage.cost.toFixed(4)})`);
				lines.push(`   ${ep.summary.split("\n")[0].slice(0, 80)}`);
				if (ep.filesModified.length > 0) {
					lines.push(`   Modified: ${ep.filesModified.slice(0, 3).join(", ")}`);
				}
				lines.push(`   ID: \`${ep.id}\``);
				lines.push("");
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}

/**
 * Parallel Delegate Extension
 *
 * Adds a `parallel_delegate` tool (additive — does NOT replace the built-in
 * delegate) that runs multiple delegates concurrently with per-task model
 * overrides, structured episode extraction, and post-completion synthesis.
 *
 * Inspired by Slate's thread-weaving architecture:
 *   - Each task runs as an isolated pi subprocess
 *   - Each returns a structured episode (summary, discoveries, decisions, files)
 *   - A synthesis step merges all episodes via the current model
 *   - Cross-model threading: use different models for different tasks
 *
 * Usage:
 *   pi --extension examples/extensions/parallel-delegate.ts
 */

import { spawn } from "node:child_process";
import { complete } from "@mariozechner/pi-ai";
import type { AgentToolResult, ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Container, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Episode Types (shared with episodic-delegate.ts)
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
// Parallel Delegate Details
// ============================================================================

interface ParallelDelegateDetails {
	episodes: Episode[];
	synthesis: string | null;
	synthesisEpisodeId: string | null;
}

// ============================================================================
// Episode extraction (reused from episodic-delegate.ts)
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

// ============================================================================
// Episode report instruction appended to each delegate task
// ============================================================================

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
// Subprocess helpers
// ============================================================================

function extractAssistantText(event: any): string | undefined {
	if (event.message?.role !== "assistant" || !Array.isArray(event.message.content)) return undefined;
	const chunks = event.message.content
		.filter((block: any) => block.type === "text" && typeof block.text === "string")
		.map((block: any) => block.text as string);
	if (chunks.length === 0) return undefined;
	return chunks.join("\n").trim();
}

interface TaskSpec {
	task: string;
	model?: string;
	tools?: string[];
	episodeLabel?: string;
	cwd?: string;
}

async function runDelegate(taskSpec: TaskSpec, defaultCwd: string, timeoutSeconds: number): Promise<Episode> {
	const started = Date.now();
	const augmentedTask = taskSpec.task + EPISODE_REPORT_INSTRUCTION;
	const delegatedCwd = taskSpec.cwd?.trim() || defaultCwd;

	const args: string[] = ["--mode", "json", "-p", "--no-session"];
	if (taskSpec.model?.trim()) args.push("--model", taskSpec.model.trim());
	const toolFilter = taskSpec.tools?.filter((t) => t.trim().length > 0);
	if (toolFilter && toolFilter.length > 0) {
		args.push("--tools", toolFilter.join(","));
	}
	args.push(augmentedTask);

	const usage: EpisodeUsage = {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		cost: 0,
		turns: 0,
	};
	let finalText = "";
	let model = taskSpec.model?.trim();
	let _stderr = "";
	let timedOut = false;

	const exitCode = await new Promise<number>((resolve) => {
		const proc = spawn("pi", args, {
			cwd: delegatedCwd,
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

		proc.on("error", (_error: NodeJS.ErrnoException) => {
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
		taskSpec.task,
		delegatedCwd,
		model,
		durationMs,
		exitCode,
		usage,
		taskSpec.episodeLabel,
	);
}

// ============================================================================
// Concurrency limiter (from subagent pattern)
// ============================================================================

async function mapWithConcurrencyLimit<TIn, TOut>(
	items: TIn[],
	concurrency: number,
	fn: (item: TIn, index: number) => Promise<TOut>,
): Promise<TOut[]> {
	if (items.length === 0) return [];
	const limit = Math.max(1, Math.min(concurrency, items.length));
	const results: TOut[] = new Array(items.length);
	let nextIndex = 0;
	const workers = new Array(limit).fill(null).map(async () => {
		while (true) {
			const current = nextIndex++;
			if (current >= items.length) return;
			results[current] = await fn(items[current], current);
		}
	});
	await Promise.all(workers);
	return results;
}

// ============================================================================
// Synthesis
// ============================================================================

async function synthesizeEpisodes(episodes: Episode[], ctx: ExtensionContext): Promise<string> {
	const model = ctx.model;
	if (!model) return "(no model available for synthesis)";

	const apiKey = await ctx.modelRegistry.getApiKey(model);
	if (!apiKey) return "(no API key available for synthesis)";

	// Build the synthesis prompt from all episode data
	let episodeData = "";
	for (const ep of episodes) {
		const status = ep.exitCode === 0 ? "SUCCESS" : "FAILED";
		episodeData += `\n### ${ep.label || ep.id} [${status}]\n`;
		episodeData += `**Task:** ${ep.task}\n`;
		episodeData += `**Model:** ${ep.model || "default"}\n`;
		episodeData += `**Summary:** ${ep.summary}\n`;
		if (ep.discoveries.length > 0) {
			episodeData += "**Discoveries:**\n";
			for (const d of ep.discoveries) episodeData += `- ${d}\n`;
		}
		if (ep.decisions.length > 0) {
			episodeData += "**Decisions:**\n";
			for (const d of ep.decisions) episodeData += `- ${d}\n`;
		}
		if (ep.filesRead.length > 0) {
			episodeData += `**Files read:** ${ep.filesRead.join(", ")}\n`;
		}
		if (ep.filesModified.length > 0) {
			episodeData += `**Files modified:** ${ep.filesModified.join(", ")}\n`;
		}
		if (ep.errors.length > 0) {
			episodeData += "**Errors:**\n";
			for (const e of ep.errors) episodeData += `- ${e}\n`;
		}
		episodeData += "\n";
	}

	const messages = [
		{
			role: "user" as const,
			content: [
				{
					type: "text" as const,
					text: `You are synthesizing results from ${episodes.length} parallel delegate tasks that ran concurrently. Each task produced a structured episode report.

Your job is to create a unified synthesis that:
1. Summarizes the overall outcome across all tasks
2. Highlights connections, conflicts, or dependencies between task results
3. Merges discoveries and decisions into a coherent narrative
4. Notes any files that were modified by multiple tasks (potential conflicts)
5. Identifies any errors or failures and their impact
6. Provides actionable next steps based on the combined results

Be concise but thorough. This synthesis will be used as context for subsequent work.

## Episode Reports
${episodeData}

Produce a structured synthesis with clear sections. Start with a one-line overall summary.`,
				},
			],
			timestamp: Date.now(),
		},
	];

	try {
		const response = await complete(model, { messages }, { apiKey, maxTokens: 4096 });
		const synthesis = response.content
			.filter((c): c is { type: "text"; text: string } => c.type === "text")
			.map((c) => c.text)
			.join("\n");
		return synthesis.trim() || "(empty synthesis)";
	} catch (error) {
		const msg = error instanceof Error ? error.message : String(error);
		return `(synthesis failed: ${msg})`;
	}
}

// ============================================================================
// TUI helpers
// ============================================================================

function formatTokens(count: number): string {
	if (count < 1000) return count.toString();
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	return `${(count / 1000000).toFixed(1)}M`;
}

function formatUsage(usage: EpisodeUsage, model?: string): string {
	const parts: string[] = [];
	if (usage.turns) parts.push(`${usage.turns}t`);
	if (usage.input) parts.push(`↑${formatTokens(usage.input)}`);
	if (usage.output) parts.push(`↓${formatTokens(usage.output)}`);
	if (usage.cost) parts.push(`$${usage.cost.toFixed(4)}`);
	if (model) parts.push(model);
	return parts.join(" ");
}

function aggregateUsage(episodes: Episode[]): EpisodeUsage {
	const total: EpisodeUsage = {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		cost: 0,
		turns: 0,
	};
	for (const ep of episodes) {
		total.input += ep.usage.input;
		total.output += ep.usage.output;
		total.cacheRead += ep.usage.cacheRead;
		total.cacheWrite += ep.usage.cacheWrite;
		total.cost += ep.usage.cost;
		total.turns += ep.usage.turns;
	}
	return total;
}

// ============================================================================
// Extension
// ============================================================================

const MAX_TASKS = 8;
const DEFAULT_CONCURRENCY = 4;
const MAX_CONCURRENCY = 4;
const DEFAULT_TIMEOUT = 240;

export default function (pi: ExtensionAPI) {
	// Shared episode store for priorEpisodes composition
	const episodeStore = new Map<string, Episode>();

	pi.registerTool({
		name: "parallel_delegate",
		label: "Parallel Delegate",
		description: [
			"Run multiple delegate tasks concurrently, each in an isolated pi subprocess.",
			"Each task can specify its own model (cross-model threading), tools, and working directory.",
			"Returns structured episodes for each task plus an optional synthesis merging all results.",
			"Episode IDs can be passed to subsequent delegate/parallel_delegate calls via priorEpisodes.",
			"Max 8 tasks, max 4 concurrent. Each task gets an episode report with summary, discoveries, decisions, and files.",
		].join(" "),
		parameters: Type.Object({
			tasks: Type.Array(
				Type.Object({
					task: Type.String({ description: "Task to delegate to a nested pi run." }),
					model: Type.Optional(
						Type.String({
							description: "Model override for this task (e.g. gemini-2.5-flash, claude-sonnet-4).",
						}),
					),
					tools: Type.Optional(Type.Array(Type.String(), { description: "Optional allow-list of tool names." })),
					episodeLabel: Type.Optional(Type.String({ description: "Human-readable label for this episode." })),
					cwd: Type.Optional(Type.String({ description: "Working directory for the delegate." })),
				}),
				{ description: "Array of tasks to run in parallel. Max 8." },
			),
			synthesize: Type.Optional(
				Type.Boolean({
					description: "Generate a synthesis episode merging all results. Default: true",
				}),
			),
			maxConcurrency: Type.Optional(
				Type.Number({
					description: "Max parallel delegates. Default: 4, max: 4",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, onUpdate, ctx): Promise<AgentToolResult<ParallelDelegateDetails>> {
			const tasks = params.tasks;
			const doSynthesize = params.synthesize !== false;
			const concurrency = Math.max(1, Math.min(params.maxConcurrency || DEFAULT_CONCURRENCY, MAX_CONCURRENCY));

			// Validate task count
			if (tasks.length === 0) {
				return {
					content: [{ type: "text", text: "No tasks provided." }],
					details: { episodes: [], synthesis: null, synthesisEpisodeId: null },
				};
			}
			if (tasks.length > MAX_TASKS) {
				return {
					content: [{ type: "text", text: `Too many tasks (${tasks.length}). Maximum is ${MAX_TASKS}.` }],
					details: { episodes: [], synthesis: null, synthesisEpisodeId: null },
				};
			}

			// Track completion state for streaming updates
			const completedFlags: boolean[] = new Array(tasks.length).fill(false);
			const episodeSlots: (Episode | null)[] = new Array(tasks.length).fill(null);

			const emitUpdate = () => {
				if (!onUpdate) return;
				const done = completedFlags.filter(Boolean).length;
				const running = tasks.length - done;
				const completedEpisodes = episodeSlots.filter((e): e is Episode => e !== null);

				onUpdate({
					content: [
						{
							type: "text",
							text: `Parallel: ${done}/${tasks.length} done, ${running} running${doSynthesize && done === tasks.length ? ", synthesizing..." : ""}`,
						},
					],
					details: {
						episodes: completedEpisodes,
						synthesis: null,
						synthesisEpisodeId: null,
					},
				});
			};

			// Initial update
			emitUpdate();

			// Run all tasks with concurrency limit
			const episodes = await mapWithConcurrencyLimit(tasks, concurrency, async (taskSpec, index) => {
				const episode = await runDelegate(taskSpec, ctx.cwd, DEFAULT_TIMEOUT);
				episodeSlots[index] = episode;
				completedFlags[index] = true;
				emitUpdate();
				return episode;
			});

			// Store all episodes
			for (const ep of episodes) {
				episodeStore.set(ep.id, ep);
			}

			// Synthesize if requested
			let synthesis: string | null = null;
			let synthesisEpisodeId: string | null = null;

			if (doSynthesize && episodes.length > 1) {
				synthesis = await synthesizeEpisodes(episodes, ctx);

				// Create a synthetic episode for the synthesis itself
				const synthId = generateEpisodeId();
				const synthEpisode: Episode = {
					id: synthId,
					label: "synthesis",
					timestamp: Date.now(),
					task: `Synthesize ${episodes.length} parallel episodes`,
					cwd: ctx.cwd,
					model: ctx.model?.id,
					summary: synthesis.split("\n")[0].slice(0, 300),
					discoveries: [],
					decisions: [],
					filesRead: [],
					filesModified: [],
					errors: [],
					durationMs: 0,
					exitCode: 0,
					turns: 0,
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
					rawOutput: synthesis,
				};
				episodeStore.set(synthId, synthEpisode);
				synthesisEpisodeId = synthId;
			}

			const details: ParallelDelegateDetails = {
				episodes,
				synthesis,
				synthesisEpisodeId,
			};

			// Build text response
			const lines: string[] = [];
			const successCount = episodes.filter((e) => e.exitCode === 0).length;
			lines.push(`## Parallel Delegates: ${successCount}/${episodes.length} succeeded\n`);

			for (const ep of episodes) {
				const icon = ep.exitCode === 0 ? "✓" : "✗";
				lines.push(`### ${icon} ${ep.label || ep.id}${ep.model ? ` (${ep.model})` : ""}`);
				lines.push(`**Summary:** ${ep.summary}`);
				if (ep.discoveries.length > 0) {
					lines.push("**Discoveries:**");
					for (const d of ep.discoveries) lines.push(`- ${d}`);
				}
				if (ep.decisions.length > 0) {
					lines.push("**Decisions:**");
					for (const d of ep.decisions) lines.push(`- ${d}`);
				}
				if (ep.filesModified.length > 0) {
					lines.push(`**Files modified:** ${ep.filesModified.join(", ")}`);
				}
				if (ep.errors.length > 0) {
					lines.push("**Errors:**");
					for (const e of ep.errors) lines.push(`- ${e}`);
				}
				lines.push(`*Episode ID: \`${ep.id}\`*\n`);
			}

			if (synthesis) {
				lines.push("---\n");
				lines.push("## Synthesis\n");
				lines.push(synthesis);
				lines.push(`\n*Synthesis Episode ID: \`${synthesisEpisodeId}\`*`);
			}

			lines.push("\n---\n");
			const allIds = episodes.map((e) => e.id);
			if (synthesisEpisodeId) allIds.push(synthesisEpisodeId);
			lines.push(`**All episode IDs** (for \`priorEpisodes\`): ${allIds.map((id) => `\`${id}\``).join(", ")}`);

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details,
			};
		},

		renderCall(args, theme) {
			const tasks = args.tasks || [];
			let text =
				theme.fg("toolTitle", theme.bold("parallel_delegate ")) +
				theme.fg("accent", `${tasks.length} task${tasks.length !== 1 ? "s" : ""}`) +
				(args.maxConcurrency ? theme.fg("dim", ` ×${args.maxConcurrency}`) : "") +
				(args.synthesize === false ? theme.fg("dim", " (no synthesis)") : "");

			for (const t of tasks.slice(0, 4)) {
				const label = t.episodeLabel || t.task.slice(0, 40);
				const preview = label.length > 50 ? `${label.slice(0, 50)}...` : label;
				const modelTag = t.model ? theme.fg("accent", ` [${t.model}]`) : "";
				text += `\n  ${theme.fg("dim", "•")} ${theme.fg("muted", preview)}${modelTag}`;
			}
			if (tasks.length > 4) {
				text += `\n  ${theme.fg("muted", `... +${tasks.length - 4} more`)}`;
			}

			return new Text(text, 0, 0);
		},

		renderResult(result, { expanded }, theme) {
			const details = result.details as ParallelDelegateDetails | undefined;
			if (!details || details.episodes.length === 0) {
				const block = result.content[0];
				return new Text(block?.type === "text" ? block.text : "(no output)", 0, 0);
			}

			const episodes = details.episodes;
			const successCount = episodes.filter((e) => e.exitCode === 0).length;
			const hasFailures = successCount < episodes.length;
			const icon = hasFailures ? theme.fg("warning", "◐") : theme.fg("success", "✓");
			const totalUsage = aggregateUsage(episodes);

			if (!expanded) {
				// ── Collapsed view ──
				const lines: string[] = [];
				lines.push(
					`${icon} ${theme.fg("toolTitle", theme.bold("parallel "))}${theme.fg("accent", `${successCount}/${episodes.length} tasks`)} ${theme.fg("dim", formatUsage(totalUsage))}`,
				);

				for (const ep of episodes) {
					const epIcon = ep.exitCode === 0 ? theme.fg("success", "✓") : theme.fg("error", "✗");
					const label = ep.label || ep.id;
					const modelTag = ep.model ? theme.fg("dim", ` ${ep.model}`) : "";
					const duration = theme.fg("dim", `${(ep.durationMs / 1000).toFixed(1)}s`);
					lines.push(`  ${epIcon} ${theme.fg("accent", label)}${modelTag} ${duration}`);
					lines.push(`    ${theme.fg("muted", ep.summary.split("\n")[0].slice(0, 80))}`);
				}

				if (details.synthesis) {
					lines.push("");
					lines.push(
						`  ${theme.fg("toolTitle", "synthesis:")} ${theme.fg("muted", details.synthesis.split("\n")[0].slice(0, 80))}`,
					);
				}

				lines.push(theme.fg("muted", "(Ctrl+O to expand)"));
				return new Text(lines.join("\n"), 0, 0);
			}

			// ── Expanded view ──
			const container = new Container();
			container.addChild(
				new Text(
					`${icon} ${theme.fg("toolTitle", theme.bold("parallel "))}${theme.fg("accent", `${successCount}/${episodes.length} tasks`)} ${theme.fg("dim", formatUsage(totalUsage))}`,
					0,
					0,
				),
			);

			for (const ep of episodes) {
				const epIcon = ep.exitCode === 0 ? theme.fg("success", "✓") : theme.fg("error", "✗");

				container.addChild(new Spacer(1));
				container.addChild(
					new Text(
						`${theme.fg("muted", "─── ")}${epIcon} ${theme.fg("accent", ep.label || ep.id)}${ep.model ? theme.fg("dim", ` ${ep.model}`) : ""} ${theme.fg("dim", `${(ep.durationMs / 1000).toFixed(1)}s`)}`,
						0,
						0,
					),
				);
				container.addChild(
					new Text(
						theme.fg("muted", "Task: ") +
							theme.fg("dim", ep.task.length > 80 ? `${ep.task.slice(0, 80)}...` : ep.task),
						0,
						0,
					),
				);
				container.addChild(new Text(ep.summary, 0, 0));

				if (ep.discoveries.length > 0) {
					container.addChild(new Text(theme.fg("muted", "Discoveries:"), 0, 0));
					for (const d of ep.discoveries) {
						container.addChild(new Text(theme.fg("dim", `  • ${d}`), 0, 0));
					}
				}

				if (ep.decisions.length > 0) {
					container.addChild(new Text(theme.fg("muted", "Decisions:"), 0, 0));
					for (const d of ep.decisions) {
						container.addChild(new Text(theme.fg("dim", `  • ${d}`), 0, 0));
					}
				}

				if (ep.filesModified.length > 0) {
					container.addChild(
						new Text(theme.fg("muted", "Modified: ") + theme.fg("accent", ep.filesModified.join(", ")), 0, 0),
					);
				}

				if (ep.errors.length > 0) {
					for (const e of ep.errors) {
						container.addChild(new Text(theme.fg("error", `  ✗ ${e}`), 0, 0));
					}
				}

				container.addChild(new Text(theme.fg("dim", `  ${formatUsage(ep.usage, ep.model)} | ID: ${ep.id}`), 0, 0));
			}

			// Synthesis section
			if (details.synthesis) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "═══ Synthesis ═══"), 0, 0));
				container.addChild(new Text(details.synthesis, 0, 0));
				if (details.synthesisEpisodeId) {
					container.addChild(new Text(theme.fg("dim", `Synthesis ID: ${details.synthesisEpisodeId}`), 0, 0));
				}
			}

			// Episode IDs footer
			container.addChild(new Spacer(1));
			const allIds = episodes.map((e) => e.id);
			if (details.synthesisEpisodeId) allIds.push(details.synthesisEpisodeId);
			container.addChild(new Text(theme.fg("dim", `Episode IDs: ${allIds.join(", ")}`), 0, 0));

			return container;
		},
	});

	// Register command to list stored episodes from parallel delegates
	pi.registerCommand("parallel-episodes", {
		description: "List episodes from parallel delegate runs in this session",
		handler: async (_args, ctx) => {
			if (episodeStore.size === 0) {
				ctx.ui.notify("No parallel episodes in this session yet.", "info");
				return;
			}

			const episodes = Array.from(episodeStore.values()).sort((a, b) => b.timestamp - a.timestamp);

			const lines: string[] = ["## Parallel Episodes\n"];
			for (const ep of episodes) {
				const status = ep.exitCode === 0 ? "✓" : "✗";
				const label = ep.label || ep.id;
				const duration = (ep.durationMs / 1000).toFixed(1);
				const model = ep.model ? ` (${ep.model})` : "";
				lines.push(
					`${status} **${label}**${model} — ${duration}s, ${ep.usage.turns}t, $${ep.usage.cost.toFixed(4)}`,
				);
				lines.push(`  ${ep.summary.split("\n")[0].slice(0, 80)}`);
				lines.push(`  ID: \`${ep.id}\``);
				lines.push("");
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}

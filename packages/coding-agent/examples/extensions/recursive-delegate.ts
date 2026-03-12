/**
 * Recursive Delegate Extension
 *
 * Adds depth-controlled recursive delegation with budget/timeout propagation.
 * A delegate can spawn its own children (who can spawn theirs), with resources
 * flowing through the tree and aggregating on return. This enables RLM-style
 * recursive decomposition with guardrails against over-decomposition.
 *
 * At max depth, children use plain LLM completion (via complete()) instead of
 * spawning agent subprocesses, providing a graceful fallback.
 *
 * Key features:
 *   - Depth-controlled recursion (default max 2, hard max 4)
 *   - Budget propagation (USD limit across the entire tree)
 *   - Timeout propagation (wall-clock seconds across the tree)
 *   - Tree-structured episode aggregation
 *   - Fallback to plain LLM at max depth
 *
 * Usage:
 *   pi --extension examples/extensions/recursive-delegate.ts
 */

import { spawn } from "node:child_process";
import { complete } from "@mariozechner/pi-ai";
import type { AgentToolResult, ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Container, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Episode Types (standalone copy)
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

interface RecursiveEpisode extends Episode {
	depth: number;
	maxDepth: number;
	childEpisodes: RecursiveEpisode[];
	budgetUsed: number;
	budgetRemaining: number;
}

interface RecursiveDelegateDetails {
	episode: RecursiveEpisode;
	priorEpisodeIds: string[];
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
			// Fall through
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
	for (const ep of resolved) {
		context += `### Episode: ${ep.label || ep.id}\n`;
		context += `**Task:** ${ep.task}\n`;
		context += `**Summary:** ${ep.summary}\n`;
		if (ep.discoveries.length > 0) {
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
 * Parse recursive delegate metadata prefix from task prompt.
 * Format: [RECURSIVE_DELEGATE depth=N maxDepth=M remainingBudget=X remainingTimeout=Y]
 */
interface RecursiveMeta {
	depth: number;
	maxDepth: number;
	remainingBudget: number | undefined;
	remainingTimeout: number;
}

function parseRecursiveMeta(task: string): { meta: RecursiveMeta | null; cleanTask: string } {
	const match = task.match(
		/^\[RECURSIVE_DELEGATE\s+depth=(\d+)\s+maxDepth=(\d+)\s+remainingBudget=([\d.]+|none)\s+remainingTimeout=(\d+)\]\s*/,
	);
	if (!match) return { meta: null, cleanTask: task };

	return {
		meta: {
			depth: parseInt(match[1], 10),
			maxDepth: parseInt(match[2], 10),
			remainingBudget: match[3] === "none" ? undefined : parseFloat(match[3]),
			remainingTimeout: parseInt(match[4], 10),
		},
		cleanTask: task.slice(match[0].length),
	};
}

function buildRecursivePrefix(
	depth: number,
	maxDepth: number,
	remainingBudget: number | undefined,
	remainingTimeout: number,
): string {
	const budget = remainingBudget !== undefined ? remainingBudget.toFixed(4) : "none";
	return `[RECURSIVE_DELEGATE depth=${depth} maxDepth=${maxDepth} remainingBudget=${budget} remainingTimeout=${Math.round(remainingTimeout)}] `;
}

// ============================================================================
// Agent subprocess execution
// ============================================================================

async function runAgentDelegate(
	task: string,
	cwd: string,
	model: string | undefined,
	tools: string[] | undefined,
	timeoutSeconds: number,
	label?: string,
): Promise<Episode> {
	const started = Date.now();
	const augmentedTask = task + EPISODE_REPORT_INSTRUCTION;

	const args: string[] = ["--mode", "json", "-p", "--no-session"];
	if (model?.trim()) args.unshift("--model", model.trim());
	const toolFilter = tools?.filter((t) => t.trim().length > 0);
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
	let resolvedModel = model?.trim();
	let _stderr = "";
	let timedOut = false;

	const exitCode = await new Promise<number>((resolve) => {
		const proc = spawn("pi", args, {
			cwd,
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
			resolvedModel = resolvedModel ?? parsed.message.model;
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
		task.slice(0, 200),
		cwd,
		resolvedModel,
		durationMs,
		exitCode,
		usage,
		label,
	);
}

// ============================================================================
// Plain LLM fallback at max depth
// ============================================================================

async function runPlainLLM(
	task: string,
	ctx: ExtensionContext,
	timeoutSeconds: number,
	label?: string,
): Promise<Episode> {
	const started = Date.now();
	const model = ctx.model;
	if (!model) {
		return extractEpisodeFromOutput(
			"(no model available for plain LLM fallback)",
			task.slice(0, 200),
			ctx.cwd,
			undefined,
			Date.now() - started,
			1,
			{ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
			label,
		);
	}

	const apiKey = await ctx.modelRegistry.getApiKey(model);
	if (!apiKey) {
		return extractEpisodeFromOutput(
			"(no API key for plain LLM fallback)",
			task.slice(0, 200),
			ctx.cwd,
			model.id,
			Date.now() - started,
			1,
			{ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
			label,
		);
	}

	try {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), timeoutSeconds * 1000);

		const response = await complete(
			model,
			{
				messages: [
					{
						role: "user" as const,
						content: [{ type: "text" as const, text: task + EPISODE_REPORT_INSTRUCTION }],
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey, maxTokens: 4096, signal: controller.signal },
		);

		clearTimeout(timeout);

		const text = response.content
			.filter((c): c is { type: "text"; text: string } => c.type === "text")
			.map((c) => c.text)
			.join("\n")
			.trim();

		const usage: EpisodeUsage = {
			input: response.usage?.input ?? 0,
			output: response.usage?.output ?? 0,
			cacheRead: response.usage?.cacheRead ?? 0,
			cacheWrite: response.usage?.cacheWrite ?? 0,
			cost: response.usage?.cost?.total ?? 0,
			turns: 1,
		};

		const durationMs = Date.now() - started;
		return extractEpisodeFromOutput(text, task.slice(0, 200), ctx.cwd, model.id, durationMs, 0, usage, label);
	} catch (error) {
		const msg = error instanceof Error ? error.message : String(error);
		const durationMs = Date.now() - started;
		return extractEpisodeFromOutput(
			`(plain LLM error: ${msg})`,
			task.slice(0, 200),
			ctx.cwd,
			model.id,
			durationMs,
			1,
			{ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
			label,
		);
	}
}

// ============================================================================
// Tree rendering
// ============================================================================

function renderTree(episode: RecursiveEpisode, indent: number = 0): string {
	const prefix = "  ".repeat(indent);
	const icon = episode.exitCode === 0 ? "✓" : "✗";
	const depthTag = `[d${episode.depth}/${episode.maxDepth}]`;
	const lines: string[] = [];

	lines.push(
		`${prefix}${icon} ${depthTag} ${episode.label || episode.id} — ${episode.summary.split("\n")[0].slice(0, 60)}`,
	);

	if (episode.childEpisodes.length > 0) {
		for (const child of episode.childEpisodes) {
			lines.push(renderTree(child, indent + 1));
		}
	}

	return lines.join("\n");
}

function _aggregateUsage(episodes: Episode[]): EpisodeUsage {
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

export default function (pi: ExtensionAPI) {
	// Global episode store
	const episodeStore = new Map<string, Episode>();

	// Store last recursive tree for /recursive-tree command
	let lastRecursiveEpisode: RecursiveEpisode | null = null;

	pi.registerTool({
		name: "recursive_delegate",
		label: "Recursive Delegate",
		description: [
			"Delegate a task with depth-controlled recursive decomposition.",
			"The child can spawn its own children (up to maxDepth), with budget and timeout propagating through the tree.",
			"At maxDepth, falls back to plain LLM completion (no agent loop).",
			"Returns a tree-structured episode aggregating results from all levels.",
			"Use for complex tasks that benefit from hierarchical decomposition.",
		].join(" "),
		parameters: Type.Object({
			task: Type.String({ description: "Task to delegate" }),
			model: Type.Optional(Type.String({ description: "Model override" })),
			tools: Type.Optional(Type.Array(Type.String(), { description: "Tool allow-list" })),
			episodeLabel: Type.Optional(Type.String({ description: "Human-readable label for this episode" })),
			cwd: Type.Optional(Type.String({ description: "Working directory" })),
			maxDepth: Type.Optional(
				Type.Number({
					description: "Max recursion depth. Default: 2. Max: 4. At maxDepth, children use plain LLM completion.",
				}),
			),
			maxBudget: Type.Optional(
				Type.Number({
					description: "Max total cost in USD across the entire recursive tree. Default: no limit.",
				}),
			),
			maxTimeout: Type.Optional(
				Type.Number({
					description: "Max total wall-clock seconds for the entire tree. Default: 600.",
				}),
			),
			priorEpisodes: Type.Optional(
				Type.Array(Type.String(), {
					description: "Episode IDs to seed context",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, onUpdate, ctx): Promise<AgentToolResult<RecursiveDelegateDetails>> {
			const started = Date.now();

			// Parse recursive metadata from task (if this is a child invocation)
			const { meta: inheritedMeta, cleanTask } = parseRecursiveMeta(params.task);

			const depth = inheritedMeta?.depth ?? 0;
			const maxDepth = Math.min(inheritedMeta?.maxDepth ?? params.maxDepth ?? 2, 4);
			const maxBudget = inheritedMeta?.remainingBudget ?? params.maxBudget;
			const maxTimeout = inheritedMeta?.remainingTimeout ?? params.maxTimeout ?? 600;

			const task = inheritedMeta ? cleanTask : params.task;
			const delegatedCwd = params.cwd?.trim() || ctx.cwd;
			const priorEpisodeIds = params.priorEpisodes || [];

			// Check budget
			if (maxBudget !== undefined && maxBudget <= 0) {
				const errorEpisode: RecursiveEpisode = {
					id: generateEpisodeId(),
					label: params.episodeLabel,
					timestamp: Date.now(),
					task,
					cwd: delegatedCwd,
					summary: "Budget exhausted — cannot spawn child.",
					discoveries: [],
					decisions: [],
					filesRead: [],
					filesModified: [],
					errors: ["Budget exhausted"],
					durationMs: 0,
					exitCode: 1,
					turns: 0,
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
					rawOutput: "",
					depth,
					maxDepth,
					childEpisodes: [],
					budgetUsed: 0,
					budgetRemaining: 0,
				};
				return {
					content: [{ type: "text", text: "Budget exhausted." }],
					details: { episode: errorEpisode, priorEpisodeIds },
				};
			}

			if (onUpdate) {
				onUpdate({
					content: [
						{
							type: "text",
							text: `Recursive delegate: depth ${depth}/${maxDepth}${maxBudget !== undefined ? `, budget $${maxBudget.toFixed(4)}` : ""}, timeout ${maxTimeout}s...`,
						},
					],
					details: {
						episode: {
							id: generateEpisodeId(),
							label: params.episodeLabel,
							timestamp: Date.now(),
							task,
							cwd: delegatedCwd,
							summary: "Recursive delegate in progress.",
							discoveries: [],
							decisions: [],
							filesRead: [],
							filesModified: [],
							errors: [],
							durationMs: 0,
							exitCode: 0,
							turns: 0,
							usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
							rawOutput: "",
							depth,
							maxDepth,
							childEpisodes: [],
							budgetUsed: 0,
							budgetRemaining: maxBudget ?? -1,
						},
						priorEpisodeIds,
					},
				});
			}

			// Build augmented task with prior episode context
			let augmentedTask = task;
			if (priorEpisodeIds.length > 0) {
				const priorContext = buildPriorEpisodesContext(episodeStore, priorEpisodeIds);
				if (priorContext) {
					augmentedTask = `${priorContext}\n\n---\n\n${augmentedTask}`;
				}
			}

			// Add recursion metadata to the task so children know their position
			const childDepth = depth + 1;
			const elapsed = (Date.now() - started) / 1000;
			const childTimeout = Math.max(maxTimeout - elapsed, 10);

			let episode: Episode;

			if (depth >= maxDepth) {
				// At max depth: use plain LLM completion (no agent loop)
				if (onUpdate) {
					onUpdate({
						content: [{ type: "text", text: `At max depth ${depth}/${maxDepth} — using plain LLM...` }],
						details: {
							episode: {
								id: generateEpisodeId(),
								label: params.episodeLabel,
								timestamp: Date.now(),
								task,
								cwd: delegatedCwd,
								summary: "Plain LLM fallback in progress.",
								discoveries: [],
								decisions: [],
								filesRead: [],
								filesModified: [],
								errors: [],
								durationMs: 0,
								exitCode: 0,
								turns: 0,
								usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
								rawOutput: "",
								depth,
								maxDepth,
								childEpisodes: [],
								budgetUsed: 0,
								budgetRemaining: maxBudget ?? -1,
							},
							priorEpisodeIds,
						},
					});
				}
				episode = await runPlainLLM(augmentedTask, ctx, childTimeout, params.episodeLabel);
			} else {
				// Not at max depth: spawn full agent subprocess with recursive metadata
				const childBudget = maxBudget; // Will be reduced after this child completes
				const childPrefix = buildRecursivePrefix(childDepth, maxDepth, childBudget, childTimeout);
				const childTask = childPrefix + augmentedTask;

				episode = await runAgentDelegate(
					childTask,
					delegatedCwd,
					params.model,
					params.tools?.filter((t) => t.trim().length > 0),
					childTimeout,
					params.episodeLabel,
				);
			}

			// Build recursive episode
			const totalCost = episode.usage.cost;
			const recursiveEpisode: RecursiveEpisode = {
				...episode,
				depth,
				maxDepth,
				childEpisodes: [], // Children are tracked by the child process itself
				budgetUsed: totalCost,
				budgetRemaining: maxBudget !== undefined ? maxBudget - totalCost : -1,
			};

			// Store episodes
			episodeStore.set(recursiveEpisode.id, recursiveEpisode);
			lastRecursiveEpisode = recursiveEpisode;

			const details: RecursiveDelegateDetails = {
				episode: recursiveEpisode,
				priorEpisodeIds,
			};

			// Build response
			const lines: string[] = [];
			const icon = episode.exitCode === 0 ? "✓" : "✗";
			lines.push(`## ${icon} Recursive Delegate [depth ${depth}/${maxDepth}]`);
			lines.push(`**Label:** ${params.episodeLabel || recursiveEpisode.id}`);
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
				`*Depth ${depth}/${maxDepth} | Cost: $${totalCost.toFixed(4)}${maxBudget !== undefined ? ` (remaining: $${(maxBudget - totalCost).toFixed(4)})` : ""} | ${(episode.durationMs / 1000).toFixed(1)}s | Episode ID: \`${recursiveEpisode.id}\`*`,
			);

			if (episode.rawOutput && episode.rawOutput !== episode.summary) {
				lines.push("\n---\n");
				lines.push(episode.rawOutput);
			}

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details,
			};
		},

		renderCall(args, theme) {
			const taskPreview = args.task.length > 60 ? `${args.task.slice(0, 60)}...` : args.task;

			// Parse depth from task if it has recursive metadata
			const { meta } = parseRecursiveMeta(args.task);
			const depthTag = meta
				? theme.fg("warning", ` [d${meta.depth}/${meta.maxDepth}]`)
				: args.maxDepth
					? theme.fg("dim", ` [max ${args.maxDepth}]`)
					: "";

			let text = theme.fg("toolTitle", theme.bold("recursive_delegate")) + depthTag;
			if (args.episodeLabel) text += ` ${theme.fg("accent", args.episodeLabel)}`;
			text += `\n  ${theme.fg("muted", taskPreview)}`;
			if (args.model) text += `\n  ${theme.fg("dim", `model=${args.model}`)}`;

			const constraints: string[] = [];
			if (args.maxBudget) constraints.push(`$${args.maxBudget}`);
			if (args.maxTimeout) constraints.push(`${args.maxTimeout}s`);
			if (constraints.length > 0) {
				text += `\n  ${theme.fg("dim", constraints.join(" | "))}`;
			}

			return new Text(text, 0, 0);
		},

		renderResult(result, { expanded }, theme) {
			const details = result.details as RecursiveDelegateDetails | undefined;
			if (!details?.episode) {
				const block = result.content[0];
				return new Text(block?.type === "text" ? block.text : "(no output)", 0, 0);
			}

			const ep = details.episode;
			const isError = ep.exitCode !== 0;
			const icon = isError ? theme.fg("error", "✗") : theme.fg("success", "✓");
			const depthTag = theme.fg("warning", `[d${ep.depth}/${ep.maxDepth}]`);

			if (!expanded) {
				const lines: string[] = [];
				lines.push(
					`${icon} ${theme.fg("toolTitle", theme.bold("recursive"))} ${depthTag} ${theme.fg("accent", ep.label || ep.id)} ${theme.fg("dim", `${(ep.durationMs / 1000).toFixed(1)}s $${ep.budgetUsed.toFixed(4)}`)}`,
				);
				lines.push(theme.fg("muted", ep.summary.split("\n")[0].slice(0, 100)));
				if (ep.filesModified.length > 0) {
					lines.push(theme.fg("dim", `files: ${ep.filesModified.slice(0, 3).join(", ")}`));
				}
				return new Text(lines.join("\n"), 0, 0);
			}

			// Expanded view with tree
			const container = new Container();
			container.addChild(
				new Text(
					`${icon} ${theme.fg("toolTitle", theme.bold("recursive_delegate"))} ${depthTag} ${theme.fg("accent", ep.label || ep.id)}`,
					0,
					0,
				),
			);

			container.addChild(new Text(theme.fg("muted", "Task: ") + theme.fg("dim", ep.task.slice(0, 120)), 0, 0));

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

			// Tree view if there are children
			if (ep.childEpisodes.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Recursive Tree ───"), 0, 0));
				container.addChild(new Text(renderTree(ep), 0, 0));
			}

			container.addChild(new Spacer(1));
			container.addChild(
				new Text(
					theme.fg(
						"dim",
						`depth ${ep.depth}/${ep.maxDepth} | ${(ep.durationMs / 1000).toFixed(1)}s | turns ${ep.usage.turns} | $${ep.budgetUsed.toFixed(4)}${ep.budgetRemaining >= 0 ? ` (remaining: $${ep.budgetRemaining.toFixed(4)})` : ""} | ID: ${ep.id}`,
					),
					0,
					0,
				),
			);

			return container;
		},
	});

	// /recursive-tree command
	pi.registerCommand("recursive-tree", {
		description: "Show the full tree of recursive episodes from the last invocation",
		handler: async (_args, ctx) => {
			if (!lastRecursiveEpisode) {
				ctx.ui.notify("No recursive delegate has been run yet.", "info");
				return;
			}

			const tree = renderTree(lastRecursiveEpisode);
			ctx.ui.notify(`## Recursive Tree\n\n\`\`\`\n${tree}\n\`\`\``, "info");
		},
	});
}

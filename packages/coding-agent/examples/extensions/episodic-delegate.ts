/**
 * Episodic Delegate Extension
 *
 * Inspired by Slate's thread-weaving architecture, this replaces the basic
 * delegate tool with one that returns structured "episodes" — compressed
 * representations of completed work that include:
 *   - Summary of what was accomplished
 *   - Key discoveries and decisions made
 *   - Files read and modified
 *   - Errors encountered
 *   - Composable context for subsequent delegates
 *
 * Episodes can be chained: pass a prior episode ID to seed the next delegate
 * with relevant context, enabling Slate-style thread weaving without full
 * context transfer.
 *
 * Usage:
 *   pi --extension examples/extensions/episodic-delegate.ts
 *
 * The extension replaces the built-in delegate tool. Use it the same way,
 * but with additional parameters:
 *   - priorEpisodes: array of episode IDs to seed context from
 *   - episodeLabel: optional human-readable label for the episode
 */

import { spawn } from "node:child_process";
import type { AgentToolResult, ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Container, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Episode Types
// ============================================================================

interface Episode {
	id: string;
	label?: string;
	timestamp: number;
	task: string;
	cwd: string;
	model?: string;

	// Structured results
	summary: string;
	discoveries: string[];
	decisions: string[];
	filesRead: string[];
	filesModified: string[];
	errors: string[];

	// Execution metadata
	durationMs: number;
	exitCode: number;
	turns: number;
	usage: EpisodeUsage;

	// Raw final output for backward compat
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

interface EpisodeStore {
	episodes: Map<string, Episode>;
}

interface EpisodicDelegateDetails {
	episode: Episode;
	priorEpisodeIds: string[];
}

// ============================================================================
// Episode extraction from delegate output
// ============================================================================

/**
 * Parse structured episode data from the delegate's final output.
 * The delegate is instructed to emit a structured block at the end.
 * Falls back to heuristic extraction if the structured block is missing.
 */
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

	// Try to find structured episode block
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

	// Heuristic extraction from raw output
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
		filesRead: extractFilePaths(rawOutput, "read"),
		filesModified: extractFilePaths(rawOutput, "modified"),
		errors: exitCode !== 0 ? [rawOutput.slice(0, 200)] : [],
		durationMs,
		exitCode,
		turns: usage.turns,
		usage,
		rawOutput,
	};
}

function generateEpisodeId(): string {
	return `ep_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function fallbackSummary(output: string): string {
	// Take first meaningful paragraph
	const lines = output.split("\n").filter((l) => l.trim().length > 0);
	if (lines.length === 0) return "(no output)";
	// Take up to 3 lines or 300 chars
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

function extractFilePaths(output: string, type: "read" | "modified"): string[] {
	const paths = new Set<string>();

	// Different patterns based on whether we're looking for read or modified files
	if (type === "modified") {
		// Look for write/edit/create patterns
		const writePatterns = [
			/(?:wrote|created|modified|updated|saved|edited)\s+(?:to\s+)?([^\s,:'"]+\.\w+)/gi,
			/(?:file|path):\s*([^\s,:'"]+\.\w+)/gi,
		];
		for (const regex of writePatterns) {
			let match: RegExpExecArray | null;
			while ((match = regex.exec(output)) !== null) {
				paths.add(match[1]);
			}
		}
	} else {
		// For reads, look for read/open/accessed patterns or just file paths
		const readPatterns = [
			/(?:read|opened|loaded|accessed)\s+([^\s,:'"]+\.\w+)/gi,
			/(?:file|path):\s*([^\s,:'"]+\.\w+)/gi,
		];
		for (const regex of readPatterns) {
			let match: RegExpExecArray | null;
			while ((match = regex.exec(output)) !== null) {
				paths.add(match[1]);
			}
		}

		// Fallback: find absolute/relative file paths if no explicit reads found
		if (paths.size === 0) {
			const pathRegex = /(?:^|\s)((?:\/|\.\/|~\/)[^\s:,'"]+\.\w+)/gm;
			let match: RegExpExecArray | null;
			while ((match = pathRegex.exec(output)) !== null) {
				paths.add(match[1]);
			}
		}
	}

	return Array.from(paths).slice(0, 20);
}

// ============================================================================
// Context building from prior episodes
// ============================================================================

function buildPriorEpisodesContext(store: EpisodeStore, episodeIds: string[]): string {
	const resolved = episodeIds.map((id) => store.episodes.get(id)).filter((e): e is Episode => e !== undefined);

	if (resolved.length === 0) return "";

	let context = "\n\n## Prior Episode Context\n\n";
	context += "The following episodes provide context from previous work in this session:\n\n";

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
		if (ep.errors.length > 0) {
			context += "**Errors encountered:**\n";
			for (const e of ep.errors) context += `- ${e}\n`;
		}
		context += "\n";
	}

	return context;
}

// ============================================================================
// Episode-aware task prompt
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
// Extension
// ============================================================================

export default function (pi: ExtensionAPI) {
	const store: EpisodeStore = { episodes: new Map() };

	// Register episodic delegate tool
	pi.registerTool({
		name: "delegate",
		label: "Episodic Delegate",
		description: [
			"Delegate a bounded task to a nested pi subprocess with isolated context.",
			"Returns a structured episode with summary, discoveries, decisions, and file operations.",
			"Pass priorEpisodes to seed context from earlier episodes (Slate-style thread weaving).",
			"Use episodeLabel for human-readable episode names.",
		].join(" "),
		parameters: Type.Object({
			task: Type.String({
				description: "Task to delegate to a nested pi run.",
			}),
			cwd: Type.Optional(
				Type.String({
					description: "Working directory for the delegated run. Defaults to current cwd.",
				}),
			),
			model: Type.Optional(
				Type.String({
					description: "Optional model override for delegated run.",
				}),
			),
			tools: Type.Optional(
				Type.Array(Type.String(), {
					description: "Optional allow-list of tool names.",
				}),
			),
			timeoutSeconds: Type.Optional(
				Type.Number({
					description: "Timeout for delegated run. Default: 240 seconds.",
				}),
			),
			outputMode: Type.Optional(Type.Union([Type.Literal("final"), Type.Literal("summary")])),
			priorEpisodes: Type.Optional(
				Type.Array(Type.String(), {
					description: "Episode IDs from prior delegates to seed as context. Enables composable thread weaving.",
				}),
			),
			episodeLabel: Type.Optional(
				Type.String({
					description: "Human-readable label for this episode (e.g. 'auth-refactor', 'test-setup').",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<AgentToolResult<EpisodicDelegateDetails>> {
			const started = Date.now();
			const timeoutSeconds = Math.min(Math.max(params.timeoutSeconds || 240, 1), 1800);

			// Build the augmented task with episode context and report instruction
			let augmentedTask = params.task;

			// Inject prior episode context
			const priorEpisodeIds = params.priorEpisodes || [];
			if (priorEpisodeIds.length > 0) {
				const priorContext = buildPriorEpisodesContext(store, priorEpisodeIds);
				if (priorContext) {
					augmentedTask = priorContext + "\n\n---\n\n" + augmentedTask;
				}
			}

			// Add episode report instruction
			augmentedTask += EPISODE_REPORT_INSTRUCTION;

			const args: string[] = ["--mode", "json", "-p", "--no-session", augmentedTask];
			if (params.model?.trim()) args.unshift("--model", params.model.trim());
			const toolFilter = params.tools?.filter((t) => t.trim().length > 0);
			if (toolFilter && toolFilter.length > 0) {
				args.push("--tools", toolFilter.join(","));
			}

			const delegatedCwd = params.cwd?.trim() || ctx.cwd;
			const usage: EpisodeUsage = {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				cost: 0,
				turns: 0,
			};
			let finalText = "";
			let model = params.model?.trim();
			let stopReason: string | undefined;
			let errorMessage: string | undefined;
			let stderr = "";
			let timedOut = false;

			// Run the delegate subprocess
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
					stopReason = parsed.message.stopReason ?? stopReason;
					errorMessage = parsed.message.errorMessage ?? errorMessage;
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
					stderr += chunk.toString();
				});

				proc.on("error", (error: NodeJS.ErrnoException) => {
					clearTimeout(timeout);
					errorMessage = error.code === "ENOENT" ? "pi executable not found in PATH." : error.message;
					resolve(127);
				});

				proc.on("close", (code: number | null) => {
					clearTimeout(timeout);
					if (buffer.trim()) parseLine(buffer);
					resolve(code ?? 0);
				});
			});

			if (timedOut) {
				errorMessage = `Delegated run timed out after ${timeoutSeconds}s.`;
			}

			const durationMs = Date.now() - started;

			// Extract structured episode
			const episode = extractEpisodeFromOutput(
				finalText,
				params.task,
				delegatedCwd,
				model,
				durationMs,
				exitCode,
				usage,
				params.episodeLabel,
			);

			// Store episode for future composition
			store.episodes.set(episode.id, episode);

			const details: EpisodicDelegateDetails = {
				episode,
				priorEpisodeIds,
			};

			if (exitCode !== 0 || errorMessage) {
				const message = errorMessage ?? (stderr.trim() || `Delegate failed with exit code ${exitCode}.`);
				return {
					content: [{ type: "text", text: `Delegate error: ${message}` }],
					details,
					isError: true,
				};
			}

			// Build the episode summary response
			const episodeResponse = formatEpisodeResponse(episode);
			return {
				content: [{ type: "text", text: episodeResponse }],
				details,
			};
		},

		renderCall(args, theme) {
			const taskPreview = args.task.length > 72 ? `${args.task.slice(0, 72)}...` : args.task;
			let text = theme.fg("toolTitle", theme.bold("delegate ")) + theme.fg("muted", `"${taskPreview}"`);
			if (args.episodeLabel) text += ` ${theme.fg("accent", `[${args.episodeLabel}]`)}`;
			if (args.priorEpisodes?.length) text += ` ${theme.fg("dim", `← ${args.priorEpisodes.length} prior`)}`;
			if (args.model) text += ` ${theme.fg("accent", `model=${args.model}`)}`;
			if (args.timeoutSeconds) text += ` ${theme.fg("dim", `${Math.min(Math.max(args.timeoutSeconds, 1), 1800)}s`)}`;
			return new Text(text, 0, 0);
		},

		renderResult(result, { expanded }, theme) {
			const details = result.details as EpisodicDelegateDetails | undefined;
			if (!details?.episode) {
				const block = result.content[0];
				return new Text(block?.type === "text" ? block.text : "", 0, 0);
			}

			const ep = details.episode;
			const isError = ep.exitCode !== 0 || ep.errors.length > 0;
			const icon = isError ? theme.fg("error", "✗") : theme.fg("success", "✓");

			if (!expanded) {
				// Compact view
				const lines = [
					`${icon} ${theme.fg("toolTitle", theme.bold("episode"))} ${theme.fg("accent", ep.label || ep.id)} ${theme.fg("dim", `${(ep.durationMs / 1000).toFixed(1)}s`)}`,
					theme.fg("muted", ep.summary.split("\n")[0].slice(0, 100)),
				];
				if (ep.filesModified.length > 0) {
					lines.push(
						theme.fg(
							"dim",
							`files: ${ep.filesModified.slice(0, 3).join(", ")}${ep.filesModified.length > 3 ? ` +${ep.filesModified.length - 3}` : ""}`,
						),
					);
				}
				lines.push(
					theme.fg(
						"dim",
						`turns ${ep.usage.turns} | in ${ep.usage.input} | out ${ep.usage.output} | $${ep.usage.cost.toFixed(4)}`,
					),
				);
				if (details.priorEpisodeIds.length > 0) {
					lines.push(theme.fg("dim", `← composed from ${details.priorEpisodeIds.length} prior episode(s)`));
				}
				return new Text(lines.join("\n"), 0, 0);
			}

			// Expanded view
			const container = new Container();
			container.addChild(
				new Text(
					`${icon} ${theme.fg("toolTitle", theme.bold("episode"))} ${theme.fg("accent", ep.label || ep.id)}`,
					0,
					0,
				),
			);
			container.addChild(new Text(theme.fg("muted", "Task: ") + theme.fg("dim", ep.task), 0, 0));
			container.addChild(new Spacer(1));
			container.addChild(new Text(theme.fg("muted", "─── Summary ───"), 0, 0));
			container.addChild(new Text(ep.summary, 0, 0));

			if (ep.discoveries.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Discoveries ───"), 0, 0));
				for (const d of ep.discoveries) {
					container.addChild(new Text(theme.fg("dim", `• ${d}`), 0, 0));
				}
			}

			if (ep.decisions.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Decisions ───"), 0, 0));
				for (const d of ep.decisions) {
					container.addChild(new Text(theme.fg("dim", `• ${d}`), 0, 0));
				}
			}

			if (ep.filesModified.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Files Modified ───"), 0, 0));
				for (const f of ep.filesModified) {
					container.addChild(new Text(theme.fg("accent", `  ${f}`), 0, 0));
				}
			}

			if (ep.errors.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("error", "─── Errors ───"), 0, 0));
				for (const e of ep.errors) {
					container.addChild(new Text(theme.fg("error", `  ${e}`), 0, 0));
				}
			}

			container.addChild(new Spacer(1));
			container.addChild(
				new Text(
					theme.fg(
						"dim",
						`${(ep.durationMs / 1000).toFixed(1)}s | turns ${ep.usage.turns} | in ${ep.usage.input} | out ${ep.usage.output} | $${ep.usage.cost.toFixed(4)}`,
					),
					0,
					0,
				),
			);

			return container;
		},
	});

	// Register a command to list stored episodes
	pi.registerCommand("episodes", {
		description: "List episodes from delegate runs in this session",
		handler: async (_args, ctx) => {
			if (store.episodes.size === 0) {
				ctx.ui.notify("No episodes in this session yet.", "info");
				return;
			}

			const episodes = Array.from(store.episodes.values()).sort((a, b) => b.timestamp - a.timestamp);

			const lines: string[] = ["## Episodes\n"];
			for (const ep of episodes) {
				const status = ep.exitCode === 0 ? "✓" : "✗";
				const label = ep.label || ep.id;
				const duration = (ep.durationMs / 1000).toFixed(1);
				lines.push(`${status} **${label}** (${duration}s, ${ep.usage.turns} turns, $${ep.usage.cost.toFixed(4)})`);
				lines.push(`  ${ep.summary.split("\n")[0].slice(0, 80)}`);
				if (ep.filesModified.length > 0) {
					lines.push(`  Modified: ${ep.filesModified.slice(0, 3).join(", ")}`);
				}
				lines.push(`  ID: \`${ep.id}\``);
				lines.push("");
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}

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

function formatEpisodeResponse(episode: Episode): string {
	const lines: string[] = [];

	lines.push(`## Episode: ${episode.label || episode.id}`);
	lines.push("");
	lines.push(`**Summary:** ${episode.summary}`);
	lines.push("");

	if (episode.discoveries.length > 0) {
		lines.push("**Discoveries:**");
		for (const d of episode.discoveries) lines.push(`- ${d}`);
		lines.push("");
	}

	if (episode.decisions.length > 0) {
		lines.push("**Decisions:**");
		for (const d of episode.decisions) lines.push(`- ${d}`);
		lines.push("");
	}

	if (episode.filesModified.length > 0) {
		lines.push(`**Files modified:** ${episode.filesModified.join(", ")}`);
	}

	if (episode.filesRead.length > 0) {
		lines.push(`**Files read:** ${episode.filesRead.join(", ")}`);
	}

	if (episode.errors.length > 0) {
		lines.push("**Errors:**");
		for (const e of episode.errors) lines.push(`- ${e}`);
	}

	lines.push("");
	lines.push(
		`*Episode ID: \`${episode.id}\` — pass this to \`priorEpisodes\` in subsequent delegates to compose context.*`,
	);

	// Include the raw output below the structured episode
	if (episode.rawOutput && episode.rawOutput !== episode.summary) {
		lines.push("");
		lines.push("---");
		lines.push("");
		lines.push(episode.rawOutput);
	}

	return lines.join("\n");
}

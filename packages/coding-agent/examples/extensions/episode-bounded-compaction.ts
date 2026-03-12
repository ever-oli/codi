/**
 * Episode-Bounded Compaction Extension
 *
 * Inspired by Slate's episodic memory architecture, this extension replaces
 * the default mechanical compaction (cut at token threshold) with semantic,
 * episode-aware compaction that:
 *
 * 1. Detects natural episode boundaries in the conversation:
 *    - Delegate/subagent completions
 *    - Git commits (detected via bash tool calls)
 *    - Todo completions
 *    - Explicit user markers (/checkpoint)
 *    - File write sequences followed by verification
 *
 * 2. Compresses completed episodes more aggressively while preserving
 *    the current in-progress episode with full fidelity.
 *
 * 3. Maintains a structured episode log that survives compaction,
 *    giving the model a reliable map of what happened even after
 *    the raw context is gone.
 *
 * The key insight from Slate: compression should happen at completion
 * boundaries, not arbitrary token thresholds. A completed task compresses
 * cleanly because it has a defined start, middle, and end. An in-progress
 * task cannot be safely compressed.
 *
 * Usage:
 *   pi --extension examples/extensions/episode-bounded-compaction.ts
 *
 * Commands:
 *   /checkpoint [label]  - Mark the current point as an episode boundary
 *   /episode-log         - View the accumulated episode log
 */

import { complete } from "@mariozechner/pi-ai";
import type { ExtensionAPI, SessionEntry } from "@mariozechner/pi-coding-agent";
import { convertToLlm, serializeConversation } from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";

// ============================================================================
// Episode Boundary Detection
// ============================================================================

interface EpisodeBoundary {
	/** Index in the entries array where this boundary sits */
	entryIndex: number;
	/** What type of boundary this is */
	type: "delegate_complete" | "git_commit" | "todo_complete" | "checkpoint" | "write_verify" | "user_turn";
	/** Human-readable label */
	label: string;
}

/**
 * Scan session entries and find natural episode boundaries.
 * These are points where a logical unit of work completed.
 */
function detectEpisodeBoundaries(entries: SessionEntry[], startIndex: number, endIndex: number): EpisodeBoundary[] {
	const boundaries: EpisodeBoundary[] = [];

	for (let i = startIndex; i < endIndex; i++) {
		const entry = entries[i];

		if (entry.type === "message") {
			const msg = entry.message;

			// Delegate completions - strong boundary
			if (
				msg.role === "toolResult" &&
				"toolName" in msg &&
				(msg.toolName === "delegate" || msg.toolName === "subagent")
			) {
				boundaries.push({
					entryIndex: i,
					type: "delegate_complete",
					label: `Delegate completed`,
				});
				continue;
			}

			// Git commit detection from bash tool results
			if (msg.role === "toolResult" && "toolName" in msg && msg.toolName === "bash") {
				const content = extractTextContent(msg.content);
				if (isGitCommit(content)) {
					boundaries.push({
						entryIndex: i,
						type: "git_commit",
						label: extractCommitMessage(content),
					});
					continue;
				}
			}

			// Bash execution with git commit
			if (msg.role === "bashExecution") {
				const bashMsg = msg as any;
				const bashText = `${bashMsg.command} ${bashMsg.output || ""}`;
				if (isGitCommit(bashText)) {
					boundaries.push({
						entryIndex: i,
						type: "git_commit",
						label: extractCommitMessage(bashText),
					});
					continue;
				}
			}

			// Todo tool completions
			if (msg.role === "toolResult" && "toolName" in msg && msg.toolName === "todo") {
				const content = extractTextContent(msg.content);
				if (content.includes("toggled") || content.includes("completed") || content.includes("[x]")) {
					boundaries.push({
						entryIndex: i,
						type: "todo_complete",
						label: "Todo item completed",
					});
					continue;
				}
			}
		}

		// Custom checkpoint entries (from /checkpoint command)
		if (entry.type === "custom_message" && entry.customType === "episode-checkpoint") {
			boundaries.push({
				entryIndex: i,
				type: "checkpoint",
				label: typeof entry.content === "string" ? entry.content : "Manual checkpoint",
			});
			continue;
		}

		// User turns - natural boundary for context shift
		if (entry.type === "message" && entry.message.role === "user") {
			const content = extractTextContent(entry.message.content);
			if (content.length > 20) {
				boundaries.push({
					entryIndex: i,
					type: "user_turn",
					label: `User: ${content.slice(0, 50)}...`,
				});
			}
		}
	}

	// Post-pass: detect write-verify sequences (write tool followed by read/bash verification)
	for (let i = startIndex; i < endIndex - 1; i++) {
		const entry = entries[i];
		const nextEntry = entries[i + 1];

		if (entry.type !== "message" || nextEntry.type !== "message") continue;
		if (entry.message.role !== "toolResult" || nextEntry.message.role !== "toolResult") continue;

		const firstTool = (entry.message as any).toolName;
		const secondTool = (nextEntry.message as any).toolName;

		// Write followed by read or bash (verification)
		if ((firstTool === "write" || firstTool === "edit") && (secondTool === "read" || secondTool === "bash")) {
			const writeContent = extractTextContent(entry.message.content);
			const fileMatch = writeContent.match(/(?:wrote|created|modified|edited)\s+(.+)/i);
			const label = fileMatch ? `Write+Verify: ${fileMatch[1].slice(0, 40)}` : "Write+Verify sequence";

			boundaries.push({
				entryIndex: i + 1,
				type: "write_verify",
				label,
			});
		}
	}

	// Sort boundaries by entry index
	boundaries.sort((a, b) => a.entryIndex - b.entryIndex);

	return boundaries;
}

function extractTextContent(content: string | Array<{ type: string; text?: string }>): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((c) => c.type === "text" && c.text)
		.map((c) => c.text!)
		.join("\n");
}

function isGitCommit(text: string): boolean {
	return (
		/git\s+commit/.test(text) ||
		/\[[\w-]+\s+[\da-f]+\]/.test(text) || // [branch abc1234] commit message
		/create mode \d+/.test(text)
	);
}

function extractCommitMessage(text: string): string {
	// Match [branch hash] message
	const match = text.match(/\[[\w/-]+\s+[\da-f]+\]\s+(.+)/);
	if (match) return `git: ${match[1].slice(0, 60)}`;
	return "git commit";
}

// ============================================================================
// Episode-aware summarization
// ============================================================================

const EPISODE_SUMMARIZATION_SYSTEM = `You are a context checkpoint summarizer for a coding agent. You create structured summaries that preserve the essential information from completed work episodes.

Your summaries must be:
- Accurate: preserve exact file paths, function names, error messages, and decisions
- Structured: use the provided format consistently
- Concise: completed episodes get compressed; in-progress work gets preserved in detail
- Composable: each episode summary should be self-contained enough to be useful on its own`;

const EPISODE_SUMMARIZATION_PROMPT = `Summarize the following conversation, organizing it by episodes (logical units of work).

The conversation contains these detected episode boundaries:
{boundaries}

Create a structured summary using this format:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements mentioned by user]

## Episode Log
{episode_format}

## Current State
### In Progress
- [ ] [Current work that hasn't completed yet]

### Blocked
- [Issues, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [What should happen next]

## Critical Context
- [Data needed to continue]

<read-files>
[paths]
</read-files>

<modified-files>
[paths]
</modified-files>

IMPORTANT:
- Completed episodes should be compressed to their essential outcomes
- The current in-progress episode (after the last boundary) should be preserved in more detail
- File paths and error messages must be exact
- Each episode entry should be self-contained`;

function buildEpisodeFormat(boundaries: EpisodeBoundary[]): string {
	if (boundaries.length === 0) {
		return `### Episode 1 (in progress)
- [Describe current work with key details]`;
	}

	let format = "";
	for (let i = 0; i < boundaries.length; i++) {
		format += `### Episode ${i + 1}: ${boundaries[i].label} ✓
- [x] [Key outcome]
- Files: [modified files]
- Decisions: [any key decisions]

`;
	}
	format += `### Episode ${boundaries.length + 1} (in progress)
- [ ] [Current work with key details preserved]`;
	return format;
}

function formatBoundaries(boundaries: EpisodeBoundary[]): string {
	if (boundaries.length === 0) return "(no completed episodes detected)";
	return boundaries.map((b, i) => `${i + 1}. [${b.type}] ${b.label} (at entry ${b.entryIndex})`).join("\n");
}

// ============================================================================
// Episode Log - persists across compactions
// ============================================================================

interface EpisodeLogEntry {
	timestamp: number;
	type: string;
	label: string;
	summary: string;
	filesModified: string[];
}

interface EpisodeLog {
	entries: EpisodeLogEntry[];
}

// ============================================================================
// Extension
// ============================================================================

export default function (pi: ExtensionAPI) {
	// Persistent episode log that survives compaction
	const episodeLog: EpisodeLog = { entries: [] };

	// Hook into compaction to provide episode-aware summarization
	pi.on("session_before_compact", async (event, ctx) => {
		const { preparation, branchEntries, signal } = event;
		const { messagesToSummarize, turnPrefixMessages, tokensBefore, firstKeptEntryId, previousSummary } = preparation;

		// Find the model for summarization
		const model = ctx.model;
		if (!model) {
			ctx.ui.notify("Episode compaction: no model available, falling back to default", "warning");
			return; // Fall through to default compaction
		}

		const apiKey = await ctx.modelRegistry.getApiKey(model);
		if (!apiKey) {
			ctx.ui.notify(`Episode compaction: no API key for ${model.provider}, falling back to default`, "warning");
			return;
		}

		// Detect episode boundaries in the entries being compacted
		// We need to map message indices to entry indices
		const allMessages = [...messagesToSummarize, ...turnPrefixMessages];
		if (allMessages.length === 0) return;

		// Detect boundaries in the full branch entries
		const boundaries = detectEpisodeBoundaries(branchEntries, 0, branchEntries.length);

		ctx.ui.notify(
			`Episode compaction: ${boundaries.length} episode boundaries detected in ${allMessages.length} messages`,
			"info",
		);

		// Convert messages to serialized text
		const llmMessages = convertToLlm(allMessages);
		const conversationText = serializeConversation(llmMessages);

		// Build the episode-aware prompt
		const boundaryText = formatBoundaries(boundaries);
		const episodeFormat = buildEpisodeFormat(boundaries);

		let prompt = EPISODE_SUMMARIZATION_PROMPT.replace("{boundaries}", boundaryText).replace(
			"{episode_format}",
			episodeFormat,
		);

		// Include previous summary if we have one
		let fullPrompt = `<conversation>\n${conversationText}\n</conversation>\n\n`;
		if (previousSummary) {
			fullPrompt += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
			prompt =
				prompt +
				"\n\nIMPORTANT: Merge with the previous summary. Preserve completed episodes from the previous summary and add new ones.";
		}

		// Include episode log if we have one
		if (episodeLog.entries.length > 0) {
			fullPrompt += `<episode-log>\n${formatEpisodeLog(episodeLog)}\n</episode-log>\n\n`;
			prompt +=
				"\n\nThe episode-log above contains episodes from before the previous compaction. Incorporate them into the Episode Log section.";
		}

		fullPrompt += prompt;

		try {
			const summaryMessages = [
				{
					role: "user" as const,
					content: [{ type: "text" as const, text: fullPrompt }],
					timestamp: Date.now(),
				},
			];

			const response = await complete(
				model,
				{
					systemPrompt: EPISODE_SUMMARIZATION_SYSTEM,
					messages: summaryMessages,
				},
				{
					apiKey,
					maxTokens: 8192,
					signal,
				},
			);

			if (response.stopReason === "aborted") {
				return { cancel: true };
			}

			const summary = response.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");

			if (!summary.trim()) {
				ctx.ui.notify("Episode compaction: empty summary, falling back to default", "warning");
				return;
			}

			// Update the episode log with newly detected completed episodes
			for (const boundary of boundaries) {
				episodeLog.entries.push({
					timestamp: Date.now(),
					type: boundary.type,
					label: boundary.label,
					summary: `Completed during compaction cycle`,
					filesModified: [],
				});
			}

			ctx.ui.notify(
				`Episode compaction: summarized ${allMessages.length} messages across ${boundaries.length} completed episodes`,
				"info",
			);

			return {
				compaction: {
					summary,
					firstKeptEntryId,
					tokensBefore,
					details: {
						episodeBoundaries: boundaries.length,
						episodeTypes: boundaries.map((b) => b.type),
						totalEpisodesLogged: episodeLog.entries.length,
					},
				},
			};
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Episode compaction failed: ${message}`, "error");
			return; // Fall back to default
		}
	});

	// Track episode boundaries in real-time via tool results
	let lastToolName: string | undefined;
	let lastWriteTarget: string | undefined;

	pi.on("tool_result", (event, _ctx) => {
		const { toolName, isError } = event;

		// Track delegate completions
		if ((toolName === "delegate" || toolName === "subagent") && !isError) {
			const content = event.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");

			episodeLog.entries.push({
				timestamp: Date.now(),
				type: "delegate_complete",
				label: `Delegate: ${content.slice(0, 60)}...`,
				summary: content.slice(0, 200),
				filesModified: [],
			});
		}

		// Track git commits
		if (toolName === "bash" && !isError) {
			const content = event.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");

			const input = event.input as { command?: string } | undefined;
			const fullText = `${input?.command || ""} ${content}`;

			if (isGitCommit(fullText)) {
				episodeLog.entries.push({
					timestamp: Date.now(),
					type: "git_commit",
					label: extractCommitMessage(fullText),
					summary: content.slice(0, 200),
					filesModified: [],
				});
			}
		}

		// Track write-verify sequences
		if ((toolName === "write" || toolName === "edit") && !isError) {
			const input = event.input as { path?: string; file_path?: string } | undefined;
			lastWriteTarget = input?.path || input?.file_path;
			lastToolName = toolName;
		} else if ((toolName === "read" || toolName === "bash") && lastWriteTarget && lastToolName) {
			// Read/bash after a write = verification step
			episodeLog.entries.push({
				timestamp: Date.now(),
				type: "write_verify",
				label: `Write+Verify: ${lastWriteTarget.split("/").pop() || lastWriteTarget}`,
				summary: `Wrote ${lastWriteTarget} then verified with ${toolName}`,
				filesModified: [lastWriteTarget],
			});
			lastWriteTarget = undefined;
			lastToolName = undefined;
		} else if (toolName !== "read" && toolName !== "bash") {
			// Reset if a different tool is used
			lastWriteTarget = undefined;
			lastToolName = undefined;
		}

		// Track todo completions
		if (toolName === "todo" && !isError) {
			const content = event.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");

			if (content.includes("toggled") || content.includes("[x]")) {
				episodeLog.entries.push({
					timestamp: Date.now(),
					type: "todo_complete",
					label: "Todo item completed",
					summary: content.slice(0, 100),
					filesModified: [],
				});
			}
		}
	});

	// /checkpoint command - manually mark an episode boundary
	pi.registerCommand("checkpoint", {
		description: "Mark the current point as an episode boundary for smarter compaction",
		handler: async (args, ctx) => {
			const label = args.trim() || `Checkpoint at ${new Date().toLocaleTimeString()}`;

			// Inject as a custom message so it appears in the conversation
			pi.sendMessage({
				customType: "episode-checkpoint",
				content: `📌 Episode checkpoint: ${label}`,
				display: true,
				details: { label, timestamp: Date.now() },
			});

			// Also add to the episode log
			episodeLog.entries.push({
				timestamp: Date.now(),
				type: "checkpoint",
				label,
				summary: "Manual checkpoint set by user",
				filesModified: [],
			});

			ctx.ui.notify(`Checkpoint set: ${label}`, "info");
		},
	});

	// /episode-log command - view the accumulated episode log
	pi.registerCommand("episode-log", {
		description: "View the accumulated episode log from this session",
		handler: async (_args, ctx) => {
			if (episodeLog.entries.length === 0) {
				ctx.ui.notify(
					"No episodes logged yet. Episodes are detected from delegate completions, git commits, todo completions, and /checkpoint markers.",
					"info",
				);
				return;
			}

			ctx.ui.notify(formatEpisodeLog(episodeLog), "info");
		},
	});

	// Render checkpoint messages
	pi.registerMessageRenderer("episode-checkpoint", (message, _options, _theme) => {
		const content =
			typeof message.content === "string"
				? message.content
				: message.content
						.filter((c: any): c is { type: "text"; text: string } => c.type === "text")
						.map((c: any) => c.text)
						.join("\n");
		return new Text(content, 0, 0);
	});
}

// ============================================================================
// Helpers
// ============================================================================

function formatEpisodeLog(log: EpisodeLog): string {
	const lines: string[] = ["## Episode Log\n"];

	for (let i = 0; i < log.entries.length; i++) {
		const entry = log.entries[i];
		const time = new Date(entry.timestamp).toLocaleTimeString();
		const icon =
			entry.type === "delegate_complete"
				? "🔄"
				: entry.type === "git_commit"
					? "📝"
					: entry.type === "todo_complete"
						? "✅"
						: entry.type === "checkpoint"
							? "📌"
							: entry.type === "write_verify"
								? "📄"
								: entry.type === "user_turn"
									? "💬"
									: "•";

		lines.push(`${i + 1}. ${icon} **${entry.label}** (${time}, ${entry.type})`);
		if (entry.summary && entry.summary !== "Manual checkpoint set by user") {
			lines.push(`   ${entry.summary.slice(0, 100)}`);
		}
		if (entry.filesModified.length > 0) {
			lines.push(`   Files: ${entry.filesModified.join(", ")}`);
		}
	}

	return lines.join("\n");
}

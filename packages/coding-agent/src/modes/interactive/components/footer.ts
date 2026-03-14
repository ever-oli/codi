import { type Component, truncateToWidth, visibleWidth } from "@mariozechner/pi-tui";
import type { AgentSession } from "../../../core/agent-session.js";
import type { ReadonlyFooterDataProvider } from "../../../core/footer-data-provider.js";
import { theme } from "../theme/theme.js";

/**
 * Sanitize text for display in a single-line status.
 * Removes newlines, tabs, carriage returns, and other control characters.
 */
function sanitizeStatusText(text: string): string {
	// Replace newlines, tabs, carriage returns with space, then collapse multiple spaces
	return text
		.replace(/[\r\n\t]/g, " ")
		.replace(/ +/g, " ")
		.trim();
}

/**
 * Format token counts (similar to web-ui)
 */
function formatTokens(count: number): string {
	if (count < 1000) return count.toString();
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
	return `${Math.round(count / 1000000)}M`;
}

/** Render a compact progress bar like [████░░░░] */
function renderProgressBar(percent: number, barWidth: number = 8): string {
	const filled = Math.round((Math.min(100, Math.max(0, percent)) / 100) * barWidth);
	return `[${"█".repeat(filled)}${"░".repeat(barWidth - filled)}]`;
}

/** Render a waveform sparkline from token rate samples using block characters. */
function renderWaveform(samples: number[], waveWidth: number = 12): string {
	if (samples.length === 0) return "─".repeat(waveWidth);
	const max = Math.max(...samples, 1);
	const blocks = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
	const step = Math.max(1, Math.floor(samples.length / waveWidth));
	const sampled: number[] = [];
	for (let i = 0; i < samples.length && sampled.length < waveWidth; i += step) {
		sampled.push(samples[i]);
	}
	return sampled
		.map((v) => blocks[Math.round((v / max) * (blocks.length - 1))])
		.join("")
		.padEnd(waveWidth, " ");
}

/**
 * Footer component that shows pwd, token stats, and context usage.
 * Computes token/context stats from session, gets git branch and extension statuses from provider.
 */
export class FooterComponent implements Component {
	private activeToolName: string | undefined;
	private lastResponseMs: number | undefined;
	private agentStartTime: number | undefined;
	// Tok/s waveform tracking
	private tokSamples: number[] = [];
	private lastSampleTime = 0;
	private lastOutputTokens = 0;
	// Breadcrumbs
	private currentFilePath: string | undefined;
	private currentLineNumber: number | undefined;
	// Rearview mirror - recent actions (#7)
	private recentActions: Array<{ action: string; time: number }> = [];
	// Touched files (#31)
	private touchedFiles: string[] = [];
	// Discovery treasures (#40)
	private discoveries: Array<{ text: string; time: number }> = [];

	constructor(
		private session: AgentSession,
		private footerData: ReadonlyFooterDataProvider,
	) {}

	setAutoCompactEnabled(enabled: boolean): void {
		void enabled;
	}

	setActiveTool(toolName: string | undefined): void {
		this.activeToolName = toolName;
	}

	markAgentStart(): void {
		this.agentStartTime = Date.now();
	}

	markAgentEnd(): void {
		if (this.agentStartTime) {
			this.lastResponseMs = Date.now() - this.agentStartTime;
			this.agentStartTime = undefined;
		}
	}

	/** Record token output sample for waveform. Called during streaming. */
	recordTokenSample(outputTokens: number): void {
		const now = Date.now();
		const elapsed = (now - this.lastSampleTime) / 1000;
		if (elapsed >= 0.5) {
			const newTokens = outputTokens - this.lastOutputTokens;
			const rate = elapsed > 0 ? newTokens / elapsed : 0;
			this.tokSamples.push(rate);
			if (this.tokSamples.length > 60) this.tokSamples.shift();
			this.lastOutputTokens = outputTokens;
			this.lastSampleTime = now;
		}
	}

	/** Set breadcrumb context (current file/line agent is working on). */
	setBreadcrumbs(filePath: string | undefined, lineNumber: number | undefined): void {
		this.currentFilePath = filePath;
		this.currentLineNumber = lineNumber;
	}

	/** Record a recent action for rearview mirror (#7). */
	recordAction(action: string): void {
		this.recentActions.push({ action, time: Date.now() });
		if (this.recentActions.length > 5) this.recentActions.shift();
	}

	/** Track a touched file (#31). */
	recordFileAccess(filePath: string): void {
		const short = filePath.split("/").slice(-2).join("/");
		this.touchedFiles = [short, ...this.touchedFiles.filter((f) => f !== short)].slice(0, 6);
	}

	/** Record a discovery/treasure (#40). */
	recordDiscovery(text: string): void {
		this.discoveries.push({ text, time: Date.now() });
		if (this.discoveries.length > 3) this.discoveries.shift();
	}

	setVerificationReport(report: string | undefined, show: boolean): void {
		void report;
		void show;
	}

	setHitlPause(reason: string | undefined): void {
		void reason;
	}

	recordCommandUsage(command: string): void {
		void command;
	}

	recordFileVisit(filePath: string): void {
		void filePath;
	}

	recordCostSnapshot(cost: number): void {
		void cost;
	}

	setPersona(persona: string | undefined): void {
		void persona;
	}

	setConstraints(constraints: readonly string[]): void {
		void constraints;
	}

	addSummaryCard(title: string, text: string): void {
		void title;
		void text;
	}

	setInterviewMode(active: boolean, question?: string): void {
		void active;
		void question;
	}

	setMemoryCount(count: number): void {
		void count;
	}

	setPendingDiffCount(count: number): void {
		void count;
	}

	setLogStreaming(active: boolean): void {
		void active;
	}

	setCondensedView(lines: readonly string[]): void {
		void lines;
	}

	/**
	 * No-op: git branch caching now handled by provider.
	 * Kept for compatibility with existing call sites in interactive-mode.
	 */
	invalidate(): void {
		// No-op: git branch is cached/invalidated by provider
	}

	/**
	 * Clean up resources.
	 * Git watcher cleanup now handled by provider.
	 */
	dispose(): void {
		// Git watcher cleanup handled by provider
	}

	render(width: number): string[] {
		const state = this.session.state;

		// Calculate cumulative usage from ALL session entries (not just post-compaction messages)
		let totalInput = 0;
		let totalOutput = 0;
		let totalCacheRead = 0;
		let totalCacheWrite = 0;
		let totalCost = 0;

		for (const entry of this.session.sessionManager.getEntries()) {
			if (entry.type === "message" && entry.message.role === "assistant") {
				totalInput += entry.message.usage.input;
				totalOutput += entry.message.usage.output;
				totalCacheRead += entry.message.usage.cacheRead;
				totalCacheWrite += entry.message.usage.cacheWrite;
				totalCost += entry.message.usage.cost.total;
			}
		}

		// Sample tok/s during active streaming
		if (this.agentStartTime) {
			this.recordTokenSample(totalOutput);
		}

		// Calculate context usage from session (handles compaction correctly).
		const contextUsage = this.session.getContextUsage();
		const contextPercentValue = contextUsage?.percent ?? 0;
		const contextPercent = contextUsage?.percent !== null ? contextPercentValue.toFixed(1) : "?";

		// Replace home directory with ~
		let pwd = process.cwd();
		const home = process.env.HOME || process.env.USERPROFILE;
		if (home && pwd.startsWith(home)) {
			pwd = `~${pwd.slice(home.length)}`;
		}

		// Add git branch if available
		const branch = this.footerData.getGitBranch();
		if (branch) {
			pwd = `${pwd} (${branch})`;
		}

		// Add session name if set
		const sessionName = this.session.sessionManager.getSessionName();
		if (sessionName) {
			pwd = `${pwd} • ${sessionName}`;
		}

		// Build stats line
		const statsParts = [];
		if (totalInput) statsParts.push(`↑${formatTokens(totalInput)}`);
		if (totalOutput) statsParts.push(`↓${formatTokens(totalOutput)}`);
		if (totalCacheRead) statsParts.push(`R${formatTokens(totalCacheRead)}`);
		if (totalCacheWrite) statsParts.push(`W${formatTokens(totalCacheWrite)}`);

		// Show cost with "(sub)" indicator if using OAuth subscription
		const usingSubscription = state.model ? this.session.modelRegistry.isUsingOAuth(state.model) : false;
		if (totalCost || usingSubscription) {
			const costStr = `$${totalCost.toFixed(3)}${usingSubscription ? " (sub)" : ""}`;
			statsParts.push(costStr);
		}

		// Show last response latency
		if (this.lastResponseMs !== undefined) {
			const latencyStr =
				this.lastResponseMs < 1000 ? `${this.lastResponseMs}ms` : `${(this.lastResponseMs / 1000).toFixed(1)}s`;
			statsParts.push(theme.fg("dim", `⏱${latencyStr}`));
		}

		// Context window bar (#15 Context Window Visualizer)
		const barWidth = 6;
		const contextBar = renderProgressBar(contextPercentValue, barWidth);
		let contextBarColored: string;
		if (contextPercentValue > 90) {
			contextBarColored = theme.fg("error", contextBar);
		} else if (contextPercentValue > 70) {
			contextBarColored = theme.fg("warning", contextBar);
		} else {
			contextBarColored = theme.fg("dim", contextBar);
		}
		statsParts.push(`${contextBarColored} ${contextPercent}%`);

		let statsLeft = statsParts.join(" ");

		// Add model name on the right side, plus thinking level if model supports it
		const modelName = state.model?.id || "no-model";

		let statsLeftWidth = visibleWidth(statsLeft);

		// If statsLeft is too wide, truncate it
		if (statsLeftWidth > width) {
			statsLeft = truncateToWidth(statsLeft, width, "...");
			statsLeftWidth = visibleWidth(statsLeft);
		}

		// Calculate available space for padding (minimum 2 spaces between stats and model)
		const minPadding = 2;

		// Add thinking level indicator if model supports reasoning
		let rightSideWithoutProvider = modelName;
		if (state.model?.reasoning) {
			const thinkingLevel = state.thinkingLevel || "off";
			rightSideWithoutProvider =
				thinkingLevel === "off" ? `${modelName} • thinking off` : `${modelName} • ${thinkingLevel}`;
		}

		// Add active tool indicator
		if (this.activeToolName) {
			const toolIcon = theme.fg("accent", `⟳ ${this.activeToolName}`);
			rightSideWithoutProvider = `${toolIcon} ${rightSideWithoutProvider}`;
		}

		// Prepend the provider in parentheses if there are multiple providers and there's enough room
		let rightSide = rightSideWithoutProvider;
		if (this.footerData.getAvailableProviderCount() > 1 && state.model) {
			rightSide = `(${state.model!.provider}) ${rightSideWithoutProvider}`;
			if (statsLeftWidth + minPadding + visibleWidth(rightSide) > width) {
				// Too wide, fall back
				rightSide = rightSideWithoutProvider;
			}
		}

		const rightSideWidth = visibleWidth(rightSide);
		const totalNeeded = statsLeftWidth + minPadding + rightSideWidth;

		let statsLine: string;
		if (totalNeeded <= width) {
			// Both fit - add padding to right-align model
			const padding = " ".repeat(width - statsLeftWidth - rightSideWidth);
			statsLine = statsLeft + padding + rightSide;
		} else {
			// Need to truncate right side
			const availableForRight = width - statsLeftWidth - minPadding;
			if (availableForRight > 0) {
				const truncatedRight = truncateToWidth(rightSide, availableForRight, "");
				const truncatedRightWidth = visibleWidth(truncatedRight);
				const padding = " ".repeat(Math.max(0, width - statsLeftWidth - truncatedRightWidth));
				statsLine = statsLeft + padding + truncatedRight;
			} else {
				// Not enough space for right side at all
				statsLine = statsLeft;
			}
		}

		// Apply dim to each part separately. statsLeft may contain color codes (for context %)
		// that end with a reset, which would clear an outer dim wrapper. So we dim the parts
		// before and after the colored section independently.
		const dimStatsLeft = theme.fg("dim", statsLeft);
		const remainder = statsLine.slice(statsLeft.length); // padding + rightSide
		const dimRemainder = theme.fg("dim", remainder);

		const pwdLine = truncateToWidth(theme.fg("dim", pwd), width, theme.fg("dim", "..."));
		const lines = [pwdLine, dimStatsLeft + dimRemainder];

		// Workflow info line: breadcrumbs + task progress + tok/s waveform
		const workflowParts: string[] = [];

		// Breadcrumbs (#8)
		if (this.currentFilePath) {
			const shortPath = this.currentFilePath.split("/").slice(-2).join("/");
			const lineInfo = this.currentLineNumber ? `:${this.currentLineNumber}` : "";
			workflowParts.push(theme.fg("muted", `📎${shortPath}${lineInfo}`));
		}

		// Task progress (#16)
		const workflow = this.session.workflow;
		if (workflow?.taskGraph) {
			const tasks = workflow.taskGraph.taskOrder;
			const doneCount = tasks.filter((id) => {
				const t = workflow.taskGraph.tasks[id];
				return t?.status === "done";
			}).length;
			if (tasks.length > 1) {
				const pct = (doneCount / tasks.length) * 100;
				workflowParts.push(theme.fg("dim", `${renderProgressBar(pct, 5)} ${doneCount}/${tasks.length}`));
			}
		}

		// Tok/s waveform (#1)
		if (this.tokSamples.length > 2) {
			const waveform = renderWaveform(this.tokSamples, 10);
			const avgRate = this.tokSamples.reduce((a, b) => a + b, 0) / this.tokSamples.length;
			workflowParts.push(`${theme.fg("accent", `⚡${waveform}`)}${theme.fg("dim", ` ${Math.round(avgRate)}/s`)}`);
		}

		if (workflowParts.length > 0) {
			lines.push(truncateToWidth(workflowParts.join("  "), width, theme.fg("dim", "...")));
		}

		// Rearview mirror / activity line (#7, #31, #40)
		const activityParts: string[] = [];

		// Recent actions (rearview mirror)
		if (this.recentActions.length > 0) {
			const recent = this.recentActions
				.slice(-3)
				.map((a) => a.action)
				.join(" → ");
			activityParts.push(theme.fg("dim", `◀ ${recent}`));
		}

		// Touched files
		if (this.touchedFiles.length > 0) {
			activityParts.push(theme.fg("muted", `📂${this.touchedFiles.slice(0, 3).join(",")}`));
		}

		// Discoveries
		if (this.discoveries.length > 0) {
			const latest = this.discoveries[this.discoveries.length - 1];
			activityParts.push(theme.fg("warning", `💎${latest.text}`));
		}

		if (activityParts.length > 0) {
			lines.push(truncateToWidth(activityParts.join("  "), width, theme.fg("dim", "...")));
		}

		// Action queue (#21) - show pending tasks
		if (workflow?.taskGraph) {
			const pendingTasks = workflow.taskGraph.taskOrder
				.filter((id) => {
					const t = workflow.taskGraph.tasks[id];
					return t && t.status !== "done" && t.status !== "waived";
				})
				.slice(0, 4);
			if (pendingTasks.length > 0) {
				const queueLine = pendingTasks
					.map((id) => {
						const t = workflow.taskGraph.tasks[id];
						const statusIcon = t?.status === "in_progress" ? "▶" : t?.status === "blocked" ? "⏸" : "○";
						return `${statusIcon} ${t?.goal?.slice(0, 30) ?? id}`;
					})
					.join("  ");
				lines.push(theme.fg("dim", truncateToWidth(`Queue: ${queueLine}`, width, "...")));
			}
		}

		// Add extension statuses on a single line, sorted by key alphabetically
		const extensionStatuses = this.footerData.getExtensionStatuses();
		if (extensionStatuses.size > 0) {
			const sortedStatuses = Array.from(extensionStatuses.entries())
				.sort(([a], [b]) => a.localeCompare(b))
				.map(([, text]) => sanitizeStatusText(text));
			const statusLine = sortedStatuses.join(" ");
			// Truncate to terminal width with dim ellipsis for consistency with footer style
			lines.push(truncateToWidth(statusLine, width, theme.fg("dim", "...")));
		}

		return lines;
	}
}

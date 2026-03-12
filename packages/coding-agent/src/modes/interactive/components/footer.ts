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
	// Verification report (#50)
	private verificationReport: Array<{ criteria: string; passed: boolean }> = [];
	private showVerificationReport = false;
	// HITL pause state (#13)
	private hitlPauseReason: string | undefined;
	private hitlPending = false;
	// Command familiarity (#3)
	private commandUsage: Map<string, number> = new Map();
	// Codebase heatmap (#6)
	private fileVisitCounts: Map<string, number> = new Map();
	// Token burn rate trend (#10)
	private costSnapshots: Array<{ cost: number; time: number }> = [];
	// Personality toggle (#26)
	private currentPersona: string | undefined;
	// Constraint toggle (#38)
	private activeConstraints: string[] = [];
	// Summary cards (#39)
	private summaryCards: Array<{ title: string; text: string }> = [];
	// Interview mode (#46)
	private interviewMode = false;
	private interviewQuestion: string | undefined;
	// Memory log count (#19)
	private memoryCount = 0;
	// Pending diff indicator (#24)
	private pendingDiffCount = 0;
	// Live log toggle (#28)
	private logStreaming = false;
	// Condensed view (#47)
	private condensedView: string[] = [];

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

	/** Set verification report for display (#50). */
	setVerificationReport(report: Array<{ criteria: string; passed: boolean }>, show: boolean = true): void {
		this.verificationReport = report;
		this.showVerificationReport = show;
	}

	/** Set HITL pause state (#13). */
	setHitlPause(reason: string | undefined): void {
		this.hitlPauseReason = reason;
		this.hitlPending = reason !== undefined;
	}

	/** Track slash command usage (#3). */
	recordCommandUsage(command: string): void {
		this.commandUsage.set(command, (this.commandUsage.get(command) ?? 0) + 1);
	}

	/** Track file visits for heatmap (#6). */
	recordFileVisit(filePath: string): void {
		const short = filePath.split("/").slice(-2).join("/");
		this.fileVisitCounts.set(short, (this.fileVisitCounts.get(short) ?? 0) + 1);
	}

	/** Record cost snapshot for burn rate (#10). */
	recordCostSnapshot(cost: number): void {
		this.costSnapshots.push({ cost, time: Date.now() });
		if (this.costSnapshots.length > 20) this.costSnapshots.shift();
	}

	/** Set current persona (#26). */
	setPersona(persona: string | undefined): void {
		this.currentPersona = persona;
	}

	/** Set active constraints (#38). */
	setConstraints(constraints: string[]): void {
		this.activeConstraints = constraints;
	}

	/** Add summary card (#39). */
	addSummaryCard(title: string, text: string): void {
		this.summaryCards.push({ title, text });
		if (this.summaryCards.length > 3) this.summaryCards.shift();
	}

	/** Set interview mode (#46). */
	setInterviewMode(active: boolean, question?: string): void {
		this.interviewMode = active;
		this.interviewQuestion = question;
	}

	/** Set memory count (#19). */
	setMemoryCount(count: number): void {
		this.memoryCount = count;
	}

	/** Set pending diff count (#24). */
	setPendingDiffCount(count: number): void {
		this.pendingDiffCount = count;
	}

	/** Set log streaming state (#28). */
	setLogStreaming(active: boolean): void {
		this.logStreaming = active;
	}

	/** Set condensed view lines (#47). */
	setCondensedView(lines: string[]): void {
		this.condensedView = lines;
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

		// Truncate path if too long to fit width
		if (pwd.length > width) {
			const half = Math.floor(width / 2) - 2;
			if (half > 1) {
				const start = pwd.slice(0, half);
				const end = pwd.slice(-(half - 1));
				pwd = `${start}...${end}`;
			} else {
				pwd = pwd.slice(0, Math.max(1, width));
			}
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
			// Truncate statsLeft to fit width (no room for right side)
			const plainStatsLeft = statsLeft.replace(/\x1b\[[0-9;]*m/g, "");
			statsLeft = `${plainStatsLeft.substring(0, width - 3)}...`;
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
			if (availableForRight > 3) {
				// Truncate to fit (strip ANSI codes for length calculation, then truncate raw string)
				const plainRightSide = rightSide.replace(/\x1b\[[0-9;]*m/g, "");
				const truncatedPlain = plainRightSide.substring(0, availableForRight);
				// For simplicity, just use plain truncated version (loses color, but fits)
				const padding = " ".repeat(width - statsLeftWidth - truncatedPlain.length);
				statsLine = statsLeft + padding + truncatedPlain;
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

		const lines = [theme.fg("dim", pwd), dimStatsLeft + dimRemainder];

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

		// Verification report (#50)
		if (this.showVerificationReport && this.verificationReport.length > 0) {
			const reportLine = this.verificationReport
				.map((r) => (r.passed ? theme.fg("success", `✓ ${r.criteria}`) : theme.fg("error", `✗ ${r.criteria}`)))
				.join("  ");
			lines.push(truncateToWidth(reportLine, width, theme.fg("dim", "...")));
		}

		// HITL pause indicator (#13)
		if (this.hitlPending && this.hitlPauseReason) {
			lines.push(theme.fg("warning", `⏸ Awaiting approval: ${this.hitlPauseReason}`));
		}

		// Status indicators line: persona, constraints, memory, diffs, logs (#5 #19 #24 #26 #28 #38 #46)
		const statusParts: string[] = [];

		// Interview mode (#46)
		if (this.interviewMode) {
			statusParts.push(theme.fg("accent", "❓ Interview"));
			if (this.interviewQuestion) {
				statusParts.push(theme.fg("muted", this.interviewQuestion.slice(0, 40)));
			}
		}

		// Personality (#26)
		if (this.currentPersona) {
			const personaIcons: Record<string, string> = { concise: "🎯", creative: "🎨", technical: "⚙️" };
			const icon = personaIcons[this.currentPersona] ?? "👤";
			statusParts.push(theme.fg("dim", `${icon}${this.currentPersona}`));
		}

		// Constraints (#38)
		if (this.activeConstraints.length > 0) {
			statusParts.push(theme.fg("dim", `🔒${this.activeConstraints.slice(0, 2).join(",")}`));
		}

		// Memory count (#19)
		if (this.memoryCount > 0) {
			statusParts.push(theme.fg("dim", `🧠${this.memoryCount}`));
		}

		// Pending diffs (#24)
		if (this.pendingDiffCount > 0) {
			statusParts.push(theme.fg("warning", `📝${this.pendingDiffCount} pending`));
		}

		// Log streaming (#28)
		if (this.logStreaming) {
			statusParts.push(theme.fg("dim", "📋streaming"));
		}

		// Command familiarity - top 3 commands (#3)
		if (this.commandUsage.size > 0) {
			const topCommands = Array.from(this.commandUsage.entries())
				.sort((a, b) => b[1] - a[1])
				.slice(0, 3)
				.map(([cmd, count]) => `${cmd}(${count})`);
			statusParts.push(theme.fg("dim", `⌨${topCommands.join(",")}`));
		}

		// Token burn rate trend (#10)
		if (this.costSnapshots.length >= 2) {
			const recent = this.costSnapshots.slice(-5);
			const trend = recent[recent.length - 1].cost - recent[0].cost;
			const trendIcon = trend > 0.01 ? "📈" : trend < -0.01 ? "📉" : "➡️";
			statusParts.push(theme.fg("dim", `${trendIcon}$${trend.toFixed(3)}/run`));
		}

		if (statusParts.length > 0) {
			lines.push(truncateToWidth(statusParts.join("  "), width, theme.fg("dim", "...")));
		}

		// Codebase heatmap - top visited files (#6)
		if (this.fileVisitCounts.size > 0) {
			const topFiles = Array.from(this.fileVisitCounts.entries())
				.sort((a, b) => b[1] - a[1])
				.slice(0, 4);
			const maxCount = topFiles[0]?.[1] ?? 1;
			const heatBlocks = [" ", "░", "▒", "▓", "█"];
			const heatLine = topFiles
				.map(([file, count]) => {
					const level = Math.min(4, Math.ceil((count / maxCount) * 4));
					return `${heatBlocks[level]}${file}`;
				})
				.join(" ");
			lines.push(truncateToWidth(theme.fg("dim", `🗺${heatLine}`), width, "..."));
		}

		// Summary cards (#39)
		if (this.summaryCards.length > 0) {
			const latest = this.summaryCards[this.summaryCards.length - 1];
			lines.push(truncateToWidth(theme.fg("muted", `📋 ${latest.title}: ${latest.text}`), width, "..."));
		}

		// Condensed view (#47)
		if (this.condensedView.length > 0) {
			lines.push(truncateToWidth(theme.fg("dim", this.condensedView.join(" › ")), width, "..."));
		}

		// Game-like controls hint (#5)
		if (this.agentStartTime) {
			lines.push(theme.fg("dim", "ESC:cancel  TAB:expand  Ctrl+C:interrupt"));
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

/**
 * Session-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import type { AgentSession } from "../../../core/agent-session.js";
import type { SessionManager } from "../../../core/session-manager.js";
import type { SettingsManager } from "../../../core/settings-manager.js";
import { getLatestTaskVerification } from "../../../core/workflow/session-orchestrator.js";
import { theme } from "../theme/theme.js";

export interface SessionCommandContext {
	readonly session: AgentSession;
	readonly sessionManager: SessionManager;
	readonly settingsManager: SettingsManager;

	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	updateTerminalTitle(): void;
	getWorkflowDisplayState(): WorkflowDisplayState;
	renderWidgets(): void;
	buildTaskExecutionContractText(taskId: string): string | undefined;
}

interface WorkflowDisplayState {
	goal: string;
	phase: string;
	status: string;
	activeTaskId?: string;
	activeTaskGoal?: string;
	activeTaskStatus?: string;
	activeTaskVerification?: string;
	activeTaskCompletion?: string;
	activeTaskCompletionReady: boolean;
	activeTaskCriteriaCount: number;
	activeTaskNotesCount: number;
	schedulableTasks: number;
	transitions: number;
	verification: number;
	artifacts: number;
}

export function handleNameCommand(ctx: SessionCommandContext, text: string): void {
	const name = text.replace(/^\/name\s*/, "").trim();
	if (!name) {
		const currentName = ctx.sessionManager.getSessionName();
		if (currentName) {
			ctx.showStatus(`Session name: ${currentName}`);
		} else {
			ctx.showWarning("Usage: /name <name>");
		}
		return;
	}

	ctx.sessionManager.appendSessionInfo(name);
	ctx.updateTerminalTitle();
	ctx.showStatus(`Session name set: ${name}`);
}

export function handleSessionCommand(ctx: SessionCommandContext): void {
	const stats = ctx.session.getSessionStats();
	const sessionName = ctx.sessionManager.getSessionName();
	const workflow = ctx.getWorkflowDisplayState();
	const workflowSnapshot = ctx.session.workflow;
	const latestVerification =
		workflow.activeTaskId !== undefined
			? getLatestTaskVerification(workflowSnapshot, workflow.activeTaskId)
			: undefined;
	const latestCommand =
		workflowSnapshot.workspace.lastCommandResults.length > 0
			? workflowSnapshot.workspace.lastCommandResults[workflowSnapshot.workspace.lastCommandResults.length - 1]
			: undefined;
	const latestTest =
		workflowSnapshot.workspace.testResults.length > 0
			? workflowSnapshot.workspace.testResults[workflowSnapshot.workspace.testResults.length - 1]
			: undefined;
	const changedFilesPreview = workflowSnapshot.workspace.changedFiles.slice(0, 3);

	let info = `${theme.bold("Session Info")}\n\n`;
	if (sessionName) {
		info += `${theme.fg("dim", "Name:")} ${sessionName}\n`;
	}
	info += `${theme.fg("dim", "File:")} ${stats.sessionFile ?? "In-memory"}\n`;
	info += `${theme.fg("dim", "ID:")} ${stats.sessionId}\n\n`;
	info += `${theme.bold("Messages")}\n`;
	info += `${theme.fg("dim", "User:")} ${stats.userMessages}\n`;
	info += `${theme.fg("dim", "Assistant:")} ${stats.assistantMessages}\n`;
	info += `${theme.fg("dim", "Tool Calls:")} ${stats.toolCalls}\n`;
	info += `${theme.fg("dim", "Tool Results:")} ${stats.toolResults}\n`;
	info += `${theme.fg("dim", "Total:")} ${stats.totalMessages}\n\n`;
	info += `${theme.bold("Tokens")}\n`;
	info += `${theme.fg("dim", "Input:")} ${stats.tokens.input.toLocaleString()}\n`;
	info += `${theme.fg("dim", "Output:")} ${stats.tokens.output.toLocaleString()}\n`;
	if (stats.tokens.cacheRead > 0) {
		info += `${theme.fg("dim", "Cache Read:")} ${stats.tokens.cacheRead.toLocaleString()}\n`;
	}
	if (stats.tokens.cacheWrite > 0) {
		info += `${theme.fg("dim", "Cache Write:")} ${stats.tokens.cacheWrite.toLocaleString()}\n`;
	}
	info += `${theme.fg("dim", "Total:")} ${stats.tokens.total.toLocaleString()}\n`;

	if (stats.cost > 0) {
		info += `\n${theme.bold("Cost")}\n`;
		info += `${theme.fg("dim", "Total:")} ${stats.cost.toFixed(4)}`;
	}

	info += `\n\n${theme.bold("Workflow")}\n`;
	info += `${theme.fg("dim", "Goal:")} ${workflow.goal}\n`;
	info += `${theme.fg("dim", "Phase:")} ${workflow.phase}\n`;
	info += `${theme.fg("dim", "Status:")} ${workflow.status}\n`;
	if (workflow.activeTaskId) {
		info += `${theme.fg("dim", "Active Task ID:")} ${workflow.activeTaskId}\n`;
	}
	if (workflow.activeTaskGoal) {
		info += `${theme.fg("dim", "Active Task:")} ${workflow.activeTaskGoal}\n`;
	}
	if (workflow.activeTaskStatus) {
		info += `${theme.fg("dim", "Task Status:")} ${workflow.activeTaskStatus}\n`;
	}
	if (workflow.activeTaskVerification) {
		info += `${theme.fg("dim", "Task Verification:")} ${workflow.activeTaskVerification}\n`;
	}
	if (workflow.activeTaskCompletion) {
		info += `${theme.fg("dim", "Completion State:")} ${workflow.activeTaskCompletion}\n`;
	}
	info += `${theme.fg("dim", "Completion Ready:")} ${workflow.activeTaskCompletionReady ? "yes" : "no"}\n`;
	info += `${theme.fg("dim", "Acceptance Criteria:")} ${workflow.activeTaskCriteriaCount}\n`;
	info += `${theme.fg("dim", "Task Notes:")} ${workflow.activeTaskNotesCount}\n`;
	info += `${theme.fg("dim", "Schedulable Tasks:")} ${workflow.schedulableTasks}\n`;
	info += `${theme.fg("dim", "Transitions:")} ${workflow.transitions}\n`;
	info += `${theme.fg("dim", "Verification Records:")} ${workflow.verification}\n`;
	info += `${theme.fg("dim", "Artifacts:")} ${workflow.artifacts}\n`;
	info += `${theme.fg("dim", "Changed Files:")} ${workflowSnapshot.workspace.changedFiles.length}\n`;
	if (changedFilesPreview.length > 0) {
		info += `${theme.fg("dim", "Recent Changes:")} ${changedFilesPreview.join(", ")}\n`;
	}
	if (workflowSnapshot.workspace.git.branch) {
		info += `${theme.fg("dim", "Git Branch:")} ${workflowSnapshot.workspace.git.branch}\n`;
	}
	if (workflowSnapshot.workspace.git.head) {
		info += `${theme.fg("dim", "Git Head:")} ${workflowSnapshot.workspace.git.head}\n`;
	}
	if (workflowSnapshot.workspace.refreshedAt) {
		info += `${theme.fg("dim", "Workspace Refreshed:")} ${workflowSnapshot.workspace.refreshedAt}\n`;
	}
	if (latestCommand) {
		info += `${theme.fg("dim", "Last Command:")} ${latestCommand.command} (exit ${latestCommand.exitCode})\n`;
	}
	if (latestTest) {
		info += `${theme.fg("dim", "Latest Test:")} ${latestTest.command} (${latestTest.passed ? "passed" : "failed"})\n`;
	}
	if (latestVerification) {
		info += `${theme.fg("dim", "Latest Verification:")} ${latestVerification.status} for ${latestVerification.taskId}\n`;
	}
	if (workflow.activeTaskId) {
		const contractText = ctx.buildTaskExecutionContractText(workflow.activeTaskId);
		if (contractText) {
			info += `${theme.fg("dim", "Execution Contract:")}\n${contractText}\n`;
		}
	}

	ctx.showStatus(info);
}

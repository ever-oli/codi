/**
 * Workflow-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import type { AgentSession } from "../../../core/agent-session.js";
import {
	getActiveTaskCompletionState,
	getLatestTaskVerification,
	getTaskCompletionLabel,
	getTaskVerificationStatus,
	WORKFLOW_PHASES,
	type WorkflowPhase,
} from "../../../core/workflow/session-orchestrator.js";
import {
	areTaskDependenciesSatisfied,
	createTaskGraphFromGoal,
	type TaskStatus,
} from "../../../core/workflow/task-graph.js";

const WORKFLOW_TASK_STATUSES: TaskStatus[] = ["pending", "ready", "in_progress", "blocked", "done", "waived"];

export interface WorkflowCommandContext {
	readonly session: AgentSession;

	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	renderWidgets(): void;
	updateEditorBorderColor(): void;
}

function formatWorkflowLabel(value: string): string {
	return value.replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTaskCompletionLabel(value: string): string {
	switch (value) {
		case "completion_ready":
			return "Completion Ready";
		case "needs_verification":
			return "Needs Verification";
		case "incomplete":
			return "Incomplete";
		default:
			return value;
	}
}

function buildTaskExecutionContractText(ctx: WorkflowCommandContext, taskId: string): string | undefined {
	const workflow = ctx.session.workflow;
	const task = workflow.taskGraph.tasks[taskId];
	if (!task) return undefined;

	const depsReady = areTaskDependenciesSatisfied(workflow.taskGraph, taskId);
	const verification = getTaskVerificationStatus(workflow, taskId);
	const completion = getTaskCompletionLabel(workflow, taskId);

	const lines = [
		`Goal: ${task.goal}`,
		`Status: ${formatWorkflowLabel(task.status)}`,
		`Dependencies satisfied: ${depsReady ? "yes" : "no"}`,
		`Verification: ${formatWorkflowLabel(verification)}`,
		`Completion: ${formatTaskCompletionLabel(completion)}`,
	];

	if (task.acceptanceCriteria.length > 0) {
		lines.push("Acceptance criteria:");
		for (const criterion of task.acceptanceCriteria) {
			lines.push(`  - ${criterion}`);
		}
	}

	if (task.notes.length > 0) {
		lines.push("Notes:");
		for (const note of task.notes) {
			lines.push(`  - ${note}`);
		}
	}

	return lines.join("\n");
}

export function handleWorkflowPlanCommand(ctx: WorkflowCommandContext, text: string): void {
	const argText = text.replace(/^\/plan\s*/, "").trim();
	const workflow = ctx.session.workflow;
	const activeTaskId = workflow.taskGraph.activeTaskId;
	const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
	const activeTaskCompletion = getActiveTaskCompletionState(workflow);

	if (!argText || argText === "show") {
		const taskSummary = activeTask ? `${activeTask.id}: ${activeTask.goal} [${activeTask.status}]` : "none";
		ctx.showStatus(
			`Plan goal: ${workflow.goal}\nPhase: ${formatWorkflowLabel(workflow.currentPhase)}\nActive task: ${taskSummary}\nCompletion ready: ${activeTaskCompletion.completionReady ? "yes" : "no"}\nTasks: ${workflow.taskGraph.taskOrder.length}`,
		);
		return;
	}

	if (argText === "start") {
		if (workflow.currentPhase === "plan") {
			ctx.showStatus("Workflow is already in Plan");
			return;
		}
		try {
			ctx.session.transitionWorkflow("plan", "Manual planning start from /plan");
			ctx.renderWidgets();
			ctx.showStatus("Workflow phase: Plan");
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	if (argText.startsWith("goal ")) {
		const nextGoal = argText.slice(5).trim();
		if (!nextGoal) {
			ctx.showWarning("Usage: /plan goal <goal>");
			return;
		}

		const activeTaskGoalShouldTrackGoal =
			activeTask && workflow.taskGraph.taskOrder.length === 1 && activeTask.goal.trim() === workflow.goal.trim();

		const nextSnapshot = {
			...workflow,
			goal: nextGoal,
			taskGraph:
				activeTaskGoalShouldTrackGoal && activeTaskId
					? {
							...workflow.taskGraph,
							tasks: {
								...workflow.taskGraph.tasks,
								[activeTaskId]: {
									...activeTask,
									goal: nextGoal,
								},
							},
						}
					: workflow.taskGraph,
		};

		ctx.session.replaceWorkflowSnapshot(nextSnapshot);
		ctx.renderWidgets();
		ctx.showStatus(`Plan goal updated: ${nextGoal}`);
		return;
	}

	if (argText === "split") {
		const nextGraph = createTaskGraphFromGoal(workflow.goal, {
			existingGraph: workflow.taskGraph,
		});
		ctx.session.replaceWorkflowTaskGraph(nextGraph);
		ctx.session.recordWorkflowArtifact({
			id: `plan-split-${Date.now()}`,
			type: "plan",
			label: "Workflow task graph generated from goal",
			producer: "interactive:/plan split",
			metadata: {
				goal: workflow.goal,
				taskCount: nextGraph.taskOrder.length,
			},
		});
		ctx.renderWidgets();
		const activeTaskAfterSplit = nextGraph.activeTaskId ? nextGraph.tasks[nextGraph.activeTaskId] : undefined;
		ctx.showStatus(
			`Plan graph updated from goal\nTasks: ${nextGraph.taskOrder.length}\nActive task: ${activeTaskAfterSplit ? `${activeTaskAfterSplit.id} (${activeTaskAfterSplit.status})` : "none"}`,
		);
		return;
	}

	ctx.showWarning("Usage: /plan [show] | /plan start | /plan goal <goal> | /plan split");
}

export function handleWorkflowPhaseCommand(ctx: WorkflowCommandContext, text: string): void {
	const argText = text.replace(/^\/phase\s*/, "").trim();
	const currentPhase = ctx.session.workflow.currentPhase;

	if (!argText) {
		const phases = WORKFLOW_PHASES.join(" -> ");
		ctx.showStatus(`Workflow phase: ${formatWorkflowLabel(currentPhase)}\nFlow: ${phases}`);
		return;
	}

	const [phaseToken, ...reasonParts] = argText.split(/\s+/);
	const nextPhase = phaseToken as WorkflowPhase;
	if (!WORKFLOW_PHASES.includes(nextPhase)) {
		ctx.showWarning(`Unknown workflow phase "${phaseToken}"`);
		return;
	}

	const reason = reasonParts.join(" ").trim() || "Manual phase update from /phase";
	try {
		ctx.session.transitionWorkflow(nextPhase, reason);
		const nextSnapshot = ctx.session.workflow;
		const completionState = getActiveTaskCompletionState(nextSnapshot);
		ctx.renderWidgets();
		ctx.updateEditorBorderColor();
		let message = `Workflow phase: ${formatWorkflowLabel(currentPhase)} -> ${formatWorkflowLabel(nextPhase)}`;
		if (nextPhase === "summarize" && !completionState.completionReady) {
			message += `\nWarning: active task is ${formatTaskCompletionLabel(completionState.completionLabel)} and is not completion-ready.`;
		}
		ctx.showStatus(message);
	} catch (error) {
		ctx.showError(error instanceof Error ? error.message : String(error));
	}
}

export function handleWorkflowTaskCommand(ctx: WorkflowCommandContext, text: string): void {
	const argText = text.replace(/^\/task\s*/, "").trim();
	const workflow = ctx.session.workflow;

	if (!argText || argText === "list") {
		const lines = workflow.taskGraph.taskOrder.map((taskId) => {
			const task = workflow.taskGraph.tasks[taskId];
			if (!task) return undefined;
			const isActive = workflow.taskGraph.activeTaskId === taskId ? " *" : "";
			const verificationStatus = getTaskVerificationStatus(workflow, taskId);
			const completionLabel = getTaskCompletionLabel(workflow, taskId);
			const dependenciesReady = areTaskDependenciesSatisfied(workflow.taskGraph, taskId);
			return `${task.id}: ${task.goal} [${task.status}]${isActive} | deps=${dependenciesReady ? "ready" : "waiting"} | verification=${verificationStatus} | completion=${formatTaskCompletionLabel(completionLabel)} (${task.acceptanceCriteria.length} criteria, ${task.notes.length} notes)`;
		});
		const visibleLines = lines.filter((line): line is string => line !== undefined);
		ctx.showStatus(
			visibleLines.length > 0 ? `Workflow tasks:\n${visibleLines.join("\n")}` : "Workflow tasks: none yet",
		);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "add") {
		const taskId = rest[0];
		const goal = rest.slice(1).join(" ").trim();
		if (!taskId || !goal) {
			ctx.showWarning("Usage: /task add <id> <goal>");
			return;
		}
		try {
			ctx.session.upsertWorkflowTask({ id: taskId, goal, status: "ready" });
			ctx.renderWidgets();
			ctx.showStatus(`Workflow task added: ${taskId}`);
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	if (subcommand === "show") {
		const taskId = rest[0];
		if (!taskId) {
			ctx.showWarning("Usage: /task show <id>");
			return;
		}
		const task = workflow.taskGraph.tasks[taskId];
		if (!task) {
			ctx.showWarning(`Unknown workflow task "${taskId}"`);
			return;
		}
		const verificationStatus = getTaskVerificationStatus(workflow, taskId);
		const completionLabel = getTaskCompletionLabel(workflow, taskId);
		const latestVerification = getLatestTaskVerification(workflow, taskId);
		const dependenciesReady = areTaskDependenciesSatisfied(workflow.taskGraph, taskId);
		const contractText = buildTaskExecutionContractText(ctx, taskId) ?? "none";
		const criteria =
			task.acceptanceCriteria.length > 0
				? task.acceptanceCriteria.map((criterion) => `- ${criterion}`).join("\n")
				: "- none";
		const notes = task.notes.length > 0 ? task.notes.map((note) => `- ${note}`).join("\n") : "- none";
		ctx.showStatus(
			`Task ${task.id}: ${task.goal}\nStatus: ${formatWorkflowLabel(task.status)}\nDependencies: ${dependenciesReady ? "Satisfied" : "Waiting"}\nVerification: ${formatWorkflowLabel(verificationStatus)}\nCompletion: ${formatTaskCompletionLabel(completionLabel)}\nAcceptance criteria:\n${criteria}\nNotes:\n${notes}\nLatest verification details: ${latestVerification?.evidence.diffSummary ?? latestVerification?.evidence.userWaiver ?? latestVerification?.evidence.commands[0]?.details ?? "none"}\nExecution contract:\n${contractText}`,
		);
		return;
	}

	if (subcommand === "active") {
		const taskId = rest[0];
		if (!taskId) {
			ctx.showWarning("Usage: /task active <id>");
			return;
		}
		try {
			ctx.session.setWorkflowActiveTask(taskId);
			ctx.renderWidgets();
			ctx.showStatus(`Workflow active task: ${taskId}`);
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	if (subcommand === "status") {
		const taskId = rest[0];
		const statusToken = rest[1] as TaskStatus | undefined;
		if (!taskId || !statusToken) {
			ctx.showWarning("Usage: /task status <id> <pending|ready|in_progress|blocked|done|waived>");
			return;
		}
		if (!WORKFLOW_TASK_STATUSES.includes(statusToken)) {
			ctx.showWarning(`Unknown task status "${statusToken}"`);
			return;
		}
		try {
			ctx.session.updateWorkflowTaskStatus(taskId, statusToken);
			const nextWorkflow = ctx.session.workflow;
			const completionLabel = getTaskCompletionLabel(nextWorkflow, taskId);
			ctx.renderWidgets();
			ctx.showStatus(
				`Workflow task ${taskId}: ${formatWorkflowLabel(statusToken)}\nCompletion: ${formatTaskCompletionLabel(completionLabel)}`,
			);
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	if (subcommand === "criteria") {
		const taskId = rest[0];
		const criterion = rest.slice(1).join(" ").trim();
		if (!taskId || !criterion) {
			ctx.showWarning("Usage: /task criteria <id> <criterion>");
			return;
		}
		const task = workflow.taskGraph.tasks[taskId];
		if (!task) {
			ctx.showWarning(`Unknown workflow task "${taskId}"`);
			return;
		}
		try {
			ctx.session.updateWorkflowTask(taskId, {
				acceptanceCriteria: [...task.acceptanceCriteria, criterion],
			});
			ctx.renderWidgets();
			ctx.showStatus(`Workflow task ${taskId}: added acceptance criterion`);
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	if (subcommand === "note") {
		const taskId = rest[0];
		const note = rest.slice(1).join(" ").trim();
		if (!taskId || !note) {
			ctx.showWarning("Usage: /task note <id> <note>");
			return;
		}
		const task = workflow.taskGraph.tasks[taskId];
		if (!task) {
			ctx.showWarning(`Unknown workflow task "${taskId}"`);
			return;
		}
		try {
			ctx.session.updateWorkflowTask(taskId, {
				notes: [...task.notes, note],
			});
			ctx.renderWidgets();
			ctx.showStatus(`Workflow task ${taskId}: added note`);
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	ctx.showWarning(
		"Usage: /task [list] | /task show <id> | /task add <id> <goal> | /task active <id> | /task status <id> <status> | /task criteria <id> <criterion> | /task note <id> <note>",
	);
}

export function handleWorkflowVerifyCommand(ctx: WorkflowCommandContext, text: string): void {
	const argText = text.replace(/^\/verify\s*/, "").trim();
	const workflow = ctx.session.workflow;
	const activeTaskId = workflow.taskGraph.activeTaskId;
	if (!activeTaskId) {
		ctx.showWarning("No active workflow task. Use /task active <id> first.");
		return;
	}

	const [statusTokenRaw, ...detailParts] = argText ? argText.split(/\s+/) : [];
	const statusToken = statusTokenRaw?.toLowerCase();
	const detailText = detailParts.join(" ").trim();
	const verificationStatus =
		statusToken === "pass" || statusToken === "passed"
			? "passed"
			: statusToken === "fail" || statusToken === "failed"
				? "failed"
				: statusToken === "waive" || statusToken === "waived"
					? "waived"
					: undefined;

	if (!verificationStatus) {
		ctx.showWarning("Usage: /verify <passed|failed|waived> <details>");
		return;
	}

	ctx.session.recordWorkflowVerification(activeTaskId, verificationStatus, {
		tests: [],
		commands: [
			{
				command: "manual:/verify",
				validated: verificationStatus === "passed",
				details: detailText || undefined,
			},
		],
		userWaiver: verificationStatus === "waived" ? detailText || "Manual waiver recorded." : undefined,
		diffSummary: detailText || undefined,
	});
	ctx.session.recordWorkflowArtifact({
		id: `manual-verify-${Date.now()}`,
		type: "verification",
		label: `Manual verification for ${activeTaskId}`,
		producer: "interactive:/verify",
		metadata: {
			status: verificationStatus,
			details: detailText || null,
		},
	});
	const nextWorkflow = ctx.session.workflow;
	const completionLabel = getTaskCompletionLabel(nextWorkflow, activeTaskId);
	ctx.renderWidgets();
	ctx.showStatus(
		`Workflow verification for ${activeTaskId}: ${formatWorkflowLabel(verificationStatus)}\nCompletion: ${formatTaskCompletionLabel(completionLabel)}`,
	);
}

export function handleWorkflowSummaryCommand(ctx: WorkflowCommandContext): void {
	const workflow = ctx.session.workflow;
	const activeTaskId = workflow.taskGraph.activeTaskId;
	const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
	const completion = getActiveTaskCompletionState(workflow);

	const lines: string[] = [
		`Phase: ${formatWorkflowLabel(workflow.currentPhase)}`,
		`Status: ${formatWorkflowLabel(workflow.status)}`,
		`Goal: ${workflow.goal}`,
		`Active task: ${activeTask ? `${activeTask.id} - ${activeTask.goal} [${formatWorkflowLabel(activeTask.status)}]` : "none"}`,
		`Verification: ${formatWorkflowLabel(completion.verificationStatus)}`,
		`Completion: ${formatTaskCompletionLabel(completion.completionLabel)}`,
		`Completion ready: ${completion.completionReady ? "yes" : "no"}`,
		`Tasks: ${workflow.taskGraph.taskOrder.length}`,
	];

	ctx.showStatus(lines.join("\n"));
}

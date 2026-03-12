import type { AgentMessage, AgentTool } from "@mariozechner/pi-agent-core";
import type { ResourceLoader } from "./resource-loader.js";
import { buildSystemPrompt } from "./system-prompt.js";
import type { StaticContextEngine } from "./workflow/context-engine.js";
import {
	getActiveTaskCompletionState,
	type WorkflowPhase,
	type WorkflowSessionSnapshot,
} from "./workflow/session-orchestrator.js";
import { buildTaskSubagentContract, type SubagentInputContract } from "./workflow/subagents.js";
import type { WorkspaceState } from "./workflow/workspace-state.js";

export interface BuildAgentSystemPromptOptions {
	cwd: string;
	toolNames: string[];
	toolRegistry: ReadonlyMap<string, AgentTool>;
	toolPromptSnippets: ReadonlyMap<string, string>;
	toolPromptGuidelines: ReadonlyMap<string, readonly string[]>;
	resourceLoader: ResourceLoader;
}

export interface WorkflowPromptContext {
	workflow: WorkflowSessionSnapshot;
	workflowContextEngine: StaticContextEngine;
}

export type WorkflowEvidenceSummary = {
	latestCommand?: WorkspaceState["lastCommandResults"][number];
	latestTest?: WorkspaceState["testResults"][number];
	latestVerification?: WorkflowSessionSnapshot["verification"][number];
	artifactIds: string[];
};

export type WorkflowTurnContext = ReturnType<StaticContextEngine["buildContext"]>;

export function buildAgentSystemPrompt(options: BuildAgentSystemPromptOptions): string {
	const validToolNames = options.toolNames.filter((name) => options.toolRegistry.has(name));
	const toolSnippets: Record<string, string> = {};
	const promptGuidelines: string[] = [];

	for (const name of validToolNames) {
		const snippet = options.toolPromptSnippets.get(name);
		if (snippet) {
			toolSnippets[name] = snippet;
		}

		const toolGuidelines = options.toolPromptGuidelines.get(name);
		if (toolGuidelines) {
			promptGuidelines.push(...toolGuidelines);
		}
	}

	const loaderSystemPrompt = options.resourceLoader.getSystemPrompt();
	const loaderAppendSystemPrompt = options.resourceLoader.getAppendSystemPrompt();
	const appendSystemPrompt = loaderAppendSystemPrompt.length > 0 ? loaderAppendSystemPrompt.join("\n\n") : undefined;
	const loadedSkills = options.resourceLoader.getSkills().skills;
	const loadedContextFiles = options.resourceLoader.getAgentsFiles().agentsFiles;

	return buildSystemPrompt({
		cwd: options.cwd,
		skills: loadedSkills,
		contextFiles: loadedContextFiles,
		customPrompt: loaderSystemPrompt,
		appendSystemPrompt,
		selectedTools: validToolNames,
		toolSnippets,
		promptGuidelines,
	});
}

export function buildWorkflowPromptAppendix(context: WorkflowPromptContext): string | undefined {
	const { workflow } = context;
	const activeTaskId = workflow.taskGraph.activeTaskId;
	const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
	const activeTaskCompletion = getActiveTaskCompletionState(workflow);
	const workspaceEvidence = getWorkflowEvidenceSummary(workflow);
	const turnContext = buildWorkflowTurnContext(context);
	const executionContract = buildActiveTaskExecutionContract(context);

	const lines = [
		"## Workflow Context",
		`- Goal: ${workflow.goal}`,
		`- Phase: ${workflow.currentPhase}`,
		`- Status: ${workflow.status}`,
	];

	const phaseGuidance: Record<WorkflowPhase, string[]> = {
		intake: [
			"Clarify the goal, constraints, and desired outcome before deeper execution.",
			"Turn the request into a plan candidate instead of jumping straight into broad implementation.",
		],
		plan: [
			"Avoid broad edits unless the user explicitly asks for implementation now.",
			"Shape tasks, dependencies, acceptance criteria, and verification strategy before execution.",
			"Prefer workflow controls like /plan, /task, and /phase when they help tighten the plan.",
		],
		execute: [
			"Focus on the active task only and avoid widening scope.",
			"Prefer concrete edits and commands that directly advance the active task acceptance criteria.",
			"Use relevant workspace evidence to stay scoped to current changed files and recent command results.",
		],
		verify: [
			"Prioritize tests, commands, diffs, and evidence collection over new implementation work.",
			"Only introduce new edits if verification exposes a blocking defect.",
			"Do not treat the task as complete without passed or waived verification evidence.",
		],
		summarize: [
			"Summarize completed work, supporting evidence, remaining gaps, and natural next steps.",
			"Avoid introducing new implementation scope while summarizing.",
		],
	};
	lines.push("- Phase directives:");
	for (const directive of phaseGuidance[workflow.currentPhase]) {
		lines.push(`  - ${directive}`);
	}

	if (activeTask) {
		lines.push(`- Active task: ${activeTask.id} (${activeTask.status}) - ${activeTask.goal}`);
		lines.push(`- Active task verification: ${activeTaskCompletion.verificationStatus}`);
		lines.push(`- Active task completion state: ${activeTaskCompletion.completionLabel}`);
		if (activeTask.acceptanceCriteria.length > 0) {
			lines.push("- Acceptance criteria:");
			for (const criterion of activeTask.acceptanceCriteria) {
				lines.push(`  - ${criterion}`);
			}
		}
		if (activeTask.notes.length > 0) {
			lines.push("- Task notes:");
			for (const note of activeTask.notes) {
				lines.push(`  - ${note}`);
			}
		}
	}

	if (workflow.workspace.git.branch || workflow.workspace.git.head) {
		lines.push(
			`- Git: branch=${workflow.workspace.git.branch ?? "unknown"}, head=${workflow.workspace.git.head ?? "unknown"}`,
		);
	}
	if (workflow.workspace.changedFiles.length > 0) {
		lines.push(`- Changed files: ${workflow.workspace.changedFiles.slice(0, 8).join(", ")}`);
	}
	if (turnContext.relevantFiles.length > 0) {
		lines.push(`- Relevant files: ${turnContext.relevantFiles.join(", ")}`);
	}
	if (turnContext.shortSummary) {
		lines.push(`- Turn summary: ${turnContext.shortSummary}`);
	}
	if (workspaceEvidence.latestCommand) {
		lines.push(
			`- Latest command: ${workspaceEvidence.latestCommand.command} (exit ${workspaceEvidence.latestCommand.exitCode})`,
		);
	}
	if (workspaceEvidence.latestTest) {
		lines.push(
			`- Latest test result: ${workspaceEvidence.latestTest.command} (${workspaceEvidence.latestTest.passed ? "passed" : "failed"})`,
		);
	}
	if (workspaceEvidence.latestVerification && activeTaskId) {
		lines.push(
			`- Latest task verification record: ${activeTaskId} -> ${workspaceEvidence.latestVerification.status}`,
		);
	}
	if (workspaceEvidence.artifactIds.length > 0) {
		lines.push(`- Recent workflow artifacts: ${workspaceEvidence.artifactIds.join(", ")}`);
	}
	if (executionContract) {
		lines.push(`- Execution contract goal: ${executionContract.goal}`);
		if (executionContract.inputs.length > 0) {
			lines.push("- Execution contract inputs:");
			for (const input of executionContract.inputs) {
				lines.push(`  - ${input}`);
			}
		}
		if (executionContract.constraints.length > 0) {
			lines.push("- Execution contract constraints:");
			for (const constraint of executionContract.constraints) {
				lines.push(`  - ${constraint}`);
			}
		}
	}

	return lines.join("\n");
}

export function buildWorkflowControlMessage(context: WorkflowPromptContext): AgentMessage | undefined {
	const { workflow } = context;
	const turnContext = buildWorkflowTurnContext(context);
	const activeTaskCompletion = getActiveTaskCompletionState(workflow);
	const executionContract = buildActiveTaskExecutionContract(context);
	const lines = [
		`Workflow phase: ${workflow.currentPhase}`,
		`Workflow status: ${workflow.status}`,
		`Goal: ${workflow.goal}`,
		`Completion state: ${activeTaskCompletion.completionLabel}`,
		turnContext.shortSummary ? `Turn summary: ${turnContext.shortSummary}` : undefined,
		turnContext.relevantFiles.length > 0 ? `Relevant files: ${turnContext.relevantFiles.join(", ")}` : undefined,
		executionContract ? `Execution contract goal: ${executionContract.goal}` : undefined,
		executionContract && executionContract.inputs.length > 0
			? `Execution contract inputs: ${executionContract.inputs.join(" | ")}`
			: undefined,
		executionContract && executionContract.constraints.length > 0
			? `Execution contract constraints: ${executionContract.constraints.join(" | ")}`
			: undefined,
	].filter((line): line is string => line !== undefined);

	if (lines.length === 0) {
		return undefined;
	}

	return {
		role: "custom",
		customType: "workflow.control",
		content: lines.join("\n"),
		display: false,
		timestamp: Date.now(),
	};
}

export function getWorkflowEvidenceSummary(workflow: WorkflowSessionSnapshot): WorkflowEvidenceSummary {
	const activeTaskId = workflow.taskGraph.activeTaskId;
	return {
		latestCommand:
			workflow.workspace.lastCommandResults.length > 0
				? workflow.workspace.lastCommandResults[workflow.workspace.lastCommandResults.length - 1]
				: undefined,
		latestTest:
			workflow.workspace.testResults.length > 0
				? workflow.workspace.testResults[workflow.workspace.testResults.length - 1]
				: undefined,
		latestVerification: activeTaskId
			? [...workflow.verification].reverse().find((record) => record.taskId === activeTaskId)
			: undefined,
		artifactIds: workflow.workspace.artifactIds.slice(-5),
	};
}

export function buildWorkflowTurnContext(context: WorkflowPromptContext): WorkflowTurnContext {
	const { workflow, workflowContextEngine } = context;
	const activeTaskId = workflow.taskGraph.activeTaskId;
	const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
	const shortSummary = activeTask
		? `${workflow.currentPhase} phase for ${activeTask.id}: ${activeTask.goal}`
		: `${workflow.currentPhase} phase with no active task selected`;

	return workflowContextEngine.buildContext({
		task: activeTask,
		workspace: workflow.workspace,
		memory: workflow.memory,
		artifacts: workflow.artifacts,
		shortSummary,
		relevantFiles: [
			...workflow.workspace.changedFiles,
			...workflow.artifacts
				.slice(-5)
				.map((artifact) => artifact.path)
				.filter((path): path is string => path !== undefined),
		],
	});
}

export function buildActiveTaskExecutionContract(context: WorkflowPromptContext): SubagentInputContract | undefined {
	const { workflow } = context;
	const turnContext = buildWorkflowTurnContext(context);
	if (!turnContext.task) {
		return undefined;
	}
	const workspaceEvidence = getWorkflowEvidenceSummary(workflow);
	return buildTaskSubagentContract(turnContext.task, {
		phase: workflow.currentPhase,
		relevantFiles: turnContext.relevantFiles,
		extraInputs: [
			turnContext.shortSummary ? `summary:${turnContext.shortSummary}` : "",
			workspaceEvidence.latestCommand ? `last-command:${workspaceEvidence.latestCommand.command}` : "",
			workspaceEvidence.latestTest
				? `latest-test:${workspaceEvidence.latestTest.command}:${workspaceEvidence.latestTest.passed ? "passed" : "failed"}`
				: "",
		].filter((input) => input.length > 0),
	});
}

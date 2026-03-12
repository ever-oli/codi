import { spawnSync } from "node:child_process";
import type { BashResult } from "../bash-executor.js";
import type { WorkflowSessionSnapshot } from "./session-orchestrator.js";
import { createWorkspaceState, type WorkspaceState } from "./workspace-state.js";

export interface GitSnapshot {
	branch?: string;
	head?: string;
	statusSummary?: string;
	changedFiles: string[];
}

export function isVerificationCommand(command: string): boolean {
	return /\b(npm run check|npm run lint|npm run typecheck|npm test|pnpm test|pnpm lint|pnpm typecheck|yarn test|yarn lint|vitest|jest|pytest|cargo test|go test|tsc\b|eslint\b|biome check)\b/i.test(
		command,
	);
}

export function readGitSnapshot(cwd: string): GitSnapshot {
	const branchResult = spawnSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], {
		cwd,
		encoding: "utf8",
	});
	const headResult = spawnSync("git", ["rev-parse", "HEAD"], {
		cwd,
		encoding: "utf8",
	});
	const statusResult = spawnSync("git", ["status", "--short"], {
		cwd,
		encoding: "utf8",
	});

	if (statusResult.status !== 0) {
		return { changedFiles: [] };
	}

	const statusLines = statusResult.stdout
		.split("\n")
		.map((line) => line.trimEnd())
		.filter((line) => line.length > 0);

	return {
		branch: branchResult.status === 0 ? branchResult.stdout.trim() || undefined : undefined,
		head: headResult.status === 0 ? headResult.stdout.trim() || undefined : undefined,
		statusSummary: statusLines.join("\n") || undefined,
		changedFiles: statusLines.map((line) => line.slice(3).trim()).filter((line) => line.length > 0),
	};
}

export function createWorkspaceStateFromCommand(options: {
	cwd: string;
	workflow: WorkflowSessionSnapshot;
	command: string;
	result: BashResult;
}): WorkspaceState {
	const git = readGitSnapshot(options.cwd);
	const now = new Date().toISOString();
	const verificationCommand = isVerificationCommand(options.command);

	return createWorkspaceState({
		cwd: options.cwd,
		git: {
			branch: git.branch,
			head: git.head,
			statusSummary: git.statusSummary,
		},
		changedFiles: git.changedFiles,
		lastCommandResults: [
			...options.workflow.workspace.lastCommandResults,
			{
				command: options.command,
				exitCode: options.result.exitCode ?? -1,
				startedAt: now,
				finishedAt: now,
				stdout: options.result.output || undefined,
			},
		].slice(-5),
		testResults: verificationCommand
			? [
					...options.workflow.workspace.testResults,
					{
						command: options.command,
						passed: options.result.exitCode === 0,
						recordedAt: now,
						details: options.result.cancelled ? "Command cancelled." : undefined,
					},
				].slice(-5)
			: options.workflow.workspace.testResults,
		artifactIds: options.workflow.artifacts.map((artifact) => artifact.id).slice(-10),
		refreshedAt: now,
	});
}

export function createWorkspaceStateAfterArtifact(options: {
	cwd: string;
	workflow: WorkflowSessionSnapshot;
}): WorkspaceState {
	const git = readGitSnapshot(options.cwd);
	const now = new Date().toISOString();

	return createWorkspaceState({
		cwd: options.cwd,
		git: {
			branch: git.branch,
			head: git.head,
			statusSummary: git.statusSummary,
		},
		changedFiles: git.changedFiles,
		lastCommandResults: options.workflow.workspace.lastCommandResults,
		testResults: options.workflow.workspace.testResults,
		artifactIds: options.workflow.artifacts.map((artifact) => artifact.id).slice(-10),
		refreshedAt: now,
	});
}

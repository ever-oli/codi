export interface GitSnapshot {
	branch?: string;
	statusSummary?: string;
	head?: string;
}

export interface CommandResultSummary {
	command: string;
	exitCode: number;
	startedAt: string;
	finishedAt: string;
	stdout?: string;
	stderr?: string;
}

export interface TestResultSummary {
	command: string;
	passed: boolean;
	recordedAt: string;
	details?: string;
}

export interface WorkspaceState {
	cwd: string;
	git: GitSnapshot;
	changedFiles: string[];
	lastCommandResults: CommandResultSummary[];
	testResults: TestResultSummary[];
	artifactIds: string[];
	refreshedAt: string;
}

export interface CreateWorkspaceStateInput {
	cwd: string;
	git?: GitSnapshot;
	changedFiles?: string[];
	lastCommandResults?: CommandResultSummary[];
	testResults?: TestResultSummary[];
	artifactIds?: string[];
	refreshedAt?: string;
}

export function createWorkspaceState(input: CreateWorkspaceStateInput): WorkspaceState {
	return {
		cwd: input.cwd,
		git: input.git ?? {},
		changedFiles: input.changedFiles ?? [],
		lastCommandResults: input.lastCommandResults ?? [],
		testResults: input.testResults ?? [],
		artifactIds: input.artifactIds ?? [],
		refreshedAt: input.refreshedAt ?? new Date().toISOString(),
	};
}

import type { WorkflowArtifact } from "./artifacts.js";
import type { MemoryRecord } from "./memory-store.js";
import type { TaskNode } from "./task-graph.js";
import type { WorkspaceState } from "./workspace-state.js";

export interface BuildContextInput {
	task?: TaskNode;
	workspace: WorkspaceState;
	memory: MemoryRecord[];
	artifacts: WorkflowArtifact[];
	shortSummary?: string;
	relevantFiles?: string[];
}

export interface BuildContextResult {
	task?: TaskNode;
	workspace: WorkspaceState;
	memory: MemoryRecord[];
	artifacts: WorkflowArtifact[];
	shortSummary?: string;
	relevantFiles: string[];
}

export interface ContextEngine {
	buildContext(input: BuildContextInput): BuildContextResult;
}

export class StaticContextEngine implements ContextEngine {
	buildContext(input: BuildContextInput): BuildContextResult {
		const candidateFiles = new Set<string>(input.relevantFiles ?? []);
		for (const changedFile of input.workspace.changedFiles) {
			candidateFiles.add(changedFile);
		}
		for (const artifact of input.artifacts) {
			if (artifact.path) {
				candidateFiles.add(artifact.path);
			}
		}

		return {
			task: input.task,
			workspace: input.workspace,
			memory: input.memory,
			artifacts: input.artifacts,
			shortSummary: input.shortSummary,
			relevantFiles: [...candidateFiles].slice(0, 8),
		};
	}
}

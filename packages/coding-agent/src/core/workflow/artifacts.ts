export type WorkflowArtifactType = "plan" | "diff" | "test" | "summary" | "decision" | "verification";

export interface WorkflowArtifact {
	id: string;
	type: WorkflowArtifactType;
	label: string;
	path?: string;
	producer?: string;
	recordedAt: string;
	metadata?: Record<string, string | number | boolean | null>;
}

export function createWorkflowArtifact(
	input: Omit<WorkflowArtifact, "recordedAt"> & { recordedAt?: string },
): WorkflowArtifact {
	return {
		...input,
		recordedAt: input.recordedAt ?? new Date().toISOString(),
	};
}

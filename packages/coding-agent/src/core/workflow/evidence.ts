export type VerificationStatus = "pending" | "passed" | "failed" | "waived";

export interface RunEvidence {
	tests: TestEvidence[];
	commands: CommandEvidence[];
	diffSummary?: string;
	userWaiver?: string;
}

export interface TestEvidence {
	command: string;
	passed: boolean;
	details?: string;
}

export interface CommandEvidence {
	command: string;
	validated: boolean;
	details?: string;
}

export interface VerificationRecord {
	taskId: string;
	status: VerificationStatus;
	evidence: RunEvidence;
	recordedAt: string;
}

export function createVerificationRecord(
	input: Omit<VerificationRecord, "recordedAt"> & { recordedAt?: string },
): VerificationRecord {
	return {
		...input,
		recordedAt: input.recordedAt ?? new Date().toISOString(),
	};
}

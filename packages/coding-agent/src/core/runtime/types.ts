export const RUNTIME_FEATURE_FLAG_NAMES = [
	"runtime.deliveryQueue",
	"runtime.namedLanes",
	"ui.marketplace",
	"ui.eventStreamViewer",
	"runtime.mailboxProtocolV2",
	"runtime.heartbeatCronCore",
	"model.roleProfiles",
] as const;

export type RuntimeFeatureFlagName = (typeof RUNTIME_FEATURE_FLAG_NAMES)[number];

export interface RuntimeFeatureFlags {
	"runtime.deliveryQueue": boolean;
	"runtime.namedLanes": boolean;
	"ui.marketplace": boolean;
	"ui.eventStreamViewer": boolean;
	"runtime.mailboxProtocolV2": boolean;
	"runtime.heartbeatCronCore": boolean;
	"model.roleProfiles": boolean;
}

export const DEFAULT_RUNTIME_FEATURE_FLAGS: RuntimeFeatureFlags = {
	"runtime.deliveryQueue": true,
	"runtime.namedLanes": true,
	"ui.marketplace": true,
	"ui.eventStreamViewer": true,
	"runtime.mailboxProtocolV2": true,
	"runtime.heartbeatCronCore": true,
	"model.roleProfiles": true,
};

export const LANE_NAMES = ["default", "delegate", "cron", "compact", "notification"] as const;
export type LaneName = (typeof LANE_NAMES)[number];

export interface LanePolicy {
	concurrency: number;
	queueStrategy: "fifo";
}

export type LanePolicies = Record<LaneName, LanePolicy>;

export const DEFAULT_LANE_POLICIES: LanePolicies = {
	default: { concurrency: 2, queueStrategy: "fifo" },
	delegate: { concurrency: 1, queueStrategy: "fifo" },
	cron: { concurrency: 1, queueStrategy: "fifo" },
	compact: { concurrency: 1, queueStrategy: "fifo" },
	notification: { concurrency: 2, queueStrategy: "fifo" },
};

export type RuntimeSeverity = "debug" | "info" | "warn" | "error";

export interface RuntimeEventRecord {
	id: string;
	sessionId?: string;
	type: string;
	severity: RuntimeSeverity;
	source: string;
	lane?: LaneName;
	payload: Record<string, unknown>;
	createdAt: number;
}

export type QueueMessageState = "queued" | "leased" | "acked" | "failed" | "dead_letter";

export interface OutboundQueueRecord {
	id: string;
	topic: string;
	payload: Record<string, unknown>;
	lane: LaneName;
	state: QueueMessageState;
	attempts: number;
	maxAttempts: number;
	dedupeKey?: string;
	availableAt: number;
	leasedUntil?: number;
	lastError?: string;
	createdAt: number;
	updatedAt: number;
}

export interface DeliveryAttemptRecord {
	id: number;
	messageId: string;
	attempt: number;
	startedAt: number;
	endedAt?: number;
	success: boolean;
	error?: string;
}

export type MailboxMessageState = "drafted" | "queued" | "leased" | "delivered" | "acked" | "failed" | "dead_letter";

export interface MailboxEnvelope {
	messageId: string;
	threadId: string;
	from: string;
	to: string;
	intent: string;
	payload: Record<string, unknown>;
	replyTo?: string;
	deadline?: string;
	priority: number;
	expectedOutputSchema?: string;
	completionCriteria?: string;
	retryPolicy?: string;
	createdAt: number;
	updatedAt: number;
	state: MailboxMessageState;
	lastError?: string;
}

export type DelegatedTaskStatus = "queued" | "running" | "blocked" | "completed" | "failed";

export interface DelegatedTaskRecord {
	delegatedTaskId: string;
	threadId: string;
	mailboxMessageId?: string;
	parentSessionId: string;
	owner: string;
	assignee: string;
	goal: string;
	summary?: string;
	status: DelegatedTaskStatus;
	lastError?: string;
	createdAt: number;
	updatedAt: number;
	completedAt?: number;
}

export interface CronJobRecord {
	id: string;
	name: string;
	intervalSeconds: number;
	intent: string;
	payload: Record<string, unknown>;
	enabled: boolean;
	nextRunAt: number;
	lastRunAt?: number;
	lastRunStatus?: "ok" | "error";
	lastError?: string;
	createdAt: number;
	updatedAt: number;
}

export const MODEL_ROLE_NAMES = ["main", "task", "compact", "quick"] as const;
export type ModelRoleName = (typeof MODEL_ROLE_NAMES)[number];

export type RoleModelProfile = Partial<Record<ModelRoleName, string>>;

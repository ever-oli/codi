import { join } from "node:path";
import type { AgentSessionEvent } from "../agent-session.js";
import type { ModelRegistry } from "../model-registry.js";
import type { SettingsManager } from "../settings-manager.js";
import { DelegatedTaskService } from "./delegated-tasks.js";
import { DeliveryQueueService } from "./delivery-queue.js";
import { RuntimeEventStream } from "./event-stream.js";
import { HeartbeatCronService } from "./heartbeat-cron.js";
import { LaneScheduler } from "./lane-scheduler.js";
import { MailboxService } from "./mailbox.js";
import { ModelRoleResolver } from "./model-roles.js";
import { RuntimeServiceRegistry } from "./service-registry.js";
import { RuntimeSqliteStore } from "./sqlite-store.js";
import {
	DEFAULT_LANE_POLICIES,
	type LaneName,
	type OutboundQueueRecord,
	type RoleModelProfile,
	type RuntimeEventRecord,
	type RuntimeFeatureFlagName,
} from "./types.js";

export interface RuntimeServicesOptions {
	agentDir: string;
	settingsManager: SettingsManager;
	modelRegistry: ModelRegistry;
}

export class RuntimeServices {
	readonly registry = new RuntimeServiceRegistry();
	readonly store: RuntimeSqliteStore;
	readonly events: RuntimeEventStream;
	readonly lanes: LaneScheduler;
	readonly queue: DeliveryQueueService;
	readonly delegatedTasks: DelegatedTaskService;
	readonly mailbox: MailboxService;
	readonly heartbeat: HeartbeatCronService;
	readonly modelRoles: ModelRoleResolver;
	private started = false;

	constructor(private readonly options: RuntimeServicesOptions) {
		const dbPath = join(options.agentDir, "runtime", "runtime.sqlite");
		this.store = this.registry.register("store", new RuntimeSqliteStore(dbPath));
		this.events = this.registry.register("events", new RuntimeEventStream(this.store));
		const policies = {
			...DEFAULT_LANE_POLICIES,
			...options.settingsManager.getLanePolicies(),
		};
		this.lanes = this.registry.register("lanes", new LaneScheduler(policies, this.events));
		this.queue = this.registry.register("queue", new DeliveryQueueService(this.store, this.lanes, this.events));
		this.delegatedTasks = this.registry.register("delegatedTasks", new DelegatedTaskService(this.store, this.events));
		this.mailbox = this.registry.register(
			"mailbox",
			new MailboxService(this.store, this.queue, this.events, this.delegatedTasks),
		);
		this.heartbeat = this.registry.register(
			"heartbeat",
			new HeartbeatCronService(this.store, this.events, this.queue, this.lanes),
		);
		this.modelRoles = this.registry.register(
			"modelRoles",
			new ModelRoleResolver(this.options.modelRegistry, () => this.options.settingsManager.getRoleModelProfile()),
		);
		this.registerDefaultQueueHandlers();
		this.syncLanePoliciesFromSettings();
	}

	start(): void {
		if (this.started) {
			return;
		}
		this.started = true;
		this.queue.recoverExpiredLeases();
		this.queue.startPolling(1000);
		if (this.isFeatureEnabled("runtime.heartbeatCronCore")) {
			this.heartbeat.start();
		}
		this.events.record({
			type: "runtime.started",
			source: "runtime",
			payload: {
				services: this.registry.keys(),
			},
		});
	}

	stop(): void {
		if (!this.started) {
			return;
		}
		this.started = false;
		this.queue.stopPolling();
		this.heartbeat.stop();
		this.events.record({
			type: "runtime.stopped",
			source: "runtime",
			payload: {},
		});
		this.store.close();
	}

	isFeatureEnabled(flag: RuntimeFeatureFlagName): boolean {
		return this.options.settingsManager.getRuntimeFeatureFlags()[flag];
	}

	syncLanePoliciesFromSettings(): void {
		const policies = {
			...DEFAULT_LANE_POLICIES,
			...this.options.settingsManager.getLanePolicies(),
		};
		for (const [lane, policy] of Object.entries(policies) as Array<
			[LaneName, { concurrency: number; queueStrategy: "fifo" }]
		>) {
			this.lanes.setPolicy(lane, policy);
			this.store.saveLanePolicy(lane, policy.concurrency, policy.queueStrategy);
		}
	}

	recordAgentSessionEvent(sessionId: string, event: AgentSessionEvent): RuntimeEventRecord {
		const source = "agent.session";
		switch (event.type) {
			case "agent_start":
				return this.events.record({
					sessionId,
					type: "session.agent_start",
					source,
					payload: {},
				});
			case "agent_end":
				return this.events.record({
					sessionId,
					type: "session.agent_end",
					source,
					payload: {},
				});
			case "message_start":
				return this.events.record({
					sessionId,
					type: "session.message_start",
					source,
					payload: { role: event.message.role },
				});
			case "message_end":
				return this.events.record({
					sessionId,
					type: "session.message_end",
					source,
					payload: { role: event.message.role },
				});
			case "tool_execution_start":
				return this.events.record({
					sessionId,
					type: "session.tool_start",
					source,
					payload: {
						toolName: event.toolName,
						toolCallId: event.toolCallId,
					},
				});
			case "tool_execution_update":
				return this.events.record({
					sessionId,
					type: "session.tool_update",
					source,
					payload: {
						toolName: event.toolName,
						toolCallId: event.toolCallId,
					},
				});
			case "tool_execution_end":
				return this.events.record({
					sessionId,
					type: "session.tool_end",
					source,
					severity: event.isError ? "warn" : "info",
					payload: {
						toolName: event.toolName,
						toolCallId: event.toolCallId,
						isError: event.isError,
					},
				});
			case "auto_compaction_start":
				return this.events.record({
					sessionId,
					type: "session.auto_compaction_start",
					source,
					lane: "compact",
					payload: {
						reason: event.reason,
					},
				});
			case "auto_compaction_end":
				return this.events.record({
					sessionId,
					type: "session.auto_compaction_end",
					source,
					lane: "compact",
					severity: event.errorMessage ? "warn" : "info",
					payload: {
						aborted: event.aborted,
						willRetry: event.willRetry,
						errorMessage: event.errorMessage,
					},
				});
			case "auto_retry_start":
				return this.events.record({
					sessionId,
					type: "session.auto_retry_start",
					source,
					payload: {
						attempt: event.attempt,
						maxAttempts: event.maxAttempts,
						delayMs: event.delayMs,
						errorMessage: event.errorMessage,
					},
				});
			case "auto_retry_end":
				return this.events.record({
					sessionId,
					type: "session.auto_retry_end",
					source,
					severity: event.success ? "info" : "warn",
					payload: {
						success: event.success,
						attempt: event.attempt,
						finalError: event.finalError,
					},
				});
			default:
				return this.events.record({
					sessionId,
					type: `session.${event.type}`,
					source,
					payload: {},
				});
		}
	}

	enqueueNotification(topic: string, payload: Record<string, unknown>, dedupeKey?: string): OutboundQueueRecord {
		return this.queue.enqueue({
			topic,
			payload,
			lane: "notification",
			dedupeKey,
			maxAttempts: 4,
		});
	}

	getRoleProfile(): RoleModelProfile {
		return this.options.settingsManager.getRoleModelProfile();
	}

	private registerDefaultQueueHandlers(): void {
		this.queue.registerHandler({
			topic: "notification.message",
			handle: async (message) => {
				this.events.record({
					type: "notification.delivered",
					source: "runtime.queue",
					lane: "notification",
					payload: {
						id: message.id,
						payload: message.payload,
					},
				});
			},
		});
	}
}

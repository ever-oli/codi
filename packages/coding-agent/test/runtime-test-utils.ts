import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DelegatedTaskService } from "../src/core/runtime/delegated-tasks.js";
import { DeliveryQueueService } from "../src/core/runtime/delivery-queue.js";
import { RuntimeEventStream } from "../src/core/runtime/event-stream.js";
import { HeartbeatCronService } from "../src/core/runtime/heartbeat-cron.js";
import { LaneScheduler } from "../src/core/runtime/lane-scheduler.js";
import { MailboxService } from "../src/core/runtime/mailbox.js";
import { RuntimeSqliteStore } from "../src/core/runtime/sqlite-store.js";
import { DEFAULT_LANE_POLICIES } from "../src/core/runtime/types.js";

export function waitMs(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function waitFor(
	condition: () => boolean,
	options?: {
		timeoutMs?: number;
		intervalMs?: number;
	},
): Promise<void> {
	const timeoutMs = options?.timeoutMs ?? 2000;
	const intervalMs = options?.intervalMs ?? 25;
	const startedAt = Date.now();
	while (Date.now() - startedAt < timeoutMs) {
		if (condition()) {
			return;
		}
		await waitMs(intervalMs);
	}
	throw new Error(`waitFor timeout after ${timeoutMs}ms`);
}

export function createRuntimeHarness(name: string): {
	tmpPath: string;
	store: RuntimeSqliteStore;
	events: RuntimeEventStream;
	lanes: LaneScheduler;
	queue: DeliveryQueueService;
	delegatedTasks: DelegatedTaskService;
	mailbox: MailboxService;
	heartbeat: HeartbeatCronService;
	cleanup: () => void;
} {
	const tmpPath = mkdtempSync(join(tmpdir(), `pi-runtime-${name}-`));
	const dbPath = join(tmpPath, "runtime.sqlite");
	const store = new RuntimeSqliteStore(dbPath);
	const events = new RuntimeEventStream(store);
	const lanes = new LaneScheduler(DEFAULT_LANE_POLICIES, events);
	const queue = new DeliveryQueueService(store, lanes, events);
	const delegatedTasks = new DelegatedTaskService(store, events);
	const mailbox = new MailboxService(store, queue, events, delegatedTasks);
	const heartbeat = new HeartbeatCronService(store, events, queue, lanes, 1_000_000);

	const cleanup = () => {
		heartbeat.stop();
		queue.stopPolling();
		store.close();
		rmSync(tmpPath, { recursive: true, force: true });
	};

	return {
		tmpPath,
		store,
		events,
		lanes,
		queue,
		delegatedTasks,
		mailbox,
		heartbeat,
		cleanup,
	};
}

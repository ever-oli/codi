import { afterEach, describe, expect, test, vi } from "vitest";
import { createRuntimeHarness, waitFor } from "./runtime-test-utils.js";

describe("HeartbeatCronService", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
		vi.restoreAllMocks();
	});

	test("dispatches due jobs via cron lane and avoids duplicate dispatch for same run window", async () => {
		const runtime = createRuntimeHarness("heartbeat-dispatch");
		cleanup = runtime.cleanup;
		let now = 1_700_000_000_000;
		vi.spyOn(runtime.store, "now").mockImplementation(() => now);

		const job = runtime.heartbeat.addJob({
			name: "nightly-compaction",
			intervalSeconds: 30,
			intent: "compact",
			payload: { scope: "session" },
		});

		now = job.nextRunAt + 1;
		await runtime.heartbeat.tick();
		await runtime.heartbeat.tick();
		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.list("acked", 20).length >= 1);

		const queueRows = runtime.store.db
			.prepare("SELECT COUNT(*) AS count FROM outbound_queue WHERE topic = 'cron.dispatch'")
			.get() as { count: number };
		expect(queueRows.count).toBe(1);

		const cronLaneStart = runtime.events
			.list({ type: "lane.task.started", lane: "cron", limit: 20 })
			.some((event) => event.type === "lane.task.started");
		expect(cronLaneStart).toBe(true);

		const jobAfterTick = runtime.heartbeat.getJob(job.id);
		expect(jobAfterTick?.lastRunStatus).toBe("ok");
		expect(jobAfterTick?.nextRunAt).toBeGreaterThan(now);
	});

	test("supports pause, resume, and remove lifecycle", async () => {
		const runtime = createRuntimeHarness("heartbeat-lifecycle");
		cleanup = runtime.cleanup;
		let now = 1_700_000_100_000;
		vi.spyOn(runtime.store, "now").mockImplementation(() => now);

		const job = runtime.heartbeat.addJob({
			name: "send-summary",
			intervalSeconds: 30,
			intent: "notify",
		});

		const paused = runtime.heartbeat.setJobEnabled(job.id, false);
		expect(paused).toBe(true);

		now = job.nextRunAt + 1;
		await runtime.heartbeat.tick();
		expect(runtime.queue.list("queued", 20)).toHaveLength(0);

		const resumed = runtime.heartbeat.setJobEnabled(job.id, true);
		expect(resumed).toBe(true);

		await runtime.heartbeat.tick();
		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.list("acked", 20).length >= 1);

		const removed = runtime.heartbeat.removeJob(job.id);
		expect(removed).toBe(true);
		expect(runtime.heartbeat.getJob(job.id)).toBeUndefined();
	});
});

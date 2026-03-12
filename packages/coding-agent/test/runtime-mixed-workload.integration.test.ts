import { afterEach, describe, expect, test, vi } from "vitest";
import { createRuntimeHarness, waitFor, waitMs } from "./runtime-test-utils.js";

describe("Runtime mixed workload integration", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
		vi.restoreAllMocks();
	});

	test("queue + delegate + cron + compact workloads complete without starvation", async () => {
		const runtime = createRuntimeHarness("mixed-workload");
		cleanup = runtime.cleanup;

		let now = 1_700_000_200_000;
		vi.spyOn(runtime.store, "now").mockImplementation(() => now);

		let notificationCount = 0;
		runtime.queue.registerHandler({
			topic: "integration.notify",
			handle: async () => {
				notificationCount += 1;
				await waitMs(5);
			},
		});

		const envelope = runtime.mailbox.send({
			from: "session-main",
			to: "session-worker",
			intent: "delegate.task",
			payload: { goal: "run integration step" },
		});

		runtime.queue.enqueue({
			topic: "integration.notify",
			lane: "notification",
			payload: { id: "n1" },
		});
		runtime.queue.enqueue({
			topic: "integration.notify",
			lane: "notification",
			payload: { id: "n2" },
		});
		runtime.queue.enqueue({
			topic: "integration.notify",
			lane: "notification",
			payload: { id: "n3" },
		});

		const compactPromise = runtime.lanes.schedule("compact", "compact-pass", async () => {
			await waitMs(5);
			return "compact-done";
		});

		const cronJob = runtime.heartbeat.addJob({
			name: "hourly-review",
			intervalSeconds: 30,
			intent: "review",
			payload: { source: "integration" },
		});
		now = cronJob.nextRunAt + 1;
		await runtime.heartbeat.tick();

		for (let index = 0; index < 10; index += 1) {
			await runtime.queue.processDue();
			if (
				runtime.mailbox.get(envelope.messageId)?.state === "delivered" &&
				runtime.queue.list("queued", 100).length === 0 &&
				runtime.queue.list("leased", 100).length === 0
			) {
				break;
			}
			await waitMs(10);
		}

		await waitFor(() => runtime.mailbox.get(envelope.messageId)?.state === "delivered");
		await waitFor(() => runtime.queue.list("acked", 100).length >= 5);
		await expect(compactPromise).resolves.toBe("compact-done");
		expect(notificationCount).toBe(3);
		expect(runtime.queue.list("dead_letter", 20)).toHaveLength(0);

		await waitFor(() =>
			runtime.lanes.getSnapshots().every((snapshot) => snapshot.active === 0 && snapshot.queued === 0),
		);
	});
});

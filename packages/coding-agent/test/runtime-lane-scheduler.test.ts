import { afterEach, describe, expect, test } from "vitest";
import { createRuntimeHarness, waitFor } from "./runtime-test-utils.js";

function createDeferred<T = void>(): {
	promise: Promise<T>;
	resolve: (value: T | PromiseLike<T>) => void;
	reject: (error?: unknown) => void;
} {
	let resolve!: (value: T | PromiseLike<T>) => void;
	let reject!: (error?: unknown) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

describe("LaneScheduler", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
	});

	test("enforces per-lane concurrency limits", async () => {
		const runtime = createRuntimeHarness("lane-concurrency");
		cleanup = runtime.cleanup;
		runtime.lanes.setPolicy("delegate", { concurrency: 1, queueStrategy: "fifo" });

		let active = 0;
		let maxActive = 0;
		const gate = createDeferred<void>();

		const first = runtime.lanes.schedule("delegate", "first", async () => {
			active += 1;
			maxActive = Math.max(maxActive, active);
			await gate.promise;
			active -= 1;
			return "first";
		});
		const second = runtime.lanes.schedule("delegate", "second", async () => {
			active += 1;
			maxActive = Math.max(maxActive, active);
			active -= 1;
			return "second";
		});

		await waitFor(() => {
			const snapshot = runtime.lanes.getSnapshots().find((entry) => entry.lane === "delegate");
			return snapshot?.active === 1 && snapshot.queued === 1;
		});

		gate.resolve();
		await expect(first).resolves.toBe("first");
		await expect(second).resolves.toBe("second");
		expect(maxActive).toBe(1);
	});

	test("isolates lanes under mixed workload", async () => {
		const runtime = createRuntimeHarness("lane-isolation");
		cleanup = runtime.cleanup;
		runtime.lanes.setPolicy("delegate", { concurrency: 1, queueStrategy: "fifo" });
		runtime.lanes.setPolicy("notification", { concurrency: 1, queueStrategy: "fifo" });

		const gate = createDeferred<void>();
		let notificationDone = false;

		const delegatePromise = runtime.lanes.schedule("delegate", "delegate-long", async () => {
			await gate.promise;
		});
		const notificationPromise = runtime.lanes.schedule("notification", "notify-fast", async () => {
			notificationDone = true;
		});

		await waitFor(() => notificationDone);
		gate.resolve();
		await delegatePromise;
		await notificationPromise;
		expect(notificationDone).toBe(true);
	});

	test("cancel on a queued task only affects that targeted task", async () => {
		const runtime = createRuntimeHarness("lane-cancel");
		cleanup = runtime.cleanup;
		runtime.lanes.setPolicy("delegate", { concurrency: 1, queueStrategy: "fifo" });

		const gate = createDeferred<void>();
		const first = runtime.lanes.schedule("delegate", "delegate-first", async () => {
			await gate.promise;
			return "first";
		});
		const second = runtime.lanes.schedule("delegate", "delegate-second", async () => "second");

		await waitFor(() => {
			const snapshot = runtime.lanes.getSnapshots().find((entry) => entry.lane === "delegate");
			return snapshot?.active === 1 && snapshot.queued === 1;
		});

		const queuedEvent = runtime.events
			.list({ type: "lane.task.queued", lane: "delegate", limit: 20 })
			.find((event) => event.payload.label === "delegate-second");
		const taskId = typeof queuedEvent?.payload.taskId === "string" ? queuedEvent.payload.taskId : "";
		expect(taskId).toBeTruthy();

		const cancelled = runtime.lanes.cancel(taskId);
		expect(cancelled).toBe(true);

		await expect(second).rejects.toThrow("cancelled");
		gate.resolve();
		await expect(first).resolves.toBe("first");
	});
});

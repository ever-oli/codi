import { afterEach, describe, expect, test } from "vitest";
import { createRuntimeHarness, waitFor } from "./runtime-test-utils.js";

describe("DeliveryQueueService", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
	});

	test("dedupe key returns the same logical queued message", () => {
		const runtime = createRuntimeHarness("queue-dedupe");
		cleanup = runtime.cleanup;
		runtime.queue.registerHandler({
			topic: "test.topic",
			handle: async () => {},
		});

		const first = runtime.queue.enqueue({
			topic: "test.topic",
			payload: { value: 1 },
			dedupeKey: "dedupe-1",
		});
		const second = runtime.queue.enqueue({
			topic: "test.topic",
			payload: { value: 2 },
			dedupeKey: "dedupe-1",
		});

		expect(second.id).toBe(first.id);
		expect(runtime.queue.list("queued", 10)).toHaveLength(1);
	});

	test("failed deliveries backoff and eventually dead-letter", async () => {
		const runtime = createRuntimeHarness("queue-dead-letter");
		cleanup = runtime.cleanup;
		runtime.queue.registerHandler({
			topic: "test.fail",
			handle: async () => {
				throw new Error("forced failure");
			},
		});

		const message = runtime.queue.enqueue({
			topic: "test.fail",
			payload: { value: 1 },
			maxAttempts: 2,
		});

		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.getById(message.id)?.attempts === 1);
		runtime.store.db
			.prepare("UPDATE outbound_queue SET available_at = ? WHERE id = ?")
			.run(runtime.store.now() - 1, message.id);

		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.getById(message.id)?.state === "dead_letter");

		const final = runtime.queue.getById(message.id);
		expect(final?.attempts).toBe(2);
		expect(final?.state).toBe("dead_letter");
		expect(final?.lastError).toContain("forced failure");
	});

	test("dead-letter retry path returns to queued and then acked", async () => {
		const runtime = createRuntimeHarness("queue-retry");
		cleanup = runtime.cleanup;
		let shouldFail = true;
		runtime.queue.registerHandler({
			topic: "test.retry",
			handle: async () => {
				if (shouldFail) {
					throw new Error("fail once");
				}
			},
		});

		const message = runtime.queue.enqueue({
			topic: "test.retry",
			payload: {},
			maxAttempts: 1,
		});

		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.getById(message.id)?.state === "dead_letter");
		shouldFail = false;
		const retried = runtime.queue.retryDeadLetter(message.id);
		expect(retried).toBe(true);

		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.getById(message.id)?.state === "acked");
	});
});

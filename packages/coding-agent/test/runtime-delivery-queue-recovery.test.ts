import { afterEach, describe, expect, test } from "vitest";
import { createRuntimeHarness, waitFor } from "./runtime-test-utils.js";

describe("DeliveryQueueService lease recovery", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
	});

	test("recoverExpiredLeases re-queues and delivers expired leased messages", async () => {
		const runtime = createRuntimeHarness("queue-recovery");
		cleanup = runtime.cleanup;
		runtime.queue.registerHandler({
			topic: "test.recover",
			handle: async () => {},
		});

		const message = runtime.queue.enqueue({
			topic: "test.recover",
			payload: { hello: "world" },
			maxAttempts: 3,
		});

		runtime.store.db
			.prepare(
				`UPDATE outbound_queue
				 SET state = 'leased',
				     leased_until = ?,
				     updated_at = ?
				 WHERE id = ?`,
			)
			.run(runtime.store.now() - 1000, runtime.store.now(), message.id);

		const recovered = runtime.queue.recoverExpiredLeases();
		expect(recovered).toBe(1);
		expect(runtime.queue.getById(message.id)?.state).toBe("queued");

		await runtime.queue.processDue();
		await waitFor(() => runtime.queue.getById(message.id)?.state === "acked");
	});
});

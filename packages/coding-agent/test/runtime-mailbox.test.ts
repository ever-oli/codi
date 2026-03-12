import { afterEach, describe, expect, test } from "vitest";
import { createRuntimeHarness, waitFor } from "./runtime-test-utils.js";

describe("MailboxService", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
	});

	test("send -> delivered -> ack flow persists and is queryable", async () => {
		const runtime = createRuntimeHarness("mailbox-flow");
		cleanup = runtime.cleanup;

		const envelope = runtime.mailbox.send({
			from: "session-a",
			to: "session-b",
			intent: "delegate",
			payload: { task: "run checks" },
			delegatedTask: { goal: "run checks" },
		});
		expect(envelope.state).toBe("queued");
		const delegatedTaskId = String(envelope.payload.delegatedTaskId ?? "");
		expect(runtime.delegatedTasks.get(delegatedTaskId)?.status).toBe("queued");

		await runtime.queue.processDue();
		await waitFor(() => runtime.mailbox.get(envelope.messageId)?.state === "delivered");
		expect(runtime.delegatedTasks.get(delegatedTaskId)?.status).toBe("running");

		const acked = runtime.mailbox.ack(envelope.messageId, "session-b");
		expect(acked).toBe(true);
		expect(runtime.mailbox.get(envelope.messageId)?.state).toBe("acked");
		expect(runtime.delegatedTasks.get(delegatedTaskId)?.status).toBe("completed");

		const inbox = runtime.mailbox.listInbox("session-b", 20);
		const outbox = runtime.mailbox.listOutbox("session-a", 20);
		expect(inbox.some((message) => message.messageId === envelope.messageId)).toBe(true);
		expect(outbox.some((message) => message.messageId === envelope.messageId)).toBe(true);
	});

	test("retry works after failed/dead-letter style transitions and preserves thread correlation", async () => {
		const runtime = createRuntimeHarness("mailbox-retry");
		cleanup = runtime.cleanup;

		const envelope = runtime.mailbox.send({
			from: "session-a",
			to: "session-b",
			intent: "delegate",
			payload: { task: "retry me" },
			delegatedTask: { goal: "retry me" },
		});
		const delegatedTaskId = String(envelope.payload.delegatedTaskId ?? "");

		const markedFailed = runtime.mailbox.markFailed(envelope.messageId, "transient failure");
		expect(markedFailed).toBe(true);
		expect(runtime.mailbox.get(envelope.messageId)?.state).toBe("failed");
		expect(runtime.delegatedTasks.get(delegatedTaskId)?.status).toBe("failed");

		const retried = runtime.mailbox.retry(envelope.messageId);
		expect(retried).toBe(true);
		expect(runtime.delegatedTasks.get(delegatedTaskId)?.status).toBe("queued");

		await runtime.queue.processDue();
		await waitFor(() => runtime.mailbox.get(envelope.messageId)?.state === "delivered");
		expect(runtime.delegatedTasks.get(delegatedTaskId)?.status).toBe("running");

		const thread = runtime.mailbox.listThread(envelope.threadId, 20);
		expect(thread).toHaveLength(1);
		expect(thread[0]?.messageId).toBe(envelope.messageId);
		expect(thread[0]?.threadId).toBe(envelope.threadId);
		expect(runtime.delegatedTasks.list({ threadId: envelope.threadId, limit: 20 })[0]?.delegatedTaskId).toBe(
			delegatedTaskId,
		);
	});
});

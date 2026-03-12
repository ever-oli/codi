import { afterEach, describe, expect, test } from "vitest";
import { createRuntimeHarness } from "./runtime-test-utils.js";

describe("DelegatedTaskService", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
	});

	test("create persists a delegated task and emits a queued event", () => {
		const runtime = createRuntimeHarness("delegated-create");
		cleanup = runtime.cleanup;

		const task = runtime.delegatedTasks.create({
			threadId: "thread-1",
			parentSessionId: "session-a",
			owner: "session-a",
			assignee: "session-b",
			goal: "Run checks",
			summary: "Initial handoff",
		});

		expect(runtime.delegatedTasks.get(task.delegatedTaskId)).toMatchObject({
			delegatedTaskId: task.delegatedTaskId,
			threadId: "thread-1",
			parentSessionId: "session-a",
			owner: "session-a",
			assignee: "session-b",
			goal: "Run checks",
			status: "queued",
		});

		const events = runtime.events.list({ type: "delegated_task.queued", limit: 10 });
		expect(events).toHaveLength(1);
		expect(events[0]?.payload.delegatedTaskId).toBe(task.delegatedTaskId);
	});

	test("valid transitions succeed and invalid transitions are rejected", () => {
		const runtime = createRuntimeHarness("delegated-transitions");
		cleanup = runtime.cleanup;

		const task = runtime.delegatedTasks.create({
			threadId: "thread-1",
			parentSessionId: "session-a",
			owner: "session-a",
			assignee: "session-b",
			goal: "Inspect repo",
		});

		expect(runtime.delegatedTasks.markCompleted(task.delegatedTaskId)).toBeUndefined();
		expect(runtime.delegatedTasks.markRunning(task.delegatedTaskId)?.status).toBe("running");
		expect(runtime.delegatedTasks.markBlocked(task.delegatedTaskId, "Waiting for approval")?.status).toBe("blocked");
		expect(runtime.delegatedTasks.markRunning(task.delegatedTaskId, "Approval received")?.status).toBe("running");
		expect(runtime.delegatedTasks.markCompleted(task.delegatedTaskId, "Done")?.status).toBe("completed");
		expect(runtime.delegatedTasks.markFailed(task.delegatedTaskId, "late failure")).toBeUndefined();
	});

	test("list supports query filters by session, thread, and status", () => {
		const runtime = createRuntimeHarness("delegated-list");
		cleanup = runtime.cleanup;

		const first = runtime.delegatedTasks.create({
			threadId: "thread-a",
			parentSessionId: "session-a",
			owner: "session-a",
			assignee: "session-b",
			goal: "First",
		});
		const second = runtime.delegatedTasks.create({
			threadId: "thread-b",
			parentSessionId: "session-a",
			owner: "session-a",
			assignee: "session-c",
			goal: "Second",
		});
		runtime.delegatedTasks.markRunning(second.delegatedTaskId);
		runtime.delegatedTasks.markFailed(second.delegatedTaskId, "No context");

		runtime.delegatedTasks.create({
			threadId: "thread-c",
			parentSessionId: "session-z",
			owner: "session-z",
			assignee: "session-y",
			goal: "Third",
		});

		expect(runtime.delegatedTasks.list({ parentSessionId: "session-a", limit: 10 })).toHaveLength(2);
		expect(
			runtime.delegatedTasks.list({ threadId: "thread-a", limit: 10 }).map((task) => task.delegatedTaskId),
		).toEqual([first.delegatedTaskId]);
		expect(runtime.delegatedTasks.list({ status: "failed", limit: 10 }).map((task) => task.delegatedTaskId)).toEqual([
			second.delegatedTaskId,
		]);
	});
});

import { afterEach, describe, expect, test, vi } from "vitest";
import { createRuntimeHarness } from "./runtime-test-utils.js";

describe("RuntimeEventStream", () => {
	let cleanup: (() => void) | undefined;

	afterEach(() => {
		cleanup?.();
		cleanup = undefined;
		vi.restoreAllMocks();
	});

	test("supports list filters for type, session, lane, severity, and time windows", () => {
		const runtime = createRuntimeHarness("events-filters");
		cleanup = runtime.cleanup;

		runtime.events.record({
			type: "queue.enqueued",
			source: "runtime.queue",
			lane: "notification",
			severity: "info",
			sessionId: "session-a",
			payload: { id: "m1" },
			createdAt: 1_000,
		});
		runtime.events.record({
			type: "queue.retry.scheduled",
			source: "runtime.queue",
			lane: "notification",
			severity: "warn",
			sessionId: "session-a",
			payload: { id: "m1" },
			createdAt: 2_000,
		});
		runtime.events.record({
			type: "lane.task.started",
			source: "runtime.lane",
			lane: "delegate",
			severity: "info",
			sessionId: "session-b",
			payload: { taskId: "t1" },
			createdAt: 3_000,
		});

		expect(runtime.events.list({ type: "queue.enqueued", limit: 10 })).toHaveLength(1);
		expect(runtime.events.list({ sessionId: "session-a", limit: 10 })).toHaveLength(2);
		expect(runtime.events.list({ lane: "notification", limit: 10 })).toHaveLength(2);
		expect(runtime.events.list({ severity: "warn", limit: 10 })).toHaveLength(1);

		const timeRange = runtime.events.list({ fromTs: 1_500, toTs: 2_500, limit: 10 });
		expect(timeRange).toHaveLength(1);
		expect(timeRange[0]?.type).toBe("queue.retry.scheduled");
	});

	test("tail order and prune retention behavior are deterministic", () => {
		const runtime = createRuntimeHarness("events-prune");
		cleanup = runtime.cleanup;
		let now = 10_000;
		vi.spyOn(runtime.store, "now").mockImplementation(() => now);

		runtime.events.record({
			type: "event.old",
			source: "test",
			payload: {},
			createdAt: 1_000,
		});
		runtime.events.record({
			type: "event.mid",
			source: "test",
			payload: {},
			createdAt: 6_000,
		});
		runtime.events.record({
			type: "event.new",
			source: "test",
			payload: {},
			createdAt: 9_000,
		});

		const tail = runtime.events.tail(3);
		expect(tail.map((event) => event.type)).toEqual(["event.new", "event.mid", "event.old"]);

		now = 10_000;
		const pruned = runtime.events.pruneByAge(3_000);
		expect(pruned).toBe(2);

		const remaining = runtime.events.list({ limit: 10 });
		expect(remaining).toHaveLength(1);
		expect(remaining[0]?.type).toBe("event.new");
	});
});

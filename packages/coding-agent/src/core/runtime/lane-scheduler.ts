import { randomUUID } from "node:crypto";
import type { RuntimeEventStream } from "./event-stream.js";
import type { LaneName, LanePolicies, LanePolicy } from "./types.js";

interface LaneTask<T> {
	id: string;
	lane: LaneName;
	label: string;
	startedAt?: number;
	run: () => Promise<T>;
	resolve: (value: T) => void;
	reject: (error: unknown) => void;
}

interface LaneQueueState {
	active: Map<string, LaneTask<unknown>>;
	queue: Array<LaneTask<unknown>>;
}

export interface LaneSnapshot {
	lane: LaneName;
	concurrency: number;
	queueStrategy: "fifo";
	active: number;
	queued: number;
}

export class LaneScheduler {
	private readonly lanes = new Map<LaneName, LaneQueueState>();
	private readonly policies = new Map<LaneName, LanePolicy>();

	constructor(
		initialPolicies: LanePolicies,
		private readonly events: RuntimeEventStream,
	) {
		for (const [lane, policy] of Object.entries(initialPolicies) as Array<[LaneName, LanePolicy]>) {
			this.policies.set(lane, { concurrency: Math.max(1, Math.floor(policy.concurrency)), queueStrategy: "fifo" });
			this.lanes.set(lane, { active: new Map(), queue: [] });
		}
	}

	setPolicy(lane: LaneName, policy: LanePolicy): void {
		this.policies.set(lane, {
			concurrency: Math.max(1, Math.floor(policy.concurrency)),
			queueStrategy: "fifo",
		});
		if (!this.lanes.has(lane)) {
			this.lanes.set(lane, { active: new Map(), queue: [] });
		}
		this.events.record({
			type: "lane.policy.updated",
			source: "runtime.lane",
			lane,
			payload: {
				concurrency: Math.max(1, Math.floor(policy.concurrency)),
				queueStrategy: "fifo",
			},
		});
		this.drain(lane);
	}

	schedule<T>(lane: LaneName, label: string, run: () => Promise<T>): Promise<T> {
		const laneState = this.ensureLane(lane);
		return new Promise<T>((resolve, reject) => {
			const task: LaneTask<T> = {
				id: randomUUID(),
				lane,
				label,
				run,
				resolve,
				reject,
			};
			laneState.queue.push(task as LaneTask<unknown>);
			this.events.record({
				type: "lane.task.queued",
				source: "runtime.lane",
				lane,
				payload: {
					taskId: task.id,
					label: task.label,
					queued: laneState.queue.length,
					active: laneState.active.size,
				},
			});
			this.drain(lane);
		});
	}

	cancel(taskId: string): boolean {
		for (const [lane, state] of this.lanes.entries()) {
			const queueIndex = state.queue.findIndex((task) => task.id === taskId);
			if (queueIndex >= 0) {
				const [task] = state.queue.splice(queueIndex, 1);
				task.reject(new Error(`Task ${taskId} cancelled`));
				this.events.record({
					type: "lane.task.cancelled",
					source: "runtime.lane",
					lane,
					payload: { taskId, stage: "queued" },
				});
				return true;
			}
		}
		return false;
	}

	getSnapshots(): LaneSnapshot[] {
		const entries = Array.from(this.policies.entries()).map(([lane, policy]) => {
			const state = this.ensureLane(lane);
			return {
				lane,
				concurrency: policy.concurrency,
				queueStrategy: "fifo",
				active: state.active.size,
				queued: state.queue.length,
			} satisfies LaneSnapshot;
		});
		entries.sort((a, b) => a.lane.localeCompare(b.lane));
		return entries;
	}

	private ensureLane(lane: LaneName): LaneQueueState {
		if (!this.lanes.has(lane)) {
			this.lanes.set(lane, { active: new Map(), queue: [] });
		}
		return this.lanes.get(lane)!;
	}

	private drain(lane: LaneName): void {
		const policy = this.policies.get(lane);
		if (!policy) {
			return;
		}
		const state = this.ensureLane(lane);
		while (state.active.size < policy.concurrency && state.queue.length > 0) {
			const task = state.queue.shift()!;
			state.active.set(task.id, task);
			task.startedAt = Date.now();
			this.events.record({
				type: "lane.task.started",
				source: "runtime.lane",
				lane,
				payload: {
					taskId: task.id,
					label: task.label,
					active: state.active.size,
					queued: state.queue.length,
				},
			});
			void this.runTask(task);
		}
	}

	private async runTask<T>(task: LaneTask<T>): Promise<void> {
		const state = this.ensureLane(task.lane);
		try {
			const result = await task.run();
			task.resolve(result);
			this.events.record({
				type: "lane.task.completed",
				source: "runtime.lane",
				lane: task.lane,
				payload: {
					taskId: task.id,
					label: task.label,
					durationMs: task.startedAt ? Date.now() - task.startedAt : undefined,
				},
			});
		} catch (error) {
			task.reject(error);
			const message = error instanceof Error ? error.message : String(error);
			this.events.record({
				type: "lane.task.failed",
				source: "runtime.lane",
				lane: task.lane,
				severity: "warn",
				payload: {
					taskId: task.id,
					label: task.label,
					error: message,
					durationMs: task.startedAt ? Date.now() - task.startedAt : undefined,
				},
			});
		} finally {
			state.active.delete(task.id);
			this.drain(task.lane);
		}
	}
}

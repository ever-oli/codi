import { randomUUID } from "node:crypto";
import type { RuntimeSqliteStore } from "./sqlite-store.js";
import type { LaneName, RuntimeEventRecord, RuntimeSeverity } from "./types.js";

export interface RuntimeEventInput {
	sessionId?: string;
	type: string;
	severity?: RuntimeSeverity;
	source: string;
	lane?: LaneName;
	payload?: Record<string, unknown>;
	createdAt?: number;
}

export interface RuntimeEventFilters {
	type?: string;
	sessionId?: string;
	lane?: LaneName;
	severity?: RuntimeSeverity;
	fromTs?: number;
	toTs?: number;
	limit?: number;
}

export class RuntimeEventStream {
	constructor(private readonly store: RuntimeSqliteStore) {}

	record(event: RuntimeEventInput): RuntimeEventRecord {
		const record: RuntimeEventRecord = {
			id: randomUUID(),
			sessionId: event.sessionId,
			type: event.type,
			severity: event.severity ?? "info",
			source: event.source,
			lane: event.lane,
			payload: event.payload ?? {},
			createdAt: event.createdAt ?? this.store.now(),
		};
		this.store.insertRuntimeEvent(record);
		return record;
	}

	list(filters: RuntimeEventFilters = {}): RuntimeEventRecord[] {
		return this.store.listRuntimeEvents({
			type: filters.type,
			sessionId: filters.sessionId,
			lane: filters.lane,
			severity: filters.severity,
			fromTs: filters.fromTs,
			toTs: filters.toTs,
			limit: Math.max(1, Math.min(500, filters.limit ?? 50)),
		});
	}

	tail(limit = 25): RuntimeEventRecord[] {
		return this.list({ limit });
	}

	pruneByAge(maxAgeMs: number): number {
		const cutoff = this.store.now() - Math.max(0, maxAgeMs);
		return this.store.pruneRuntimeEvents(cutoff);
	}
}

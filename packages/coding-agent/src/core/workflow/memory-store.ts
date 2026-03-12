export type MemoryScope = "user" | "repo" | "decision";

export interface MemoryRecord {
	key: string;
	scope: MemoryScope;
	value: string;
	updatedAt: string;
}

export interface MemoryStore {
	getAll(): MemoryRecord[];
	getByScope(scope: MemoryScope): MemoryRecord[];
	get(key: string): MemoryRecord | undefined;
	set(record: MemoryRecord): void;
}

export class InMemoryMemoryStore implements MemoryStore {
	#records = new Map<string, MemoryRecord>();

	getAll(): MemoryRecord[] {
		return Array.from(this.#records.values());
	}

	getByScope(scope: MemoryScope): MemoryRecord[] {
		return this.getAll().filter((record) => record.scope === scope);
	}

	get(key: string): MemoryRecord | undefined {
		return this.#records.get(key);
	}

	set(record: MemoryRecord): void {
		this.#records.set(record.key, record);
	}
}

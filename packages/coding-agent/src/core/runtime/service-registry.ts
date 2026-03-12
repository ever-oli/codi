export class RuntimeServiceRegistry {
	private readonly services = new Map<string, unknown>();

	register<T>(name: string, service: T): T {
		this.services.set(name, service);
		return service;
	}

	get<T>(name: string): T | undefined {
		return this.services.get(name) as T | undefined;
	}

	has(name: string): boolean {
		return this.services.has(name);
	}

	keys(): string[] {
		return Array.from(this.services.keys());
	}
}

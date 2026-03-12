import type { Api, Model } from "@mariozechner/pi-ai";
import type { ModelRegistry } from "../model-registry.js";
import type { ModelRoleName, RoleModelProfile } from "./types.js";

function normalizeModelRef(value: string): string {
	return value.trim();
}

function splitModelRef(value: string): { provider: string; modelId: string } | undefined {
	const normalized = normalizeModelRef(value);
	const sep = normalized.indexOf("/");
	if (sep <= 0 || sep === normalized.length - 1) {
		return undefined;
	}
	return {
		provider: normalized.slice(0, sep).trim(),
		modelId: normalized.slice(sep + 1).trim(),
	};
}

export class ModelRoleResolver {
	constructor(
		private readonly registry: ModelRegistry,
		private readonly profileProvider: () => RoleModelProfile,
	) {}

	getProfile(): RoleModelProfile {
		return { ...this.profileProvider() };
	}

	getRoleRef(role: ModelRoleName): string | undefined {
		const profile = this.getProfile();
		const value = profile[role];
		return value?.trim() ? value.trim() : undefined;
	}

	resolveRoleModel(role: ModelRoleName): Model<Api> | undefined {
		const ref = this.getRoleRef(role);
		if (!ref) {
			return undefined;
		}
		const parsed = splitModelRef(ref);
		if (!parsed) {
			return undefined;
		}
		return this.registry.find(parsed.provider, parsed.modelId);
	}
}

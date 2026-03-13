/**
 * Model-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import type { Model } from "@mariozechner/pi-ai";
import type { AgentSession } from "../../../core/agent-session.js";
import type { ModelRoleName, RuntimeServices } from "../../../core/runtime/index.js";
import { MODEL_ROLE_NAMES } from "../../../core/runtime/index.js";
import type { SettingsManager } from "../../../core/settings-manager.js";
import { theme } from "../theme/theme.js";

export interface ModelCommandContext {
	readonly session: AgentSession;
	readonly settingsManager: SettingsManager;
	readonly runtimeServices: RuntimeServices | undefined;

	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	renderRuntimePanel(title: string, lines: string[]): void;
	getRuntimeOrWarn(flag?: string): RuntimeServices | undefined;
	updateEditorBorderColor(): void;
	updateFooter(): void;
	checkDaxnutsEasterEgg(model: { provider: string; id: string }): void;
	ensureRoleModel(role: ModelRoleName): Promise<void>;
	showModelSelector(initialSearchInput?: string): void;
}

export async function handleModelCommand(ctx: ModelCommandContext, searchTerm?: string): Promise<void> {
	if (!searchTerm) {
		ctx.showModelSelector();
		return;
	}

	const model = await findExactModelMatch(ctx, searchTerm);
	if (model) {
		try {
			await ctx.session.setModel(model);
			ctx.updateFooter();
			ctx.updateEditorBorderColor();
			ctx.showStatus(`Model: ${model.id}`);
			ctx.checkDaxnutsEasterEgg(model);
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
		}
		return;
	}

	ctx.showModelSelector(searchTerm);
}

export async function findExactModelMatch(
	ctx: ModelCommandContext,
	searchTerm: string,
): Promise<Model<any> | undefined> {
	const term = searchTerm.trim();
	if (!term) return undefined;

	let targetProvider: string | undefined;
	let targetModelId = "";

	if (term.includes("/")) {
		const parts = term.split("/", 2);
		targetProvider = parts[0]?.trim().toLowerCase();
		targetModelId = parts[1]?.trim().toLowerCase() ?? "";
	} else {
		targetModelId = term.toLowerCase();
	}

	if (!targetModelId) return undefined;

	const models = await getModelCandidates(ctx);
	const exactMatches = models.filter((item) => {
		const idMatch = item.id.toLowerCase() === targetModelId;
		const providerMatch = !targetProvider || item.provider.toLowerCase() === targetProvider;
		return idMatch && providerMatch;
	});

	return exactMatches.length === 1 ? exactMatches[0] : undefined;
}

export async function getModelCandidates(ctx: ModelCommandContext): Promise<Model<any>[]> {
	if (ctx.session.scopedModels.length > 0) {
		return ctx.session.scopedModels.map((scoped) => scoped.model);
	}

	ctx.session.modelRegistry.refresh();
	try {
		return await ctx.session.modelRegistry.getAvailable();
	} catch {
		return [];
	}
}

export async function handleModelsCommand(ctx: ModelCommandContext, text: string): Promise<void> {
	if (text === "/models" || text.trim() === "/models") {
		ctx.showStatus("Usage: /models roles [show|set|clear]");
		return;
	}
	if (text.startsWith("/models roles")) {
		await handleModelRolesCommand(ctx, text);
		return;
	}
	ctx.showWarning("Unknown /models command. Use /models roles.");
}

export async function handleModelRolesCommand(ctx: ModelCommandContext, text: string): Promise<void> {
	const runtime = ctx.getRuntimeOrWarn("model.roleProfiles");
	if (!runtime) return;
	const argText = text.replace(/^\/models\s+roles\s*/, "").trim();
	if (!argText || argText === "show") {
		const profile = ctx.settingsManager.getRoleModelProfile();
		const lines = MODEL_ROLE_NAMES.map((role) => {
			const value = profile[role];
			const resolved = runtime.modelRoles.resolveRoleModel(role);
			const resolvedText = resolved ? `${resolved.provider}/${resolved.id}` : "unresolved";
			return `${theme.fg("accent", role)}: ${value ?? theme.fg("dim", "not set")} ${theme.fg("muted", `(${resolvedText})`)}`;
		});
		lines.push("");
		lines.push("Usage: /models roles set <main|task|compact|quick> <provider>/<model>");
		lines.push("       /models roles clear <main|task|compact|quick>");
		ctx.renderRuntimePanel("Model Roles", lines);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "set") {
		const role = rest[0] as ModelRoleName | undefined;
		const modelRef = rest.slice(1).join(" ").trim();
		if (!role || !(MODEL_ROLE_NAMES as readonly string[]).includes(role) || !modelRef) {
			ctx.showWarning("Usage: /models roles set <main|task|compact|quick> <provider>/<model>");
			return;
		}
		if (!modelRef.includes("/")) {
			ctx.showWarning("Model reference must be provider/model.");
			return;
		}
		ctx.settingsManager.setRoleModel(role, modelRef);
		runtime.events.record({
			type: "model.roles.updated",
			source: "interactive:/models roles set",
			payload: {
				role,
				modelRef,
			},
		});
		if (role === "main") {
			try {
				await ctx.ensureRoleModel("main");
			} catch {
				// Preserve setting even when model cannot be switched immediately.
			}
		}
		ctx.showStatus(`Model role ${role} set to ${modelRef}`);
		return;
	}
	if (subcommand === "clear") {
		const role = rest[0] as ModelRoleName | undefined;
		if (!role || !(MODEL_ROLE_NAMES as readonly string[]).includes(role)) {
			ctx.showWarning("Usage: /models roles clear <main|task|compact|quick>");
			return;
		}
		ctx.settingsManager.setRoleModel(role, undefined);
		runtime.events.record({
			type: "model.roles.updated",
			source: "interactive:/models roles clear",
			payload: { role, modelRef: null },
		});
		ctx.showStatus(`Model role ${role} cleared`);
		return;
	}

	ctx.showWarning(
		"Usage: /models roles [show] | /models roles set <role> <provider/model> | /models roles clear <role>",
	);
}

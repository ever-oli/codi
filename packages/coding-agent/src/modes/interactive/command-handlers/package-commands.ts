/**
 * Package-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import { getAgentDir } from "../../../config.js";
import type { AgentSession } from "../../../core/agent-session.js";
import { DefaultPackageManager } from "../../../core/package-manager.js";
import type { SettingsManager } from "../../../core/settings-manager.js";
import { theme } from "../theme/theme.js";

export interface PackageCommandContext {
	readonly session: AgentSession;
	readonly settingsManager: SettingsManager;

	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	renderRuntimePanel(title: string, lines: string[]): void;
	handleReloadCommand(): Promise<void>;
	showPackageManageSelector(packageManager: DefaultPackageManager): Promise<void>;
	isRuntimeFeatureEnabled(flag: string): boolean;
}

export async function handlePackagesCommand(ctx: PackageCommandContext, text: string): Promise<void> {
	if (!ctx.isRuntimeFeatureEnabled("ui.marketplace")) {
		ctx.showWarning("ui.marketplace is disabled. Enable it in settings.runtime.featureFlags.");
		return;
	}
	if (ctx.session.isStreaming || ctx.session.isCompacting) {
		ctx.showWarning("Wait until the current task is idle before managing packages.");
		return;
	}
	const argText = text.replace(/^\/packages\s*/, "").trim();
	const packageManager = new DefaultPackageManager({
		cwd: process.cwd(),
		agentDir: getAgentDir(),
		settingsManager: ctx.settingsManager,
	});
	packageManager.setProgressCallback((event) => {
		if (event.type === "start") {
			ctx.showStatus(event.message || "Working...");
		}
	});

	if (!argText) {
		const globalPackages = ctx.settingsManager.getGlobalSettings().packages ?? [];
		const projectPackages = ctx.settingsManager.getProjectSettings().packages ?? [];
		ctx.renderRuntimePanel("Packages", [
			`${theme.fg("accent", "Installed")} user=${globalPackages.length} project=${projectPackages.length}`,
			"Commands:",
			"  /packages list",
			"  /packages install <source> [--local]",
			"  /packages remove <source> [--local]",
			"  /packages update [source]",
			"  /packages manage",
		]);
		return;
	}

	const [subcommand, ...rest] = argText.split(/\s+/);
	if (subcommand === "list") {
		const globalPackages = ctx.settingsManager.getGlobalSettings().packages ?? [];
		const projectPackages = ctx.settingsManager.getProjectSettings().packages ?? [];
		const lines: string[] = [];
		if (globalPackages.length > 0) {
			lines.push(theme.bold("User packages:"));
			for (const pkg of globalPackages) {
				const source = typeof pkg === "string" ? pkg : pkg.source;
				lines.push(`  ${source}`);
			}
		}
		if (projectPackages.length > 0) {
			lines.push(theme.bold("Project packages:"));
			for (const pkg of projectPackages) {
				const source = typeof pkg === "string" ? pkg : pkg.source;
				lines.push(`  ${source}`);
			}
		}
		if (lines.length === 0) {
			lines.push(theme.fg("dim", "No packages configured."));
		}
		ctx.renderRuntimePanel("Packages", lines);
		return;
	}

	if (subcommand === "manage") {
		await ctx.showPackageManageSelector(packageManager);
		return;
	}

	if (subcommand === "install" || subcommand === "remove" || subcommand === "update") {
		const local = rest.includes("--local") || rest.includes("-l");
		const source = rest.find((token) => !token.startsWith("-"));
		try {
			if (subcommand === "install") {
				if (!source) {
					ctx.showWarning("Usage: /packages install <source> [--local]");
					return;
				}
				await packageManager.install(source, { local });
				packageManager.addSourceToSettings(source, { local });
				ctx.showStatus(`Installed ${source}`);
				await ctx.handleReloadCommand();
				return;
			}
			if (subcommand === "remove") {
				if (!source) {
					ctx.showWarning("Usage: /packages remove <source> [--local]");
					return;
				}
				await packageManager.remove(source, { local });
				packageManager.removeSourceFromSettings(source, { local });
				ctx.showStatus(`Removed ${source}`);
				await ctx.handleReloadCommand();
				return;
			}
			await packageManager.update(source);
			ctx.showStatus(source ? `Updated ${source}` : "Updated configured packages");
			await ctx.handleReloadCommand();
			return;
		} catch (error) {
			ctx.showError(error instanceof Error ? error.message : String(error));
			return;
		}
	}

	ctx.showWarning(
		"Usage: /packages [list] | /packages install <source> [--local] | /packages remove <source> [--local] | /packages update [source] | /packages manage",
	);
}

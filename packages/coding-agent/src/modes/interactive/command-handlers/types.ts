/**
 * Context interface for slash command handlers.
 * Provides the subset of InteractiveMode state that handlers need.
 */

import type { Model } from "@mariozechner/pi-ai";
import type { AgentSession } from "../../../core/agent-session.js";
import type { ReadonlyFooterDataProvider } from "../../../core/footer-data-provider.js";
import type { ModelRoleName, RuntimeServices } from "../../../core/runtime/index.js";
import type { SettingsManager } from "../../../core/settings-manager.js";

/**
 * Base context that all command handlers receive.
 * Each handler file may extend this with additional typed callbacks.
 */
export interface CommandHandlerContext {
	// Core references
	readonly session: AgentSession;
	readonly settingsManager: SettingsManager;
	readonly runtimeServices: RuntimeServices | undefined;
	readonly footerDataProvider: ReadonlyFooterDataProvider;

	// UI feedback
	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	renderRuntimePanel(title: string, lines: string[]): void;
	getRuntimeOrWarn(flag?: string): RuntimeServices | undefined;

	// UI actions
	updateEditorBorderColor(): void;
	updateFooter(): void;

	// Easter egg
	checkDaxnutsEasterEgg(model: { provider: string; id: string }): void;

	// Model operations
	ensureRoleModel(role: ModelRoleName): Promise<void>;
	getModelCandidates(): Promise<Model<any>[]>;
	findExactModelMatch(searchTerm: string): Promise<Model<any> | undefined>;
	updateAvailableProviderCount(): Promise<void>;

	// Selector display
	showModelSelector(initialSearchInput?: string): void;
	showModelsSelector(): Promise<void>;
	showSettingsSelector(): void;
	showUserMessageSelector(): void;
	showTreeSelector(initialSelectedId?: string): void;
	showSessionSelector(): Promise<void>;
	showOAuthSelector(mode: "login" | "logout"): Promise<void>;

	// Session operations
	promptWithMainRole(text: string, options?: { streamingBehavior: "steer" }): Promise<void>;
	shutdown(): Promise<void>;
	handleClearCommand(): Promise<void>;
	handleCompactCommand(customInstructions?: string): Promise<void>;
	handleReloadCommand(): Promise<void>;
	handleBashCommand(command: string, excludeFromContext: boolean): Promise<void>;
	isExtensionCommand(text: string): boolean;
	queueCompactionMessage(text: string, mode: "steer" | "followUp"): void;
	updatePendingMessagesDisplay(): void;
	flushPendingBashComponents(): void;
}

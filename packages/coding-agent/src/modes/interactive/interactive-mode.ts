/**
 * Interactive mode for the coding agent.
 * Handles TUI rendering and user interaction, delegating business logic to AgentSession.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, ImageContent, Message, Model, OAuthProviderId } from "@mariozechner/pi-ai";
import { getOAuthProviders } from "@mariozechner/pi-ai/oauth";
import type {
	AutocompleteItem,
	EditorAction,
	EditorComponent,
	EditorTheme,
	KeyId,
	MarkdownTheme,
	OverlayHandle,
	OverlayOptions,
	SlashCommand,
} from "@mariozechner/pi-tui";
import {
	CombinedAutocompleteProvider,
	type Component,
	Container,
	fuzzyFilter,
	Loader,
	Markdown,
	matchesKey,
	ProcessTerminal,
	Spacer,
	Text,
	TruncatedText,
	TUI,
	truncateToWidth,
	visibleWidth,
} from "@mariozechner/pi-tui";
import { spawn, spawnSync } from "child_process";
import {
	APP_NAME,
	getAgentDir,
	getAuthPath,
	getDebugLogPath,
	getShareViewerUrl,
	getUpdateInstruction,
	VERSION,
} from "../../config.js";
import { type AgentSession, type AgentSessionEvent, parseSkillBlock } from "../../core/agent-session.js";
import type { CompactionResult } from "../../core/compaction/index.js";
import type {
	ExtensionContext,
	ExtensionRunner,
	ExtensionUIContext,
	ExtensionUIDialogOptions,
	ExtensionWidgetOptions,
} from "../../core/extensions/index.js";
import { FooterDataProvider, type ReadonlyFooterDataProvider } from "../../core/footer-data-provider.js";
import { type AppAction, KeybindingsManager } from "../../core/keybindings.js";
import { createCompactionSummaryMessage } from "../../core/messages.js";
import { resolveModelScope } from "../../core/model-resolver.js";
import { DefaultPackageManager } from "../../core/package-manager.js";
import type { ResourceDiagnostic } from "../../core/resource-loader.js";
import {
	type DelegatedTaskRecord,
	type DelegatedTaskStatus,
	LANE_NAMES,
	type LaneName,
	MODEL_ROLE_NAMES,
	type ModelRoleName,
	RUNTIME_FEATURE_FLAG_NAMES,
	type RuntimeFeatureFlagName,
	type RuntimeServices,
} from "../../core/runtime/index.js";
import { type SessionContext, SessionManager } from "../../core/session-manager.js";
import type { StartupDensity } from "../../core/settings-manager.js";
import { BUILTIN_SLASH_COMMANDS } from "../../core/slash-commands.js";
import type { TruncationResult } from "../../core/tools/truncate.js";
import {
	getActiveTaskCompletionState,
	getLatestTaskVerification,
	getTaskCompletionLabel,
	getTaskVerificationStatus,
	WORKFLOW_PHASES,
	type WorkflowPhase,
} from "../../core/workflow/session-orchestrator.js";
import { buildTaskSubagentContract } from "../../core/workflow/subagents.js";
import {
	areTaskDependenciesSatisfied,
	createTaskGraphFromGoal,
	getSchedulableTasks,
	type TaskStatus,
} from "../../core/workflow/task-graph.js";
import { getChangelogPath, getNewEntries, parseChangelog } from "../../utils/changelog.js";
import { copyToClipboard } from "../../utils/clipboard.js";
import { extensionForImageMimeType, readClipboardImage } from "../../utils/clipboard-image.js";
import { ensureTool } from "../../utils/tools-manager.js";
import { handleInteractiveAgentEvent } from "./agent-event-handler.js";
import { ArminComponent } from "./components/armin.js";
import { AssistantMessageComponent } from "./components/assistant-message.js";
import { BashExecutionComponent } from "./components/bash-execution.js";
import { BorderedLoader } from "./components/bordered-loader.js";
import { BranchSummaryMessageComponent } from "./components/branch-summary-message.js";
import { CompactionSummaryMessageComponent } from "./components/compaction-summary-message.js";
import { ConfigSelectorComponent } from "./components/config-selector.js";
import { CustomEditor } from "./components/custom-editor.js";
import { CustomMessageComponent } from "./components/custom-message.js";
import { DaxnutsComponent } from "./components/daxnuts.js";
import { DynamicBorder } from "./components/dynamic-border.js";
import { ExtensionEditorComponent } from "./components/extension-editor.js";
import { ExtensionInputComponent } from "./components/extension-input.js";
import { ExtensionSelectorComponent } from "./components/extension-selector.js";
import { FooterComponent } from "./components/footer.js";
import { appKey, appKeyHint, editorKey, keyHint, rawKeyHint } from "./components/keybinding-hints.js";
import { LoginDialogComponent } from "./components/login-dialog.js";
import { ModelSelectorComponent } from "./components/model-selector.js";
import { OAuthSelectorComponent } from "./components/oauth-selector.js";
import { ScopedModelsSelectorComponent } from "./components/scoped-models-selector.js";
import { SessionSelectorComponent } from "./components/session-selector.js";
import { SettingsSelectorComponent } from "./components/settings-selector.js";
import { SkillInvocationMessageComponent } from "./components/skill-invocation-message.js";
import { ToolExecutionComponent } from "./components/tool-execution.js";
import { TreeSelectorComponent } from "./components/tree-selector.js";
import { UserMessageComponent } from "./components/user-message.js";
import { UserMessageSelectorComponent } from "./components/user-message-selector.js";
import { handleInteractiveSubmit } from "./submit-dispatch.js";
import {
	getAvailableThemes,
	getAvailableThemesWithPaths,
	getEditorTheme,
	getMarkdownTheme,
	getThemeByName,
	initTheme,
	onThemeChange,
	setRegisteredThemes,
	setTheme,
	setThemeInstance,
	Theme,
	type ThemeColor,
	theme,
} from "./theme/theme.js";

/** Interface for components that can be expanded/collapsed */
interface Expandable {
	setExpanded(expanded: boolean): void;
}

const WORKFLOW_TASK_STATUSES: TaskStatus[] = ["pending", "ready", "in_progress", "blocked", "done", "waived"];

function isExpandable(obj: unknown): obj is Expandable {
	return typeof obj === "object" && obj !== null && "setExpanded" in obj && typeof obj.setExpanded === "function";
}

type CompactionQueuedMessage = {
	text: string;
	mode: "steer" | "followUp";
};

/**
 * Options for InteractiveMode initialization.
 */
export interface InteractiveModeOptions {
	/** Providers that were migrated to auth.json (shows warning) */
	migratedProviders?: string[];
	/** Warning message if session model couldn't be restored */
	modelFallbackMessage?: string;
	/** Initial message to send on startup (can include @file content) */
	initialMessage?: string;
	/** Images to attach to the initial message */
	initialImages?: ImageContent[];
	/** Additional messages to send after the initial message */
	initialMessages?: string[];
	/** Force verbose startup (overrides quietStartup setting) */
	verbose?: boolean;
	/** Runtime services for durable queue/lane/event/mailbox/heartbeat orchestration. */
	runtimeServices?: RuntimeServices;
}

export class InteractiveMode {
	private session: AgentSession;
	private ui: TUI;
	private chatContainer: Container;
	private pendingMessagesContainer: Container;
	private statusContainer: Container;
	private defaultEditor: CustomEditor;
	private editor: EditorComponent;
	private autocompleteProvider: CombinedAutocompleteProvider | undefined;
	private fdPath: string | undefined;
	private editorContainer: Container;
	private footer: FooterComponent;
	private footerDataProvider: FooterDataProvider;
	private keybindings: KeybindingsManager;
	private version: string;
	private startupLogoLines: string[] = [];
	private isInitialized = false;
	private onInputCallback?: (text: string) => void;
	private loadingAnimation: Loader | undefined = undefined;
	private pendingWorkingMessage: string | undefined = undefined;
	private readonly defaultWorkingMessage = "Working...";

	private lastSigintTime = 0;
	private lastEscapeTime = 0;
	private changelogMarkdown: string | undefined = undefined;

	// Status line tracking (for mutating immediately-sequential status updates)
	private lastStatusSpacer: Spacer | undefined = undefined;
	private lastStatusText: Text | undefined = undefined;

	// Streaming message tracking
	private streamingComponent: AssistantMessageComponent | undefined = undefined;
	private streamingMessage: AssistantMessage | undefined = undefined;

	// Tool execution tracking: toolCallId -> component
	private pendingTools = new Map<string, ToolExecutionComponent>();

	// Tool output expansion state
	private toolOutputExpanded = false;

	// Thinking block visibility state
	private hideThinkingBlock = false;

	// Skill commands: command name -> skill file path
	private skillCommands = new Map<string, string>();

	// Agent subscription unsubscribe function
	private unsubscribe?: () => void;

	// Track if editor is in bash mode (text starts with !)
	private isBashMode = false;

	// Track current bash execution component
	private bashComponent: BashExecutionComponent | undefined = undefined;

	// Track pending bash components (shown in pending area, moved to chat on submit)
	private pendingBashComponents: BashExecutionComponent[] = [];

	// Auto-compaction state
	private autoCompactionLoader: Loader | undefined = undefined;
	private autoCompactionEscapeHandler?: () => void;

	// Auto-retry state
	private retryLoader: Loader | undefined = undefined;
	private retryEscapeHandler?: () => void;

	// Messages queued while compaction is running
	private compactionQueuedMessages: CompactionQueuedMessage[] = [];

	// Shutdown state
	private shutdownRequested = false;

	// Extension UI state
	private extensionSelector: ExtensionSelectorComponent | undefined = undefined;
	private extensionInput: ExtensionInputComponent | undefined = undefined;
	private extensionEditor: ExtensionEditorComponent | undefined = undefined;
	private extensionTerminalInputUnsubscribers = new Set<() => void>();

	// Extension widgets (components rendered above/below the editor)
	private extensionWidgetsAbove = new Map<string, Component & { dispose?(): void }>();
	private extensionWidgetsBelow = new Map<string, Component & { dispose?(): void }>();
	private widgetContainerAbove!: Container;
	private widgetContainerBelow!: Container;
	private workflowStripComponent: Component | undefined = undefined;

	// Custom footer from extension (undefined = use built-in footer)
	private customFooter: (Component & { dispose?(): void }) | undefined = undefined;

	// Header container that holds the built-in or custom header
	private headerContainer: Container;

	// Built-in header (logo + keybinding hints + changelog)
	private builtInHeader: Component | undefined = undefined;

	// Custom header from extension (undefined = use built-in header)
	private customHeader: (Component & { dispose?(): void }) | undefined = undefined;
	private readonly runtimeServices: RuntimeServices | undefined;
	private eventTailInterval: ReturnType<typeof setInterval> | undefined;
	private eventTailLastTs = 0;
	private startupResourceListingMode: "none" | "summary" | "summaryPreview" | "full" = "summary";

	// Convenience accessors
	private get agent() {
		return this.session.agent;
	}
	private get sessionManager() {
		return this.session.sessionManager;
	}
	private get settingsManager() {
		return this.session.settingsManager;
	}

	constructor(
		session: AgentSession,
		private options: InteractiveModeOptions = {},
	) {
		this.session = session;
		this.runtimeServices = options.runtimeServices;
		this.version = VERSION;
		this.ui = new TUI(new ProcessTerminal(), this.settingsManager.getShowHardwareCursor());
		this.ui.setClearOnShrink(this.settingsManager.getClearOnShrink());
		this.headerContainer = new Container();
		this.chatContainer = new Container();
		this.pendingMessagesContainer = new Container();
		this.statusContainer = new Container();
		this.widgetContainerAbove = new Container();
		this.widgetContainerBelow = new Container();
		this.keybindings = KeybindingsManager.create();
		const editorPaddingX = this.settingsManager.getEditorPaddingX();
		const autocompleteMaxVisible = this.settingsManager.getAutocompleteMaxVisible();
		this.defaultEditor = new CustomEditor(this.ui, getEditorTheme(), this.keybindings, {
			paddingX: editorPaddingX,
			autocompleteMaxVisible,
		});
		this.editor = this.defaultEditor;
		this.editorContainer = new Container();
		this.editorContainer.addChild(this.editor as Component);
		this.footerDataProvider = new FooterDataProvider();
		this.footer = new FooterComponent(session, this.footerDataProvider);
		this.footer.setAutoCompactEnabled(session.autoCompactionEnabled);
		this.startupLogoLines = this.loadStartupLogo();

		// Load hide thinking block setting
		this.hideThinkingBlock = this.settingsManager.getHideThinkingBlock();

		// Register themes from resource loader and initialize
		setRegisteredThemes(this.session.resourceLoader.getThemes().themes);
		initTheme(this.settingsManager.getTheme(), true);
	}

	private loadStartupLogo(): string[] {
		if (APP_NAME.toLowerCase() !== "codi") {
			return [];
		}

		const envPath = process.env.CODI_UI_LOGO_PATH?.trim();
		const candidates = [
			envPath,
			"/Users/ever/Downloads/2025/spike.txt",
			path.join(os.homedir(), ".pi", "agent", "spike.txt"),
		]
			.filter((p): p is string => Boolean(p))
			.map((p) => (p.startsWith("~/") ? path.join(os.homedir(), p.slice(2)) : p));

		let raw = "";
		for (const filePath of candidates) {
			try {
				raw = fs.readFileSync(filePath, "utf8");
				if (raw.trim()) break;
			} catch {
				// continue to next candidate
			}
		}
		if (!raw.trim()) {
			return [];
		}

		return raw
			.split("\n")
			.map((line) => line.replace(/\r/g, "").replace(/\s+$/g, ""))
			.filter((line) => line.length > 0);
	}

	private getStartupLogoBlock(verbose: boolean): string[] {
		if (this.startupLogoLines.length === 0) {
			return [];
		}

		// Fit by height first so tall ASCII logos don't get aggressively cut.
		const terminalRows = this.ui.terminal.rows || process.stdout.rows || 24;
		const terminalCols = this.ui.terminal.columns || process.stdout.columns || 80;
		const reservedRows = verbose ? 24 : 13;
		const minLogoRows = verbose ? 4 : 3;
		const preferredLogoRows = verbose ? 16 : 9;
		const maxLinesByHeight = Math.max(minLogoRows, terminalRows - reservedRows);
		const maxLines = Math.min(preferredLogoRows, maxLinesByHeight, this.startupLogoLines.length);

		const preferredCols = verbose ? 100 : 72;
		const maxCols = Math.max(24, Math.min(preferredCols, terminalCols - 4));
		return this.startupLogoLines
			.slice(0, maxLines)
			.map((line) => theme.fg("muted", truncateToWidth(line, maxCols, "…")));
	}

	private setupAutocomplete(fdPath: string | undefined): void {
		// Define commands for autocomplete
		const slashCommands: SlashCommand[] = BUILTIN_SLASH_COMMANDS.map((command) => ({
			name: command.name,
			description: command.description,
		}));

		const modelCommand = slashCommands.find((command) => command.name === "model");
		if (modelCommand) {
			modelCommand.getArgumentCompletions = (prefix: string): AutocompleteItem[] | null => {
				// Get available models (scoped or from registry)
				const models =
					this.session.scopedModels.length > 0
						? this.session.scopedModels.map((s) => s.model)
						: this.session.modelRegistry.getAvailable();

				if (models.length === 0) return null;

				// Create items with provider/id format
				const items = models.map((m) => ({
					id: m.id,
					provider: m.provider,
					label: `${m.provider}/${m.id}`,
				}));

				// Fuzzy filter by model ID + provider (allows "opus anthropic" to match)
				const filtered = fuzzyFilter(items, prefix, (item) => `${item.id} ${item.provider}`);

				if (filtered.length === 0) return null;

				return filtered.map((item) => ({
					value: item.label,
					label: item.id,
					description: item.provider,
				}));
			};
		}

		// Convert prompt templates to SlashCommand format for autocomplete
		const templateCommands: SlashCommand[] = this.session.promptTemplates.map((cmd) => ({
			name: cmd.name,
			description: cmd.description,
		}));

		// Convert extension commands to SlashCommand format
		const builtinCommandNames = new Set(slashCommands.map((c) => c.name));
		const extensionCommands: SlashCommand[] = (
			this.session.extensionRunner?.getRegisteredCommands(builtinCommandNames) ?? []
		).map((cmd) => ({
			name: cmd.name,
			description: cmd.description ?? "(extension command)",
			getArgumentCompletions: cmd.getArgumentCompletions,
		}));

		// Build skill commands from session.skills (if enabled)
		this.skillCommands.clear();
		const skillCommandList: SlashCommand[] = [];
		if (this.settingsManager.getEnableSkillCommands()) {
			for (const skill of this.session.resourceLoader.getSkills().skills) {
				const commandName = `skill:${skill.name}`;
				this.skillCommands.set(commandName, skill.filePath);
				skillCommandList.push({ name: commandName, description: skill.description });
			}
		}

		// Setup autocomplete
		this.autocompleteProvider = new CombinedAutocompleteProvider(
			[...slashCommands, ...templateCommands, ...extensionCommands, ...skillCommandList],
			process.cwd(),
			fdPath,
		);
		this.defaultEditor.setAutocompleteProvider(this.autocompleteProvider);
		if (this.editor !== this.defaultEditor) {
			this.editor.setAutocompleteProvider?.(this.autocompleteProvider);
		}
	}

	async init(): Promise<void> {
		if (this.isInitialized) return;

		// Load changelog (only show new entries, skip for resumed sessions)
		this.changelogMarkdown = this.getChangelogForDisplay();

		// Ensure fd and rg are available (downloads if missing, adds to PATH via getBinDir)
		// Both are needed: fd for autocomplete, rg for grep tool and bash commands
		const [fdPath] = await Promise.all([ensureTool("fd"), ensureTool("rg")]);
		this.fdPath = fdPath;

		// Add header container as first child
		this.ui.addChild(this.headerContainer);

		const startupPresentation = this.getStartupPresentation();
		this.startupResourceListingMode = startupPresentation.listingMode;
		this.builtInHeader = new Text(this.buildStartupHeader(startupPresentation.headerVariant), 1, 0);

		// Setup UI layout
		this.headerContainer.addChild(new Spacer(1));
		this.headerContainer.addChild(this.builtInHeader);
		this.headerContainer.addChild(new Spacer(1));

		// Add changelog if provided
		if (this.changelogMarkdown) {
			this.headerContainer.addChild(new DynamicBorder());
			const versionMatch = this.changelogMarkdown.match(/##\s+\[?(\d+\.\d+\.\d+)\]?/);
			const latestVersion = versionMatch ? versionMatch[1] : this.version;
			const condensedText = `Updated to v${latestVersion}. Use ${theme.bold("/changelog")} to view full changelog.`;
			if (startupPresentation.headerVariant === "verbose" && !this.settingsManager.getCollapseChangelog()) {
				this.headerContainer.addChild(new Text(theme.bold(theme.fg("accent", "What's New")), 1, 0));
				this.headerContainer.addChild(new Spacer(1));
				this.headerContainer.addChild(
					new Markdown(this.changelogMarkdown.trim(), 1, 0, this.getMarkdownThemeWithSettings()),
				);
				this.headerContainer.addChild(new Spacer(1));
			} else {
				this.headerContainer.addChild(new Text(condensedText, 1, 0));
			}
			this.headerContainer.addChild(new DynamicBorder());
		}

		this.ui.addChild(this.chatContainer);
		this.ui.addChild(this.pendingMessagesContainer);
		this.ui.addChild(this.statusContainer);
		this.renderWidgets(); // Initialize with default spacer
		this.ui.addChild(this.widgetContainerAbove);
		this.ui.addChild(this.editorContainer);
		this.ui.addChild(this.widgetContainerBelow);
		this.ui.addChild(this.footer);
		this.ui.setFocus(this.editor);

		this.setupKeyHandlers();
		this.setupEditorSubmitHandler();

		// Initialize extensions first so resources are shown before messages
		await this.initExtensions();

		// Render initial messages AFTER showing loaded resources
		this.renderInitialMessages();

		// Start the UI
		this.ui.start();
		this.isInitialized = true;

		// Set terminal title
		this.updateTerminalTitle();

		// Subscribe to agent events
		this.subscribeToAgent();

		// Set up theme file watcher
		onThemeChange(() => {
			this.ui.invalidate();
			this.updateEditorBorderColor();
			this.ui.requestRender();
		});

		// Set up git branch watcher (uses provider instead of footer)
		this.footerDataProvider.onBranchChange(() => {
			this.ui.requestRender();
		});

		// Initialize available provider count for footer display
		await this.updateAvailableProviderCount();
	}

	/**
	 * Update terminal title with session name and cwd.
	 */
	private updateTerminalTitle(): void {
		const cwdBasename = path.basename(process.cwd());
		const sessionName = this.sessionManager.getSessionName();
		if (sessionName) {
			this.ui.terminal.setTitle(`π - ${sessionName} - ${cwdBasename}`);
		} else {
			this.ui.terminal.setTitle(`π - ${cwdBasename}`);
		}
	}

	/**
	 * Run the interactive mode. This is the main entry point.
	 * Initializes the UI, shows warnings, processes initial messages, and starts the interactive loop.
	 */
	async run(): Promise<void> {
		await this.init();

		// Start version check asynchronously
		this.checkForNewVersion().then((newVersion) => {
			if (newVersion) {
				this.showNewVersionNotification(newVersion);
			}
		});

		// Show startup warnings
		const { migratedProviders, modelFallbackMessage, initialMessage, initialImages, initialMessages } = this.options;

		if (migratedProviders && migratedProviders.length > 0) {
			this.showWarning(`Migrated credentials to auth.json: ${migratedProviders.join(", ")}`);
		}

		const modelsJsonError = this.session.modelRegistry.getError();
		if (modelsJsonError) {
			this.showError(`models.json error: ${modelsJsonError}`);
		}

		if (modelFallbackMessage) {
			this.showWarning(modelFallbackMessage);
		}

		// Process initial messages
		if (initialMessage) {
			try {
				await this.promptWithMainRole(initialMessage, { images: initialImages });
			} catch (error: unknown) {
				const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
				this.showError(errorMessage);
			}
		}

		if (initialMessages) {
			for (const message of initialMessages) {
				try {
					await this.promptWithMainRole(message);
				} catch (error: unknown) {
					const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
					this.showError(errorMessage);
				}
			}
		}

		// Main interactive loop
		while (true) {
			const userInput = await this.getUserInput();
			try {
				await this.promptWithMainRole(userInput);
			} catch (error: unknown) {
				const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
				this.showError(errorMessage);
			}
		}
	}

	/**
	 * Check npm registry for a newer version.
	 */
	private async checkForNewVersion(): Promise<string | undefined> {
		if (process.env.PI_SKIP_VERSION_CHECK || process.env.PI_OFFLINE) return undefined;

		try {
			const response = await fetch("https://registry.npmjs.org/@mariozechner/pi-coding-agent/latest", {
				signal: AbortSignal.timeout(10000),
			});
			if (!response.ok) return undefined;

			const data = (await response.json()) as { version?: string };
			const latestVersion = data.version;

			if (latestVersion && latestVersion !== this.version) {
				return latestVersion;
			}

			return undefined;
		} catch {
			return undefined;
		}
	}

	/**
	 * Get changelog entries to display on startup.
	 * Only shows new entries since last seen version, skips for resumed sessions.
	 */
	private getChangelogForDisplay(): string | undefined {
		// Skip changelog for resumed/continued sessions (already have messages)
		if (this.session.state.messages.length > 0) {
			return undefined;
		}

		const lastVersion = this.settingsManager.getLastChangelogVersion();
		const changelogPath = getChangelogPath();
		const entries = parseChangelog(changelogPath);

		if (!lastVersion) {
			// Fresh install - just record the version, don't show changelog
			this.settingsManager.setLastChangelogVersion(VERSION);
			return undefined;
		} else {
			const newEntries = getNewEntries(entries, lastVersion);
			if (newEntries.length > 0) {
				this.settingsManager.setLastChangelogVersion(VERSION);
				return newEntries.map((e) => e.content).join("\n\n");
			}
		}

		return undefined;
	}

	private getMarkdownThemeWithSettings(): MarkdownTheme {
		return {
			...getMarkdownTheme(),
			codeBlockIndent: this.settingsManager.getCodeBlockIndent(),
		};
	}

	// =========================================================================
	// Extension System
	// =========================================================================

	private formatDisplayPath(p: string): string {
		const home = os.homedir();
		let result = p;

		// Replace home directory with ~
		if (result.startsWith(home)) {
			result = `~${result.slice(home.length)}`;
		}

		return result;
	}

	/**
	 * Get a short path relative to the package root for display.
	 */
	private getShortPath(fullPath: string, source: string): string {
		// For npm packages, show path relative to node_modules/pkg/
		const npmMatch = fullPath.match(/node_modules\/(@?[^/]+(?:\/[^/]+)?)\/(.*)/);
		if (npmMatch && source.startsWith("npm:")) {
			return npmMatch[2];
		}

		// For git packages, show path relative to repo root
		const gitMatch = fullPath.match(/git\/[^/]+\/[^/]+\/(.*)/);
		if (gitMatch && source.startsWith("git:")) {
			return gitMatch[1];
		}

		// For local/auto, just use formatDisplayPath
		return this.formatDisplayPath(fullPath);
	}

	private getDisplaySourceInfo(
		source: string,
		scope: string,
	): { label: string; scopeLabel?: string; color: "accent" | "muted" } {
		if (source === "local") {
			if (scope === "user") {
				return { label: "user", color: "muted" };
			}
			if (scope === "project") {
				return { label: "project", color: "muted" };
			}
			if (scope === "temporary") {
				return { label: "path", scopeLabel: "temp", color: "muted" };
			}
			return { label: "path", color: "muted" };
		}

		if (source === "cli") {
			return { label: "path", scopeLabel: scope === "temporary" ? "temp" : undefined, color: "muted" };
		}

		const scopeLabel =
			scope === "user" ? "user" : scope === "project" ? "project" : scope === "temporary" ? "temp" : undefined;
		return { label: source, scopeLabel, color: "accent" };
	}

	private getScopeGroup(source: string, scope: string): "user" | "project" | "path" {
		if (source === "cli" || scope === "temporary") return "path";
		if (scope === "user") return "user";
		if (scope === "project") return "project";
		return "path";
	}

	private isPackageSource(source: string): boolean {
		return source.startsWith("npm:") || source.startsWith("git:");
	}

	private buildScopeGroups(
		paths: string[],
		metadata: Map<string, { source: string; scope: string; origin: string }>,
	): Array<{ scope: "user" | "project" | "path"; paths: string[]; packages: Map<string, string[]> }> {
		const groups: Record<
			"user" | "project" | "path",
			{ scope: "user" | "project" | "path"; paths: string[]; packages: Map<string, string[]> }
		> = {
			user: { scope: "user", paths: [], packages: new Map() },
			project: { scope: "project", paths: [], packages: new Map() },
			path: { scope: "path", paths: [], packages: new Map() },
		};

		for (const p of paths) {
			const meta = this.findMetadata(p, metadata);
			const source = meta?.source ?? "local";
			const scope = meta?.scope ?? "project";
			const groupKey = this.getScopeGroup(source, scope);
			const group = groups[groupKey];

			if (this.isPackageSource(source)) {
				const list = group.packages.get(source) ?? [];
				list.push(p);
				group.packages.set(source, list);
			} else {
				group.paths.push(p);
			}
		}

		return [groups.project, groups.user, groups.path].filter(
			(group) => group.paths.length > 0 || group.packages.size > 0,
		);
	}

	private countScopeGroupEntries(group: {
		scope: "user" | "project" | "path";
		paths: string[];
		packages: Map<string, string[]>;
	}): number {
		let total = group.paths.length;
		for (const paths of group.packages.values()) {
			total += paths.length;
		}
		return total;
	}

	private summarizeScopeGroups(
		groups: Array<{ scope: "user" | "project" | "path"; paths: string[]; packages: Map<string, string[]> }>,
	): string {
		return groups
			.map((group) => ({ scope: group.scope, count: this.countScopeGroupEntries(group) }))
			.filter((group) => group.count > 0)
			.map((group) => `${group.scope} ${group.count}`)
			.join(", ");
	}

	private formatInlinePreview(items: string[], maxVisible = 3): string {
		const visible = items.slice(0, maxVisible);
		const hiddenCount = Math.max(0, items.length - visible.length);
		const preview = visible.join(", ");
		if (hiddenCount === 0) {
			return preview;
		}
		return preview ? `${preview}, +${hiddenCount} more` : `+${hiddenCount} more`;
	}

	private formatCount(count: number, singular: string, plural = `${singular}s`): string {
		return `${count} ${count === 1 ? singular : plural}`;
	}

	private formatScopeGroups(
		groups: Array<{ scope: "user" | "project" | "path"; paths: string[]; packages: Map<string, string[]> }>,
		options: {
			formatPath: (p: string) => string;
			formatPackagePath: (p: string, source: string) => string;
		},
	): string {
		const lines: string[] = [];

		for (const group of groups) {
			lines.push(`  ${theme.fg("accent", group.scope)}`);

			const sortedPaths = [...group.paths].sort((a, b) => a.localeCompare(b));
			for (const p of sortedPaths) {
				lines.push(theme.fg("dim", `    ${options.formatPath(p)}`));
			}

			const sortedPackages = Array.from(group.packages.entries()).sort(([a], [b]) => a.localeCompare(b));
			for (const [source, paths] of sortedPackages) {
				lines.push(`    ${theme.fg("mdLink", source)}`);
				const sortedPackagePaths = [...paths].sort((a, b) => a.localeCompare(b));
				for (const p of sortedPackagePaths) {
					lines.push(theme.fg("dim", `      ${options.formatPackagePath(p, source)}`));
				}
			}
		}

		return lines.join("\n");
	}

	/**
	 * Find metadata for a path, checking parent directories if exact match fails.
	 * Package manager stores metadata for directories, but we display file paths.
	 */
	private findMetadata(
		p: string,
		metadata: Map<string, { source: string; scope: string; origin: string }>,
	): { source: string; scope: string; origin: string } | undefined {
		// Try exact match first
		const exact = metadata.get(p);
		if (exact) return exact;

		// Try parent directories (package manager stores directory paths)
		let current = p;
		while (current.includes("/")) {
			current = current.substring(0, current.lastIndexOf("/"));
			const parent = metadata.get(current);
			if (parent) return parent;
		}

		return undefined;
	}

	/**
	 * Format a path with its source/scope info from metadata.
	 */
	private formatPathWithSource(
		p: string,
		metadata: Map<string, { source: string; scope: string; origin: string }>,
	): string {
		const meta = this.findMetadata(p, metadata);
		if (meta) {
			const shortPath = this.getShortPath(p, meta.source);
			const { label, scopeLabel } = this.getDisplaySourceInfo(meta.source, meta.scope);
			const labelText = scopeLabel ? `${label} (${scopeLabel})` : label;
			return `${labelText} ${shortPath}`;
		}
		return this.formatDisplayPath(p);
	}

	/**
	 * Format resource diagnostics with nice collision display using metadata.
	 */
	private formatDiagnostics(
		diagnostics: readonly ResourceDiagnostic[],
		metadata: Map<string, { source: string; scope: string; origin: string }>,
	): string {
		const lines: string[] = [];

		// Group collision diagnostics by name
		const collisions = new Map<string, ResourceDiagnostic[]>();
		const otherDiagnostics: ResourceDiagnostic[] = [];

		for (const d of diagnostics) {
			if (d.type === "collision" && d.collision) {
				const list = collisions.get(d.collision.name) ?? [];
				list.push(d);
				collisions.set(d.collision.name, list);
			} else {
				otherDiagnostics.push(d);
			}
		}

		// Format collision diagnostics grouped by name
		for (const [name, collisionList] of collisions) {
			const first = collisionList[0]?.collision;
			if (!first) continue;
			lines.push(theme.fg("warning", `  "${name}" collision:`));
			// Show winner
			lines.push(
				theme.fg("dim", `    ${theme.fg("success", "✓")} ${this.formatPathWithSource(first.winnerPath, metadata)}`),
			);
			// Show all losers
			for (const d of collisionList) {
				if (d.collision) {
					lines.push(
						theme.fg(
							"dim",
							`    ${theme.fg("warning", "✗")} ${this.formatPathWithSource(d.collision.loserPath, metadata)} (skipped)`,
						),
					);
				}
			}
		}

		// Format other diagnostics (skill name collisions, parse errors, etc.)
		for (const d of otherDiagnostics) {
			if (d.path) {
				// Use metadata-aware formatting for paths
				const sourceInfo = this.formatPathWithSource(d.path, metadata);
				lines.push(theme.fg(d.type === "error" ? "error" : "warning", `  ${sourceInfo}`));
				lines.push(theme.fg(d.type === "error" ? "error" : "warning", `    ${d.message}`));
			} else {
				lines.push(theme.fg(d.type === "error" ? "error" : "warning", `  ${d.message}`));
			}
		}

		return lines.join("\n");
	}

	private resolveStartupDensity(): StartupDensity {
		if (this.options.verbose === true) {
			return "verbose";
		}
		return this.settingsManager.getStartupDensity();
	}

	private getStartupPresentation(): {
		density: StartupDensity;
		headerVariant: "single" | "compact" | "verbose";
		listingMode: "none" | "summary" | "summaryPreview" | "full";
	} {
		const density = this.resolveStartupDensity();
		if (density === "verbose") {
			return { density, headerVariant: "verbose", listingMode: "full" };
		}
		if (density === "compact") {
			return { density, headerVariant: "compact", listingMode: "summary" };
		}
		const terminalRows = this.ui.terminal.rows || process.stdout.rows || 24;
		if (terminalRows < 34) {
			return { density, headerVariant: "single", listingMode: "none" };
		}
		if (terminalRows < 48) {
			return { density, headerVariant: "compact", listingMode: "summary" };
		}
		return { density, headerVariant: "compact", listingMode: "summaryPreview" };
	}

	private buildStartupHeader(variant: "single" | "compact" | "verbose"): string {
		const logo = theme.bold(theme.fg("accent", APP_NAME)) + theme.fg("dim", ` v${this.version}`);
		const logoBlock = this.getStartupLogoBlock(variant === "verbose");
		const kb = this.keybindings;
		const hint = (action: AppAction, desc: string) => appKeyHint(kb, action, desc);

		if (variant === "verbose") {
			const instructions = [
				hint("interrupt", "to interrupt"),
				hint("clear", "to clear"),
				rawKeyHint(`${appKey(kb, "clear")} twice`, "to exit"),
				hint("exit", "to exit (empty)"),
				hint("suspend", "to suspend"),
				keyHint("deleteToLineEnd", "to delete to end"),
				hint("cycleThinkingLevel", "to cycle thinking level"),
				rawKeyHint(`${appKey(kb, "cycleModelForward")}/${appKey(kb, "cycleModelBackward")}`, "to cycle models"),
				hint("selectModel", "to select model"),
				hint("expandTools", "to expand tools"),
				hint("toggleThinking", "to expand thinking"),
				hint("externalEditor", "for external editor"),
				rawKeyHint("/", "for commands"),
				rawKeyHint("!", "to run bash"),
				rawKeyHint("!!", "to run bash (no context)"),
				hint("followUp", "to queue follow-up"),
				hint("dequeue", "to edit all queued messages"),
				hint("pasteImage", "to paste image"),
				rawKeyHint("drop files", "to attach"),
			].join("\n");
			if (logoBlock.length > 0) {
				return `${logo}\n${logoBlock.join("\n")}\n${instructions}`;
			}
			return `${logo}\n${instructions}`;
		}

		const compactPrimary = [
			hint("interrupt", "interrupt"),
			hint("selectModel", "model"),
			hint("expandTools", "tools"),
			hint("toggleThinking", "thinking"),
			hint("externalEditor", "editor"),
			rawKeyHint("/", "commands"),
		].join(theme.fg("dim", " | "));
		const compactSecondary = [
			hint("clear", "clear"),
			hint("followUp", "queue"),
			rawKeyHint("!", "bash"),
			theme.fg("dim", "Use /hotkeys for full list"),
		].join(theme.fg("dim", " | "));

		if (variant === "single") {
			return `${logo} ${theme.fg("dim", "|")} ${compactPrimary}`;
		}
		return `${logo} ${theme.fg("dim", "|")} ${compactPrimary}\n${compactSecondary}`;
	}

	private showLoadedResources(options?: {
		extensionPaths?: string[];
		listingMode?: "none" | "summary" | "summaryPreview" | "full";
		issuesOnly?: boolean;
		showDiagnostics?: boolean;
	}): void {
		const listingMode = options?.issuesOnly ? "none" : (options?.listingMode ?? "summary");
		const showListing = listingMode !== "none";
		const showDetailedListing = listingMode === "full";
		const showPreview = listingMode === "summaryPreview";
		const showDiagnostics = options?.showDiagnostics ?? true;
		if (!showListing && !showDiagnostics) {
			return;
		}

		const metadata = this.session.resourceLoader.getPathMetadata();
		const sectionHeader = (name: string, color: ThemeColor = "mdHeading") => theme.fg(color, `[${name}]`);

		const skillsResult = this.session.resourceLoader.getSkills();
		const promptsResult = this.session.resourceLoader.getPrompts();
		const themesResult = this.session.resourceLoader.getThemes();

		if (showListing) {
			const contextFiles = this.session.resourceLoader.getAgentsFiles().agentsFiles;
			const skills = skillsResult.skills;
			const templates = this.session.promptTemplates;
			const extensionPaths = options?.extensionPaths ?? [];
			const loadedThemes = themesResult.themes;
			const customThemes = loadedThemes.filter((t) => t.sourcePath);
			if (showDetailedListing) {
				if (contextFiles.length > 0) {
					this.chatContainer.addChild(new Spacer(1));
					const contextList = contextFiles
						.map((f) => theme.fg("dim", `  ${this.formatDisplayPath(f.path)}`))
						.join("\n");
					this.chatContainer.addChild(new Text(`${sectionHeader("Context")}\n${contextList}`, 0, 0));
					this.chatContainer.addChild(new Spacer(1));
				}

				if (skills.length > 0) {
					const skillPaths = skills.map((s) => s.filePath);
					const groups = this.buildScopeGroups(skillPaths, metadata);
					const skillList = this.formatScopeGroups(groups, {
						formatPath: (p) => this.formatDisplayPath(p),
						formatPackagePath: (p, source) => this.getShortPath(p, source),
					});
					this.chatContainer.addChild(new Text(`${sectionHeader("Skills")}\n${skillList}`, 0, 0));
					this.chatContainer.addChild(new Spacer(1));
				}

				if (templates.length > 0) {
					const templatePaths = templates.map((t) => t.filePath);
					const groups = this.buildScopeGroups(templatePaths, metadata);
					const templateByPath = new Map(templates.map((t) => [t.filePath, t]));
					const templateList = this.formatScopeGroups(groups, {
						formatPath: (p) => {
							const template = templateByPath.get(p);
							return template ? `/${template.name}` : this.formatDisplayPath(p);
						},
						formatPackagePath: (p) => {
							const template = templateByPath.get(p);
							return template ? `/${template.name}` : this.formatDisplayPath(p);
						},
					});
					this.chatContainer.addChild(new Text(`${sectionHeader("Prompts")}\n${templateList}`, 0, 0));
					this.chatContainer.addChild(new Spacer(1));
				}

				if (extensionPaths.length > 0) {
					const groups = this.buildScopeGroups(extensionPaths, metadata);
					const extList = this.formatScopeGroups(groups, {
						formatPath: (p) => this.formatDisplayPath(p),
						formatPackagePath: (p, source) => this.getShortPath(p, source),
					});
					this.chatContainer.addChild(new Text(`${sectionHeader("Extensions", "mdHeading")}\n${extList}`, 0, 0));
					this.chatContainer.addChild(new Spacer(1));
				}

				if (customThemes.length > 0) {
					const themePaths = customThemes.map((t) => t.sourcePath!);
					const groups = this.buildScopeGroups(themePaths, metadata);
					const themeList = this.formatScopeGroups(groups, {
						formatPath: (p) => this.formatDisplayPath(p),
						formatPackagePath: (p, source) => this.getShortPath(p, source),
					});
					this.chatContainer.addChild(new Text(`${sectionHeader("Themes")}\n${themeList}`, 0, 0));
					this.chatContainer.addChild(new Spacer(1));
				}
			} else {
				const summaryLines: string[] = [];

				if (contextFiles.length > 0) {
					if (showPreview) {
						const contextPreview = this.formatInlinePreview(
							contextFiles.map((file) => this.formatDisplayPath(file.path)),
							2,
						);
						const suffix = contextPreview ? `: ${theme.fg("dim", contextPreview)}` : "";
						summaryLines.push(
							`${sectionHeader("Context")} ${this.formatCount(contextFiles.length, "file")}${suffix}`,
						);
					} else {
						summaryLines.push(`${sectionHeader("Context")} ${this.formatCount(contextFiles.length, "file")}`);
					}
				}

				if (skills.length > 0) {
					const skillPaths = skills.map((s) => s.filePath);
					const groups = this.buildScopeGroups(skillPaths, metadata);
					if (showPreview) {
						const scopeSummary = this.summarizeScopeGroups(groups);
						const suffix = scopeSummary ? ` ${theme.fg("dim", `(${scopeSummary})`)}` : "";
						summaryLines.push(
							`${sectionHeader("Skills")} ${this.formatCount(skills.length, "loaded skill")}${suffix}`,
						);
					} else {
						summaryLines.push(`${sectionHeader("Skills")} ${this.formatCount(skills.length, "loaded skill")}`);
					}
				}

				if (templates.length > 0) {
					if (showPreview) {
						const templatePreview = this.formatInlinePreview(
							templates.map((template) => `/${template.name}`),
							2,
						);
						const suffix = templatePreview ? `: ${theme.fg("dim", templatePreview)}` : "";
						summaryLines.push(
							`${sectionHeader("Prompts")} ${this.formatCount(templates.length, "prompt")}${suffix}`,
						);
					} else {
						summaryLines.push(`${sectionHeader("Prompts")} ${this.formatCount(templates.length, "prompt")}`);
					}
				}

				if (extensionPaths.length > 0) {
					const groups = this.buildScopeGroups(extensionPaths, metadata);
					if (showPreview) {
						const scopeSummary = this.summarizeScopeGroups(groups);
						const suffix = scopeSummary ? ` ${theme.fg("dim", `(${scopeSummary})`)}` : "";
						summaryLines.push(
							`${sectionHeader("Extensions")} ${this.formatCount(extensionPaths.length, "loaded extension")}${suffix}`,
						);
					} else {
						summaryLines.push(
							`${sectionHeader("Extensions")} ${this.formatCount(extensionPaths.length, "loaded extension")}`,
						);
					}
				}

				if (customThemes.length > 0) {
					const themePaths = customThemes.map((t) => t.sourcePath!);
					const groups = this.buildScopeGroups(themePaths, metadata);
					if (showPreview) {
						const scopeSummary = this.summarizeScopeGroups(groups);
						const suffix = scopeSummary ? ` ${theme.fg("dim", `(${scopeSummary})`)}` : "";
						summaryLines.push(
							`${sectionHeader("Themes")} ${this.formatCount(customThemes.length, "theme")}${suffix}`,
						);
					} else {
						summaryLines.push(`${sectionHeader("Themes")} ${this.formatCount(customThemes.length, "theme")}`);
					}
				}

				if (summaryLines.length > 0) {
					summaryLines.push(theme.fg("dim", "Use /resources full for detailed startup listings."));
					this.chatContainer.addChild(new Spacer(1));
					this.chatContainer.addChild(new Text(summaryLines.join("\n"), 0, 0));
					this.chatContainer.addChild(new Spacer(1));
				}
			}
		}

		if (showDiagnostics) {
			const skillDiagnostics = skillsResult.diagnostics;
			if (skillDiagnostics.length > 0) {
				const warningLines = this.formatDiagnostics(skillDiagnostics, metadata);
				this.chatContainer.addChild(new Text(`${theme.fg("warning", "[Skill conflicts]")}\n${warningLines}`, 0, 0));
				this.chatContainer.addChild(new Spacer(1));
			}

			const promptDiagnostics = promptsResult.diagnostics;
			if (promptDiagnostics.length > 0) {
				const warningLines = this.formatDiagnostics(promptDiagnostics, metadata);
				this.chatContainer.addChild(
					new Text(`${theme.fg("warning", "[Prompt conflicts]")}\n${warningLines}`, 0, 0),
				);
				this.chatContainer.addChild(new Spacer(1));
			}

			const extensionDiagnostics: ResourceDiagnostic[] = [];
			const extensionErrors = this.session.resourceLoader.getExtensions().errors;
			if (extensionErrors.length > 0) {
				for (const error of extensionErrors) {
					extensionDiagnostics.push({ type: "error", message: error.error, path: error.path });
				}
			}

			const commandDiagnostics = this.session.extensionRunner?.getCommandDiagnostics() ?? [];
			extensionDiagnostics.push(...commandDiagnostics);

			const shortcutDiagnostics = this.session.extensionRunner?.getShortcutDiagnostics() ?? [];
			extensionDiagnostics.push(...shortcutDiagnostics);

			if (extensionDiagnostics.length > 0) {
				const warningLines = this.formatDiagnostics(extensionDiagnostics, metadata);
				this.chatContainer.addChild(
					new Text(`${theme.fg("warning", "[Extension issues]")}\n${warningLines}`, 0, 0),
				);
				this.chatContainer.addChild(new Spacer(1));
			}

			const themeDiagnostics = themesResult.diagnostics;
			if (themeDiagnostics.length > 0) {
				const warningLines = this.formatDiagnostics(themeDiagnostics, metadata);
				this.chatContainer.addChild(new Text(`${theme.fg("warning", "[Theme conflicts]")}\n${warningLines}`, 0, 0));
				this.chatContainer.addChild(new Spacer(1));
			}
		}
	}

	/**
	 * Initialize the extension system with TUI-based UI context.
	 */
	private async initExtensions(): Promise<void> {
		const uiContext = this.createExtensionUIContext();
		await this.session.bindExtensions({
			uiContext,
			commandContextActions: {
				waitForIdle: () => this.session.agent.waitForIdle(),
				newSession: async (options) => {
					if (this.loadingAnimation) {
						this.loadingAnimation.stop();
						this.loadingAnimation = undefined;
					}
					this.statusContainer.clear();

					// Delegate to AgentSession (handles setup + agent state sync)
					const success = await this.session.newSession(options);
					if (!success) {
						return { cancelled: true };
					}

					// Clear UI state
					this.chatContainer.clear();
					this.pendingMessagesContainer.clear();
					this.compactionQueuedMessages = [];
					this.streamingComponent = undefined;
					this.streamingMessage = undefined;
					this.pendingTools.clear();

					// Render any messages added via setup, or show empty session
					this.renderInitialMessages();
					this.ui.requestRender();

					return { cancelled: false };
				},
				fork: async (entryId) => {
					const result = await this.session.fork(entryId);
					if (result.cancelled) {
						return { cancelled: true };
					}

					this.chatContainer.clear();
					this.renderInitialMessages();
					this.editor.setText(result.selectedText);
					this.showStatus("Forked to new session");

					return { cancelled: false };
				},
				navigateTree: async (targetId, options) => {
					const result = await this.session.navigateTree(targetId, {
						summarize: options?.summarize,
						customInstructions: options?.customInstructions,
						replaceInstructions: options?.replaceInstructions,
						label: options?.label,
					});
					if (result.cancelled) {
						return { cancelled: true };
					}

					this.chatContainer.clear();
					this.renderInitialMessages();
					if (result.editorText && !this.editor.getText().trim()) {
						this.editor.setText(result.editorText);
					}
					this.showStatus("Navigated to selected point");

					return { cancelled: false };
				},
				switchSession: async (sessionPath) => {
					await this.handleResumeSession(sessionPath);
					return { cancelled: false };
				},
				reload: async () => {
					await this.handleReloadCommand();
				},
			},
			shutdownHandler: () => {
				this.shutdownRequested = true;
				if (!this.session.isStreaming) {
					void this.shutdown();
				}
			},
			onError: (error) => {
				this.showExtensionError(error.extensionPath, error.error, error.stack);
			},
		});

		setRegisteredThemes(this.session.resourceLoader.getThemes().themes);
		this.setupAutocomplete(this.fdPath);

		const extensionRunner = this.session.extensionRunner;
		if (!extensionRunner) {
			this.showLoadedResources({
				extensionPaths: [],
				listingMode: this.startupResourceListingMode,
				showDiagnostics: true,
			});
			return;
		}

		this.setupExtensionShortcuts(extensionRunner);
		this.showLoadedResources({
			extensionPaths: extensionRunner.getExtensionPaths(),
			listingMode: this.startupResourceListingMode,
			showDiagnostics: true,
		});
	}

	/**
	 * Get a registered tool definition by name (for custom rendering).
	 */
	private getRegisteredToolDefinition(toolName: string) {
		const tools = this.session.extensionRunner?.getAllRegisteredTools() ?? [];
		const registeredTool = tools.find((t) => t.definition.name === toolName);
		return registeredTool?.definition;
	}

	/**
	 * Set up keyboard shortcuts registered by extensions.
	 */
	private setupExtensionShortcuts(extensionRunner: ExtensionRunner): void {
		const shortcuts = extensionRunner.getShortcuts(this.keybindings.getEffectiveConfig());
		if (shortcuts.size === 0) return;

		// Create a context for shortcut handlers
		const createContext = (): ExtensionContext => ({
			ui: this.createExtensionUIContext(),
			hasUI: true,
			cwd: process.cwd(),
			sessionManager: this.sessionManager,
			modelRegistry: this.session.modelRegistry,
			model: this.session.model,
			isIdle: () => !this.session.isStreaming,
			abort: () => this.session.abort(),
			hasPendingMessages: () => this.session.pendingMessageCount > 0,
			shutdown: () => {
				this.shutdownRequested = true;
			},
			getContextUsage: () => this.session.getContextUsage(),
			compact: (options) => {
				void (async () => {
					try {
						const result = await this.executeCompaction(options?.customInstructions, false);
						if (result) {
							options?.onComplete?.(result);
						}
					} catch (error) {
						const err = error instanceof Error ? error : new Error(String(error));
						options?.onError?.(err);
					}
				})();
			},
			getSystemPrompt: () => this.session.systemPrompt,
		});

		// Set up the extension shortcut handler on the default editor
		this.defaultEditor.onExtensionShortcut = (data: string) => {
			for (const [shortcutStr, shortcut] of shortcuts) {
				// Cast to KeyId - extension shortcuts use the same format
				if (matchesKey(data, shortcutStr as KeyId)) {
					// Run handler async, don't block input
					Promise.resolve(shortcut.handler(createContext())).catch((err) => {
						this.showError(`Shortcut handler error: ${err instanceof Error ? err.message : String(err)}`);
					});
					return true;
				}
			}
			return false;
		};
	}

	/**
	 * Set extension status text in the footer.
	 */
	private setExtensionStatus(key: string, text: string | undefined): void {
		this.footerDataProvider.setExtensionStatus(key, text);
		this.ui.requestRender();
	}

	/**
	 * Set an extension widget (string array or custom component).
	 */
	private setExtensionWidget(
		key: string,
		content: string[] | ((tui: TUI, thm: Theme) => Component & { dispose?(): void }) | undefined,
		options?: ExtensionWidgetOptions,
	): void {
		const placement = options?.placement ?? "aboveEditor";
		const removeExisting = (map: Map<string, Component & { dispose?(): void }>) => {
			const existing = map.get(key);
			if (existing?.dispose) existing.dispose();
			map.delete(key);
		};

		removeExisting(this.extensionWidgetsAbove);
		removeExisting(this.extensionWidgetsBelow);

		if (content === undefined) {
			this.renderWidgets();
			return;
		}

		let component: Component & { dispose?(): void };

		if (Array.isArray(content)) {
			// Wrap string array in a Container with Text components
			const container = new Container();
			for (const line of content.slice(0, InteractiveMode.MAX_WIDGET_LINES)) {
				container.addChild(new Text(line, 1, 0));
			}
			if (content.length > InteractiveMode.MAX_WIDGET_LINES) {
				container.addChild(new Text(theme.fg("muted", "... (widget truncated)"), 1, 0));
			}
			component = container;
		} else {
			// Factory function - create component
			component = content(this.ui, theme);
		}

		const targetMap = placement === "belowEditor" ? this.extensionWidgetsBelow : this.extensionWidgetsAbove;
		targetMap.set(key, component);
		this.renderWidgets();
	}

	private clearExtensionWidgets(): void {
		for (const widget of this.extensionWidgetsAbove.values()) {
			widget.dispose?.();
		}
		for (const widget of this.extensionWidgetsBelow.values()) {
			widget.dispose?.();
		}
		this.extensionWidgetsAbove.clear();
		this.extensionWidgetsBelow.clear();
		this.renderWidgets();
	}

	private resetExtensionUI(): void {
		if (this.extensionSelector) {
			this.hideExtensionSelector();
		}
		if (this.extensionInput) {
			this.hideExtensionInput();
		}
		if (this.extensionEditor) {
			this.hideExtensionEditor();
		}
		this.ui.hideOverlay();
		this.clearExtensionTerminalInputListeners();
		this.setExtensionFooter(undefined);
		this.setExtensionHeader(undefined);
		this.clearExtensionWidgets();
		this.footerDataProvider.clearExtensionStatuses();
		this.footer.invalidate();
		this.setCustomEditorComponent(undefined);
		this.defaultEditor.onExtensionShortcut = undefined;
		this.updateTerminalTitle();
		if (this.loadingAnimation) {
			this.loadingAnimation.setMessage(
				`${this.defaultWorkingMessage} (${appKey(this.keybindings, "interrupt")} to interrupt)`,
			);
		}
	}

	// Maximum total widget lines to prevent viewport overflow
	private static readonly MAX_WIDGET_LINES = 10;

	/**
	 * Render all extension widgets to the widget container.
	 */
	private renderWidgets(): void {
		if (!this.widgetContainerAbove || !this.widgetContainerBelow) return;
		this.updateWorkflowStrip();
		this.renderWidgetContainer(this.widgetContainerAbove, this.extensionWidgetsAbove, true, true);
		this.renderWidgetContainer(this.widgetContainerBelow, this.extensionWidgetsBelow, false, false);
		this.ui.requestRender();
	}

	private renderWidgetContainer(
		container: Container,
		widgets: Map<string, Component & { dispose?(): void }>,
		spacerWhenEmpty: boolean,
		leadingSpacer: boolean,
	): void {
		container.clear();
		const includeWorkflowStrip = container === this.widgetContainerAbove && this.workflowStripComponent !== undefined;

		if (widgets.size === 0 && !includeWorkflowStrip) {
			if (spacerWhenEmpty) {
				container.addChild(new Spacer(1));
			}
			return;
		}

		if (leadingSpacer) {
			container.addChild(new Spacer(1));
		}
		if (includeWorkflowStrip && this.workflowStripComponent) {
			container.addChild(this.workflowStripComponent);
		}
		for (const component of widgets.values()) {
			container.addChild(component);
		}
	}

	/**
	 * Set a custom footer component, or restore the built-in footer.
	 */
	private setExtensionFooter(
		factory:
			| ((tui: TUI, thm: Theme, footerData: ReadonlyFooterDataProvider) => Component & { dispose?(): void })
			| undefined,
	): void {
		// Dispose existing custom footer
		if (this.customFooter?.dispose) {
			this.customFooter.dispose();
		}

		// Remove current footer from UI
		if (this.customFooter) {
			this.ui.removeChild(this.customFooter);
		} else {
			this.ui.removeChild(this.footer);
		}

		if (factory) {
			// Create and add custom footer, passing the data provider
			this.customFooter = factory(this.ui, theme, this.footerDataProvider);
			this.ui.addChild(this.customFooter);
		} else {
			// Restore built-in footer
			this.customFooter = undefined;
			this.ui.addChild(this.footer);
		}

		this.ui.requestRender();
	}

	/**
	 * Set a custom header component, or restore the built-in header.
	 */
	private setExtensionHeader(factory: ((tui: TUI, thm: Theme) => Component & { dispose?(): void }) | undefined): void {
		// Header may not be initialized yet if called during early initialization
		if (!this.builtInHeader) {
			return;
		}

		// Dispose existing custom header
		if (this.customHeader?.dispose) {
			this.customHeader.dispose();
		}

		// Find the index of the current header in the header container
		const currentHeader = this.customHeader || this.builtInHeader;
		const index = this.headerContainer.children.indexOf(currentHeader);

		if (factory) {
			// Create and add custom header
			this.customHeader = factory(this.ui, theme);
			if (index !== -1) {
				this.headerContainer.children[index] = this.customHeader;
			} else {
				// If not found (e.g. builtInHeader was never added), add at the top
				this.headerContainer.children.unshift(this.customHeader);
			}
		} else {
			// Restore built-in header
			this.customHeader = undefined;
			if (index !== -1) {
				this.headerContainer.children[index] = this.builtInHeader;
			}
		}

		this.ui.requestRender();
	}

	private addExtensionTerminalInputListener(
		handler: (data: string) => { consume?: boolean; data?: string } | undefined,
	): () => void {
		const unsubscribe = this.ui.addInputListener(handler);
		this.extensionTerminalInputUnsubscribers.add(unsubscribe);
		return () => {
			unsubscribe();
			this.extensionTerminalInputUnsubscribers.delete(unsubscribe);
		};
	}

	private clearExtensionTerminalInputListeners(): void {
		for (const unsubscribe of this.extensionTerminalInputUnsubscribers) {
			unsubscribe();
		}
		this.extensionTerminalInputUnsubscribers.clear();
	}

	/**
	 * Create the ExtensionUIContext for extensions.
	 */
	private createExtensionUIContext(): ExtensionUIContext {
		return {
			select: (title, options, opts) => this.showExtensionSelector(title, options, opts),
			confirm: (title, message, opts) => this.showExtensionConfirm(title, message, opts),
			input: (title, placeholder, opts) => this.showExtensionInput(title, placeholder, opts),
			notify: (message, type) => this.showExtensionNotify(message, type),
			onTerminalInput: (handler) => this.addExtensionTerminalInputListener(handler),
			setStatus: (key, text) => this.setExtensionStatus(key, text),
			setWorkingMessage: (message) => {
				if (this.loadingAnimation) {
					if (message) {
						this.loadingAnimation.setMessage(message);
					} else {
						this.loadingAnimation.setMessage(
							`${this.defaultWorkingMessage} (${appKey(this.keybindings, "interrupt")} to interrupt)`,
						);
					}
				} else {
					// Queue message for when loadingAnimation is created (handles agent_start race)
					this.pendingWorkingMessage = message;
				}
			},
			setWidget: (key, content, options) => this.setExtensionWidget(key, content, options),
			setFooter: (factory) => this.setExtensionFooter(factory),
			setHeader: (factory) => this.setExtensionHeader(factory),
			setTitle: (title) => this.ui.terminal.setTitle(title),
			custom: (factory, options) => this.showExtensionCustom(factory, options),
			pasteToEditor: (text) => this.editor.handleInput(`\x1b[200~${text}\x1b[201~`),
			setEditorText: (text) => this.editor.setText(text),
			getEditorText: () => this.editor.getText(),
			editor: (title, prefill) => this.showExtensionEditor(title, prefill),
			setEditorComponent: (factory) => this.setCustomEditorComponent(factory),
			get theme() {
				return theme;
			},
			getAllThemes: () => getAvailableThemesWithPaths(),
			getTheme: (name) => getThemeByName(name),
			setTheme: (themeOrName) => {
				if (themeOrName instanceof Theme) {
					setThemeInstance(themeOrName);
					this.ui.requestRender();
					return { success: true };
				}
				const result = setTheme(themeOrName, true);
				if (result.success) {
					if (this.settingsManager.getTheme() !== themeOrName) {
						this.settingsManager.setTheme(themeOrName);
					}
					this.ui.requestRender();
				}
				return result;
			},
			getToolsExpanded: () => this.toolOutputExpanded,
			setToolsExpanded: (expanded) => this.setToolsExpanded(expanded),
		};
	}

	/**
	 * Show a selector for extensions.
	 */
	private showExtensionSelector(
		title: string,
		options: string[],
		opts?: ExtensionUIDialogOptions,
	): Promise<string | undefined> {
		return new Promise((resolve) => {
			if (opts?.signal?.aborted) {
				resolve(undefined);
				return;
			}

			const onAbort = () => {
				this.hideExtensionSelector();
				resolve(undefined);
			};
			opts?.signal?.addEventListener("abort", onAbort, { once: true });

			this.extensionSelector = new ExtensionSelectorComponent(
				title,
				options,
				(option) => {
					opts?.signal?.removeEventListener("abort", onAbort);
					this.hideExtensionSelector();
					resolve(option);
				},
				() => {
					opts?.signal?.removeEventListener("abort", onAbort);
					this.hideExtensionSelector();
					resolve(undefined);
				},
				{ tui: this.ui, timeout: opts?.timeout },
			);

			this.editorContainer.clear();
			this.editorContainer.addChild(this.extensionSelector);
			this.ui.setFocus(this.extensionSelector);
			this.ui.requestRender();
		});
	}

	/**
	 * Hide the extension selector.
	 */
	private hideExtensionSelector(): void {
		this.extensionSelector?.dispose();
		this.editorContainer.clear();
		this.editorContainer.addChild(this.editor);
		this.extensionSelector = undefined;
		this.ui.setFocus(this.editor);
		this.ui.requestRender();
	}

	/**
	 * Show a confirmation dialog for extensions.
	 */
	private async showExtensionConfirm(
		title: string,
		message: string,
		opts?: ExtensionUIDialogOptions,
	): Promise<boolean> {
		const result = await this.showExtensionSelector(`${title}\n${message}`, ["Yes", "No"], opts);
		return result === "Yes";
	}

	/**
	 * Show a text input for extensions.
	 */
	private showExtensionInput(
		title: string,
		placeholder?: string,
		opts?: ExtensionUIDialogOptions,
	): Promise<string | undefined> {
		return new Promise((resolve) => {
			if (opts?.signal?.aborted) {
				resolve(undefined);
				return;
			}

			const onAbort = () => {
				this.hideExtensionInput();
				resolve(undefined);
			};
			opts?.signal?.addEventListener("abort", onAbort, { once: true });

			this.extensionInput = new ExtensionInputComponent(
				title,
				placeholder,
				(value) => {
					opts?.signal?.removeEventListener("abort", onAbort);
					this.hideExtensionInput();
					resolve(value);
				},
				() => {
					opts?.signal?.removeEventListener("abort", onAbort);
					this.hideExtensionInput();
					resolve(undefined);
				},
				{ tui: this.ui, timeout: opts?.timeout },
			);

			this.editorContainer.clear();
			this.editorContainer.addChild(this.extensionInput);
			this.ui.setFocus(this.extensionInput);
			this.ui.requestRender();
		});
	}

	/**
	 * Hide the extension input.
	 */
	private hideExtensionInput(): void {
		this.extensionInput?.dispose();
		this.editorContainer.clear();
		this.editorContainer.addChild(this.editor);
		this.extensionInput = undefined;
		this.ui.setFocus(this.editor);
		this.ui.requestRender();
	}

	/**
	 * Show a multi-line editor for extensions (with Ctrl+G support).
	 */
	private showExtensionEditor(title: string, prefill?: string): Promise<string | undefined> {
		return new Promise((resolve) => {
			this.extensionEditor = new ExtensionEditorComponent(
				this.ui,
				this.keybindings,
				title,
				prefill,
				(value) => {
					this.hideExtensionEditor();
					resolve(value);
				},
				() => {
					this.hideExtensionEditor();
					resolve(undefined);
				},
			);

			this.editorContainer.clear();
			this.editorContainer.addChild(this.extensionEditor);
			this.ui.setFocus(this.extensionEditor);
			this.ui.requestRender();
		});
	}

	/**
	 * Hide the extension editor.
	 */
	private hideExtensionEditor(): void {
		this.editorContainer.clear();
		this.editorContainer.addChild(this.editor);
		this.extensionEditor = undefined;
		this.ui.setFocus(this.editor);
		this.ui.requestRender();
	}

	/**
	 * Set a custom editor component from an extension.
	 * Pass undefined to restore the default editor.
	 */
	private setCustomEditorComponent(
		factory: ((tui: TUI, theme: EditorTheme, keybindings: KeybindingsManager) => EditorComponent) | undefined,
	): void {
		// Save text from current editor before switching
		const currentText = this.editor.getText();

		this.editorContainer.clear();

		if (factory) {
			// Create the custom editor with tui, theme, and keybindings
			const newEditor = factory(this.ui, getEditorTheme(), this.keybindings);

			// Wire up callbacks from the default editor
			newEditor.onSubmit = this.defaultEditor.onSubmit;
			newEditor.onChange = this.defaultEditor.onChange;

			// Copy text from previous editor
			newEditor.setText(currentText);

			// Copy appearance settings if supported
			if (newEditor.borderColor !== undefined) {
				newEditor.borderColor = this.defaultEditor.borderColor;
			}
			if (newEditor.setPaddingX !== undefined) {
				newEditor.setPaddingX(this.defaultEditor.getPaddingX());
			}

			// Set autocomplete if supported
			if (newEditor.setAutocompleteProvider && this.autocompleteProvider) {
				newEditor.setAutocompleteProvider(this.autocompleteProvider);
			}

			// If extending CustomEditor, copy app-level handlers
			// Use duck typing since instanceof fails across jiti module boundaries
			const customEditor = newEditor as unknown as Record<string, unknown>;
			if ("actionHandlers" in customEditor && customEditor.actionHandlers instanceof Map) {
				customEditor.onEscape = () => this.defaultEditor.onEscape?.();
				customEditor.onCtrlD = () => this.defaultEditor.onCtrlD?.();
				customEditor.onPasteImage = () => this.defaultEditor.onPasteImage?.();
				customEditor.onExtensionShortcut = (data: string) => this.defaultEditor.onExtensionShortcut?.(data);
				// Copy action handlers (clear, suspend, model switching, etc.)
				for (const [action, handler] of this.defaultEditor.actionHandlers) {
					(customEditor.actionHandlers as Map<string, () => void>).set(action, handler);
				}
			}

			this.editor = newEditor;
		} else {
			// Restore default editor with text from custom editor
			this.defaultEditor.setText(currentText);
			this.editor = this.defaultEditor;
		}

		this.editorContainer.addChild(this.editor as Component);
		this.ui.setFocus(this.editor as Component);
		this.ui.requestRender();
	}

	/**
	 * Show a notification for extensions.
	 */
	private showExtensionNotify(message: string, type?: "info" | "warning" | "error"): void {
		if (type === "error") {
			this.showError(message);
		} else if (type === "warning") {
			this.showWarning(message);
		} else {
			this.showStatus(message);
		}
	}

	/** Show a custom component with keyboard focus. Overlay mode renders on top of existing content. */
	private async showExtensionCustom<T>(
		factory: (
			tui: TUI,
			theme: Theme,
			keybindings: KeybindingsManager,
			done: (result: T) => void,
		) => (Component & { dispose?(): void }) | Promise<Component & { dispose?(): void }>,
		options?: {
			overlay?: boolean;
			overlayOptions?: OverlayOptions | (() => OverlayOptions);
			onHandle?: (handle: OverlayHandle) => void;
		},
	): Promise<T> {
		const savedText = this.editor.getText();
		const isOverlay = options?.overlay ?? false;

		const restoreEditor = () => {
			this.editorContainer.clear();
			this.editorContainer.addChild(this.editor);
			this.editor.setText(savedText);
			this.ui.setFocus(this.editor);
			this.ui.requestRender();
		};

		return new Promise((resolve, reject) => {
			let component: Component & { dispose?(): void };
			let closed = false;

			const close = (result: T) => {
				if (closed) return;
				closed = true;
				if (isOverlay) this.ui.hideOverlay();
				else restoreEditor();
				// Note: both branches above already call requestRender
				resolve(result);
				try {
					component?.dispose?.();
				} catch {
					/* ignore dispose errors */
				}
			};

			Promise.resolve(factory(this.ui, theme, this.keybindings, close))
				.then((c) => {
					if (closed) return;
					component = c;
					if (isOverlay) {
						// Resolve overlay options - can be static or dynamic function
						const resolveOptions = (): OverlayOptions | undefined => {
							if (options?.overlayOptions) {
								const opts =
									typeof options.overlayOptions === "function"
										? options.overlayOptions()
										: options.overlayOptions;
								return opts;
							}
							// Fallback: use component's width property if available
							const w = (component as { width?: number }).width;
							return w ? { width: w } : undefined;
						};
						const handle = this.ui.showOverlay(component, resolveOptions());
						// Expose handle to caller for visibility control
						options?.onHandle?.(handle);
					} else {
						this.editorContainer.clear();
						this.editorContainer.addChild(component);
						this.ui.setFocus(component);
						this.ui.requestRender();
					}
				})
				.catch((err) => {
					if (closed) return;
					if (!isOverlay) restoreEditor();
					reject(err);
				});
		});
	}

	/**
	 * Show an extension error in the UI.
	 */
	private showExtensionError(extensionPath: string, error: string, stack?: string): void {
		const errorMsg = `Extension "${extensionPath}" error: ${error}`;
		const errorText = new Text(theme.fg("error", errorMsg), 1, 0);
		this.chatContainer.addChild(errorText);
		if (stack) {
			// Show stack trace in dim color, indented
			const stackLines = stack
				.split("\n")
				.slice(1) // Skip first line (duplicates error message)
				.map((line) => theme.fg("dim", `  ${line.trim()}`))
				.join("\n");
			if (stackLines) {
				this.chatContainer.addChild(new Text(stackLines, 1, 0));
			}
		}
		this.ui.requestRender();
	}

	// =========================================================================
	// Key Handlers
	// =========================================================================

	private setupKeyHandlers(): void {
		// Set up handlers on defaultEditor - they use this.editor for text access
		// so they work correctly regardless of which editor is active
		this.defaultEditor.onEscape = () => {
			if (this.loadingAnimation) {
				this.restoreQueuedMessagesToEditor({ abort: true });
			} else if (this.session.isBashRunning) {
				this.session.abortBash();
			} else if (this.isBashMode) {
				this.editor.setText("");
				this.isBashMode = false;
				this.updateEditorBorderColor();
			} else if (!this.editor.getText().trim()) {
				// Double-escape with empty editor triggers /tree, /fork, or nothing based on setting
				const action = this.settingsManager.getDoubleEscapeAction();
				if (action !== "none") {
					const now = Date.now();
					if (now - this.lastEscapeTime < 500) {
						if (action === "tree") {
							this.showTreeSelector();
						} else {
							this.showUserMessageSelector();
						}
						this.lastEscapeTime = 0;
					} else {
						this.lastEscapeTime = now;
					}
				}
			}
		};

		// Register app action handlers
		this.defaultEditor.onAction("clear", () => this.handleCtrlC());
		this.defaultEditor.onCtrlD = () => this.handleCtrlD();
		this.defaultEditor.onAction("suspend", () => this.handleCtrlZ());
		this.defaultEditor.onAction("cycleThinkingLevel", () => this.cycleThinkingLevel());
		this.defaultEditor.onAction("cycleModelForward", () => this.cycleModel("forward"));
		this.defaultEditor.onAction("cycleModelBackward", () => this.cycleModel("backward"));

		// Global debug handler on TUI (works regardless of focus)
		this.ui.onDebug = () => this.handleDebugCommand();
		this.defaultEditor.onAction("selectModel", () => this.showModelSelector());
		this.defaultEditor.onAction("expandTools", () => this.toggleToolOutputExpansion());
		this.defaultEditor.onAction("toggleThinking", () => this.toggleThinkingBlockVisibility());
		this.defaultEditor.onAction("externalEditor", () => this.openExternalEditor());
		this.defaultEditor.onAction("followUp", () => this.handleFollowUp());
		this.defaultEditor.onAction("dequeue", () => this.handleDequeue());
		this.defaultEditor.onAction("newSession", () => this.handleClearCommand());
		this.defaultEditor.onAction("tree", () => this.showTreeSelector());
		this.defaultEditor.onAction("fork", () => this.showUserMessageSelector());
		this.defaultEditor.onAction("resume", () => this.showSessionSelector());

		this.defaultEditor.onChange = (text: string) => {
			const wasBashMode = this.isBashMode;
			this.isBashMode = text.trimStart().startsWith("!");
			if (wasBashMode !== this.isBashMode) {
				this.updateEditorBorderColor();
			}
		};

		// Handle clipboard image paste (triggered on Ctrl+V)
		this.defaultEditor.onPasteImage = () => {
			this.handleClipboardImagePaste();
		};
	}

	private async handleClipboardImagePaste(): Promise<void> {
		try {
			const image = await readClipboardImage();
			if (!image) {
				return;
			}

			// Write to temp file
			const tmpDir = os.tmpdir();
			const ext = extensionForImageMimeType(image.mimeType) ?? "png";
			const fileName = `pi-clipboard-${crypto.randomUUID()}.${ext}`;
			const filePath = path.join(tmpDir, fileName);
			fs.writeFileSync(filePath, Buffer.from(image.bytes));

			// Insert file path directly
			this.editor.insertTextAtCursor?.(filePath);
			this.ui.requestRender();
		} catch {
			// Silently ignore clipboard errors (may not have permission, etc.)
		}
	}

	private isRuntimeFeatureEnabled(flag: Parameters<RuntimeServices["isFeatureEnabled"]>[0]): boolean {
		return this.runtimeServices?.isFeatureEnabled(flag) ?? false;
	}

	private async ensureRoleModel(role: ModelRoleName): Promise<void> {
		if (!this.runtimeServices || !this.isRuntimeFeatureEnabled("model.roleProfiles")) {
			return;
		}
		const roleModel = this.runtimeServices.modelRoles.resolveRoleModel(role);
		if (!roleModel) {
			return;
		}
		const currentModel = this.session.model;
		if (currentModel && currentModel.provider === roleModel.provider && currentModel.id === roleModel.id) {
			return;
		}
		await this.session.setModel(roleModel);
		this.footer.invalidate();
		this.updateEditorBorderColor();
	}

	private async withTemporaryRoleModel<T>(role: ModelRoleName, run: () => Promise<T>): Promise<T> {
		if (!this.runtimeServices || !this.isRuntimeFeatureEnabled("model.roleProfiles")) {
			return run();
		}
		const roleModel = this.runtimeServices.modelRoles.resolveRoleModel(role);
		const previous = this.session.model;
		const shouldSwitch =
			roleModel && (!previous || previous.provider !== roleModel.provider || previous.id !== roleModel.id);
		if (!shouldSwitch) {
			return run();
		}
		await this.session.setModel(roleModel!);
		this.footer.invalidate();
		this.updateEditorBorderColor();
		try {
			return await run();
		} finally {
			if (previous) {
				try {
					await this.session.setModel(previous);
					this.footer.invalidate();
					this.updateEditorBorderColor();
				} catch {
					// Keep current model if restoration fails.
				}
			}
		}
	}

	private async promptWithMainRole(text: string, options?: Parameters<AgentSession["prompt"]>[1]): Promise<void> {
		await this.ensureRoleModel("main");
		await this.session.prompt(text, options);
	}

	private setupEditorSubmitHandler(): void {
		this.defaultEditor.onSubmit = async (text: string) => {
			await handleInteractiveSubmit(
				{
					editor: this.editor,
					session: this.session,
					onInputCallback: this.onInputCallback,
					setBashMode: (enabled) => {
						this.isBashMode = enabled;
					},
					updateEditorBorderColor: () => {
						this.updateEditorBorderColor();
					},
					showWarning: (message) => {
						this.showWarning(message);
					},
					showSettingsSelector: () => {
						this.showSettingsSelector();
					},
					showModelsSelector: async () => {
						await this.showModelsSelector();
					},
					handleModelCommand: async (searchTerm) => {
						await this.handleModelCommand(searchTerm);
					},
					handleExportCommand: async (submitText) => {
						await this.handleExportCommand(submitText);
					},
					handleShareCommand: async () => {
						await this.handleShareCommand();
					},
					handleCopyCommand: () => {
						this.handleCopyCommand();
					},
					handleNameCommand: (submitText) => {
						this.handleNameCommand(submitText);
					},
					handleSessionCommand: () => {
						this.handleSessionCommand();
					},
					handleEventsCommand: async (submitText) => {
						await this.handleEventsCommand(submitText);
					},
					handleQueueCommand: async (submitText) => {
						await this.handleQueueCommand(submitText);
					},
					handleLanesCommand: async (submitText) => {
						await this.handleLanesCommand(submitText);
					},
					handlePackagesCommand: async (submitText) => {
						await this.handlePackagesCommand(submitText);
					},
					handleMailboxCommand: async (submitText) => {
						await this.handleMailboxCommand(submitText);
					},
					handleDelegatedCommand: async (submitText) => {
						await this.handleDelegatedCommand(submitText);
					},
					handleHeartbeatCommand: async (submitText) => {
						await this.handleHeartbeatCommand(submitText);
					},
					handleModelsCommand: async (submitText) => {
						await this.handleModelsCommand(submitText);
					},
					handleOpsCommand: async (submitText) => {
						await this.handleOpsCommand(submitText);
					},
					handleWorkflowPlanCommand: (submitText) => {
						this.handleWorkflowPlanCommand(submitText);
					},
					handleWorkflowPhaseCommand: (submitText) => {
						this.handleWorkflowPhaseCommand(submitText);
					},
					handleWorkflowTaskCommand: (submitText) => {
						this.handleWorkflowTaskCommand(submitText);
					},
					handleWorkflowVerifyCommand: (submitText) => {
						this.handleWorkflowVerifyCommand(submitText);
					},
					handleWorkflowSummaryCommand: () => {
						this.handleWorkflowSummaryCommand();
					},
					handleResourcesCommand: (submitText) => {
						this.handleResourcesCommand(submitText);
					},
					handleChangelogCommand: () => {
						this.handleChangelogCommand();
					},
					handleHotkeysCommand: () => {
						this.handleHotkeysCommand();
					},
					showUserMessageSelector: () => {
						this.showUserMessageSelector();
					},
					showTreeSelector: () => {
						this.showTreeSelector();
					},
					showOAuthSelector: (mode) => {
						this.showOAuthSelector(mode);
					},
					handleClearCommand: async () => {
						await this.handleClearCommand();
					},
					handleCompactCommand: async (customInstructions) => {
						await this.handleCompactCommand(customInstructions);
					},
					handleReloadCommand: async () => {
						await this.handleReloadCommand();
					},
					handleDebugCommand: () => {
						this.handleDebugCommand();
					},
					handleArminSaysHi: () => {
						this.handleArminSaysHi();
					},
					showSessionSelector: () => {
						this.showSessionSelector();
					},
					shutdown: async () => {
						await this.shutdown();
					},
					handleBashCommand: async (command, isExcluded) => {
						await this.handleBashCommand(command, isExcluded);
					},
					isExtensionCommand: (submitText) => {
						return this.isExtensionCommand(submitText);
					},
					queueCompactionMessage: (submitText, mode) => {
						this.queueCompactionMessage(submitText, mode);
					},
					promptWithMainRole: async (submitText, options) => {
						await this.promptWithMainRole(submitText, options);
					},
					updatePendingMessagesDisplay: () => {
						this.updatePendingMessagesDisplay();
					},
					requestRender: () => {
						this.ui.requestRender();
					},
					flushPendingBashComponents: () => {
						this.flushPendingBashComponents();
					},
				},
				text,
			);
		};
	}

	private subscribeToAgent(): void {
		this.unsubscribe = this.session.subscribe(async (event) => {
			await this.handleEvent(event);
		});
	}

	private async handleEvent(event: AgentSessionEvent): Promise<void> {
		const state = {
			retryEscapeHandler: this.retryEscapeHandler,
			retryLoader: this.retryLoader,
			loadingAnimation: this.loadingAnimation,
			pendingWorkingMessage: this.pendingWorkingMessage,
			streamingComponent: this.streamingComponent,
			streamingMessage: this.streamingMessage,
			autoCompactionEscapeHandler: this.autoCompactionEscapeHandler,
			autoCompactionLoader: this.autoCompactionLoader,
		};

		await handleInteractiveAgentEvent(
			{
				state,
				isInitialized: this.isInitialized,
				init: async () => {
					await this.init();
				},
				runtimeServices: this.runtimeServices,
				sessionId: this.session.sessionId,
				defaultEditor: this.defaultEditor,
				session: {
					abortCompaction: () => {
						this.session.abortCompaction();
					},
					abortRetry: () => {
						this.session.abortRetry();
					},
					retryAttempt: this.session.retryAttempt,
				},
				statusContainer: this.statusContainer,
				chatContainer: this.chatContainer,
				ui: this.ui,
				defaultWorkingMessage: this.defaultWorkingMessage,
				interruptKeyLabel: appKey(this.keybindings, "interrupt"),
				hideThinkingBlock: this.hideThinkingBlock,
				settingsManager: this.settingsManager,
				toolOutputExpanded: this.toolOutputExpanded,
				pendingTools: this.pendingTools,
				footerInvalidate: () => {
					this.footer.invalidate();
				},
				getMarkdownThemeWithSettings: () => {
					return this.getMarkdownThemeWithSettings();
				},
				getRegisteredToolDefinition: (toolName) => {
					return this.getRegisteredToolDefinition(toolName);
				},
				addMessageToChat: (message) => {
					this.addMessageToChat(message);
				},
				updatePendingMessagesDisplay: () => {
					this.updatePendingMessagesDisplay();
				},
				renderWidgets: () => {
					this.renderWidgets();
				},
				checkShutdownRequested: async () => {
					await this.checkShutdownRequested();
				},
				rebuildChatFromMessages: () => {
					this.rebuildChatFromMessages();
				},
				flushCompactionQueue: async (options) => {
					await this.flushCompactionQueue(options);
				},
				showStatus: (message) => {
					this.showStatus(message);
				},
				showError: (message) => {
					this.showError(message);
				},
				updateEditorBorderColor: () => {
					this.updateEditorBorderColor();
				},
				setActiveTool: (toolName) => {
					this.footer.setActiveTool(toolName);
				},
			},
			event,
		);
		this.retryEscapeHandler = state.retryEscapeHandler;
		this.retryLoader = state.retryLoader;
		this.loadingAnimation = state.loadingAnimation;
		this.pendingWorkingMessage = state.pendingWorkingMessage;
		this.streamingComponent = state.streamingComponent;
		this.streamingMessage = state.streamingMessage;
		this.autoCompactionEscapeHandler = state.autoCompactionEscapeHandler;
		this.autoCompactionLoader = state.autoCompactionLoader;
	}

	/** Extract text content from a user message */
	private getUserMessageText(message: Message): string {
		if (message.role !== "user") return "";
		const textBlocks =
			typeof message.content === "string"
				? [{ type: "text", text: message.content }]
				: message.content.filter((c: { type: string }) => c.type === "text");
		return textBlocks.map((c) => (c as { text: string }).text).join("");
	}

	/**
	 * Show a status message in the chat.
	 *
	 * If multiple status messages are emitted back-to-back (without anything else being added to the chat),
	 * we update the previous status line instead of appending new ones to avoid log spam.
	 */
	private showStatus(message: string): void {
		const children = this.chatContainer.children;
		const last = children.length > 0 ? children[children.length - 1] : undefined;
		const secondLast = children.length > 1 ? children[children.length - 2] : undefined;

		if (last && secondLast && last === this.lastStatusText && secondLast === this.lastStatusSpacer) {
			this.lastStatusText.setText(theme.fg("dim", message));
			this.ui.requestRender();
			return;
		}

		const spacer = new Spacer(1);
		const text = new Text(theme.fg("dim", message), 1, 0);
		this.chatContainer.addChild(spacer);
		this.chatContainer.addChild(text);
		this.lastStatusSpacer = spacer;
		this.lastStatusText = text;
		this.ui.requestRender();
	}

	private formatWorkflowLabel(value: string): string {
		return value.replace(/\b\w/g, (char) => char.toUpperCase());
	}

	private formatTaskCompletionLabel(value: string): string {
		switch (value) {
			case "completion_ready":
				return "Completion Ready";
			case "needs_verification":
				return "Needs Verification";
			case "failed_verification":
				return "Failed Verification";
			default:
				return this.formatWorkflowLabel(value.replace(/_/g, " "));
		}
	}

	private buildTaskExecutionContractText(taskId: string): string | undefined {
		const workflow = this.session.workflow;
		const task = workflow.taskGraph.tasks[taskId];
		if (!task) {
			return undefined;
		}
		const contract = buildTaskSubagentContract(task, {
			phase: workflow.currentPhase,
			relevantFiles: workflow.workspace.changedFiles.slice(0, 8),
			extraInputs:
				workflow.workspace.lastCommandResults.length > 0
					? [
							`last-command:${workflow.workspace.lastCommandResults[workflow.workspace.lastCommandResults.length - 1]!.command}`,
						]
					: [],
		});
		return [
			`Goal: ${contract.goal}`,
			contract.inputs.length > 0 ? `Inputs: ${contract.inputs.join(" | ")}` : "Inputs: none",
			contract.constraints.length > 0 ? `Constraints: ${contract.constraints.join(" | ")}` : "Constraints: none",
		].join("\n");
	}

	private getWorkflowDisplayState(): {
		goal: string;
		phase: string;
		status: string;
		activeTaskId?: string;
		activeTaskGoal?: string;
		activeTaskStatus?: string;
		activeTaskVerification?: string;
		activeTaskCompletion?: string;
		activeTaskCompletionReady: boolean;
		activeTaskCriteriaCount: number;
		activeTaskNotesCount: number;
		schedulableTasks: number;
		artifacts: number;
		verification: number;
		transitions: number;
	} {
		const workflow = this.session.workflow;
		const activeTaskId = workflow.taskGraph.activeTaskId;
		const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
		const activeTaskCompletion = getActiveTaskCompletionState(workflow);

		return {
			goal: workflow.goal,
			phase: this.formatWorkflowLabel(workflow.currentPhase),
			status: this.formatWorkflowLabel(workflow.status),
			activeTaskId,
			activeTaskGoal: activeTask?.goal,
			activeTaskStatus: activeTask ? this.formatWorkflowLabel(activeTask.status) : undefined,
			activeTaskVerification: activeTask
				? this.formatWorkflowLabel(activeTaskCompletion.verificationStatus)
				: undefined,
			activeTaskCompletion: activeTask
				? this.formatTaskCompletionLabel(activeTaskCompletion.completionLabel)
				: undefined,
			activeTaskCompletionReady: activeTaskCompletion.completionReady,
			activeTaskCriteriaCount: activeTask?.acceptanceCriteria.length ?? 0,
			activeTaskNotesCount: activeTask?.notes.length ?? 0,
			schedulableTasks: getSchedulableTasks(workflow.taskGraph).length,
			artifacts: workflow.artifacts.length,
			verification: workflow.verification.length,
			transitions: workflow.transitions.length,
		};
	}

	private buildWorkflowStripLines(maxWidth: number, maxHeight: number): string[] {
		const workflow = this.getWorkflowDisplayState();
		const changedFilesCount = this.session.workflow.workspace.changedFiles.length;
		const width = Math.max(40, maxWidth - 2);
		const lines: string[] = [];
		const idleSegments: string[] = [];
		if (!workflow.activeTaskGoal && maxWidth >= 90) {
			if (workflow.schedulableTasks > 0) {
				idleSegments.push(`${theme.fg("dim", "schedulable:")} ${workflow.schedulableTasks}`);
			}
			if (changedFilesCount > 0) {
				idleSegments.push(`${theme.fg("dim", "files:")} ${changedFilesCount}`);
			}
		}
		const workflowLine = `${theme.bold(theme.fg("accent", "Workflow"))} ${theme.fg("dim", `${workflow.phase} | ${workflow.status}`)}${idleSegments.length > 0 ? ` ${theme.fg("dim", "•")} ${idleSegments.join(theme.fg("dim", " • "))}` : ""}`;
		lines.push(truncateToWidth(workflowLine, width, "…"));

		if (!workflow.activeTaskGoal) {
			return lines;
		}

		const detailSegments: string[] = [];
		if (workflow.activeTaskVerification) {
			detailSegments.push(`${theme.fg("dim", "verify:")} ${workflow.activeTaskVerification}`);
		}
		if (workflow.activeTaskCompletion) {
			detailSegments.push(`${theme.fg("dim", "complete:")} ${workflow.activeTaskCompletion}`);
		}
		if (workflow.activeTaskCompletionReady) {
			detailSegments.push(theme.fg("success", "ready"));
		}
		if (changedFilesCount > 0 && maxWidth >= 78) {
			detailSegments.push(`${theme.fg("dim", "files:")} ${changedFilesCount}`);
		}

		const taskLine = `${theme.fg("accent", "Task:")} ${theme.fg("text", workflow.activeTaskGoal)}`;
		const allowThirdLine = detailSegments.length > 0 && maxWidth >= 100 && maxHeight >= 40;
		if (allowThirdLine) {
			lines.push(truncateToWidth(taskLine, width, "…"));
			lines.push(truncateToWidth(detailSegments.join(theme.fg("dim", " • ")), width, "…"));
			return lines;
		}

		const mergedTaskLine =
			detailSegments.length > 0
				? `${taskLine} ${theme.fg("dim", "•")} ${detailSegments.join(theme.fg("dim", " • "))}`
				: taskLine;
		lines.push(truncateToWidth(mergedTaskLine, width, "…"));
		return lines;
	}

	private updateWorkflowStrip(): void {
		const container = new Container();
		const maxWidth = this.ui.terminal.columns || process.stdout.columns || 80;
		const maxHeight = this.ui.terminal.rows || process.stdout.rows || 24;
		for (const line of this.buildWorkflowStripLines(maxWidth, maxHeight)) {
			container.addChild(new TruncatedText(line, 1, 0));
		}
		this.workflowStripComponent = container;
	}

	private addMessageToChat(message: AgentMessage, options?: { populateHistory?: boolean }): void {
		switch (message.role) {
			case "bashExecution": {
				const component = new BashExecutionComponent(message.command, this.ui, message.excludeFromContext);
				if (message.output) {
					component.appendOutput(message.output);
				}
				component.setComplete(
					message.exitCode,
					message.cancelled,
					message.truncated ? ({ truncated: true } as TruncationResult) : undefined,
					message.fullOutputPath,
				);
				this.chatContainer.addChild(component);
				break;
			}
			case "custom": {
				if (message.display) {
					const renderer = this.session.extensionRunner?.getMessageRenderer(message.customType);
					const component = new CustomMessageComponent(message, renderer, this.getMarkdownThemeWithSettings());
					component.setExpanded(this.toolOutputExpanded);
					this.chatContainer.addChild(component);
				}
				break;
			}
			case "compactionSummary": {
				this.chatContainer.addChild(new Spacer(1));
				const component = new CompactionSummaryMessageComponent(message, this.getMarkdownThemeWithSettings());
				component.setExpanded(this.toolOutputExpanded);
				this.chatContainer.addChild(component);
				break;
			}
			case "branchSummary": {
				this.chatContainer.addChild(new Spacer(1));
				const component = new BranchSummaryMessageComponent(message, this.getMarkdownThemeWithSettings());
				component.setExpanded(this.toolOutputExpanded);
				this.chatContainer.addChild(component);
				break;
			}
			case "user": {
				const textContent = this.getUserMessageText(message);
				if (textContent) {
					const skillBlock = parseSkillBlock(textContent);
					if (skillBlock) {
						// Render skill block (collapsible)
						this.chatContainer.addChild(new Spacer(1));
						const component = new SkillInvocationMessageComponent(
							skillBlock,
							this.getMarkdownThemeWithSettings(),
						);
						component.setExpanded(this.toolOutputExpanded);
						this.chatContainer.addChild(component);
						// Render user message separately if present
						if (skillBlock.userMessage) {
							const userComponent = new UserMessageComponent(
								skillBlock.userMessage,
								this.getMarkdownThemeWithSettings(),
							);
							this.chatContainer.addChild(userComponent);
						}
					} else {
						const userComponent = new UserMessageComponent(textContent, this.getMarkdownThemeWithSettings());
						this.chatContainer.addChild(userComponent);
					}
					if (options?.populateHistory) {
						this.editor.addToHistory?.(textContent);
					}
				}
				break;
			}
			case "assistant": {
				const assistantComponent = new AssistantMessageComponent(
					message,
					this.hideThinkingBlock,
					this.getMarkdownThemeWithSettings(),
				);
				this.chatContainer.addChild(assistantComponent);
				break;
			}
			case "toolResult": {
				// Tool results are rendered inline with tool calls, handled separately
				break;
			}
			default: {
				const _exhaustive: never = message;
			}
		}
	}

	/**
	 * Render session context to chat. Used for initial load and rebuild after compaction.
	 * @param sessionContext Session context to render
	 * @param options.updateFooter Update footer state
	 * @param options.populateHistory Add user messages to editor history
	 */
	private renderSessionContext(
		sessionContext: SessionContext,
		options: { updateFooter?: boolean; populateHistory?: boolean } = {},
	): void {
		this.pendingTools.clear();

		if (options.updateFooter) {
			this.footer.invalidate();
			this.updateEditorBorderColor();
		}

		for (const message of sessionContext.messages) {
			// Assistant messages need special handling for tool calls
			if (message.role === "assistant") {
				this.addMessageToChat(message);
				// Render tool call components
				for (const content of message.content) {
					if (content.type === "toolCall") {
						const component = new ToolExecutionComponent(
							content.name,
							content.arguments,
							{ showImages: this.settingsManager.getShowImages() },
							this.getRegisteredToolDefinition(content.name),
							this.ui,
						);
						component.setExpanded(this.toolOutputExpanded);
						this.chatContainer.addChild(component);

						if (message.stopReason === "aborted" || message.stopReason === "error") {
							let errorMessage: string;
							if (message.stopReason === "aborted") {
								const retryAttempt = this.session.retryAttempt;
								errorMessage =
									retryAttempt > 0
										? `Aborted after ${retryAttempt} retry attempt${retryAttempt > 1 ? "s" : ""}`
										: "Operation aborted";
							} else {
								errorMessage = message.errorMessage || "Error";
							}
							component.updateResult({ content: [{ type: "text", text: errorMessage }], isError: true });
						} else {
							this.pendingTools.set(content.id, component);
						}
					}
				}
			} else if (message.role === "toolResult") {
				// Match tool results to pending tool components
				const component = this.pendingTools.get(message.toolCallId);
				if (component) {
					component.updateResult(message);
					this.pendingTools.delete(message.toolCallId);
				}
			} else {
				// All other messages use standard rendering
				this.addMessageToChat(message, options);
			}
		}

		this.pendingTools.clear();
		this.renderWidgets();
		this.ui.requestRender();
	}

	renderInitialMessages(): void {
		// Get aligned messages and entries from session context
		const context = this.sessionManager.buildSessionContext();
		this.renderSessionContext(context, {
			updateFooter: true,
			populateHistory: true,
		});

		// Show compaction info if session was compacted
		const allEntries = this.sessionManager.getEntries();
		const compactionCount = allEntries.filter((e) => e.type === "compaction").length;
		const statusMessages: string[] = [];
		if (compactionCount > 0) {
			const times = compactionCount === 1 ? "1 time" : `${compactionCount} times`;
			statusMessages.unshift(`Session compacted ${times}`);
		}
		if (statusMessages.length > 0) {
			this.showStatus(statusMessages.join("\n"));
		}
	}

	async getUserInput(): Promise<string> {
		return new Promise((resolve) => {
			this.onInputCallback = (text: string) => {
				this.onInputCallback = undefined;
				resolve(text);
			};
		});
	}

	private rebuildChatFromMessages(): void {
		this.chatContainer.clear();
		const context = this.sessionManager.buildSessionContext();
		this.renderSessionContext(context);
	}

	// =========================================================================
	// Key handlers
	// =========================================================================

	private handleCtrlC(): void {
		const now = Date.now();
		if (now - this.lastSigintTime < 500) {
			void this.shutdown();
		} else {
			this.clearEditor();
			this.lastSigintTime = now;
		}
	}

	private handleCtrlD(): void {
		// Only called when editor is empty (enforced by CustomEditor)
		void this.shutdown();
	}

	/**
	 * Gracefully shutdown the agent.
	 * Emits shutdown event to extensions, then exits.
	 */
	private isShuttingDown = false;

	private async shutdown(): Promise<void> {
		if (this.isShuttingDown) return;
		this.isShuttingDown = true;

		// Emit shutdown event to extensions
		const extensionRunner = this.session.extensionRunner;
		if (extensionRunner?.hasHandlers("session_shutdown")) {
			await extensionRunner.emit({
				type: "session_shutdown",
			});
		}

		// Wait for any pending renders to complete
		// requestRender() uses process.nextTick(), so we wait one tick
		await new Promise((resolve) => process.nextTick(resolve));

		// Drain any in-flight Kitty key release events before stopping.
		// This prevents escape sequences from leaking to the parent shell over slow SSH.
		await this.ui.terminal.drainInput(1000);

		this.stop();
		process.exit(0);
	}

	/**
	 * Check if shutdown was requested and perform shutdown if so.
	 */
	private async checkShutdownRequested(): Promise<void> {
		if (!this.shutdownRequested) return;
		await this.shutdown();
	}

	private handleCtrlZ(): void {
		// Ignore SIGINT while suspended so Ctrl+C in the terminal does not
		// kill the backgrounded process. The handler is removed on resume.
		const ignoreSigint = () => {};
		process.on("SIGINT", ignoreSigint);

		// Set up handler to restore TUI when resumed
		process.once("SIGCONT", () => {
			process.removeListener("SIGINT", ignoreSigint);
			this.ui.start();
			this.ui.requestRender(true);
		});

		// Stop the TUI (restore terminal to normal mode)
		this.ui.stop();

		// Send SIGTSTP to process group (pid=0 means all processes in group)
		process.kill(0, "SIGTSTP");
	}

	private async handleFollowUp(): Promise<void> {
		const text = (this.editor.getExpandedText?.() ?? this.editor.getText()).trim();
		if (!text) return;

		// Queue input during compaction (extension commands execute immediately)
		if (this.session.isCompacting) {
			if (this.isExtensionCommand(text)) {
				this.editor.addToHistory?.(text);
				this.editor.setText("");
				await this.session.prompt(text);
			} else {
				this.queueCompactionMessage(text, "followUp");
			}
			return;
		}

		// Alt+Enter queues a follow-up message (waits until agent finishes)
		// This handles extension commands (execute immediately), prompt template expansion, and queueing
		if (this.session.isStreaming) {
			this.editor.addToHistory?.(text);
			this.editor.setText("");
			await this.promptWithMainRole(text, { streamingBehavior: "followUp" });
			this.updatePendingMessagesDisplay();
			this.ui.requestRender();
		}
		// If not streaming, Alt+Enter acts like regular Enter (trigger onSubmit)
		else if (this.editor.onSubmit) {
			this.editor.onSubmit(text);
		}
	}

	private handleDequeue(): void {
		const restored = this.restoreQueuedMessagesToEditor();
		if (restored === 0) {
			this.showStatus("No queued messages to restore");
		} else {
			this.showStatus(`Restored ${restored} queued message${restored > 1 ? "s" : ""} to editor`);
		}
	}

	private updateEditorBorderColor(): void {
		if (this.isBashMode) {
			this.editor.borderColor = theme.getBashModeBorderColor();
		} else {
			const level = this.session.thinkingLevel || "off";
			// If thinking is active, show thinking color; otherwise show workflow phase color
			if (level !== "off") {
				this.editor.borderColor = theme.getThinkingBorderColor(level);
			} else {
				const phase = this.session.workflow?.currentPhase;
				this.editor.borderColor = phase ? theme.getWorkflowPhaseColor(phase) : theme.getThinkingBorderColor("off");
			}
		}
		this.ui.requestRender();
	}

	private cycleThinkingLevel(): void {
		const newLevel = this.session.cycleThinkingLevel();
		if (newLevel === undefined) {
			this.showStatus("Current model does not support thinking");
		} else {
			this.footer.invalidate();
			this.updateEditorBorderColor();
			this.showStatus(`Thinking level: ${newLevel}`);
		}
	}

	private async cycleModel(direction: "forward" | "backward"): Promise<void> {
		try {
			const result = await this.session.cycleModel(direction);
			if (result === undefined) {
				const msg = this.session.scopedModels.length > 0 ? "Only one model in scope" : "Only one model available";
				this.showStatus(msg);
			} else {
				this.footer.invalidate();
				this.updateEditorBorderColor();
				const thinkingStr =
					result.model.reasoning && result.thinkingLevel !== "off" ? ` (thinking: ${result.thinkingLevel})` : "";
				this.showStatus(`Switched to ${result.model.name || result.model.id}${thinkingStr}`);
			}
		} catch (error) {
			this.showError(error instanceof Error ? error.message : String(error));
		}
	}

	private toggleToolOutputExpansion(): void {
		this.setToolsExpanded(!this.toolOutputExpanded);
	}

	private setToolsExpanded(expanded: boolean): void {
		this.toolOutputExpanded = expanded;
		for (const child of this.chatContainer.children) {
			if (isExpandable(child)) {
				child.setExpanded(expanded);
			}
		}
		this.ui.requestRender();
	}

	private toggleThinkingBlockVisibility(): void {
		this.hideThinkingBlock = !this.hideThinkingBlock;
		this.settingsManager.setHideThinkingBlock(this.hideThinkingBlock);

		// Rebuild chat from session messages
		this.chatContainer.clear();
		this.rebuildChatFromMessages();

		// If streaming, re-add the streaming component with updated visibility and re-render
		if (this.streamingComponent && this.streamingMessage) {
			this.streamingComponent.setHideThinkingBlock(this.hideThinkingBlock);
			this.streamingComponent.updateContent(this.streamingMessage);
			this.chatContainer.addChild(this.streamingComponent);
		}

		this.showStatus(`Thinking blocks: ${this.hideThinkingBlock ? "hidden" : "visible"}`);
	}

	private openExternalEditor(): void {
		// Determine editor (respect $VISUAL, then $EDITOR)
		const editorCmd = process.env.VISUAL || process.env.EDITOR;
		if (!editorCmd) {
			this.showWarning("No editor configured. Set $VISUAL or $EDITOR environment variable.");
			return;
		}

		const currentText = this.editor.getExpandedText?.() ?? this.editor.getText();
		const tmpFile = path.join(os.tmpdir(), `pi-editor-${Date.now()}.pi.md`);

		try {
			// Write current content to temp file
			fs.writeFileSync(tmpFile, currentText, "utf-8");

			// Stop TUI to release terminal
			this.ui.stop();

			// Split by space to support editor arguments (e.g., "code --wait")
			const [editor, ...editorArgs] = editorCmd.split(" ");

			// Spawn editor synchronously with inherited stdio for interactive editing
			const result = spawnSync(editor, [...editorArgs, tmpFile], {
				stdio: "inherit",
			});

			// On successful exit (status 0), replace editor content
			if (result.status === 0) {
				const newContent = fs.readFileSync(tmpFile, "utf-8").replace(/\n$/, "");
				this.editor.setText(newContent);
			}
			// On non-zero exit, keep original text (no action needed)
		} finally {
			// Clean up temp file
			try {
				fs.unlinkSync(tmpFile);
			} catch {
				// Ignore cleanup errors
			}

			// Restart TUI
			this.ui.start();
			// Force full re-render since external editor uses alternate screen
			this.ui.requestRender(true);
		}
	}

	// =========================================================================
	// UI helpers
	// =========================================================================

	clearEditor(): void {
		this.editor.setText("");
		this.ui.requestRender();
	}

	showError(errorMessage: string): void {
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Text(theme.fg("error", `Error: ${errorMessage}`), 1, 0));
		this.ui.requestRender();
	}

	showWarning(warningMessage: string): void {
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Text(theme.fg("warning", `Warning: ${warningMessage}`), 1, 0));
		this.ui.requestRender();
	}

	showNewVersionNotification(newVersion: string): void {
		const action = theme.fg("accent", getUpdateInstruction("@mariozechner/pi-coding-agent"));
		const updateInstruction = theme.fg("muted", `New version ${newVersion} is available. `) + action;
		const changelogUrl = theme.fg(
			"accent",
			"https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md",
		);
		const changelogLine = theme.fg("muted", "Changelog: ") + changelogUrl;

		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new DynamicBorder((text) => theme.fg("warning", text)));
		this.chatContainer.addChild(
			new Text(
				`${theme.bold(theme.fg("warning", "Update Available"))}\n${updateInstruction}\n${changelogLine}`,
				1,
				0,
			),
		);
		this.chatContainer.addChild(new DynamicBorder((text) => theme.fg("warning", text)));
		this.ui.requestRender();
	}

	/**
	 * Get all queued messages (read-only).
	 * Combines session queue and compaction queue.
	 */
	private getAllQueuedMessages(): { steering: string[]; followUp: string[] } {
		return {
			steering: [
				...this.session.getSteeringMessages(),
				...this.compactionQueuedMessages.filter((msg) => msg.mode === "steer").map((msg) => msg.text),
			],
			followUp: [
				...this.session.getFollowUpMessages(),
				...this.compactionQueuedMessages.filter((msg) => msg.mode === "followUp").map((msg) => msg.text),
			],
		};
	}

	/**
	 * Clear all queued messages and return their contents.
	 * Clears both session queue and compaction queue.
	 */
	private clearAllQueues(): { steering: string[]; followUp: string[] } {
		const { steering, followUp } = this.session.clearQueue();
		const compactionSteering = this.compactionQueuedMessages
			.filter((msg) => msg.mode === "steer")
			.map((msg) => msg.text);
		const compactionFollowUp = this.compactionQueuedMessages
			.filter((msg) => msg.mode === "followUp")
			.map((msg) => msg.text);
		this.compactionQueuedMessages = [];
		return {
			steering: [...steering, ...compactionSteering],
			followUp: [...followUp, ...compactionFollowUp],
		};
	}

	private updatePendingMessagesDisplay(): void {
		this.pendingMessagesContainer.clear();
		const { steering: steeringMessages, followUp: followUpMessages } = this.getAllQueuedMessages();
		if (steeringMessages.length > 0 || followUpMessages.length > 0) {
			this.pendingMessagesContainer.addChild(new Spacer(1));
			for (const message of steeringMessages) {
				const text = theme.fg("dim", `Steering: ${message}`);
				this.pendingMessagesContainer.addChild(new TruncatedText(text, 1, 0));
			}
			for (const message of followUpMessages) {
				const text = theme.fg("dim", `Follow-up: ${message}`);
				this.pendingMessagesContainer.addChild(new TruncatedText(text, 1, 0));
			}
			const dequeueHint = this.getAppKeyDisplay("dequeue");
			const hintText = theme.fg("dim", `↳ ${dequeueHint} to edit all queued messages`);
			this.pendingMessagesContainer.addChild(new TruncatedText(hintText, 1, 0));
		}
	}

	private restoreQueuedMessagesToEditor(options?: { abort?: boolean; currentText?: string }): number {
		const { steering, followUp } = this.clearAllQueues();
		const allQueued = [...steering, ...followUp];
		if (allQueued.length === 0) {
			this.updatePendingMessagesDisplay();
			if (options?.abort) {
				this.agent.abort();
			}
			return 0;
		}
		const queuedText = allQueued.join("\n\n");
		const currentText = options?.currentText ?? this.editor.getText();
		const combinedText = [queuedText, currentText].filter((t) => t.trim()).join("\n\n");
		this.editor.setText(combinedText);
		this.updatePendingMessagesDisplay();
		if (options?.abort) {
			this.agent.abort();
		}
		return allQueued.length;
	}

	private queueCompactionMessage(text: string, mode: "steer" | "followUp"): void {
		this.compactionQueuedMessages.push({ text, mode });
		this.editor.addToHistory?.(text);
		this.editor.setText("");
		this.updatePendingMessagesDisplay();
		this.showStatus("Queued message for after compaction");
	}

	private isExtensionCommand(text: string): boolean {
		if (!text.startsWith("/")) return false;

		const extensionRunner = this.session.extensionRunner;
		if (!extensionRunner) return false;

		const spaceIndex = text.indexOf(" ");
		const commandName = spaceIndex === -1 ? text.slice(1) : text.slice(1, spaceIndex);
		return !!extensionRunner.getCommand(commandName);
	}

	private async flushCompactionQueue(options?: { willRetry?: boolean }): Promise<void> {
		if (this.compactionQueuedMessages.length === 0) {
			return;
		}

		const queuedMessages = [...this.compactionQueuedMessages];
		this.compactionQueuedMessages = [];
		this.updatePendingMessagesDisplay();

		const restoreQueue = (error: unknown) => {
			this.session.clearQueue();
			this.compactionQueuedMessages = queuedMessages;
			this.updatePendingMessagesDisplay();
			this.showError(
				`Failed to send queued message${queuedMessages.length > 1 ? "s" : ""}: ${
					error instanceof Error ? error.message : String(error)
				}`,
			);
		};

		try {
			if (options?.willRetry) {
				// When retry is pending, queue messages for the retry turn
				for (const message of queuedMessages) {
					if (this.isExtensionCommand(message.text)) {
						await this.session.prompt(message.text);
					} else if (message.mode === "followUp") {
						await this.session.followUp(message.text);
					} else {
						await this.session.steer(message.text);
					}
				}
				this.updatePendingMessagesDisplay();
				return;
			}

			// Find first non-extension-command message to use as prompt
			const firstPromptIndex = queuedMessages.findIndex((message) => !this.isExtensionCommand(message.text));
			if (firstPromptIndex === -1) {
				// All extension commands - execute them all
				for (const message of queuedMessages) {
					await this.session.prompt(message.text);
				}
				return;
			}

			// Execute any extension commands before the first prompt
			const preCommands = queuedMessages.slice(0, firstPromptIndex);
			const firstPrompt = queuedMessages[firstPromptIndex];
			const rest = queuedMessages.slice(firstPromptIndex + 1);

			for (const message of preCommands) {
				await this.session.prompt(message.text);
			}

			// Send first prompt (starts streaming)
			const promptPromise = this.promptWithMainRole(firstPrompt.text).catch((error) => {
				restoreQueue(error);
			});

			// Queue remaining messages
			for (const message of rest) {
				if (this.isExtensionCommand(message.text)) {
					await this.session.prompt(message.text);
				} else if (message.mode === "followUp") {
					await this.session.followUp(message.text);
				} else {
					await this.session.steer(message.text);
				}
			}
			this.updatePendingMessagesDisplay();
			void promptPromise;
		} catch (error) {
			restoreQueue(error);
		}
	}

	/** Move pending bash components from pending area to chat */
	private flushPendingBashComponents(): void {
		for (const component of this.pendingBashComponents) {
			this.pendingMessagesContainer.removeChild(component);
			this.chatContainer.addChild(component);
		}
		this.pendingBashComponents = [];
	}

	// =========================================================================
	// Selectors
	// =========================================================================

	/**
	 * Shows a selector component in place of the editor.
	 * @param create Factory that receives a `done` callback and returns the component and focus target
	 */
	private showSelector(create: (done: () => void) => { component: Component; focus: Component }): void {
		const done = () => {
			this.editorContainer.clear();
			this.editorContainer.addChild(this.editor);
			this.ui.setFocus(this.editor);
		};
		const { component, focus } = create(done);
		this.editorContainer.clear();
		this.editorContainer.addChild(component);
		this.ui.setFocus(focus);
		this.ui.requestRender();
	}

	private showSettingsSelector(): void {
		this.showSelector((done) => {
			const selector = new SettingsSelectorComponent(
				{
					autoCompact: this.session.autoCompactionEnabled,
					showImages: this.settingsManager.getShowImages(),
					autoResizeImages: this.settingsManager.getImageAutoResize(),
					blockImages: this.settingsManager.getBlockImages(),
					enableSkillCommands: this.settingsManager.getEnableSkillCommands(),
					steeringMode: this.session.steeringMode,
					followUpMode: this.session.followUpMode,
					transport: this.settingsManager.getTransport(),
					thinkingLevel: this.session.thinkingLevel,
					availableThinkingLevels: this.session.getAvailableThinkingLevels(),
					currentTheme: this.settingsManager.getTheme() || "dark",
					availableThemes: getAvailableThemes(),
					hideThinkingBlock: this.hideThinkingBlock,
					collapseChangelog: this.settingsManager.getCollapseChangelog(),
					doubleEscapeAction: this.settingsManager.getDoubleEscapeAction(),
					showHardwareCursor: this.settingsManager.getShowHardwareCursor(),
					editorPaddingX: this.settingsManager.getEditorPaddingX(),
					autocompleteMaxVisible: this.settingsManager.getAutocompleteMaxVisible(),
					quietStartup: this.settingsManager.getQuietStartup(),
					startupDensity: this.settingsManager.getStartupDensity(),
					clearOnShrink: this.settingsManager.getClearOnShrink(),
					runtimeFeatureFlags: this.settingsManager.getRuntimeFeatureFlags(),
					laneConcurrency: {
						default: this.settingsManager.getLanePolicies().default.concurrency,
						delegate: this.settingsManager.getLanePolicies().delegate.concurrency,
						cron: this.settingsManager.getLanePolicies().cron.concurrency,
						compact: this.settingsManager.getLanePolicies().compact.concurrency,
						notification: this.settingsManager.getLanePolicies().notification.concurrency,
					},
				},
				{
					onAutoCompactChange: (enabled) => {
						this.session.setAutoCompactionEnabled(enabled);
						this.footer.setAutoCompactEnabled(enabled);
					},
					onShowImagesChange: (enabled) => {
						this.settingsManager.setShowImages(enabled);
						for (const child of this.chatContainer.children) {
							if (child instanceof ToolExecutionComponent) {
								child.setShowImages(enabled);
							}
						}
					},
					onAutoResizeImagesChange: (enabled) => {
						this.settingsManager.setImageAutoResize(enabled);
					},
					onBlockImagesChange: (blocked) => {
						this.settingsManager.setBlockImages(blocked);
					},
					onEnableSkillCommandsChange: (enabled) => {
						this.settingsManager.setEnableSkillCommands(enabled);
						this.setupAutocomplete(this.fdPath);
					},
					onSteeringModeChange: (mode) => {
						this.session.setSteeringMode(mode);
					},
					onFollowUpModeChange: (mode) => {
						this.session.setFollowUpMode(mode);
					},
					onTransportChange: (transport) => {
						this.settingsManager.setTransport(transport);
						this.session.agent.setTransport(transport);
					},
					onThinkingLevelChange: (level) => {
						this.session.setThinkingLevel(level);
						this.footer.invalidate();
						this.updateEditorBorderColor();
					},
					onThemeChange: (themeName) => {
						const result = setTheme(themeName, true);
						this.settingsManager.setTheme(themeName);
						this.ui.invalidate();
						if (!result.success) {
							this.showError(`Failed to load theme "${themeName}": ${result.error}\nFell back to dark theme.`);
						}
					},
					onThemePreview: (themeName) => {
						const result = setTheme(themeName, true);
						if (result.success) {
							this.ui.invalidate();
							this.ui.requestRender();
						}
					},
					onHideThinkingBlockChange: (hidden) => {
						this.hideThinkingBlock = hidden;
						this.settingsManager.setHideThinkingBlock(hidden);
						for (const child of this.chatContainer.children) {
							if (child instanceof AssistantMessageComponent) {
								child.setHideThinkingBlock(hidden);
							}
						}
						this.chatContainer.clear();
						this.rebuildChatFromMessages();
					},
					onCollapseChangelogChange: (collapsed) => {
						this.settingsManager.setCollapseChangelog(collapsed);
					},
					onStartupDensityChange: (density) => {
						this.settingsManager.setStartupDensity(density);
					},
					onDoubleEscapeActionChange: (action) => {
						this.settingsManager.setDoubleEscapeAction(action);
					},
					onShowHardwareCursorChange: (enabled) => {
						this.settingsManager.setShowHardwareCursor(enabled);
						this.ui.setShowHardwareCursor(enabled);
					},
					onEditorPaddingXChange: (padding) => {
						this.settingsManager.setEditorPaddingX(padding);
						this.defaultEditor.setPaddingX(padding);
						if (this.editor !== this.defaultEditor && this.editor.setPaddingX !== undefined) {
							this.editor.setPaddingX(padding);
						}
					},
					onAutocompleteMaxVisibleChange: (maxVisible) => {
						this.settingsManager.setAutocompleteMaxVisible(maxVisible);
						this.defaultEditor.setAutocompleteMaxVisible(maxVisible);
						if (this.editor !== this.defaultEditor && this.editor.setAutocompleteMaxVisible !== undefined) {
							this.editor.setAutocompleteMaxVisible(maxVisible);
						}
					},
					onClearOnShrinkChange: (enabled) => {
						this.settingsManager.setClearOnShrink(enabled);
						this.ui.setClearOnShrink(enabled);
					},
					onRuntimeFeatureFlagChange: (flag, enabled) => {
						this.settingsManager.setRuntimeFeatureFlag(flag, enabled);
						if (flag === "runtime.heartbeatCronCore") {
							if (enabled) {
								this.runtimeServices?.heartbeat.start();
							} else {
								this.runtimeServices?.heartbeat.stop();
							}
						}
						if (flag === "ui.eventStreamViewer" && !enabled) {
							this.stopEventTail(true);
						}
					},
					onLaneConcurrencyChange: (lane, concurrency) => {
						this.settingsManager.setLaneConcurrency(lane, concurrency);
						this.runtimeServices?.syncLanePoliciesFromSettings();
					},
					onCancel: () => {
						done();
						this.ui.requestRender();
					},
				},
			);
			return { component: selector, focus: selector.getSettingsList() };
		});
	}

	private async handleModelCommand(searchTerm?: string): Promise<void> {
		if (!searchTerm) {
			this.showModelSelector();
			return;
		}

		const model = await this.findExactModelMatch(searchTerm);
		if (model) {
			try {
				await this.session.setModel(model);
				this.footer.invalidate();
				this.updateEditorBorderColor();
				this.showStatus(`Model: ${model.id}`);
				this.checkDaxnutsEasterEgg(model);
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		this.showModelSelector(searchTerm);
	}

	private async findExactModelMatch(searchTerm: string): Promise<Model<any> | undefined> {
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

		const models = await this.getModelCandidates();
		const exactMatches = models.filter((item) => {
			const idMatch = item.id.toLowerCase() === targetModelId;
			const providerMatch = !targetProvider || item.provider.toLowerCase() === targetProvider;
			return idMatch && providerMatch;
		});

		return exactMatches.length === 1 ? exactMatches[0] : undefined;
	}

	private async getModelCandidates(): Promise<Model<any>[]> {
		if (this.session.scopedModels.length > 0) {
			return this.session.scopedModels.map((scoped) => scoped.model);
		}

		this.session.modelRegistry.refresh();
		try {
			return await this.session.modelRegistry.getAvailable();
		} catch {
			return [];
		}
	}

	/** Update the footer's available provider count from current model candidates */
	private async updateAvailableProviderCount(): Promise<void> {
		const models = await this.getModelCandidates();
		const uniqueProviders = new Set(models.map((m) => m.provider));
		this.footerDataProvider.setAvailableProviderCount(uniqueProviders.size);
	}

	private showModelSelector(initialSearchInput?: string): void {
		this.showSelector((done) => {
			const selector = new ModelSelectorComponent(
				this.ui,
				this.session.model,
				this.settingsManager,
				this.session.modelRegistry,
				this.session.scopedModels,
				async (model) => {
					try {
						await this.session.setModel(model);
						this.footer.invalidate();
						this.updateEditorBorderColor();
						done();
						this.showStatus(`Model: ${model.id}`);
						this.checkDaxnutsEasterEgg(model);
					} catch (error) {
						done();
						this.showError(error instanceof Error ? error.message : String(error));
					}
				},
				() => {
					done();
					this.ui.requestRender();
				},
				initialSearchInput,
			);
			return { component: selector, focus: selector };
		});
	}

	private async showModelsSelector(): Promise<void> {
		// Get all available models
		this.session.modelRegistry.refresh();
		const allModels = this.session.modelRegistry.getAvailable();

		if (allModels.length === 0) {
			this.showStatus("No models available");
			return;
		}

		// Check if session has scoped models (from previous session-only changes or CLI --models)
		const sessionScopedModels = this.session.scopedModels;
		const hasSessionScope = sessionScopedModels.length > 0;

		// Build enabled model IDs from session state or settings
		const enabledModelIds = new Set<string>();
		let hasFilter = false;

		if (hasSessionScope) {
			// Use current session's scoped models
			for (const sm of sessionScopedModels) {
				enabledModelIds.add(`${sm.model.provider}/${sm.model.id}`);
			}
			hasFilter = true;
		} else {
			// Fall back to settings
			const patterns = this.settingsManager.getEnabledModels();
			if (patterns !== undefined && patterns.length > 0) {
				hasFilter = true;
				const scopedModels = await resolveModelScope(patterns, this.session.modelRegistry);
				for (const sm of scopedModels) {
					enabledModelIds.add(`${sm.model.provider}/${sm.model.id}`);
				}
			}
		}

		// Track current enabled state (session-only until persisted)
		const currentEnabledIds = new Set(enabledModelIds);
		let currentHasFilter = hasFilter;

		// Helper to update session's scoped models (session-only, no persist)
		const updateSessionModels = async (enabledIds: Set<string>) => {
			if (enabledIds.size > 0 && enabledIds.size < allModels.length) {
				// Use current session thinking level, not settings default
				const currentThinkingLevel = this.session.thinkingLevel;
				const newScopedModels = await resolveModelScope(Array.from(enabledIds), this.session.modelRegistry);
				this.session.setScopedModels(
					newScopedModels.map((sm) => ({
						model: sm.model,
						thinkingLevel: sm.thinkingLevel ?? currentThinkingLevel,
					})),
				);
			} else {
				// All enabled or none enabled = no filter
				this.session.setScopedModels([]);
			}
			await this.updateAvailableProviderCount();
			this.ui.requestRender();
		};

		this.showSelector((done) => {
			const selector = new ScopedModelsSelectorComponent(
				{
					allModels,
					enabledModelIds: currentEnabledIds,
					hasEnabledModelsFilter: currentHasFilter,
				},
				{
					onModelToggle: async (modelId, enabled) => {
						if (enabled) {
							currentEnabledIds.add(modelId);
						} else {
							currentEnabledIds.delete(modelId);
						}
						currentHasFilter = true;
						await updateSessionModels(currentEnabledIds);
					},
					onEnableAll: async (allModelIds) => {
						currentEnabledIds.clear();
						for (const id of allModelIds) {
							currentEnabledIds.add(id);
						}
						currentHasFilter = false;
						await updateSessionModels(currentEnabledIds);
					},
					onClearAll: async () => {
						currentEnabledIds.clear();
						currentHasFilter = true;
						await updateSessionModels(currentEnabledIds);
					},
					onToggleProvider: async (_provider, modelIds, enabled) => {
						for (const id of modelIds) {
							if (enabled) {
								currentEnabledIds.add(id);
							} else {
								currentEnabledIds.delete(id);
							}
						}
						currentHasFilter = true;
						await updateSessionModels(currentEnabledIds);
					},
					onPersist: (enabledIds) => {
						// Persist to settings
						const newPatterns =
							enabledIds.length === allModels.length
								? undefined // All enabled = clear filter
								: enabledIds;
						this.settingsManager.setEnabledModels(newPatterns);
						this.showStatus("Model selection saved to settings");
					},
					onCancel: () => {
						done();
						this.ui.requestRender();
					},
				},
			);
			return { component: selector, focus: selector };
		});
	}

	private showUserMessageSelector(): void {
		const userMessages = this.session.getUserMessagesForForking();

		if (userMessages.length === 0) {
			this.showStatus("No messages to fork from");
			return;
		}

		this.showSelector((done) => {
			const selector = new UserMessageSelectorComponent(
				userMessages.map((m) => ({ id: m.entryId, text: m.text })),
				async (entryId) => {
					const result = await this.session.fork(entryId);
					if (result.cancelled) {
						// Extension cancelled the fork
						done();
						this.ui.requestRender();
						return;
					}

					this.chatContainer.clear();
					this.renderInitialMessages();
					this.editor.setText(result.selectedText);
					done();
					this.showStatus("Branched to new session");
				},
				() => {
					done();
					this.ui.requestRender();
				},
			);
			return { component: selector, focus: selector.getMessageList() };
		});
	}

	private showTreeSelector(initialSelectedId?: string): void {
		const tree = this.sessionManager.getTree();
		const realLeafId = this.sessionManager.getLeafId();

		if (tree.length === 0) {
			this.showStatus("No entries in session");
			return;
		}

		this.showSelector((done) => {
			const selector = new TreeSelectorComponent(
				tree,
				realLeafId,
				this.ui.terminal.rows,
				async (entryId) => {
					// Selecting the current leaf is a no-op (already there)
					if (entryId === realLeafId) {
						done();
						this.showStatus("Already at this point");
						return;
					}

					// Ask about summarization
					done(); // Close selector first

					// Loop until user makes a complete choice or cancels to tree
					let wantsSummary = false;
					let customInstructions: string | undefined;

					while (true) {
						const summaryChoice = await this.showExtensionSelector("Summarize branch?", [
							"No summary",
							"Summarize",
							"Summarize with custom prompt",
						]);

						if (summaryChoice === undefined) {
							// User pressed escape - re-show tree selector with same selection
							this.showTreeSelector(entryId);
							return;
						}

						wantsSummary = summaryChoice !== "No summary";

						if (summaryChoice === "Summarize with custom prompt") {
							customInstructions = await this.showExtensionEditor("Custom summarization instructions");
							if (customInstructions === undefined) {
								// User cancelled - loop back to summary selector
								continue;
							}
						}

						// User made a complete choice
						break;
					}

					// Set up escape handler and loader if summarizing
					let summaryLoader: Loader | undefined;
					const originalOnEscape = this.defaultEditor.onEscape;

					if (wantsSummary) {
						this.defaultEditor.onEscape = () => {
							this.session.abortBranchSummary();
						};
						this.chatContainer.addChild(new Spacer(1));
						summaryLoader = new Loader(
							this.ui,
							(spinner) => theme.fg("accent", spinner),
							(text) => theme.fg("muted", text),
							`Summarizing branch... (${appKey(this.keybindings, "interrupt")} to cancel)`,
						);
						this.statusContainer.addChild(summaryLoader);
						this.ui.requestRender();
					}

					try {
						const result = await this.session.navigateTree(entryId, {
							summarize: wantsSummary,
							customInstructions,
						});

						if (result.aborted) {
							// Summarization aborted - re-show tree selector with same selection
							this.showStatus("Branch summarization cancelled");
							this.showTreeSelector(entryId);
							return;
						}
						if (result.cancelled) {
							this.showStatus("Navigation cancelled");
							return;
						}

						// Update UI
						this.chatContainer.clear();
						this.renderInitialMessages();
						if (result.editorText && !this.editor.getText().trim()) {
							this.editor.setText(result.editorText);
						}
						this.showStatus("Navigated to selected point");
					} catch (error) {
						this.showError(error instanceof Error ? error.message : String(error));
					} finally {
						if (summaryLoader) {
							summaryLoader.stop();
							this.statusContainer.clear();
						}
						this.defaultEditor.onEscape = originalOnEscape;
					}
				},
				() => {
					done();
					this.ui.requestRender();
				},
				(entryId, label) => {
					this.sessionManager.appendLabelChange(entryId, label);
					this.ui.requestRender();
				},
				initialSelectedId,
			);
			return { component: selector, focus: selector };
		});
	}

	private showSessionSelector(): void {
		this.showSelector((done) => {
			const selector = new SessionSelectorComponent(
				(onProgress) =>
					SessionManager.list(this.sessionManager.getCwd(), this.sessionManager.getSessionDir(), onProgress),
				SessionManager.listAll,
				async (sessionPath) => {
					done();
					await this.handleResumeSession(sessionPath);
				},
				() => {
					done();
					this.ui.requestRender();
				},
				() => {
					void this.shutdown();
				},
				() => this.ui.requestRender(),
				{
					renameSession: async (sessionFilePath: string, nextName: string | undefined) => {
						const next = (nextName ?? "").trim();
						if (!next) return;
						const mgr = SessionManager.open(sessionFilePath);
						mgr.appendSessionInfo(next);
					},
					showRenameHint: true,
					keybindings: this.keybindings,
				},

				this.sessionManager.getSessionFile(),
			);
			return { component: selector, focus: selector };
		});
	}

	private async handleResumeSession(sessionPath: string): Promise<void> {
		// Stop loading animation
		if (this.loadingAnimation) {
			this.loadingAnimation.stop();
			this.loadingAnimation = undefined;
		}
		this.statusContainer.clear();

		// Clear UI state
		this.pendingMessagesContainer.clear();
		this.compactionQueuedMessages = [];
		this.streamingComponent = undefined;
		this.streamingMessage = undefined;
		this.pendingTools.clear();

		// Switch session via AgentSession (emits extension session events)
		await this.session.switchSession(sessionPath);

		// Clear and re-render the chat
		this.chatContainer.clear();
		this.renderInitialMessages();
		this.showStatus("Resumed session");
	}

	private async showOAuthSelector(mode: "login" | "logout"): Promise<void> {
		if (mode === "logout") {
			const providers = this.session.modelRegistry.authStorage.list();
			const loggedInProviders = providers.filter(
				(p) => this.session.modelRegistry.authStorage.get(p)?.type === "oauth",
			);
			if (loggedInProviders.length === 0) {
				this.showStatus("No OAuth providers logged in. Use /login first.");
				return;
			}
		}

		this.showSelector((done) => {
			const selector = new OAuthSelectorComponent(
				mode,
				this.session.modelRegistry.authStorage,
				async (providerId: string) => {
					done();

					if (mode === "login") {
						await this.showLoginDialog(providerId);
					} else {
						// Logout flow
						const providerInfo = getOAuthProviders().find((p) => p.id === providerId);
						const providerName = providerInfo?.name || providerId;

						try {
							this.session.modelRegistry.authStorage.logout(providerId);
							this.session.modelRegistry.refresh();
							await this.updateAvailableProviderCount();
							this.showStatus(`Logged out of ${providerName}`);
						} catch (error: unknown) {
							this.showError(`Logout failed: ${error instanceof Error ? error.message : String(error)}`);
						}
					}
				},
				() => {
					done();
					this.ui.requestRender();
				},
			);
			return { component: selector, focus: selector };
		});
	}

	private async showLoginDialog(providerId: string): Promise<void> {
		const providerInfo = getOAuthProviders().find((p) => p.id === providerId);
		const providerName = providerInfo?.name || providerId;

		// Providers that use callback servers (can paste redirect URL)
		const usesCallbackServer = providerInfo?.usesCallbackServer ?? false;

		// Create login dialog component
		const dialog = new LoginDialogComponent(this.ui, providerId, (_success, _message) => {
			// Completion handled below
		});

		// Show dialog in editor container
		this.editorContainer.clear();
		this.editorContainer.addChild(dialog);
		this.ui.setFocus(dialog);
		this.ui.requestRender();

		// Promise for manual code input (racing with callback server)
		let manualCodeResolve: ((code: string) => void) | undefined;
		let manualCodeReject: ((err: Error) => void) | undefined;
		const manualCodePromise = new Promise<string>((resolve, reject) => {
			manualCodeResolve = resolve;
			manualCodeReject = reject;
		});

		// Restore editor helper
		const restoreEditor = () => {
			this.editorContainer.clear();
			this.editorContainer.addChild(this.editor);
			this.ui.setFocus(this.editor);
			this.ui.requestRender();
		};

		try {
			await this.session.modelRegistry.authStorage.login(providerId as OAuthProviderId, {
				onAuth: (info: { url: string; instructions?: string }) => {
					dialog.showAuth(info.url, info.instructions);

					if (usesCallbackServer) {
						// Show input for manual paste, racing with callback
						dialog
							.showManualInput("Paste redirect URL below, or complete login in browser:")
							.then((value) => {
								if (value && manualCodeResolve) {
									manualCodeResolve(value);
									manualCodeResolve = undefined;
								}
							})
							.catch(() => {
								if (manualCodeReject) {
									manualCodeReject(new Error("Login cancelled"));
									manualCodeReject = undefined;
								}
							});
					} else if (providerId === "github-copilot") {
						// GitHub Copilot polls after onAuth
						dialog.showWaiting("Waiting for browser authentication...");
					}
					// For Anthropic: onPrompt is called immediately after
				},

				onPrompt: async (prompt: { message: string; placeholder?: string }) => {
					return dialog.showPrompt(prompt.message, prompt.placeholder);
				},

				onProgress: (message: string) => {
					dialog.showProgress(message);
				},

				onManualCodeInput: () => manualCodePromise,

				signal: dialog.signal,
			});

			// Success
			restoreEditor();
			this.session.modelRegistry.refresh();
			await this.updateAvailableProviderCount();
			this.showStatus(`Logged in to ${providerName}. Credentials saved to ${getAuthPath()}`);
		} catch (error: unknown) {
			restoreEditor();
			const errorMsg = error instanceof Error ? error.message : String(error);
			if (errorMsg !== "Login cancelled") {
				this.showError(`Failed to login to ${providerName}: ${errorMsg}`);
			}
		}
	}

	// =========================================================================
	// Command handlers
	// =========================================================================

	private async handleReloadCommand(): Promise<void> {
		if (this.session.isStreaming) {
			this.showWarning("Wait for the current response to finish before reloading.");
			return;
		}
		if (this.session.isCompacting) {
			this.showWarning("Wait for compaction to finish before reloading.");
			return;
		}

		this.resetExtensionUI();

		const loader = new BorderedLoader(this.ui, theme, "Reloading extensions, skills, prompts, themes...", {
			cancellable: false,
		});
		const previousEditor = this.editor;
		this.editorContainer.clear();
		this.editorContainer.addChild(loader);
		this.ui.setFocus(loader);
		this.ui.requestRender();

		const dismissLoader = (editor: Component) => {
			loader.dispose();
			this.editorContainer.clear();
			this.editorContainer.addChild(editor);
			this.ui.setFocus(editor);
			this.ui.requestRender();
		};

		try {
			await this.session.reload();
			setRegisteredThemes(this.session.resourceLoader.getThemes().themes);
			this.hideThinkingBlock = this.settingsManager.getHideThinkingBlock();
			const themeName = this.settingsManager.getTheme();
			const themeResult = themeName ? setTheme(themeName, true) : { success: true };
			if (!themeResult.success) {
				this.showError(`Failed to load theme "${themeName}": ${themeResult.error}\nFell back to dark theme.`);
			}
			const editorPaddingX = this.settingsManager.getEditorPaddingX();
			const autocompleteMaxVisible = this.settingsManager.getAutocompleteMaxVisible();
			this.defaultEditor.setPaddingX(editorPaddingX);
			this.defaultEditor.setAutocompleteMaxVisible(autocompleteMaxVisible);
			if (this.editor !== this.defaultEditor) {
				this.editor.setPaddingX?.(editorPaddingX);
				this.editor.setAutocompleteMaxVisible?.(autocompleteMaxVisible);
			}
			this.ui.setShowHardwareCursor(this.settingsManager.getShowHardwareCursor());
			this.ui.setClearOnShrink(this.settingsManager.getClearOnShrink());
			this.runtimeServices?.syncLanePoliciesFromSettings();
			this.setupAutocomplete(this.fdPath);
			const runner = this.session.extensionRunner;
			if (runner) {
				this.setupExtensionShortcuts(runner);
			}
			this.rebuildChatFromMessages();
			dismissLoader(this.editor as Component);
			this.showLoadedResources({
				extensionPaths: runner?.getExtensionPaths() ?? [],
				listingMode: "summary",
				showDiagnostics: true,
			});
			const modelsJsonError = this.session.modelRegistry.getError();
			if (modelsJsonError) {
				this.showError(`models.json error: ${modelsJsonError}`);
			}
			this.showStatus("Reloaded extensions, skills, prompts, themes");
		} catch (error) {
			dismissLoader(previousEditor as Component);
			this.showError(`Reload failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	private handleResourcesCommand(text: string): void {
		const argText = text
			.replace(/^\/resources\s*/, "")
			.trim()
			.toLowerCase();
		if (!argText || argText === "summary") {
			this.showLoadedResources({
				extensionPaths: this.session.extensionRunner?.getExtensionPaths() ?? [],
				listingMode: "summary",
				showDiagnostics: true,
			});
			return;
		}
		if (argText === "issues") {
			this.showLoadedResources({
				extensionPaths: this.session.extensionRunner?.getExtensionPaths() ?? [],
				issuesOnly: true,
				showDiagnostics: true,
			});
			return;
		}
		if (argText === "full") {
			this.showLoadedResources({
				extensionPaths: this.session.extensionRunner?.getExtensionPaths() ?? [],
				listingMode: "full",
				showDiagnostics: true,
			});
			return;
		}
		this.showWarning("Usage: /resources [summary|issues|full]");
	}

	private async handleExportCommand(text: string): Promise<void> {
		const parts = text.split(/\s+/);
		const outputPath = parts.length > 1 ? parts[1] : undefined;

		try {
			const filePath = await this.session.exportToHtml(outputPath);
			this.showStatus(`Session exported to: ${filePath}`);
		} catch (error: unknown) {
			this.showError(`Failed to export session: ${error instanceof Error ? error.message : "Unknown error"}`);
		}
	}

	private async handleShareCommand(): Promise<void> {
		// Check if gh is available and logged in
		try {
			const authResult = spawnSync("gh", ["auth", "status"], { encoding: "utf-8" });
			if (authResult.status !== 0) {
				this.showError("GitHub CLI is not logged in. Run 'gh auth login' first.");
				return;
			}
		} catch {
			this.showError("GitHub CLI (gh) is not installed. Install it from https://cli.github.com/");
			return;
		}

		// Export to a temp file
		const tmpFile = path.join(os.tmpdir(), "session.html");
		try {
			await this.session.exportToHtml(tmpFile);
		} catch (error: unknown) {
			this.showError(`Failed to export session: ${error instanceof Error ? error.message : "Unknown error"}`);
			return;
		}

		// Show cancellable loader, replacing the editor
		const loader = new BorderedLoader(this.ui, theme, "Creating gist...");
		this.editorContainer.clear();
		this.editorContainer.addChild(loader);
		this.ui.setFocus(loader);
		this.ui.requestRender();

		const restoreEditor = () => {
			loader.dispose();
			this.editorContainer.clear();
			this.editorContainer.addChild(this.editor);
			this.ui.setFocus(this.editor);
			try {
				fs.unlinkSync(tmpFile);
			} catch {
				// Ignore cleanup errors
			}
		};

		// Create a secret gist asynchronously
		let proc: ReturnType<typeof spawn> | null = null;

		loader.onAbort = () => {
			proc?.kill();
			restoreEditor();
			this.showStatus("Share cancelled");
		};

		try {
			const result = await new Promise<{ stdout: string; stderr: string; code: number | null }>((resolve) => {
				proc = spawn("gh", ["gist", "create", "--public=false", tmpFile]);
				let stdout = "";
				let stderr = "";
				proc.stdout?.on("data", (data) => {
					stdout += data.toString();
				});
				proc.stderr?.on("data", (data) => {
					stderr += data.toString();
				});
				proc.on("close", (code) => resolve({ stdout, stderr, code }));
			});

			if (loader.signal.aborted) return;

			restoreEditor();

			if (result.code !== 0) {
				const errorMsg = result.stderr?.trim() || "Unknown error";
				this.showError(`Failed to create gist: ${errorMsg}`);
				return;
			}

			// Extract gist ID from the URL returned by gh
			// gh returns something like: https://gist.github.com/username/GIST_ID
			const gistUrl = result.stdout?.trim();
			const gistId = gistUrl?.split("/").pop();
			if (!gistId) {
				this.showError("Failed to parse gist ID from gh output");
				return;
			}

			// Create the preview URL
			const previewUrl = getShareViewerUrl(gistId);
			this.showStatus(`Share URL: ${previewUrl}\nGist: ${gistUrl}`);
		} catch (error: unknown) {
			if (!loader.signal.aborted) {
				restoreEditor();
				this.showError(`Failed to create gist: ${error instanceof Error ? error.message : "Unknown error"}`);
			}
		}
	}

	private handleCopyCommand(): void {
		const text = this.session.getLastAssistantText();
		if (!text) {
			this.showError("No agent messages to copy yet.");
			return;
		}

		try {
			copyToClipboard(text);
			this.showStatus("Copied last agent message to clipboard");
		} catch (error) {
			this.showError(error instanceof Error ? error.message : String(error));
		}
	}

	private handleNameCommand(text: string): void {
		const name = text.replace(/^\/name\s*/, "").trim();
		if (!name) {
			const currentName = this.sessionManager.getSessionName();
			if (currentName) {
				this.chatContainer.addChild(new Spacer(1));
				this.chatContainer.addChild(new Text(theme.fg("dim", `Session name: ${currentName}`), 1, 0));
			} else {
				this.showWarning("Usage: /name <name>");
			}
			this.ui.requestRender();
			return;
		}

		this.sessionManager.appendSessionInfo(name);
		this.updateTerminalTitle();
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Text(theme.fg("dim", `Session name set: ${name}`), 1, 0));
		this.ui.requestRender();
	}

	private handleSessionCommand(): void {
		const stats = this.session.getSessionStats();
		const sessionName = this.sessionManager.getSessionName();
		const workflow = this.getWorkflowDisplayState();
		const workflowSnapshot = this.session.workflow;
		const latestVerification =
			workflow.activeTaskId !== undefined
				? getLatestTaskVerification(workflowSnapshot, workflow.activeTaskId)
				: undefined;
		const latestCommand =
			workflowSnapshot.workspace.lastCommandResults.length > 0
				? workflowSnapshot.workspace.lastCommandResults[workflowSnapshot.workspace.lastCommandResults.length - 1]
				: undefined;
		const latestTest =
			workflowSnapshot.workspace.testResults.length > 0
				? workflowSnapshot.workspace.testResults[workflowSnapshot.workspace.testResults.length - 1]
				: undefined;
		const changedFilesPreview = workflowSnapshot.workspace.changedFiles.slice(0, 3);

		let info = `${theme.bold("Session Info")}\n\n`;
		if (sessionName) {
			info += `${theme.fg("dim", "Name:")} ${sessionName}\n`;
		}
		info += `${theme.fg("dim", "File:")} ${stats.sessionFile ?? "In-memory"}\n`;
		info += `${theme.fg("dim", "ID:")} ${stats.sessionId}\n\n`;
		info += `${theme.bold("Messages")}\n`;
		info += `${theme.fg("dim", "User:")} ${stats.userMessages}\n`;
		info += `${theme.fg("dim", "Assistant:")} ${stats.assistantMessages}\n`;
		info += `${theme.fg("dim", "Tool Calls:")} ${stats.toolCalls}\n`;
		info += `${theme.fg("dim", "Tool Results:")} ${stats.toolResults}\n`;
		info += `${theme.fg("dim", "Total:")} ${stats.totalMessages}\n\n`;
		info += `${theme.bold("Tokens")}\n`;
		info += `${theme.fg("dim", "Input:")} ${stats.tokens.input.toLocaleString()}\n`;
		info += `${theme.fg("dim", "Output:")} ${stats.tokens.output.toLocaleString()}\n`;
		if (stats.tokens.cacheRead > 0) {
			info += `${theme.fg("dim", "Cache Read:")} ${stats.tokens.cacheRead.toLocaleString()}\n`;
		}
		if (stats.tokens.cacheWrite > 0) {
			info += `${theme.fg("dim", "Cache Write:")} ${stats.tokens.cacheWrite.toLocaleString()}\n`;
		}
		info += `${theme.fg("dim", "Total:")} ${stats.tokens.total.toLocaleString()}\n`;

		if (stats.cost > 0) {
			info += `\n${theme.bold("Cost")}\n`;
			info += `${theme.fg("dim", "Total:")} ${stats.cost.toFixed(4)}`;
		}

		info += `\n\n${theme.bold("Workflow")}\n`;
		info += `${theme.fg("dim", "Goal:")} ${workflow.goal}\n`;
		info += `${theme.fg("dim", "Phase:")} ${workflow.phase}\n`;
		info += `${theme.fg("dim", "Status:")} ${workflow.status}\n`;
		if (workflow.activeTaskId) {
			info += `${theme.fg("dim", "Active Task ID:")} ${workflow.activeTaskId}\n`;
		}
		if (workflow.activeTaskGoal) {
			info += `${theme.fg("dim", "Active Task:")} ${workflow.activeTaskGoal}\n`;
		}
		if (workflow.activeTaskStatus) {
			info += `${theme.fg("dim", "Task Status:")} ${workflow.activeTaskStatus}\n`;
		}
		if (workflow.activeTaskVerification) {
			info += `${theme.fg("dim", "Task Verification:")} ${workflow.activeTaskVerification}\n`;
		}
		if (workflow.activeTaskCompletion) {
			info += `${theme.fg("dim", "Completion State:")} ${workflow.activeTaskCompletion}\n`;
		}
		info += `${theme.fg("dim", "Completion Ready:")} ${workflow.activeTaskCompletionReady ? "yes" : "no"}\n`;
		info += `${theme.fg("dim", "Acceptance Criteria:")} ${workflow.activeTaskCriteriaCount}\n`;
		info += `${theme.fg("dim", "Task Notes:")} ${workflow.activeTaskNotesCount}\n`;
		info += `${theme.fg("dim", "Schedulable Tasks:")} ${workflow.schedulableTasks}\n`;
		info += `${theme.fg("dim", "Transitions:")} ${workflow.transitions}\n`;
		info += `${theme.fg("dim", "Verification Records:")} ${workflow.verification}\n`;
		info += `${theme.fg("dim", "Artifacts:")} ${workflow.artifacts}\n`;
		info += `${theme.fg("dim", "Changed Files:")} ${workflowSnapshot.workspace.changedFiles.length}\n`;
		if (changedFilesPreview.length > 0) {
			info += `${theme.fg("dim", "Recent Changes:")} ${changedFilesPreview.join(", ")}\n`;
		}
		if (workflowSnapshot.workspace.git.branch) {
			info += `${theme.fg("dim", "Git Branch:")} ${workflowSnapshot.workspace.git.branch}\n`;
		}
		if (workflowSnapshot.workspace.git.head) {
			info += `${theme.fg("dim", "Git Head:")} ${workflowSnapshot.workspace.git.head}\n`;
		}
		if (workflowSnapshot.workspace.refreshedAt) {
			info += `${theme.fg("dim", "Workspace Refreshed:")} ${workflowSnapshot.workspace.refreshedAt}\n`;
		}
		if (latestCommand) {
			info += `${theme.fg("dim", "Last Command:")} ${latestCommand.command} (exit ${latestCommand.exitCode})\n`;
		}
		if (latestTest) {
			info += `${theme.fg("dim", "Latest Test:")} ${latestTest.command} (${latestTest.passed ? "passed" : "failed"})\n`;
		}
		if (latestVerification) {
			info += `${theme.fg("dim", "Latest Verification:")} ${latestVerification.status} for ${latestVerification.taskId}\n`;
		}
		if (workflow.activeTaskId) {
			const contractText = this.buildTaskExecutionContractText(workflow.activeTaskId);
			if (contractText) {
				info += `${theme.fg("dim", "Execution Contract:")}\n${contractText}\n`;
			}
		}

		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Text(info, 1, 0));
		this.ui.requestRender();
	}

	private renderRuntimePanel(title: string, lines: string[]): void {
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new DynamicBorder());
		this.chatContainer.addChild(new Text(theme.bold(theme.fg("accent", title)), 1, 0));
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Text(lines.join("\n"), 1, 0));
		this.chatContainer.addChild(new DynamicBorder());
		this.ui.requestRender();
	}

	private getRuntimeOrWarn(flag?: Parameters<RuntimeServices["isFeatureEnabled"]>[0]): RuntimeServices | undefined {
		if (!this.runtimeServices) {
			this.showWarning("Runtime services are unavailable in this mode.");
			return undefined;
		}
		if (flag && !this.runtimeServices.isFeatureEnabled(flag)) {
			this.showWarning(`${flag} is disabled. Enable it in settings.runtime.featureFlags.`);
			return undefined;
		}
		return this.runtimeServices;
	}

	private formatRuntimeEventLine(event: {
		id: string;
		type: string;
		severity: string;
		source: string;
		lane?: string;
		createdAt: number;
		payload: Record<string, unknown>;
	}): string {
		const ts = new Date(event.createdAt).toLocaleTimeString();
		const lane = event.lane ? ` lane=${event.lane}` : "";
		if (event.type.startsWith("delegated_task.")) {
			const delegatedTaskId =
				typeof event.payload.delegatedTaskId === "string" ? event.payload.delegatedTaskId.slice(0, 8) : "unknown";
			const assignee = typeof event.payload.assignee === "string" ? event.payload.assignee : "unknown";
			const status = typeof event.payload.status === "string" ? event.payload.status : "unknown";
			const goal = typeof event.payload.goal === "string" ? event.payload.goal : "";
			const summary = typeof event.payload.summary === "string" ? event.payload.summary : "";
			const lastError = typeof event.payload.lastError === "string" ? event.payload.lastError : "";
			const detail = [
				goal ? `goal=${JSON.stringify(goal)}` : "",
				summary ? `summary=${JSON.stringify(summary)}` : "",
				lastError ? `error=${JSON.stringify(lastError)}` : "",
			]
				.filter((part) => part.length > 0)
				.join(" ");
			return `${theme.fg("dim", ts)} ${theme.fg("accent", event.type)} ${theme.fg("muted", `[${event.severity}]`)} ${theme.fg("dim", event.source)}${lane} task=${delegatedTaskId} assignee=${assignee} status=${status}${detail ? ` ${detail}` : ""}`;
		}
		const payloadSummary = Object.keys(event.payload).length > 0 ? ` ${JSON.stringify(event.payload)}` : "";
		const payloadText = payloadSummary.length > 120 ? `${payloadSummary.slice(0, 117)}...` : payloadSummary;
		return `${theme.fg("dim", ts)} ${theme.fg("accent", event.type)} ${theme.fg("muted", `[${event.severity}]`)} ${theme.fg("dim", event.source)}${lane}${payloadText}`;
	}

	private stopEventTail(silent = false): void {
		if (this.eventTailInterval) {
			clearInterval(this.eventTailInterval);
			this.eventTailInterval = undefined;
		}
		if (!silent) {
			this.showStatus("Event tail stopped");
		}
	}

	private startEventTail(limit: number): void {
		const runtime = this.getRuntimeOrWarn("ui.eventStreamViewer");
		if (!runtime) return;

		this.stopEventTail(true);
		const latest = runtime.events.tail(1);
		this.eventTailLastTs = latest[0]?.createdAt ?? 0;
		this.eventTailInterval = setInterval(() => {
			if (!this.runtimeServices) return;
			const rows = this.runtimeServices.events.list({
				fromTs: this.eventTailLastTs + 1,
				limit: Math.max(1, Math.min(100, limit)),
			});
			if (rows.length === 0) {
				return;
			}
			const ordered = [...rows].reverse();
			this.eventTailLastTs = Math.max(this.eventTailLastTs, ...ordered.map((row) => row.createdAt));
			for (const row of ordered) {
				this.chatContainer.addChild(new Text(this.formatRuntimeEventLine(row), 1, 0));
			}
			this.ui.requestRender();
		}, 1000);
		this.showStatus("Event tail started (/events tail off to stop)");
	}

	private async handleEventsCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("ui.eventStreamViewer");
		if (!runtime) return;
		const argText = text.replace(/^\/events\s*/, "").trim();
		if (!argText) {
			const events = runtime.events.list({ limit: 40 });
			const lines =
				events.length > 0
					? events.map((event) => this.formatRuntimeEventLine(event))
					: [theme.fg("dim", "No events yet.")];
			this.renderRuntimePanel("Runtime Events", lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "tail") {
			const mode = rest[0];
			if (mode === "off") {
				this.stopEventTail();
				return;
			}
			const limit = Number.parseInt(rest[1] ?? rest[0] ?? "20", 10);
			this.startEventTail(Number.isFinite(limit) ? limit : 20);
			return;
		}

		if (subcommand === "prune") {
			const days = Number.parseInt(rest[0] ?? "7", 10);
			if (!Number.isFinite(days) || days <= 0) {
				this.showWarning("Usage: /events prune <days>");
				return;
			}
			const removed = runtime.events.pruneByAge(days * 24 * 60 * 60 * 1000);
			this.showStatus(`Pruned ${removed} runtime events older than ${days} day(s).`);
			return;
		}

		const filters: {
			type?: string;
			lane?: LaneName;
			severity?: "debug" | "info" | "warn" | "error";
			limit: number;
		} = { limit: 50 };
		for (const token of argText.split(/\s+/)) {
			if (!token.includes("=")) continue;
			const [key, value] = token.split("=", 2);
			if (key === "type" && value) filters.type = value;
			if (key === "lane" && (LANE_NAMES as readonly string[]).includes(value)) filters.lane = value as LaneName;
			if (key === "severity" && ["debug", "info", "warn", "error"].includes(value)) {
				filters.severity = value as "debug" | "info" | "warn" | "error";
			}
			if (key === "limit" && value) {
				const parsed = Number.parseInt(value, 10);
				if (Number.isFinite(parsed)) {
					filters.limit = Math.max(1, Math.min(parsed, 200));
				}
			}
		}
		const events = runtime.events.list(filters);
		const lines =
			events.length > 0
				? events.map((event) => this.formatRuntimeEventLine(event))
				: [theme.fg("dim", "No matching events.")];
		this.renderRuntimePanel("Runtime Events", lines);
	}

	private formatQueueLine(message: {
		id: string;
		topic: string;
		state: string;
		lane: string;
		attempts: number;
		maxAttempts: number;
		availableAt: number;
		lastError?: string;
	}): string {
		const available = new Date(message.availableAt).toLocaleTimeString();
		const error = message.lastError ? ` error=${message.lastError}` : "";
		return `${theme.fg("accent", message.id.slice(0, 8))} ${message.topic} ${theme.fg("muted", `[${message.state}]`)} lane=${message.lane} attempts=${message.attempts}/${message.maxAttempts} at=${available}${error}`;
	}

	private async handleQueueCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("runtime.deliveryQueue");
		if (!runtime) return;
		const argText = text.replace(/^\/queue\s*/, "").trim();
		if (!argText || argText === "list") {
			const messages = runtime.queue.list(undefined, 50);
			const lines =
				messages.length > 0
					? messages.map((message) => this.formatQueueLine(message))
					: [theme.fg("dim", "Queue is empty.")];
			this.renderRuntimePanel("Delivery Queue", lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "dead-letter") {
			const messages = runtime.queue.list("dead_letter", 50);
			const lines =
				messages.length > 0
					? messages.map((message) => this.formatQueueLine(message))
					: [theme.fg("dim", "No dead-letter messages.")];
			this.renderRuntimePanel("Queue Dead Letter", lines);
			return;
		}
		if (subcommand === "retry") {
			const id = rest[0];
			if (!id) {
				this.showWarning("Usage: /queue retry <messageId>");
				return;
			}
			const retried = runtime.queue.retryDeadLetter(id);
			if (!retried) {
				this.showWarning(`No dead-letter message found for ${id}`);
				return;
			}
			this.showStatus(`Queue message retried: ${id}`);
			return;
		}
		if (subcommand === "process") {
			await runtime.queue.processDue();
			this.showStatus("Queue processing tick complete.");
			return;
		}

		this.showWarning("Usage: /queue [list] | /queue dead-letter | /queue retry <id> | /queue process");
	}

	private async handleLanesCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("runtime.namedLanes");
		if (!runtime) return;
		const argText = text.replace(/^\/lanes\s*/, "").trim();
		if (!argText || argText === "list") {
			const lines = runtime.lanes
				.getSnapshots()
				.map(
					(snapshot) =>
						`${theme.fg("accent", snapshot.lane)} concurrency=${snapshot.concurrency} active=${snapshot.active} queued=${snapshot.queued}`,
				);
			this.renderRuntimePanel("Lane Scheduler", lines.length > 0 ? lines : [theme.fg("dim", "No lane data.")]);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "set") {
			const lane = rest[0];
			const concurrency = Number.parseInt(rest[1] ?? "", 10);
			if (!(LANE_NAMES as readonly string[]).includes(lane ?? "") || !Number.isFinite(concurrency)) {
				this.showWarning("Usage: /lanes set <default|delegate|cron|compact|notification> <concurrency>");
				return;
			}
			this.settingsManager.setLaneConcurrency(lane as LaneName, concurrency);
			runtime.syncLanePoliciesFromSettings();
			this.showStatus(`Lane ${lane} concurrency set to ${Math.max(1, Math.floor(concurrency))}`);
			return;
		}
		if (subcommand === "run") {
			const lane = rest[0];
			const label = rest.slice(1).join(" ").trim() || "manual-task";
			if (!(LANE_NAMES as readonly string[]).includes(lane ?? "")) {
				this.showWarning("Usage: /lanes run <default|delegate|cron|compact|notification> [label]");
				return;
			}
			void runtime.lanes.schedule(lane as LaneName, label, async () => {
				await new Promise<void>((resolve) => setTimeout(resolve, 500));
			});
			this.showStatus(`Scheduled synthetic lane task on ${lane}: ${label}`);
			return;
		}

		this.showWarning("Usage: /lanes [list] | /lanes set <lane> <concurrency> | /lanes run <lane> [label]");
	}

	private async handlePackagesCommand(text: string): Promise<void> {
		if (!this.isRuntimeFeatureEnabled("ui.marketplace")) {
			this.showWarning("ui.marketplace is disabled. Enable it in settings.runtime.featureFlags.");
			return;
		}
		if (this.session.isStreaming || this.session.isCompacting) {
			this.showWarning("Wait until the current task is idle before managing packages.");
			return;
		}
		const argText = text.replace(/^\/packages\s*/, "").trim();
		const packageManager = new DefaultPackageManager({
			cwd: process.cwd(),
			agentDir: getAgentDir(),
			settingsManager: this.settingsManager,
		});
		packageManager.setProgressCallback((event) => {
			if (event.type === "start") {
				this.showStatus(event.message || "Working...");
			}
		});

		if (!argText) {
			const globalPackages = this.settingsManager.getGlobalSettings().packages ?? [];
			const projectPackages = this.settingsManager.getProjectSettings().packages ?? [];
			this.renderRuntimePanel("Packages", [
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
			const globalPackages = this.settingsManager.getGlobalSettings().packages ?? [];
			const projectPackages = this.settingsManager.getProjectSettings().packages ?? [];
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
			this.renderRuntimePanel("Packages", lines);
			return;
		}

		if (subcommand === "manage") {
			await this.showPackageManageSelector(packageManager);
			return;
		}

		if (subcommand === "install" || subcommand === "remove" || subcommand === "update") {
			const local = rest.includes("--local") || rest.includes("-l");
			const source = rest.find((token) => !token.startsWith("-"));
			try {
				if (subcommand === "install") {
					if (!source) {
						this.showWarning("Usage: /packages install <source> [--local]");
						return;
					}
					await packageManager.install(source, { local });
					packageManager.addSourceToSettings(source, { local });
					this.showStatus(`Installed ${source}`);
					await this.handleReloadCommand();
					return;
				}
				if (subcommand === "remove") {
					if (!source) {
						this.showWarning("Usage: /packages remove <source> [--local]");
						return;
					}
					await packageManager.remove(source, { local });
					packageManager.removeSourceFromSettings(source, { local });
					this.showStatus(`Removed ${source}`);
					await this.handleReloadCommand();
					return;
				}
				await packageManager.update(source);
				this.showStatus(source ? `Updated ${source}` : "Updated configured packages");
				await this.handleReloadCommand();
				return;
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
				return;
			}
		}

		this.showWarning(
			"Usage: /packages [list] | /packages install <source> [--local] | /packages remove <source> [--local] | /packages update [source] | /packages manage",
		);
	}

	private async showPackageManageSelector(packageManager: DefaultPackageManager): Promise<void> {
		const resolvedPaths = await packageManager.resolve();
		await new Promise<void>((resolve) => {
			this.showSelector((done) => {
				const selector = new ConfigSelectorComponent(
					resolvedPaths,
					this.settingsManager,
					process.cwd(),
					getAgentDir(),
					() => {
						done();
						resolve();
					},
					() => {
						done();
						void this.shutdown();
						resolve();
					},
					() => this.ui.requestRender(),
				);
				return { component: selector, focus: selector.getResourceList() };
			});
		});
		this.showStatus("Package resource enablement updated.");
		await this.handleReloadCommand();
	}

	private formatMailboxLine(message: {
		messageId: string;
		threadId: string;
		from: string;
		to: string;
		intent: string;
		state: string;
		priority: number;
		updatedAt: number;
		lastError?: string;
	}): string {
		const ts = new Date(message.updatedAt).toLocaleTimeString();
		const error = message.lastError ? ` error=${message.lastError}` : "";
		return `${theme.fg("accent", message.messageId.slice(0, 8))} thread=${message.threadId.slice(0, 8)} ${message.from} -> ${message.to} intent=${message.intent} ${theme.fg("muted", `[${message.state}]`)} p=${message.priority} ${ts}${error}`;
	}

	private async handleMailboxCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("runtime.mailboxProtocolV2");
		if (!runtime) return;
		const argText = text.replace(/^\/mailbox\s*/, "").trim();
		const actor = this.session.sessionId;

		if (!argText || argText === "inbox") {
			const inbox = runtime.mailbox.listInbox(actor, 50);
			const lines =
				inbox.length > 0 ? inbox.map((msg) => this.formatMailboxLine(msg)) : [theme.fg("dim", "Inbox empty.")];
			this.renderRuntimePanel(`Mailbox Inbox (${actor.slice(0, 8)})`, lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "outbox") {
			const outbox = runtime.mailbox.listOutbox(actor, 50);
			const lines =
				outbox.length > 0 ? outbox.map((msg) => this.formatMailboxLine(msg)) : [theme.fg("dim", "Outbox empty.")];
			this.renderRuntimePanel(`Mailbox Outbox (${actor.slice(0, 8)})`, lines);
			return;
		}
		if (subcommand === "thread") {
			const threadId = rest[0];
			if (!threadId) {
				this.showWarning("Usage: /mailbox thread <threadId>");
				return;
			}
			const messages = runtime.mailbox.listThread(threadId, 100);
			const lines =
				messages.length > 0
					? messages.map((msg) => this.formatMailboxLine(msg))
					: [theme.fg("dim", "Thread empty.")];
			this.renderRuntimePanel(`Mailbox Thread ${threadId}`, lines);
			return;
		}
		if (subcommand === "send") {
			const to = rest[0];
			const intent = rest[1];
			const payloadText = rest.slice(2).join(" ").trim();
			if (!to || !intent) {
				this.showWarning("Usage: /mailbox send <to> <intent> [payload]");
				return;
			}
			const envelope = runtime.mailbox.send({
				from: actor,
				to,
				intent,
				payload: payloadText ? { text: payloadText } : {},
				completionCriteria: "Acknowledge and include outcome summary.",
				retryPolicy: "exponential_backoff:max_5",
				delegatedTask:
					intent === "delegate"
						? {
								goal: payloadText || "Delegated task",
								summary: "Queued from interactive mailbox send.",
							}
						: undefined,
			});
			const delegatedTaskId =
				typeof envelope.payload.delegatedTaskId === "string"
					? `\ndelegated=${envelope.payload.delegatedTaskId}`
					: "";
			this.showStatus(
				`Mailbox message queued: ${envelope.messageId}\nthread=${envelope.threadId}\nfrom=${envelope.from} to=${envelope.to}${delegatedTaskId}`,
			);
			return;
		}
		if (subcommand === "ack") {
			const messageId = rest[0];
			if (!messageId) {
				this.showWarning("Usage: /mailbox ack <messageId>");
				return;
			}
			const acked = runtime.mailbox.ack(messageId, actor);
			if (!acked) {
				this.showWarning(`Unable to ack ${messageId}.`);
				return;
			}
			this.showStatus(`Mailbox acked: ${messageId}`);
			return;
		}
		if (subcommand === "retry") {
			const messageId = rest[0];
			if (!messageId) {
				this.showWarning("Usage: /mailbox retry <messageId>");
				return;
			}
			const retried = runtime.mailbox.retry(messageId);
			if (!retried) {
				this.showWarning(`Unable to retry ${messageId}.`);
				return;
			}
			this.showStatus(`Mailbox retried: ${messageId}`);
			return;
		}

		this.showWarning(
			"Usage: /mailbox [inbox] | /mailbox outbox | /mailbox thread <threadId> | /mailbox send <to> <intent> [payload] | /mailbox ack <messageId> | /mailbox retry <messageId>",
		);
	}

	private formatDelegatedTaskLine(task: DelegatedTaskRecord): string {
		const ts = new Date(task.updatedAt).toLocaleTimeString();
		const summary = task.summary ? ` summary=${JSON.stringify(task.summary)}` : "";
		const error = task.lastError ? ` error=${JSON.stringify(task.lastError)}` : "";
		return `${theme.fg("accent", task.delegatedTaskId.slice(0, 8))} ${task.owner} -> ${task.assignee} ${theme.fg("muted", `[${task.status}]`)} ${theme.fg("dim", ts)} goal=${JSON.stringify(task.goal)}${summary}${error}`;
	}

	private async handleDelegatedCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("runtime.mailboxProtocolV2");
		if (!runtime) return;
		const argText = text.replace(/^\/delegated\s*/, "").trim();
		const actor = this.session.sessionId;

		if (!argText || argText === "list") {
			const tasks = runtime.delegatedTasks.list({ parentSessionId: actor, limit: 50 });
			const lines =
				tasks.length > 0
					? tasks.map((task) => this.formatDelegatedTaskLine(task))
					: [theme.fg("dim", "No delegated tasks.")];
			this.renderRuntimePanel(`Delegated Tasks (${actor.slice(0, 8)})`, lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "thread") {
			const threadId = rest[0];
			if (!threadId) {
				this.showWarning("Usage: /delegated thread <threadId>");
				return;
			}
			const tasks = runtime.delegatedTasks.list({ threadId, limit: 50 });
			const lines =
				tasks.length > 0
					? tasks.map((task) => this.formatDelegatedTaskLine(task))
					: [theme.fg("dim", "No delegated tasks for thread.")];
			this.renderRuntimePanel(`Delegated Thread ${threadId}`, lines);
			return;
		}
		if (subcommand === "start" || subcommand === "block" || subcommand === "complete" || subcommand === "fail") {
			const delegatedTaskId = rest[0];
			const detail = rest.slice(1).join(" ").trim();
			if (!delegatedTaskId) {
				this.showWarning(
					`Usage: /delegated ${subcommand} <delegatedTaskId> ${subcommand === "fail" || subcommand === "block" ? "<details>" : "[details]"}`,
				);
				return;
			}
			let updated: DelegatedTaskRecord | undefined;
			if (subcommand === "start") {
				updated = runtime.delegatedTasks.markRunning(
					delegatedTaskId,
					detail || "Started from interactive command.",
				);
			}
			if (subcommand === "block") {
				if (!detail) {
					this.showWarning("Usage: /delegated block <delegatedTaskId> <details>");
					return;
				}
				updated = runtime.delegatedTasks.markBlocked(delegatedTaskId, detail);
			}
			if (subcommand === "complete") {
				updated = runtime.delegatedTasks.markCompleted(
					delegatedTaskId,
					detail || "Completed from interactive command.",
				);
			}
			if (subcommand === "fail") {
				if (!detail) {
					this.showWarning("Usage: /delegated fail <delegatedTaskId> <details>");
					return;
				}
				updated = runtime.delegatedTasks.markFailed(delegatedTaskId, detail);
			}
			if (!updated) {
				this.showWarning(`Unable to update delegated task ${delegatedTaskId}.`);
				return;
			}
			this.showStatus(`Delegated task ${updated.delegatedTaskId}: ${updated.status}`);
			return;
		}

		const filters: { status?: DelegatedTaskStatus; limit: number } = { limit: 50 };
		for (const token of argText.split(/\s+/)) {
			if (!token.includes("=")) continue;
			const [key, value] = token.split("=", 2);
			if (
				key === "status" &&
				(value === "queued" ||
					value === "running" ||
					value === "blocked" ||
					value === "completed" ||
					value === "failed")
			) {
				filters.status = value;
			}
			if (key === "limit" && value) {
				const parsed = Number.parseInt(value, 10);
				if (Number.isFinite(parsed)) {
					filters.limit = Math.max(1, Math.min(parsed, 200));
				}
			}
		}
		const tasks = runtime.delegatedTasks.list({
			parentSessionId: actor,
			status: filters.status,
			limit: filters.limit,
		});
		const lines =
			tasks.length > 0
				? tasks.map((task) => this.formatDelegatedTaskLine(task))
				: [theme.fg("dim", "No matching delegated tasks.")];
		this.renderRuntimePanel(`Delegated Tasks (${actor.slice(0, 8)})`, lines);
	}

	private async handleHeartbeatCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("runtime.heartbeatCronCore");
		if (!runtime) return;
		const argText = text.replace(/^\/heartbeat\s*/, "").trim();
		if (!argText || argText === "status") {
			const status = runtime.heartbeat.getStatus();
			const jobs = runtime.heartbeat.listJobs(10);
			const lines = [
				`running=${status.running ? "yes" : "no"} intervalMs=${status.intervalMs} ticks=${status.tickCount} jobsEnabled=${status.jobsEnabled}`,
				`lastTick=${status.lastTickAt ? new Date(status.lastTickAt).toISOString() : "never"}`,
				"",
				theme.bold("Cron jobs:"),
			];
			if (jobs.length === 0) {
				lines.push(theme.fg("dim", "  none"));
			} else {
				for (const job of jobs) {
					lines.push(
						`  ${theme.fg("accent", job.id.slice(0, 8))} ${job.name} every ${job.intervalSeconds}s ${job.enabled ? "enabled" : "paused"} next=${new Date(job.nextRunAt).toLocaleTimeString()}`,
					);
				}
			}
			this.renderRuntimePanel("Heartbeat + Cron", lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "tick") {
			await runtime.heartbeat.tick();
			this.showStatus("Heartbeat tick executed.");
			return;
		}
		if (subcommand === "add") {
			const name = rest[0];
			const intervalSeconds = Number.parseInt(rest[1] ?? "", 10);
			const intent = rest[2];
			const payloadText = rest.slice(3).join(" ").trim();
			if (!name || !Number.isFinite(intervalSeconds) || !intent) {
				this.showWarning("Usage: /heartbeat add <name> <intervalSeconds> <intent> [payload]");
				return;
			}
			const job = runtime.heartbeat.addJob({
				name,
				intervalSeconds,
				intent,
				payload: payloadText ? { text: payloadText } : {},
			});
			this.showStatus(`Cron job added: ${job.id} (${job.name})`);
			return;
		}
		if (subcommand === "pause" || subcommand === "resume") {
			const jobId = rest[0];
			if (!jobId) {
				this.showWarning(`Usage: /heartbeat ${subcommand} <jobId>`);
				return;
			}
			const ok = runtime.heartbeat.setJobEnabled(jobId, subcommand === "resume");
			if (!ok) {
				this.showWarning(`Unknown cron job: ${jobId}`);
				return;
			}
			this.showStatus(`Cron job ${subcommand}d: ${jobId}`);
			return;
		}
		if (subcommand === "remove") {
			const jobId = rest[0];
			if (!jobId) {
				this.showWarning("Usage: /heartbeat remove <jobId>");
				return;
			}
			const ok = runtime.heartbeat.removeJob(jobId);
			if (!ok) {
				this.showWarning(`Unknown cron job: ${jobId}`);
				return;
			}
			this.showStatus(`Cron job removed: ${jobId}`);
			return;
		}
		if (subcommand === "list") {
			const jobs = runtime.heartbeat.listJobs(100);
			const lines =
				jobs.length > 0
					? jobs.map(
							(job) =>
								`${theme.fg("accent", job.id.slice(0, 8))} ${job.name} every ${job.intervalSeconds}s ${job.enabled ? "enabled" : "paused"} next=${new Date(job.nextRunAt).toISOString()}`,
						)
					: [theme.fg("dim", "No cron jobs.")];
			this.renderRuntimePanel("Heartbeat Jobs", lines);
			return;
		}

		this.showWarning(
			"Usage: /heartbeat [status] | /heartbeat tick | /heartbeat list | /heartbeat add <name> <intervalSeconds> <intent> [payload] | /heartbeat pause <jobId> | /heartbeat resume <jobId> | /heartbeat remove <jobId>",
		);
	}

	private async handleModelsCommand(text: string): Promise<void> {
		if (text === "/models" || text.trim() === "/models") {
			this.showStatus("Usage: /models roles [show|set|clear]");
			return;
		}
		if (text.startsWith("/models roles")) {
			await this.handleModelRolesCommand(text);
			return;
		}
		this.showWarning("Unknown /models command. Use /models roles.");
	}

	private async handleModelRolesCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn("model.roleProfiles");
		if (!runtime) return;
		const argText = text.replace(/^\/models\s+roles\s*/, "").trim();
		if (!argText || argText === "show") {
			const profile = this.settingsManager.getRoleModelProfile();
			const lines = MODEL_ROLE_NAMES.map((role) => {
				const value = profile[role];
				const resolved = runtime.modelRoles.resolveRoleModel(role);
				const resolvedText = resolved ? `${resolved.provider}/${resolved.id}` : "unresolved";
				return `${theme.fg("accent", role)}: ${value ?? theme.fg("dim", "not set")} ${theme.fg("muted", `(${resolvedText})`)}`;
			});
			lines.push("");
			lines.push("Usage: /models roles set <main|task|compact|quick> <provider>/<model>");
			lines.push("       /models roles clear <main|task|compact|quick>");
			this.renderRuntimePanel("Model Roles", lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "set") {
			const role = rest[0] as ModelRoleName | undefined;
			const modelRef = rest.slice(1).join(" ").trim();
			if (!role || !(MODEL_ROLE_NAMES as readonly string[]).includes(role) || !modelRef) {
				this.showWarning("Usage: /models roles set <main|task|compact|quick> <provider>/<model>");
				return;
			}
			if (!modelRef.includes("/")) {
				this.showWarning("Model reference must be provider/model.");
				return;
			}
			this.settingsManager.setRoleModel(role, modelRef);
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
					await this.ensureRoleModel("main");
				} catch {
					// Preserve setting even when model cannot be switched immediately.
				}
			}
			this.showStatus(`Model role ${role} set to ${modelRef}`);
			return;
		}
		if (subcommand === "clear") {
			const role = rest[0] as ModelRoleName | undefined;
			if (!role || !(MODEL_ROLE_NAMES as readonly string[]).includes(role)) {
				this.showWarning("Usage: /models roles clear <main|task|compact|quick>");
				return;
			}
			this.settingsManager.setRoleModel(role, undefined);
			runtime.events.record({
				type: "model.roles.updated",
				source: "interactive:/models roles clear",
				payload: { role, modelRef: null },
			});
			this.showStatus(`Model role ${role} cleared`);
			return;
		}

		this.showWarning(
			"Usage: /models roles [show] | /models roles set <role> <provider/model> | /models roles clear <role>",
		);
	}

	private async handleOpsCommand(text: string): Promise<void> {
		const runtime = this.getRuntimeOrWarn();
		if (!runtime) return;

		const argText = text.replace(/^\/ops\s*/, "").trim();
		if (!argText) {
			const flags = this.settingsManager.getRuntimeFeatureFlags();
			const queueQueued = runtime.queue.list("queued", 200).length;
			const queueLeased = runtime.queue.list("leased", 200).length;
			const queueDead = runtime.queue.list("dead_letter", 200).length;
			const actor = this.session.sessionId;
			const heartbeat = runtime.heartbeat.getStatus();
			const laneSnapshots = runtime.lanes
				.getSnapshots()
				.map((snapshot) => `${snapshot.lane}:${snapshot.active}/${snapshot.queued} c=${snapshot.concurrency}`)
				.join(" | ");

			const lines = [
				theme.bold("Runtime"),
				`queue queued=${queueQueued} leased=${queueLeased} dead=${queueDead}`,
				`lanes ${laneSnapshots || "none"}`,
				`mailbox inbox=${runtime.mailbox.listInbox(actor, 200).length} outbox=${runtime.mailbox.listOutbox(actor, 200).length}`,
				`delegated tasks=${runtime.delegatedTasks.list({ parentSessionId: actor, limit: 200 }).length}`,
				`heartbeat running=${heartbeat.running ? "yes" : "no"} jobs=${heartbeat.jobsEnabled} ticks=${heartbeat.tickCount}`,
				"",
				theme.bold("Flags"),
				...RUNTIME_FEATURE_FLAG_NAMES.map((flag) => `${flag}=${flags[flag] ? "on" : "off"}`),
				"",
				"Use /ops <events|queue|lanes|packages|mailbox|delegated|heartbeat|models roles|flags>",
			];
			this.renderRuntimePanel("Ops", lines);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "events") {
			await this.handleEventsCommand(`/events ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "queue") {
			await this.handleQueueCommand(`/queue ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "lanes") {
			await this.handleLanesCommand(`/lanes ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "packages") {
			await this.handlePackagesCommand(`/packages ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "mailbox") {
			await this.handleMailboxCommand(`/mailbox ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "delegated") {
			await this.handleDelegatedCommand(`/delegated ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "heartbeat") {
			await this.handleHeartbeatCommand(`/heartbeat ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "models") {
			await this.handleModelsCommand(`/models ${rest.join(" ").trim()}`.trim());
			return;
		}
		if (subcommand === "flags") {
			const action = rest[0];
			const flagName = rest[1] as RuntimeFeatureFlagName | undefined;
			if (!action || action === "list") {
				const flags = this.settingsManager.getRuntimeFeatureFlags();
				const lines = RUNTIME_FEATURE_FLAG_NAMES.map((flag) => `${flag}=${flags[flag] ? "on" : "off"}`);
				lines.push("");
				lines.push("Usage: /ops flags [list] | /ops flags enable <flag> | /ops flags disable <flag>");
				this.renderRuntimePanel("Runtime Flags", lines);
				return;
			}
			if ((action === "enable" || action === "disable") && flagName) {
				if (!(RUNTIME_FEATURE_FLAG_NAMES as readonly string[]).includes(flagName)) {
					this.showWarning(`Unknown flag "${flagName}"`);
					return;
				}
				const enabled = action === "enable";
				this.settingsManager.setRuntimeFeatureFlag(flagName, enabled);
				if (flagName === "runtime.heartbeatCronCore") {
					if (enabled) runtime.heartbeat.start();
					else runtime.heartbeat.stop();
				}
				if (flagName === "ui.eventStreamViewer" && !enabled) {
					this.stopEventTail(true);
				}
				this.showStatus(`${flagName} ${enabled ? "enabled" : "disabled"}`);
				return;
			}
			this.showWarning("Usage: /ops flags [list] | /ops flags enable <flag> | /ops flags disable <flag>");
			return;
		}

		this.showWarning("Usage: /ops [events|queue|lanes|packages|mailbox|delegated|heartbeat|models roles|flags]");
	}

	private handleWorkflowPlanCommand(text: string): void {
		const argText = text.replace(/^\/plan\s*/, "").trim();
		const workflow = this.session.workflow;
		const activeTaskId = workflow.taskGraph.activeTaskId;
		const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
		const activeTaskCompletion = getActiveTaskCompletionState(workflow);

		if (!argText || argText === "show") {
			const taskSummary = activeTask ? `${activeTask.id}: ${activeTask.goal} [${activeTask.status}]` : "none";
			this.showStatus(
				`Plan goal: ${workflow.goal}\nPhase: ${this.formatWorkflowLabel(workflow.currentPhase)}\nActive task: ${taskSummary}\nCompletion ready: ${activeTaskCompletion.completionReady ? "yes" : "no"}\nTasks: ${workflow.taskGraph.taskOrder.length}`,
			);
			return;
		}

		if (argText === "start") {
			if (workflow.currentPhase === "plan") {
				this.showStatus("Workflow is already in Plan");
				return;
			}
			try {
				this.session.transitionWorkflow("plan", "Manual planning start from /plan");
				this.renderWidgets();
				this.showStatus("Workflow phase: Plan");
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		if (argText.startsWith("goal ")) {
			const nextGoal = argText.slice(5).trim();
			if (!nextGoal) {
				this.showWarning("Usage: /plan goal <goal>");
				return;
			}

			const activeTaskGoalShouldTrackGoal =
				activeTask && workflow.taskGraph.taskOrder.length === 1 && activeTask.goal.trim() === workflow.goal.trim();

			const nextSnapshot = {
				...workflow,
				goal: nextGoal,
				taskGraph:
					activeTaskGoalShouldTrackGoal && activeTaskId
						? {
								...workflow.taskGraph,
								tasks: {
									...workflow.taskGraph.tasks,
									[activeTaskId]: {
										...activeTask,
										goal: nextGoal,
									},
								},
							}
						: workflow.taskGraph,
			};

			this.session.replaceWorkflowSnapshot(nextSnapshot);
			this.renderWidgets();
			this.showStatus(`Plan goal updated: ${nextGoal}`);
			return;
		}

		if (argText === "split") {
			const nextGraph = createTaskGraphFromGoal(workflow.goal, {
				existingGraph: workflow.taskGraph,
			});
			this.session.replaceWorkflowTaskGraph(nextGraph);
			this.session.recordWorkflowArtifact({
				id: `plan-split-${Date.now()}`,
				type: "plan",
				label: "Workflow task graph generated from goal",
				producer: "interactive:/plan split",
				metadata: {
					goal: workflow.goal,
					taskCount: nextGraph.taskOrder.length,
				},
			});
			this.renderWidgets();
			const activeTaskAfterSplit = nextGraph.activeTaskId ? nextGraph.tasks[nextGraph.activeTaskId] : undefined;
			this.showStatus(
				`Plan graph updated from goal\nTasks: ${nextGraph.taskOrder.length}\nActive task: ${activeTaskAfterSplit ? `${activeTaskAfterSplit.id} (${activeTaskAfterSplit.status})` : "none"}`,
			);
			return;
		}

		this.showWarning("Usage: /plan [show] | /plan start | /plan goal <goal> | /plan split");
	}

	private handleWorkflowPhaseCommand(text: string): void {
		const argText = text.replace(/^\/phase\s*/, "").trim();
		const currentPhase = this.session.workflow.currentPhase;

		if (!argText) {
			const phases = WORKFLOW_PHASES.join(" -> ");
			this.showStatus(`Workflow phase: ${this.formatWorkflowLabel(currentPhase)}\nFlow: ${phases}`);
			return;
		}

		const [phaseToken, ...reasonParts] = argText.split(/\s+/);
		const nextPhase = phaseToken as WorkflowPhase;
		if (!WORKFLOW_PHASES.includes(nextPhase)) {
			this.showWarning(`Unknown workflow phase "${phaseToken}"`);
			return;
		}

		const reason = reasonParts.join(" ").trim() || `Manual phase update from /phase`;
		try {
			this.session.transitionWorkflow(nextPhase, reason);
			const nextSnapshot = this.session.workflow;
			const completionState = getActiveTaskCompletionState(nextSnapshot);
			this.renderWidgets();
			this.updateEditorBorderColor();
			let message = `Workflow phase: ${this.formatWorkflowLabel(currentPhase)} -> ${this.formatWorkflowLabel(nextPhase)}`;
			if (nextPhase === "summarize" && !completionState.completionReady) {
				message += `\nWarning: active task is ${this.formatTaskCompletionLabel(completionState.completionLabel)} and is not completion-ready.`;
			}
			this.showStatus(message);
		} catch (error) {
			this.showError(error instanceof Error ? error.message : String(error));
		}
	}

	private handleWorkflowTaskCommand(text: string): void {
		const argText = text.replace(/^\/task\s*/, "").trim();
		const workflow = this.session.workflow;

		if (!argText || argText === "list") {
			const lines = workflow.taskGraph.taskOrder.map((taskId) => {
				const task = workflow.taskGraph.tasks[taskId];
				if (!task) return undefined;
				const isActive = workflow.taskGraph.activeTaskId === taskId ? " *" : "";
				const verificationStatus = getTaskVerificationStatus(workflow, taskId);
				const completionLabel = getTaskCompletionLabel(workflow, taskId);
				const dependenciesReady = areTaskDependenciesSatisfied(workflow.taskGraph, taskId);
				return `${task.id}: ${task.goal} [${task.status}]${isActive} | deps=${dependenciesReady ? "ready" : "waiting"} | verification=${verificationStatus} | completion=${this.formatTaskCompletionLabel(completionLabel)} (${task.acceptanceCriteria.length} criteria, ${task.notes.length} notes)`;
			});
			const visibleLines = lines.filter((line): line is string => line !== undefined);
			this.showStatus(
				visibleLines.length > 0 ? `Workflow tasks:\n${visibleLines.join("\n")}` : "Workflow tasks: none yet",
			);
			return;
		}

		const [subcommand, ...rest] = argText.split(/\s+/);
		if (subcommand === "add") {
			const taskId = rest[0];
			const goal = rest.slice(1).join(" ").trim();
			if (!taskId || !goal) {
				this.showWarning("Usage: /task add <id> <goal>");
				return;
			}
			try {
				this.session.upsertWorkflowTask({ id: taskId, goal, status: "ready" });
				this.renderWidgets();
				this.showStatus(`Workflow task added: ${taskId}`);
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		if (subcommand === "show") {
			const taskId = rest[0];
			if (!taskId) {
				this.showWarning("Usage: /task show <id>");
				return;
			}
			const task = workflow.taskGraph.tasks[taskId];
			if (!task) {
				this.showWarning(`Unknown workflow task "${taskId}"`);
				return;
			}
			const verificationStatus = getTaskVerificationStatus(workflow, taskId);
			const completionLabel = getTaskCompletionLabel(workflow, taskId);
			const latestVerification = getLatestTaskVerification(workflow, taskId);
			const dependenciesReady = areTaskDependenciesSatisfied(workflow.taskGraph, taskId);
			const contractText = this.buildTaskExecutionContractText(taskId) ?? "none";
			const criteria =
				task.acceptanceCriteria.length > 0
					? task.acceptanceCriteria.map((criterion) => `- ${criterion}`).join("\n")
					: "- none";
			const notes = task.notes.length > 0 ? task.notes.map((note) => `- ${note}`).join("\n") : "- none";
			this.showStatus(
				`Task ${task.id}: ${task.goal}\nStatus: ${this.formatWorkflowLabel(task.status)}\nDependencies: ${dependenciesReady ? "Satisfied" : "Waiting"}\nVerification: ${this.formatWorkflowLabel(verificationStatus)}\nCompletion: ${this.formatTaskCompletionLabel(completionLabel)}\nAcceptance criteria:\n${criteria}\nNotes:\n${notes}\nLatest verification details: ${latestVerification?.evidence.diffSummary ?? latestVerification?.evidence.userWaiver ?? latestVerification?.evidence.commands[0]?.details ?? "none"}\nExecution contract:\n${contractText}`,
			);
			return;
		}

		if (subcommand === "active") {
			const taskId = rest[0];
			if (!taskId) {
				this.showWarning("Usage: /task active <id>");
				return;
			}
			try {
				this.session.setWorkflowActiveTask(taskId);
				this.renderWidgets();
				this.showStatus(`Workflow active task: ${taskId}`);
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		if (subcommand === "status") {
			const taskId = rest[0];
			const statusToken = rest[1] as TaskStatus | undefined;
			if (!taskId || !statusToken) {
				this.showWarning("Usage: /task status <id> <pending|ready|in_progress|blocked|done|waived>");
				return;
			}
			if (!WORKFLOW_TASK_STATUSES.includes(statusToken)) {
				this.showWarning(`Unknown task status "${statusToken}"`);
				return;
			}
			try {
				this.session.updateWorkflowTaskStatus(taskId, statusToken);
				const nextWorkflow = this.session.workflow;
				const completionLabel = getTaskCompletionLabel(nextWorkflow, taskId);
				this.renderWidgets();
				this.showStatus(
					`Workflow task ${taskId}: ${this.formatWorkflowLabel(statusToken)}\nCompletion: ${this.formatTaskCompletionLabel(completionLabel)}`,
				);
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		if (subcommand === "criteria") {
			const taskId = rest[0];
			const criterion = rest.slice(1).join(" ").trim();
			if (!taskId || !criterion) {
				this.showWarning("Usage: /task criteria <id> <criterion>");
				return;
			}
			const task = workflow.taskGraph.tasks[taskId];
			if (!task) {
				this.showWarning(`Unknown workflow task "${taskId}"`);
				return;
			}
			try {
				this.session.updateWorkflowTask(taskId, {
					acceptanceCriteria: [...task.acceptanceCriteria, criterion],
				});
				this.renderWidgets();
				this.showStatus(`Workflow task ${taskId}: added acceptance criterion`);
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		if (subcommand === "note") {
			const taskId = rest[0];
			const note = rest.slice(1).join(" ").trim();
			if (!taskId || !note) {
				this.showWarning("Usage: /task note <id> <note>");
				return;
			}
			const task = workflow.taskGraph.tasks[taskId];
			if (!task) {
				this.showWarning(`Unknown workflow task "${taskId}"`);
				return;
			}
			try {
				this.session.updateWorkflowTask(taskId, {
					notes: [...task.notes, note],
				});
				this.renderWidgets();
				this.showStatus(`Workflow task ${taskId}: added note`);
			} catch (error) {
				this.showError(error instanceof Error ? error.message : String(error));
			}
			return;
		}

		this.showWarning(
			"Usage: /task [list] | /task show <id> | /task add <id> <goal> | /task active <id> | /task status <id> <status> | /task criteria <id> <criterion> | /task note <id> <note>",
		);
	}

	private handleWorkflowVerifyCommand(text: string): void {
		const argText = text.replace(/^\/verify\s*/, "").trim();
		const workflow = this.session.workflow;
		const activeTaskId = workflow.taskGraph.activeTaskId;
		if (!activeTaskId) {
			this.showWarning("No active workflow task. Use /task active <id> first.");
			return;
		}

		const [statusTokenRaw, ...detailParts] = argText ? argText.split(/\s+/) : [];
		const statusToken = statusTokenRaw?.toLowerCase();
		const detailText = detailParts.join(" ").trim();
		const verificationStatus =
			statusToken === "pass" || statusToken === "passed"
				? "passed"
				: statusToken === "fail" || statusToken === "failed"
					? "failed"
					: statusToken === "waive" || statusToken === "waived"
						? "waived"
						: undefined;

		if (!verificationStatus) {
			this.showWarning("Usage: /verify <passed|failed|waived> <details>");
			return;
		}

		this.session.recordWorkflowVerification(activeTaskId, verificationStatus, {
			tests: [],
			commands: [
				{
					command: "manual:/verify",
					validated: verificationStatus === "passed",
					details: detailText || undefined,
				},
			],
			userWaiver: verificationStatus === "waived" ? detailText || "Manual waiver recorded." : undefined,
			diffSummary: detailText || undefined,
		});
		this.session.recordWorkflowArtifact({
			id: `manual-verify-${Date.now()}`,
			type: "verification",
			label: `Manual verification for ${activeTaskId}`,
			producer: "interactive:/verify",
			metadata: {
				status: verificationStatus,
				details: detailText || null,
			},
		});
		const nextWorkflow = this.session.workflow;
		const completionLabel = getTaskCompletionLabel(nextWorkflow, activeTaskId);
		this.renderWidgets();
		this.showStatus(
			`Workflow verification for ${activeTaskId}: ${this.formatWorkflowLabel(verificationStatus)}\nCompletion: ${this.formatTaskCompletionLabel(completionLabel)}`,
		);
	}

	private handleWorkflowSummaryCommand(): void {
		const workflow = this.session.workflow;
		const activeTaskId = workflow.taskGraph.activeTaskId;
		const activeTask = activeTaskId ? workflow.taskGraph.tasks[activeTaskId] : undefined;
		const completion = getActiveTaskCompletionState(workflow);

		const lines: string[] = [
			`Phase: ${this.formatWorkflowLabel(workflow.currentPhase)}`,
			`Status: ${this.formatWorkflowLabel(workflow.status)}`,
			`Goal: ${workflow.goal}`,
			`Active task: ${activeTask ? `${activeTask.id} - ${activeTask.goal} [${this.formatWorkflowLabel(activeTask.status)}]` : "none"}`,
			`Verification: ${this.formatWorkflowLabel(completion.verificationStatus)}`,
			`Completion: ${this.formatTaskCompletionLabel(completion.completionLabel)}`,
			`Completion ready: ${completion.completionReady ? "yes" : "no"}`,
			`Tasks: ${workflow.taskGraph.taskOrder.length}`,
		];

		this.showStatus(lines.join("\n"));
	}

	private handleChangelogCommand(): void {
		const changelogPath = getChangelogPath();
		const allEntries = parseChangelog(changelogPath);

		const changelogMarkdown =
			allEntries.length > 0
				? allEntries
						.reverse()
						.map((e) => e.content)
						.join("\n\n")
				: "No changelog entries found.";

		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new DynamicBorder());
		this.chatContainer.addChild(new Text(theme.bold(theme.fg("accent", "What's New")), 1, 0));
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Markdown(changelogMarkdown, 1, 1, this.getMarkdownThemeWithSettings()));
		this.chatContainer.addChild(new DynamicBorder());
		this.ui.requestRender();
	}

	/**
	 * Capitalize keybinding for display (e.g., "ctrl+c" -> "Ctrl+C").
	 */
	private capitalizeKey(key: string): string {
		return key
			.split("/")
			.map((k) =>
				k
					.split("+")
					.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
					.join("+"),
			)
			.join("/");
	}

	/**
	 * Get capitalized display string for an app keybinding action.
	 */
	private getAppKeyDisplay(action: AppAction): string {
		return this.capitalizeKey(appKey(this.keybindings, action));
	}

	/**
	 * Get capitalized display string for an editor keybinding action.
	 */
	private getEditorKeyDisplay(action: EditorAction): string {
		return this.capitalizeKey(editorKey(action));
	}

	private handleHotkeysCommand(): void {
		// Navigation keybindings
		const cursorWordLeft = this.getEditorKeyDisplay("cursorWordLeft");
		const cursorWordRight = this.getEditorKeyDisplay("cursorWordRight");
		const cursorLineStart = this.getEditorKeyDisplay("cursorLineStart");
		const cursorLineEnd = this.getEditorKeyDisplay("cursorLineEnd");
		const jumpForward = this.getEditorKeyDisplay("jumpForward");
		const jumpBackward = this.getEditorKeyDisplay("jumpBackward");
		const pageUp = this.getEditorKeyDisplay("pageUp");
		const pageDown = this.getEditorKeyDisplay("pageDown");

		// Editing keybindings
		const submit = this.getEditorKeyDisplay("submit");
		const newLine = this.getEditorKeyDisplay("newLine");
		const deleteWordBackward = this.getEditorKeyDisplay("deleteWordBackward");
		const deleteWordForward = this.getEditorKeyDisplay("deleteWordForward");
		const deleteToLineStart = this.getEditorKeyDisplay("deleteToLineStart");
		const deleteToLineEnd = this.getEditorKeyDisplay("deleteToLineEnd");
		const yank = this.getEditorKeyDisplay("yank");
		const yankPop = this.getEditorKeyDisplay("yankPop");
		const undo = this.getEditorKeyDisplay("undo");
		const tab = this.getEditorKeyDisplay("tab");

		// App keybindings
		const interrupt = this.getAppKeyDisplay("interrupt");
		const clear = this.getAppKeyDisplay("clear");
		const exit = this.getAppKeyDisplay("exit");
		const suspend = this.getAppKeyDisplay("suspend");
		const cycleThinkingLevel = this.getAppKeyDisplay("cycleThinkingLevel");
		const cycleModelForward = this.getAppKeyDisplay("cycleModelForward");
		const selectModel = this.getAppKeyDisplay("selectModel");
		const expandTools = this.getAppKeyDisplay("expandTools");
		const toggleThinking = this.getAppKeyDisplay("toggleThinking");
		const externalEditor = this.getAppKeyDisplay("externalEditor");
		const followUp = this.getAppKeyDisplay("followUp");
		const dequeue = this.getAppKeyDisplay("dequeue");

		let hotkeys = `
**Navigation**
| Key | Action |
|-----|--------|
| \`Arrow keys\` | Move cursor / browse history (Up when empty) |
| \`${cursorWordLeft}\` / \`${cursorWordRight}\` | Move by word |
| \`${cursorLineStart}\` | Start of line |
| \`${cursorLineEnd}\` | End of line |
| \`${jumpForward}\` | Jump forward to character |
| \`${jumpBackward}\` | Jump backward to character |
| \`${pageUp}\` / \`${pageDown}\` | Scroll by page |

**Editing**
| Key | Action |
|-----|--------|
| \`${submit}\` | Send message |
| \`${newLine}\` | New line${process.platform === "win32" ? " (Ctrl+Enter on Windows Terminal)" : ""} |
| \`${deleteWordBackward}\` | Delete word backwards |
| \`${deleteWordForward}\` | Delete word forwards |
| \`${deleteToLineStart}\` | Delete to start of line |
| \`${deleteToLineEnd}\` | Delete to end of line |
| \`${yank}\` | Paste the most-recently-deleted text |
| \`${yankPop}\` | Cycle through the deleted text after pasting |
| \`${undo}\` | Undo |

**Other**
| Key | Action |
|-----|--------|
| \`${tab}\` | Path completion / accept autocomplete |
| \`${interrupt}\` | Cancel autocomplete / abort streaming |
| \`${clear}\` | Clear editor (first) / exit (second) |
| \`${exit}\` | Exit (when editor is empty) |
| \`${suspend}\` | Suspend to background |
| \`${cycleThinkingLevel}\` | Cycle thinking level |
| \`${cycleModelForward}\` | Cycle models |
| \`${selectModel}\` | Open model selector |
| \`${expandTools}\` | Toggle tool output expansion |
| \`${toggleThinking}\` | Toggle thinking block visibility |
| \`${externalEditor}\` | Edit message in external editor |
| \`${followUp}\` | Queue follow-up message |
| \`${dequeue}\` | Restore queued messages |
| \`Ctrl+V\` | Paste image from clipboard |
| \`/\` | Slash commands |
| \`!\` | Run bash command |
| \`!!\` | Run bash command (excluded from context) |
`;

		// Add extension-registered shortcuts
		const extensionRunner = this.session.extensionRunner;
		if (extensionRunner) {
			const shortcuts = extensionRunner.getShortcuts(this.keybindings.getEffectiveConfig());
			if (shortcuts.size > 0) {
				hotkeys += `
**Extensions**
| Key | Action |
|-----|--------|
`;
				for (const [key, shortcut] of shortcuts) {
					const description = shortcut.description ?? shortcut.extensionPath;
					const keyDisplay = key.replace(/\b\w/g, (c) => c.toUpperCase());
					hotkeys += `| \`${keyDisplay}\` | ${description} |\n`;
				}
			}
		}

		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new DynamicBorder());
		this.chatContainer.addChild(new Text(theme.bold(theme.fg("accent", "Keyboard Shortcuts")), 1, 0));
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Markdown(hotkeys.trim(), 1, 1, this.getMarkdownThemeWithSettings()));
		this.chatContainer.addChild(new DynamicBorder());
		this.ui.requestRender();
	}

	private async handleClearCommand(): Promise<void> {
		// Stop loading animation
		if (this.loadingAnimation) {
			this.loadingAnimation.stop();
			this.loadingAnimation = undefined;
		}
		this.statusContainer.clear();

		// New session via session (emits extension session events)
		await this.session.newSession();

		// Clear UI state
		this.chatContainer.clear();
		this.pendingMessagesContainer.clear();
		this.compactionQueuedMessages = [];
		this.streamingComponent = undefined;
		this.streamingMessage = undefined;
		this.pendingTools.clear();

		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new Text(`${theme.fg("accent", "✓ New session started")}`, 1, 1));
		this.renderWidgets();
		this.ui.requestRender();
	}

	private handleDebugCommand(): void {
		const width = this.ui.terminal.columns;
		const height = this.ui.terminal.rows;
		const allLines = this.ui.render(width);

		const debugLogPath = getDebugLogPath();
		const debugData = [
			`Debug output at ${new Date().toISOString()}`,
			`Terminal: ${width}x${height}`,
			`Total lines: ${allLines.length}`,
			"",
			"=== All rendered lines with visible widths ===",
			...allLines.map((line, idx) => {
				const vw = visibleWidth(line);
				const escaped = JSON.stringify(line);
				return `[${idx}] (w=${vw}) ${escaped}`;
			}),
			"",
			"=== Agent messages (JSONL) ===",
			...this.session.messages.map((msg) => JSON.stringify(msg)),
			"",
		].join("\n");

		fs.mkdirSync(path.dirname(debugLogPath), { recursive: true });
		fs.writeFileSync(debugLogPath, debugData);

		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(
			new Text(`${theme.fg("accent", "✓ Debug log written")}\n${theme.fg("muted", debugLogPath)}`, 1, 1),
		);
		this.ui.requestRender();
	}

	private handleArminSaysHi(): void {
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new ArminComponent(this.ui));
		this.ui.requestRender();
	}

	private handleDaxnuts(): void {
		this.chatContainer.addChild(new Spacer(1));
		this.chatContainer.addChild(new DaxnutsComponent(this.ui));
		this.ui.requestRender();
	}

	private checkDaxnutsEasterEgg(model: { provider: string; id: string }): void {
		if (model.provider === "opencode" && model.id.toLowerCase().includes("kimi-k2.5")) {
			this.handleDaxnuts();
		}
	}

	private async handleBashCommand(command: string, excludeFromContext = false): Promise<void> {
		const extensionRunner = this.session.extensionRunner;

		// Emit user_bash event to let extensions intercept
		const eventResult = extensionRunner
			? await extensionRunner.emitUserBash({
					type: "user_bash",
					command,
					excludeFromContext,
					cwd: process.cwd(),
				})
			: undefined;

		// If extension returned a full result, use it directly
		if (eventResult?.result) {
			const result = eventResult.result;

			// Create UI component for display
			this.bashComponent = new BashExecutionComponent(command, this.ui, excludeFromContext);
			if (this.session.isStreaming) {
				this.pendingMessagesContainer.addChild(this.bashComponent);
				this.pendingBashComponents.push(this.bashComponent);
			} else {
				this.chatContainer.addChild(this.bashComponent);
			}

			// Show output and complete
			if (result.output) {
				this.bashComponent.appendOutput(result.output);
			}
			this.bashComponent.setComplete(
				result.exitCode,
				result.cancelled,
				result.truncated ? ({ truncated: true, content: result.output } as TruncationResult) : undefined,
				result.fullOutputPath,
			);

			// Record the result in session
			this.session.recordBashResult(command, result, { excludeFromContext });
			this.bashComponent = undefined;
			this.ui.requestRender();
			return;
		}

		// Normal execution path (possibly with custom operations)
		const isDeferred = this.session.isStreaming;
		this.bashComponent = new BashExecutionComponent(command, this.ui, excludeFromContext);

		if (isDeferred) {
			// Show in pending area when agent is streaming
			this.pendingMessagesContainer.addChild(this.bashComponent);
			this.pendingBashComponents.push(this.bashComponent);
		} else {
			// Show in chat immediately when agent is idle
			this.chatContainer.addChild(this.bashComponent);
		}
		this.ui.requestRender();

		try {
			const result = await this.session.executeBash(
				command,
				(chunk) => {
					if (this.bashComponent) {
						this.bashComponent.appendOutput(chunk);
						this.ui.requestRender();
					}
				},
				{ excludeFromContext, operations: eventResult?.operations },
			);

			if (this.bashComponent) {
				this.bashComponent.setComplete(
					result.exitCode,
					result.cancelled,
					result.truncated ? ({ truncated: true, content: result.output } as TruncationResult) : undefined,
					result.fullOutputPath,
				);
			}
		} catch (error) {
			if (this.bashComponent) {
				this.bashComponent.setComplete(undefined, false);
			}
			this.showError(`Bash command failed: ${error instanceof Error ? error.message : "Unknown error"}`);
		}

		this.bashComponent = undefined;
		this.ui.requestRender();
	}

	private async handleCompactCommand(customInstructions?: string): Promise<void> {
		const entries = this.sessionManager.getEntries();
		const messageCount = entries.filter((e) => e.type === "message").length;

		if (messageCount < 2) {
			this.showWarning("Nothing to compact (no messages yet)");
			return;
		}

		await this.withTemporaryRoleModel("compact", async () => {
			await this.executeCompaction(customInstructions, false);
		});
	}

	private async executeCompaction(customInstructions?: string, isAuto = false): Promise<CompactionResult | undefined> {
		// Stop loading animation
		if (this.loadingAnimation) {
			this.loadingAnimation.stop();
			this.loadingAnimation = undefined;
		}
		this.statusContainer.clear();

		// Set up escape handler during compaction
		const originalOnEscape = this.defaultEditor.onEscape;
		this.defaultEditor.onEscape = () => {
			this.session.abortCompaction();
		};

		// Show compacting status
		this.chatContainer.addChild(new Spacer(1));
		const cancelHint = `(${appKey(this.keybindings, "interrupt")} to cancel)`;
		const label = isAuto ? `Auto-compacting context... ${cancelHint}` : `Compacting context... ${cancelHint}`;
		const compactingLoader = new Loader(
			this.ui,
			(spinner) => theme.fg("accent", spinner),
			(text) => theme.fg("muted", text),
			label,
		);
		this.statusContainer.addChild(compactingLoader);
		this.ui.requestRender();

		let result: CompactionResult | undefined;

		try {
			result = await this.session.compact(customInstructions);

			// Rebuild UI
			this.rebuildChatFromMessages();

			// Add compaction component at bottom so user sees it without scrolling
			const msg = createCompactionSummaryMessage(result.summary, result.tokensBefore, new Date().toISOString());
			this.addMessageToChat(msg);

			this.footer.invalidate();
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			if (message === "Compaction cancelled" || (error instanceof Error && error.name === "AbortError")) {
				this.showError("Compaction cancelled");
			} else {
				this.showError(`Compaction failed: ${message}`);
			}
		} finally {
			compactingLoader.stop();
			this.statusContainer.clear();
			this.defaultEditor.onEscape = originalOnEscape;
		}
		void this.flushCompactionQueue({ willRetry: false });
		return result;
	}

	stop(): void {
		if (this.loadingAnimation) {
			this.loadingAnimation.stop();
			this.loadingAnimation = undefined;
		}
		this.stopEventTail(true);
		this.clearExtensionTerminalInputListeners();
		this.footer.dispose();
		this.footerDataProvider.dispose();
		this.runtimeServices?.stop();
		if (this.unsubscribe) {
			this.unsubscribe();
		}
		if (this.isInitialized) {
			this.ui.stop();
			this.isInitialized = false;
		}
	}
}

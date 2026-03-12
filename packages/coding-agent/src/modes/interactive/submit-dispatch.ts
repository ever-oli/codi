import type { EditorComponent } from "@mariozechner/pi-tui";

type SubmitCommandHandler = (text: string, context: InteractiveSubmitDispatchContext) => Promise<void> | void;

type SubmitCommand = {
	matches: (text: string) => boolean;
	run: SubmitCommandHandler;
};

type InteractiveSessionState = {
	isBashRunning: boolean;
	isCompacting: boolean;
	isStreaming: boolean;
	prompt: (text: string) => Promise<void>;
};

export interface InteractiveSubmitDispatchContext {
	editor: Pick<EditorComponent, "setText" | "addToHistory">;
	session: InteractiveSessionState;
	onInputCallback?: (text: string) => void;
	setBashMode: (enabled: boolean) => void;
	updateEditorBorderColor: () => void;
	showWarning: (message: string) => void;
	showSettingsSelector: () => void;
	showModelsSelector: () => Promise<void>;
	handleModelCommand: (searchTerm?: string) => Promise<void>;
	handleExportCommand: (text: string) => Promise<void>;
	handleShareCommand: () => Promise<void>;
	handleCopyCommand: () => void;
	handleNameCommand: (text: string) => void;
	handleSessionCommand: () => void;
	handleEventsCommand: (text: string) => Promise<void>;
	handleQueueCommand: (text: string) => Promise<void>;
	handleLanesCommand: (text: string) => Promise<void>;
	handlePackagesCommand: (text: string) => Promise<void>;
	handleMailboxCommand: (text: string) => Promise<void>;
	handleDelegatedCommand: (text: string) => Promise<void>;
	handleHeartbeatCommand: (text: string) => Promise<void>;
	handleModelsCommand: (text: string) => Promise<void>;
	handleOpsCommand: (text: string) => Promise<void>;
	handleWorkflowPlanCommand: (text: string) => void;
	handleWorkflowPhaseCommand: (text: string) => void;
	handleWorkflowTaskCommand: (text: string) => void;
	handleWorkflowVerifyCommand: (text: string) => void;
	handleWorkflowSummaryCommand: () => void;
	handleResourcesCommand: (text: string) => void;
	handleChangelogCommand: () => void;
	handleHotkeysCommand: () => void;
	showUserMessageSelector: () => void;
	showTreeSelector: () => void;
	showOAuthSelector: (mode: "login" | "logout") => void;
	handleClearCommand: () => Promise<void>;
	handleCompactCommand: (customInstructions?: string) => Promise<void>;
	handleReloadCommand: () => Promise<void>;
	handleDebugCommand: () => void;
	handleArminSaysHi: () => void;
	showSessionSelector: () => void;
	shutdown: () => Promise<void>;
	handleBashCommand: (command: string, isExcluded: boolean) => Promise<void>;
	isExtensionCommand: (text: string) => boolean;
	queueCompactionMessage: (text: string, mode: "steer" | "followUp") => void;
	promptWithMainRole: (text: string, options: { streamingBehavior: "steer" }) => Promise<void>;
	updatePendingMessagesDisplay: () => void;
	requestRender: () => void;
	flushPendingBashComponents: () => void;
	recordCommandUsage: (command: string) => void;
}

const exactMatch = (command: string, run: SubmitCommandHandler): SubmitCommand => ({
	matches: (text) => text === command,
	run,
});

const exactOrPrefixMatch = (command: string, run: SubmitCommandHandler): SubmitCommand => ({
	matches: (text) => text === command || text.startsWith(`${command} `),
	run,
});

const prefixMatch = (command: string, run: SubmitCommandHandler): SubmitCommand => ({
	matches: (text) => text.startsWith(command),
	run,
});

const submitCommands: SubmitCommand[] = [
	exactMatch("/settings", (_text, context) => {
		context.showSettingsSelector();
		context.editor.setText("");
	}),
	exactMatch("/scoped-models", async (_text, context) => {
		context.editor.setText("");
		await context.showModelsSelector();
	}),
	exactOrPrefixMatch("/model", async (text, context) => {
		const searchTerm = text.startsWith("/model ") ? text.slice(7).trim() : undefined;
		context.editor.setText("");
		await context.handleModelCommand(searchTerm);
	}),
	prefixMatch("/export", async (text, context) => {
		await context.handleExportCommand(text);
		context.editor.setText("");
	}),
	exactMatch("/share", async (_text, context) => {
		await context.handleShareCommand();
		context.editor.setText("");
	}),
	exactMatch("/copy", (_text, context) => {
		context.handleCopyCommand();
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/name", (text, context) => {
		context.handleNameCommand(text);
		context.editor.setText("");
	}),
	exactMatch("/session", (_text, context) => {
		context.handleSessionCommand();
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/events", async (text, context) => {
		await context.handleEventsCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/queue", async (text, context) => {
		await context.handleQueueCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/lanes", async (text, context) => {
		await context.handleLanesCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/packages", async (text, context) => {
		await context.handlePackagesCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/mailbox", async (text, context) => {
		await context.handleMailboxCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/delegated", async (text, context) => {
		await context.handleDelegatedCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/heartbeat", async (text, context) => {
		await context.handleHeartbeatCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/models", async (text, context) => {
		await context.handleModelsCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/ops", async (text, context) => {
		await context.handleOpsCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/plan", (text, context) => {
		context.handleWorkflowPlanCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/phase", (text, context) => {
		context.handleWorkflowPhaseCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/task", (text, context) => {
		context.handleWorkflowTaskCommand(text);
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/verify", (text, context) => {
		context.handleWorkflowVerifyCommand(text);
		context.editor.setText("");
	}),
	exactMatch("/workflow", (_text, context) => {
		context.handleWorkflowSummaryCommand();
		context.editor.setText("");
	}),
	exactOrPrefixMatch("/resources", (text, context) => {
		context.handleResourcesCommand(text);
		context.editor.setText("");
	}),
	exactMatch("/changelog", (_text, context) => {
		context.handleChangelogCommand();
		context.editor.setText("");
	}),
	exactMatch("/hotkeys", (_text, context) => {
		context.handleHotkeysCommand();
		context.editor.setText("");
	}),
	exactMatch("/fork", (_text, context) => {
		context.showUserMessageSelector();
		context.editor.setText("");
	}),
	exactMatch("/tree", (_text, context) => {
		context.showTreeSelector();
		context.editor.setText("");
	}),
	exactMatch("/login", (_text, context) => {
		context.showOAuthSelector("login");
		context.editor.setText("");
	}),
	exactMatch("/logout", (_text, context) => {
		context.showOAuthSelector("logout");
		context.editor.setText("");
	}),
	exactMatch("/new", async (_text, context) => {
		context.editor.setText("");
		await context.handleClearCommand();
	}),
	exactOrPrefixMatch("/compact", async (text, context) => {
		const customInstructions = text.startsWith("/compact ") ? text.slice(9).trim() : undefined;
		context.editor.setText("");
		await context.handleCompactCommand(customInstructions);
	}),
	exactMatch("/reload", async (_text, context) => {
		context.editor.setText("");
		await context.handleReloadCommand();
	}),
	exactMatch("/debug", (_text, context) => {
		context.handleDebugCommand();
		context.editor.setText("");
	}),
	exactMatch("/arminsayshi", (_text, context) => {
		context.handleArminSaysHi();
		context.editor.setText("");
	}),
	exactMatch("/resume", (_text, context) => {
		context.showSessionSelector();
		context.editor.setText("");
	}),
	exactMatch("/quit", async (_text, context) => {
		context.editor.setText("");
		await context.shutdown();
	}),
];

export async function handleInteractiveSubmit(
	context: InteractiveSubmitDispatchContext,
	rawText: string,
): Promise<void> {
	const text = rawText.trim();
	if (!text) return;

	for (const command of submitCommands) {
		if (command.matches(text)) {
			// Track command usage (#3)
			const cmdName = text.split(/\s+/)[0];
			if (cmdName) context.recordCommandUsage(cmdName);
			await command.run(text, context);
			return;
		}
	}

	if (text.startsWith("!")) {
		const isExcluded = text.startsWith("!!");
		const command = isExcluded ? text.slice(2).trim() : text.slice(1).trim();
		if (command) {
			if (context.session.isBashRunning) {
				context.showWarning("A bash command is already running. Press Esc to cancel it first.");
				context.editor.setText(text);
				return;
			}
			context.editor.addToHistory?.(text);
			await context.handleBashCommand(command, isExcluded);
			context.setBashMode(false);
			context.updateEditorBorderColor();
			return;
		}
	}

	if (context.session.isCompacting) {
		if (context.isExtensionCommand(text)) {
			context.editor.addToHistory?.(text);
			context.editor.setText("");
			await context.session.prompt(text);
		} else {
			context.queueCompactionMessage(text, "steer");
		}
		return;
	}

	if (context.session.isStreaming) {
		context.editor.addToHistory?.(text);
		context.editor.setText("");
		await context.promptWithMainRole(text, { streamingBehavior: "steer" });
		context.updatePendingMessagesDisplay();
		context.requestRender();
		return;
	}

	context.flushPendingBashComponents();
	context.onInputCallback?.(text);
	context.editor.addToHistory?.(text);
}

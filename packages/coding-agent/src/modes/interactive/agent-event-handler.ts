import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { AssistantMessage } from "@mariozechner/pi-ai";
import { type Container, Loader, type MarkdownTheme, Spacer, Text, type TUI } from "@mariozechner/pi-tui";
import type { AgentSessionEvent } from "../../core/agent-session.js";
import type { ToolDefinition } from "../../core/extensions/index.js";
import type { RuntimeServices } from "../../core/runtime/index.js";
import { AssistantMessageComponent } from "./components/assistant-message.js";
import { ToolExecutionComponent } from "./components/tool-execution.js";
import { theme } from "./theme/theme.js";

export interface InteractiveEventState {
	retryEscapeHandler?: (() => void) | undefined;
	retryLoader?: Loader;
	loadingAnimation?: Loader;
	pendingWorkingMessage?: string;
	streamingComponent?: AssistantMessageComponent;
	streamingMessage?: AssistantMessage;
	autoCompactionEscapeHandler?: (() => void) | undefined;
	autoCompactionLoader?: Loader;
}

export interface InteractiveEventHandlerContext {
	state: InteractiveEventState;
	isInitialized: boolean;
	init: () => Promise<void>;
	runtimeServices?: RuntimeServices;
	sessionId: string;
	defaultEditor: {
		onEscape?: (() => void) | undefined;
	};
	session: {
		abortCompaction: () => void;
		abortRetry: () => void;
		retryAttempt: number;
	};
	statusContainer: Container;
	chatContainer: Container;
	ui: TUI;
	defaultWorkingMessage: string;
	interruptKeyLabel: string;
	hideThinkingBlock: boolean;
	settingsManager: {
		getShowImages: () => boolean;
	};
	toolOutputExpanded: boolean;
	pendingTools: Map<string, ToolExecutionComponent>;
	footerInvalidate: () => void;
	getMarkdownThemeWithSettings: () => MarkdownTheme;
	getRegisteredToolDefinition: (toolName: string) => ToolDefinition | undefined;
	addMessageToChat: (message: AgentMessage) => void;
	updatePendingMessagesDisplay: () => void;
	renderWidgets: () => void;
	checkShutdownRequested: () => Promise<void>;
	rebuildChatFromMessages: () => void;
	flushCompactionQueue: (options?: { willRetry?: boolean }) => Promise<void>;
	showStatus: (message: string) => void;
	showError: (message: string) => void;
	updateEditorBorderColor: () => void;
	setActiveTool: (toolName: string | undefined) => void;
	markAgentStart: () => void;
	markAgentEnd: () => void;
	recordAction: (action: string) => void;
	recordFileAccess: (filePath: string) => void;
	recordDiscovery: (text: string) => void;
}

export async function handleInteractiveAgentEvent(
	context: InteractiveEventHandlerContext,
	event: AgentSessionEvent,
): Promise<void> {
	const { state } = context;

	if (!context.isInitialized) {
		await context.init();
	}
	if (context.runtimeServices) {
		try {
			context.runtimeServices.recordAgentSessionEvent(context.sessionId, event);
		} catch {
			// Runtime observability must not break interactive flow.
		}
	}

	context.footerInvalidate();

	switch (event.type) {
		case "agent_start":
			if (state.retryEscapeHandler) {
				context.defaultEditor.onEscape = state.retryEscapeHandler;
				state.retryEscapeHandler = undefined;
			}
			if (state.retryLoader) {
				state.retryLoader.stop();
				state.retryLoader = undefined;
			}
			if (state.loadingAnimation) {
				state.loadingAnimation.stop();
			}
			context.statusContainer.clear();
			state.loadingAnimation = new Loader(
				context.ui,
				(spinner) => theme.fg("accent", spinner),
				(message) => theme.fg("muted", message),
				context.defaultWorkingMessage,
			);
			context.statusContainer.addChild(state.loadingAnimation);
			if (state.pendingWorkingMessage !== undefined) {
				if (state.pendingWorkingMessage) {
					state.loadingAnimation.setMessage(state.pendingWorkingMessage);
				}
				state.pendingWorkingMessage = undefined;
			}
			context.updateEditorBorderColor();
			context.markAgentStart();
			context.ui.requestRender();
			break;

		case "message_start":
			if (event.message.role === "custom") {
				context.addMessageToChat(event.message);
				context.ui.requestRender();
			} else if (event.message.role === "user") {
				context.addMessageToChat(event.message);
				context.updatePendingMessagesDisplay();
				context.ui.requestRender();
			} else if (event.message.role === "assistant") {
				state.streamingComponent = new AssistantMessageComponent(
					undefined,
					context.hideThinkingBlock,
					context.getMarkdownThemeWithSettings(),
				);
				state.streamingMessage = event.message;
				context.chatContainer.addChild(state.streamingComponent);
				state.streamingComponent.updateContent(state.streamingMessage);
				context.ui.requestRender();
			}
			break;

		case "message_update":
			if (state.streamingComponent && event.message.role === "assistant") {
				state.streamingMessage = event.message;
				state.streamingComponent.updateContent(state.streamingMessage);

				for (const content of state.streamingMessage.content) {
					if (content.type === "toolCall") {
						if (!context.pendingTools.has(content.id)) {
							const component = new ToolExecutionComponent(
								content.name,
								content.arguments,
								{
									showImages: context.settingsManager.getShowImages(),
								},
								context.getRegisteredToolDefinition(content.name),
								context.ui,
							);
							component.setExpanded(context.toolOutputExpanded);
							context.chatContainer.addChild(component);
							context.pendingTools.set(content.id, component);
						} else {
							const component = context.pendingTools.get(content.id);
							if (component) {
								component.updateArgs(content.arguments);
							}
						}
					}
				}
				context.ui.requestRender();
			}
			break;

		case "message_end":
			if (event.message.role === "user") break;
			if (state.streamingComponent && event.message.role === "assistant") {
				state.streamingMessage = event.message;
				let errorMessage: string | undefined;
				if (state.streamingMessage.stopReason === "aborted") {
					errorMessage =
						context.session.retryAttempt > 0
							? `Aborted after ${context.session.retryAttempt} retry attempt${context.session.retryAttempt > 1 ? "s" : ""}`
							: "Operation aborted";
					state.streamingMessage.errorMessage = errorMessage;
				}
				state.streamingComponent.updateContent(state.streamingMessage);

				if (state.streamingMessage.stopReason === "aborted" || state.streamingMessage.stopReason === "error") {
					if (!errorMessage) {
						errorMessage = state.streamingMessage.errorMessage || "Error";
					}
					for (const [, component] of context.pendingTools.entries()) {
						component.updateResult({
							content: [{ type: "text", text: errorMessage }],
							isError: true,
						});
					}
					context.pendingTools.clear();
				} else {
					for (const [, component] of context.pendingTools.entries()) {
						component.setArgsComplete();
					}
				}
				state.streamingComponent = undefined;
				state.streamingMessage = undefined;
				context.footerInvalidate();
			}
			context.renderWidgets();
			context.ui.requestRender();
			break;

		case "tool_execution_start": {
			if (!context.pendingTools.has(event.toolCallId)) {
				const component = new ToolExecutionComponent(
					event.toolName,
					event.args,
					{
						showImages: context.settingsManager.getShowImages(),
					},
					context.getRegisteredToolDefinition(event.toolName),
					context.ui,
				);
				component.setExpanded(context.toolOutputExpanded);
				context.chatContainer.addChild(component);
				context.pendingTools.set(event.toolCallId, component);
			}
			context.setActiveTool(event.toolName);
			context.recordAction(event.toolName);
			context.footerInvalidate();
			context.ui.requestRender();
			break;
		}

		case "tool_execution_update": {
			const component = context.pendingTools.get(event.toolCallId);
			if (component) {
				component.updateResult({ ...event.partialResult, isError: false }, true);
				context.ui.requestRender();
			}
			break;
		}

		case "tool_execution_end": {
			const component = context.pendingTools.get(event.toolCallId);
			if (component) {
				component.updateResult({ ...event.result, isError: event.isError });
				context.pendingTools.delete(event.toolCallId);
			}
			// Clear active tool if no more pending tools
			if (context.pendingTools.size === 0) {
				context.setActiveTool(undefined);
			}
			context.footerInvalidate();
			context.renderWidgets();
			context.ui.requestRender();
			break;
		}

		case "agent_end":
			if (state.loadingAnimation) {
				state.loadingAnimation.stop();
				state.loadingAnimation = undefined;
				context.statusContainer.clear();
			}
			if (state.streamingComponent) {
				context.chatContainer.removeChild(state.streamingComponent);
				state.streamingComponent = undefined;
				state.streamingMessage = undefined;
			}
			context.pendingTools.clear();
			context.setActiveTool(undefined);
			context.markAgentEnd();

			await context.checkShutdownRequested();

			context.updateEditorBorderColor();
			context.footerInvalidate();
			context.renderWidgets();
			context.ui.requestRender();
			break;

		case "auto_compaction_start": {
			state.autoCompactionEscapeHandler = context.defaultEditor.onEscape;
			context.defaultEditor.onEscape = () => {
				context.session.abortCompaction();
			};
			context.statusContainer.clear();
			const reasonText = event.reason === "overflow" ? "Context overflow detected, " : "";
			state.autoCompactionLoader = new Loader(
				context.ui,
				(spinner) => theme.fg("accent", spinner),
				(message) => theme.fg("muted", message),
				`${reasonText}Auto-compacting... (${context.interruptKeyLabel} to cancel)`,
			);
			context.statusContainer.addChild(state.autoCompactionLoader);
			context.ui.requestRender();
			break;
		}

		case "auto_compaction_end": {
			if (state.autoCompactionEscapeHandler) {
				context.defaultEditor.onEscape = state.autoCompactionEscapeHandler;
				state.autoCompactionEscapeHandler = undefined;
			}
			if (state.autoCompactionLoader) {
				state.autoCompactionLoader.stop();
				state.autoCompactionLoader = undefined;
				context.statusContainer.clear();
			}
			if (event.aborted) {
				context.showStatus("Auto-compaction cancelled");
			} else if (event.result) {
				context.chatContainer.clear();
				context.rebuildChatFromMessages();
				context.addMessageToChat({
					role: "compactionSummary",
					tokensBefore: event.result.tokensBefore,
					summary: event.result.summary,
					timestamp: Date.now(),
				});
				context.footerInvalidate();
			} else if (event.errorMessage) {
				context.chatContainer.addChild(new Spacer(1));
				context.chatContainer.addChild(new Text(theme.fg("error", event.errorMessage), 1, 0));
			}
			void context.flushCompactionQueue({ willRetry: event.willRetry });
			context.renderWidgets();
			context.ui.requestRender();
			break;
		}

		case "auto_retry_start": {
			state.retryEscapeHandler = context.defaultEditor.onEscape;
			context.defaultEditor.onEscape = () => {
				context.session.abortRetry();
			};
			context.statusContainer.clear();
			const delaySeconds = Math.round(event.delayMs / 1000);
			state.retryLoader = new Loader(
				context.ui,
				(spinner) => theme.fg("warning", spinner),
				(message) => theme.fg("muted", message),
				`Retrying (${event.attempt}/${event.maxAttempts}) in ${delaySeconds}s... (${context.interruptKeyLabel} to cancel)`,
			);
			context.statusContainer.addChild(state.retryLoader);
			context.ui.requestRender();
			break;
		}

		case "auto_retry_end": {
			if (state.retryEscapeHandler) {
				context.defaultEditor.onEscape = state.retryEscapeHandler;
				state.retryEscapeHandler = undefined;
			}
			if (state.retryLoader) {
				state.retryLoader.stop();
				state.retryLoader = undefined;
				context.statusContainer.clear();
			}
			if (!event.success) {
				context.showError(`Retry failed after ${event.attempt} attempts: ${event.finalError || "Unknown error"}`);
			}
			context.ui.requestRender();
			break;
		}
	}
}

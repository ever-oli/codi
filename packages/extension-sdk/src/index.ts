import type { ImageContent, TextContent } from "@mariozechner/pi-ai";
import type { AgentToolResult, ExtensionFactory, ToolDefinition } from "@mariozechner/pi-coding-agent";
import type { TSchema } from "@sinclair/typebox";

export type { ImageContent, TextContent } from "@mariozechner/pi-ai";
export { StringEnum } from "@mariozechner/pi-ai";
export type {
	AgentToolResult,
	AgentToolUpdateCallback,
	CustomEntry,
	CustomMessageEntry,
	ExecOptions,
	ExecResult,
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
	ExtensionFactory,
	InputEvent,
	InputEventResult,
	MessageRenderer,
	MessageRenderOptions,
	ProviderConfig,
	ProviderModelConfig,
	RegisteredCommand,
	RegisteredTool,
	SessionEntry,
	SessionMessageEntry,
	Theme,
	ToolCallEvent,
	ToolDefinition,
	ToolInfo,
	ToolRenderResultOptions,
	ToolResultEvent,
} from "@mariozechner/pi-coding-agent";
export {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	isBashToolResult,
	isEditToolResult,
	isFindToolResult,
	isGrepToolResult,
	isLsToolResult,
	isReadToolResult,
	isToolCallEventType,
	isWriteToolResult,
	keyHint,
	truncateHead,
	truncateLine,
	truncateTail,
} from "@mariozechner/pi-coding-agent";
export type { Component, EditorTheme, TUI } from "@mariozechner/pi-tui";
export { Editor, Key, matchesKey, Text, truncateToWidth } from "@mariozechner/pi-tui";
export type { Static, TSchema } from "@sinclair/typebox";
export { Type } from "@sinclair/typebox";

export function defineExtension<T extends ExtensionFactory>(extension: T): T {
	return extension;
}

export function defineTool<TParams extends TSchema = TSchema, TDetails = unknown>(
	tool: ToolDefinition<TParams, TDetails>,
): ToolDefinition<TParams, TDetails> {
	return tool;
}

export function textContent(text: string): TextContent {
	return { type: "text", text };
}

export function imageContent(data: string, mimeType: string): ImageContent {
	return { type: "image", data, mimeType };
}

export function textToolResult<TDetails = unknown>(text: string, details?: TDetails): AgentToolResult<TDetails> {
	return {
		content: [textContent(text)],
		details: details as TDetails,
	};
}

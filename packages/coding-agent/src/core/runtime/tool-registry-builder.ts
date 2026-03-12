import type { AgentTool } from "@mariozechner/pi-agent-core";
import {
	type ExtensionRunner,
	type ToolDefinition,
	wrapRegisteredTools,
	wrapToolsWithExtensions,
} from "../extensions/index.js";

export interface BuildToolRegistryOptions {
	baseToolRegistry: ReadonlyMap<string, AgentTool>;
	extensionRunner?: ExtensionRunner;
	customTools: ToolDefinition[];
	previousRegistryNames: Iterable<string>;
	previousActiveToolNames: readonly string[];
	activeToolNames?: readonly string[];
	includeAllExtensionTools?: boolean;
	normalizePromptSnippet: (snippet?: string) => string | undefined;
	normalizePromptGuidelines: (guidelines?: string[]) => string[];
}

export interface ToolRegistryBuildResult {
	toolRegistry: Map<string, AgentTool>;
	toolPromptSnippets: Map<string, string>;
	toolPromptGuidelines: Map<string, string[]>;
	activeToolNames: string[];
}

export function buildToolRegistryState(options: BuildToolRegistryOptions): ToolRegistryBuildResult {
	const previousRegistryNames = new Set(options.previousRegistryNames);
	const registeredTools = options.extensionRunner?.getAllRegisteredTools() ?? [];
	const allCustomTools = [
		...registeredTools,
		...options.customTools.map((definition) => ({ definition, extensionPath: "<sdk>" })),
	];

	const toolPromptSnippets = new Map(
		allCustomTools
			.map((registeredTool) => {
				const snippet = options.normalizePromptSnippet(
					registeredTool.definition.promptSnippet ?? registeredTool.definition.description,
				);
				return snippet ? ([registeredTool.definition.name, snippet] as const) : undefined;
			})
			.filter((entry): entry is readonly [string, string] => entry !== undefined),
	);

	const toolPromptGuidelines = new Map(
		allCustomTools
			.map((registeredTool) => {
				const guidelines = options.normalizePromptGuidelines(registeredTool.definition.promptGuidelines);
				return guidelines.length > 0 ? ([registeredTool.definition.name, guidelines] as const) : undefined;
			})
			.filter((entry): entry is readonly [string, string[]] => entry !== undefined),
	);

	const wrappedExtensionTools = options.extensionRunner
		? (wrapRegisteredTools(allCustomTools, options.extensionRunner) as AgentTool[])
		: [];

	const toolRegistry = new Map(options.baseToolRegistry);
	for (const tool of wrappedExtensionTools) {
		toolRegistry.set(tool.name, tool);
	}

	const nextToolRegistry = options.extensionRunner
		? new Map(
				wrapToolsWithExtensions(Array.from(toolRegistry.values()), options.extensionRunner).map((tool) => [
					tool.name,
					tool,
				]),
			)
		: toolRegistry;

	const nextActiveToolNames = options.activeToolNames
		? [...options.activeToolNames]
		: [...options.previousActiveToolNames];

	if (options.includeAllExtensionTools) {
		for (const tool of wrappedExtensionTools) {
			nextActiveToolNames.push(tool.name);
		}
	} else if (!options.activeToolNames) {
		for (const toolName of nextToolRegistry.keys()) {
			if (!previousRegistryNames.has(toolName)) {
				nextActiveToolNames.push(toolName);
			}
		}
	}

	return {
		toolRegistry: nextToolRegistry,
		toolPromptSnippets,
		toolPromptGuidelines,
		activeToolNames: [...new Set(nextActiveToolNames)],
	};
}

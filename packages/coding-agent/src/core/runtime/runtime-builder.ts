import type { AgentTool } from "@mariozechner/pi-agent-core";
import { ExtensionRunner, type ToolDefinition } from "../extensions/index.js";
import type { ModelRegistry } from "../model-registry.js";
import type { ResourceLoader } from "../resource-loader.js";
import type { SessionManager } from "../session-manager.js";
import { createAllTools } from "../tools/index.js";

export interface BuildSessionRuntimeOptions {
	cwd: string;
	autoResizeImages: boolean;
	shellCommandPrefix?: string;
	baseToolsOverride?: Record<string, AgentTool>;
	resourceLoader: ResourceLoader;
	customTools: ToolDefinition[];
	sessionManager: SessionManager;
	modelRegistry: ModelRegistry;
	flagValues?: ReadonlyMap<string, boolean | string>;
}

export interface SessionRuntimeBuildResult {
	baseToolRegistry: Map<string, AgentTool>;
	extensionRunner?: ExtensionRunner;
	defaultActiveToolNames: string[];
}

export function buildSessionRuntime(options: BuildSessionRuntimeOptions): SessionRuntimeBuildResult {
	const baseTools = options.baseToolsOverride
		? options.baseToolsOverride
		: createAllTools(options.cwd, {
				read: { autoResizeImages: options.autoResizeImages },
				bash: { commandPrefix: options.shellCommandPrefix },
			});

	const baseToolRegistry = new Map(Object.entries(baseTools).map(([name, tool]) => [name, tool as AgentTool]));
	const extensionsResult = options.resourceLoader.getExtensions();

	if (options.flagValues) {
		for (const [name, value] of options.flagValues) {
			extensionsResult.runtime.flagValues.set(name, value);
		}
	}

	const hasExtensions = extensionsResult.extensions.length > 0;
	const hasCustomTools = options.customTools.length > 0;
	const extensionRunner =
		hasExtensions || hasCustomTools
			? new ExtensionRunner(
					extensionsResult.extensions,
					extensionsResult.runtime,
					options.cwd,
					options.sessionManager,
					options.modelRegistry,
				)
			: undefined;

	return {
		baseToolRegistry,
		extensionRunner,
		defaultActiveToolNames: options.baseToolsOverride
			? Object.keys(options.baseToolsOverride)
			: ["read", "bash", "edit", "write"],
	};
}

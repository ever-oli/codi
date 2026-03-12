/**
 * REPL Orchestrator Extension
 *
 * Implements the RLM (Reinforcement Learning with Language Models) paper's core
 * idea: give the model a JavaScript REPL where it can programmatically orchestrate
 * sub-LM calls, enabling "programming in action space."
 *
 * Instead of making individual tool calls turn-by-turn, the model writes code that
 * dynamically decides how many sub-calls to make, how to chunk data, how to branch
 * on intermediate results, and how to compose results — all in a single tool invocation.
 *
 * Key features:
 *   - Sandboxed execution via node:vm
 *   - Persistent REPL state across invocations
 *   - llm_query() / llm_query_batched() for direct LLM completions
 *   - rlm_query() / rlm_query_batched() for recursive agent calls
 *   - FINAL() sentinel for returning results
 *   - Streaming updates during execution
 *
 * Usage:
 *   pi --extension examples/extensions/repl-orchestrator.ts
 */

import { spawn } from "node:child_process";
import * as vm from "node:vm";
import { complete } from "@mariozechner/pi-ai";
import type { AgentToolResult, ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Container, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Types
// ============================================================================

/** Sentinel thrown by FINAL() to stop execution and return a value */
class FinalSentinel {
	constructor(public readonly value: any) {}
}

interface ReplResult {
	consoleOutput: string[];
	finalValue: any | undefined;
	hasFinal: boolean;
	variableChanges: Map<string, { before: string; after: string }>;
	error: string | undefined;
}

interface ReplDetails {
	code: string;
	result: ReplResult;
}

// ============================================================================
// Helpers
// ============================================================================

function extractAssistantText(event: any): string | undefined {
	if (event.message?.role !== "assistant" || !Array.isArray(event.message.content)) return undefined;
	const chunks = event.message.content
		.filter((block: any) => block.type === "text" && typeof block.text === "string")
		.map((block: any) => block.text as string);
	if (chunks.length === 0) return undefined;
	return chunks.join("\n").trim();
}

async function mapWithConcurrencyLimit<TIn, TOut>(
	items: TIn[],
	concurrency: number,
	fn: (item: TIn, index: number) => Promise<TOut>,
): Promise<TOut[]> {
	if (items.length === 0) return [];
	const limit = Math.max(1, Math.min(concurrency, items.length));
	const results: TOut[] = new Array(items.length);
	let nextIndex = 0;
	const workers = new Array(limit).fill(null).map(async () => {
		while (true) {
			const current = nextIndex++;
			if (current >= items.length) return;
			results[current] = await fn(items[current], current);
		}
	});
	await Promise.all(workers);
	return results;
}

function previewValue(value: any): string {
	if (value === undefined) return "undefined";
	if (value === null) return "null";
	const type = typeof value;
	if (type === "string") {
		return value.length > 80 ? `"${value.slice(0, 80)}..."` : `"${value}"`;
	}
	if (type === "number" || type === "boolean") return String(value);
	if (Array.isArray(value)) return `Array(${value.length})`;
	if (type === "object") {
		const keys = Object.keys(value);
		return `{${keys.slice(0, 3).join(", ")}${keys.length > 3 ? ", ..." : ""}}`;
	}
	return String(value).slice(0, 80);
}

function typeOf(value: any): string {
	if (value === null) return "null";
	if (value === undefined) return "undefined";
	if (Array.isArray(value)) return `Array(${value.length})`;
	if (value instanceof Map) return `Map(${value.size})`;
	if (value instanceof Set) return `Set(${value.size})`;
	return typeof value;
}

// ============================================================================
// RLM Query (agent subprocess)
// ============================================================================

async function runRlmQuery(
	prompt: string,
	model: string | undefined,
	cwd: string,
	timeoutSeconds: number,
): Promise<string> {
	const args: string[] = ["--mode", "json", "-p", "--no-session"];
	if (model?.trim()) args.unshift("--model", model.trim());
	args.push(prompt);

	let finalText = "";
	let stderr = "";
	let timedOut = false;

	const exitCode = await new Promise<number>((resolve) => {
		const proc = spawn("pi", args, {
			cwd,
			stdio: ["ignore", "pipe", "pipe"],
			shell: false,
		});
		let buffer = "";
		const timeout = setTimeout(() => {
			timedOut = true;
			proc.kill("SIGTERM");
		}, timeoutSeconds * 1000);

		const parseLine = (line: string) => {
			const trimmed = line.trim();
			if (!trimmed) return;
			let parsed: any;
			try {
				parsed = JSON.parse(trimmed);
			} catch {
				return;
			}
			if (parsed.type !== "message_end" || !parsed.message) return;
			if (parsed.message.role !== "assistant") return;
			const text = extractAssistantText(parsed);
			if (text) finalText = text;
		};

		proc.stdout.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) parseLine(line);
		});

		proc.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString();
		});

		proc.on("error", (_error: NodeJS.ErrnoException) => {
			clearTimeout(timeout);
			resolve(127);
		});

		proc.on("close", (code: number | null) => {
			clearTimeout(timeout);
			if (buffer.trim()) parseLine(buffer);
			resolve(code ?? 0);
		});
	});

	if (timedOut) return `(timed out after ${timeoutSeconds}s)`;
	if (exitCode !== 0 && !finalText) return `(rlm_query failed: exit ${exitCode} — ${stderr.slice(0, 200)})`;
	return finalText;
}

// ============================================================================
// Extension
// ============================================================================

export default function (pi: ExtensionAPI) {
	// Persistent REPL state across tool invocations
	const replVariables = new Map<string, any>();

	pi.registerTool({
		name: "repl_execute",
		label: "REPL Orchestrator",
		description: [
			"Execute JavaScript code in an orchestration REPL with access to LLM sub-calls.",
			"Has access to: llm_query(prompt, model?), llm_query_batched(prompts, model?, concurrency?),",
			"rlm_query(prompt, model?) (spawns a full agent), rlm_query_batched(prompts, model?, concurrency?),",
			"readVariable(name), setVariable(name, value), SHOW_VARS(), FINAL(value) (returns result and stops).",
			"Variables persist across invocations. Use for dynamic orchestration: chunking data, branching on results,",
			"composing sub-calls programmatically. Code runs in an async context (await works).",
		].join(" "),
		parameters: Type.Object({
			code: Type.String({
				description:
					"JavaScript code to execute in the orchestration REPL. Has access to llm_query(), llm_query_batched(), rlm_query(), readVariable(), setVariable(), FINAL(), SHOW_VARS(), and standard Node.js APIs.",
			}),
			resetState: Type.Optional(
				Type.Boolean({
					description: "Clear all REPL variables before execution. Default: false",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, onUpdate, ctx): Promise<AgentToolResult<ReplDetails>> {
			const { code, resetState } = params;

			if (resetState) {
				replVariables.clear();
			}

			// Snapshot variable state before execution
			const beforeSnapshot = new Map<string, string>();
			for (const [k, v] of replVariables) {
				beforeSnapshot.set(k, previewValue(v));
			}

			// Capture console output
			const consoleOutput: string[] = [];
			const capturedConsole = {
				log: (...args: any[]) => {
					const line = args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
					consoleOutput.push(line);
					if (onUpdate && consoleOutput.length % 3 === 0) {
						onUpdate({
							content: [
								{
									type: "text",
									text: `REPL executing... (${consoleOutput.length} lines output)`,
								},
							],
						});
					}
				},
				warn: (...args: any[]) => {
					const line = `[warn] ${args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ")}`;
					consoleOutput.push(line);
				},
				error: (...args: any[]) => {
					const line = `[error] ${args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ")}`;
					consoleOutput.push(line);
				},
				info: (...args: any[]) => {
					const line = args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
					consoleOutput.push(line);
				},
			};

			// Build injected functions
			const resolveModel = (modelId?: string) => {
				if (!modelId) return ctx.model;
				// Search all models for a matching ID
				const found = ctx.modelRegistry.getAll().find((m) => m.id === modelId);
				return found ?? ctx.model;
			};

			const llm_query = async (prompt: string, model?: string): Promise<string> => {
				const targetModel = resolveModel(model);
				if (!targetModel) return "(no model available)";
				const apiKey = await ctx.modelRegistry.getApiKey(targetModel);
				if (!apiKey) return "(no API key available)";
				try {
					const response = await complete(
						targetModel,
						{
							messages: [
								{
									role: "user" as const,
									content: [{ type: "text" as const, text: prompt }],
									timestamp: Date.now(),
								},
							],
						},
						{ apiKey, maxTokens: 4096 },
					);
					return response.content
						.filter((c): c is { type: "text"; text: string } => c.type === "text")
						.map((c) => c.text)
						.join("\n")
						.trim();
				} catch (error) {
					return `(llm_query error: ${error instanceof Error ? error.message : String(error)})`;
				}
			};

			const llm_query_batched = async (
				prompts: string[],
				model?: string,
				concurrency?: number,
			): Promise<string[]> => {
				return mapWithConcurrencyLimit(prompts, concurrency || 4, async (prompt) => llm_query(prompt, model));
			};

			const rlm_query = async (prompt: string, model?: string): Promise<string> => {
				return runRlmQuery(prompt, model, ctx.cwd, 240);
			};

			const rlm_query_batched = async (
				prompts: string[],
				model?: string,
				concurrency?: number,
			): Promise<string[]> => {
				return mapWithConcurrencyLimit(prompts, concurrency || 4, async (prompt) => rlm_query(prompt, model));
			};

			const readVariable = (name: string): any => {
				return replVariables.get(name);
			};

			const setVariable = (name: string, value: any): void => {
				replVariables.set(name, value);
			};

			const SHOW_VARS = (): Record<string, string> => {
				const result: Record<string, string> = {};
				for (const [k, v] of replVariables) {
					result[k] = `${typeOf(v)}: ${previewValue(v)}`;
				}
				return result;
			};

			let finalValue: any;
			let hasFinal = false;

			const FINAL = (value: any): void => {
				finalValue = value;
				hasFinal = true;
				throw new FinalSentinel(value);
			};

			// Create vm context with injected globals
			const sandbox: Record<string, any> = {
				console: capturedConsole,
				JSON,
				Math,
				Date,
				Array,
				Object,
				Map,
				Set,
				RegExp,
				Promise,
				setTimeout,
				parseInt,
				parseFloat,
				Buffer,
				TextEncoder,
				TextDecoder,
				llm_query,
				llm_query_batched,
				rlm_query,
				rlm_query_batched,
				readVariable,
				setVariable,
				SHOW_VARS,
				FINAL,
			};

			const context = vm.createContext(sandbox);

			// Wrap code in async IIFE so await works
			const wrappedCode = `(async () => { ${code} })()`;

			let errorMessage: string | undefined;

			if (onUpdate) {
				onUpdate({
					content: [{ type: "text", text: "REPL executing..." }],
				});
			}

			try {
				const result = vm.runInContext(wrappedCode, context, {
					timeout: 30000,
					filename: "repl-orchestrator",
				});
				// The result is a Promise from the async IIFE — await it
				await result;
			} catch (error) {
				if (error instanceof FinalSentinel) {
					// Expected — FINAL() was called
				} else if (error instanceof Error) {
					if (error.message?.includes("Script execution timed out")) {
						errorMessage = "REPL execution timed out (30s limit)";
					} else {
						errorMessage = `${error.name}: ${error.message}`;
					}
				} else {
					errorMessage = String(error);
				}
			}

			// Compute variable changes
			const variableChanges = new Map<string, { before: string; after: string }>();
			const allKeys = new Set([...beforeSnapshot.keys(), ...replVariables.keys()]);
			for (const key of allKeys) {
				const before = beforeSnapshot.get(key) ?? "(unset)";
				const after = replVariables.has(key) ? previewValue(replVariables.get(key)) : "(deleted)";
				if (before !== after) {
					variableChanges.set(key, { before, after });
				}
			}

			const replResult: ReplResult = {
				consoleOutput,
				finalValue,
				hasFinal,
				variableChanges,
				error: errorMessage,
			};

			const details: ReplDetails = { code, result: replResult };

			// Build text response
			const lines: string[] = [];

			if (errorMessage) {
				lines.push(`**Error:** ${errorMessage}`);
				lines.push("");
			}

			if (hasFinal) {
				const finalStr = typeof finalValue === "string" ? finalValue : JSON.stringify(finalValue, null, 2);
				lines.push("**Final Result:**");
				lines.push(finalStr);
				lines.push("");
			}

			if (consoleOutput.length > 0) {
				lines.push("**Console Output:**");
				lines.push("```");
				lines.push(...consoleOutput.slice(-50));
				if (consoleOutput.length > 50) {
					lines.push(`... (${consoleOutput.length - 50} earlier lines omitted)`);
				}
				lines.push("```");
				lines.push("");
			}

			if (variableChanges.size > 0) {
				lines.push("**Variable Changes:**");
				for (const [key, { before, after }] of variableChanges) {
					lines.push(`- \`${key}\`: ${before} → ${after}`);
				}
				lines.push("");
			}

			if (lines.length === 0) {
				lines.push("(no output, no errors, no variable changes)");
			}

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details,
				isError: !!errorMessage && !hasFinal,
			};
		},

		renderCall(args, theme) {
			const codeLines = args.code.split("\n");
			const preview = codeLines
				.slice(0, 5)
				.map((l: string) => l.slice(0, 80))
				.join("\n");
			const more = codeLines.length > 5 ? `\n${theme.fg("dim", `... +${codeLines.length - 5} lines`)}` : "";
			let text = theme.fg("toolTitle", theme.bold("repl_execute"));
			if (args.resetState) text += theme.fg("warning", " (reset)");
			text += `\n${theme.fg("muted", preview)}${more}`;
			return new Text(text, 0, 0);
		},

		renderResult(result, { expanded }, theme) {
			const details = result.details as ReplDetails | undefined;
			if (!details) {
				const block = result.content[0];
				return new Text(block?.type === "text" ? block.text : "(no output)", 0, 0);
			}

			const r = details.result;
			const hasError = !!r.error;
			const icon = hasError && !r.hasFinal ? theme.fg("error", "✗") : theme.fg("success", "✓");

			if (!expanded) {
				const lines: string[] = [];
				lines.push(
					`${icon} ${theme.fg("toolTitle", theme.bold("repl"))}${r.hasFinal ? theme.fg("accent", " → FINAL") : ""} ${theme.fg("dim", `${r.consoleOutput.length} lines`)}`,
				);
				if (r.hasFinal) {
					lines.push(theme.fg("muted", previewValue(r.finalValue).slice(0, 100)));
				} else if (r.consoleOutput.length > 0) {
					lines.push(theme.fg("muted", r.consoleOutput[r.consoleOutput.length - 1].slice(0, 100)));
				}
				if (r.error && !r.hasFinal) {
					lines.push(theme.fg("error", r.error.slice(0, 100)));
				}
				if (r.variableChanges.size > 0) {
					lines.push(theme.fg("dim", `${r.variableChanges.size} var(s) changed`));
				}
				return new Text(lines.join("\n"), 0, 0);
			}

			// Expanded view
			const container = new Container();
			container.addChild(new Text(`${icon} ${theme.fg("toolTitle", theme.bold("repl_execute"))}`, 0, 0));

			if (r.error) {
				container.addChild(new Text(theme.fg("error", `Error: ${r.error}`), 0, 0));
			}

			if (r.hasFinal) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("accent", "─── Final Result ───"), 0, 0));
				const finalStr = typeof r.finalValue === "string" ? r.finalValue : JSON.stringify(r.finalValue, null, 2);
				container.addChild(new Text(finalStr.slice(0, 500), 0, 0));
			}

			if (r.consoleOutput.length > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", `─── Console (${r.consoleOutput.length} lines) ───`), 0, 0));
				for (const line of r.consoleOutput.slice(-20)) {
					container.addChild(new Text(theme.fg("dim", line), 0, 0));
				}
				if (r.consoleOutput.length > 20) {
					container.addChild(
						new Text(theme.fg("dim", `... ${r.consoleOutput.length - 20} earlier lines omitted`), 0, 0),
					);
				}
			}

			if (r.variableChanges.size > 0) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Variable Changes ───"), 0, 0));
				for (const [key, { before, after }] of r.variableChanges) {
					container.addChild(
						new Text(`  ${theme.fg("accent", key)}: ${theme.fg("dim", before)} → ${after}`, 0, 0),
					);
				}
			}

			return container;
		},
	});

	// Register /repl-vars command
	pi.registerCommand("repl-vars", {
		description: "Show all current REPL variables and their values",
		handler: async (_args, ctx) => {
			if (replVariables.size === 0) {
				ctx.ui.notify("No REPL variables set.", "info");
				return;
			}

			const lines: string[] = ["## REPL Variables\n"];
			for (const [key, value] of replVariables) {
				lines.push(`- **${key}** (${typeOf(value)}): ${previewValue(value)}`);
			}
			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}

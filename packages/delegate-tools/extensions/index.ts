import { spawn } from "node:child_process";
import { StringEnum, Text, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

interface DelegateUsage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cost: number;
	turns: number;
}

interface DelegateDetails {
	mode: "single";
	task: string;
	cwd: string;
	model?: string;
	toolFilter?: string[];
	durationMs: number;
	exitCode: number;
	stopReason?: string;
	errorMessage?: string;
	usage: DelegateUsage;
	stderr?: string;
}

const DELEGATE_PARAMS = Type.Object({
	task: Type.String({ description: "Task to delegate to a nested pi run." }),
	cwd: Type.Optional(Type.String({ description: "Working directory for the delegated run. Defaults to current cwd." })),
	model: Type.Optional(Type.String({ description: "Optional model override for delegated run." })),
	tools: Type.Optional(Type.Array(Type.String(), { description: "Optional allow-list of tool names." })),
	timeoutSeconds: Type.Optional(Type.Number({ description: "Timeout for delegated run. Default: 240 seconds." })),
	outputMode: Type.Optional(StringEnum(["final", "summary"] as const)),
});

const DEFAULT_TIMEOUT_SECONDS = 240;
const MAX_TIMEOUT_SECONDS = 1800;

interface StreamEvent {
	type?: string;
	message?: {
		role?: string;
		model?: string;
		stopReason?: string;
		errorMessage?: string;
		usage?: {
			input?: number;
			output?: number;
			cacheRead?: number;
			cacheWrite?: number;
			cost?: { total?: number };
		};
		content?: Array<{ type: string; text?: string }>;
	};
}

function clampTimeout(seconds: number | undefined): number {
	if (!seconds || Number.isNaN(seconds) || seconds <= 0) return DEFAULT_TIMEOUT_SECONDS;
	return Math.min(seconds, MAX_TIMEOUT_SECONDS);
}

function extractAssistantText(event: StreamEvent): string | undefined {
	if (event.message?.role !== "assistant" || !Array.isArray(event.message.content)) return undefined;
	const chunks = event.message.content
		.filter((block) => block.type === "text" && typeof block.text === "string")
		.map((block) => block.text as string);
	if (chunks.length === 0) return undefined;
	return chunks.join("\n").trim();
}

function summarizeText(text: string): string {
	const normalized = text.replace(/\s+/g, " ").trim();
	if (normalized.length <= 500) return normalized;
	return `${normalized.slice(0, 500)}...`;
}

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool<typeof DELEGATE_PARAMS, DelegateDetails>({
			name: "delegate",
			label: "Delegate",
			description:
				"Delegate a bounded task to a nested pi subprocess with isolated context, then return the delegated result summary.",
			promptSnippet:
				"Run focused delegated work in a nested pi subprocess when isolation is helpful or when parallel execution is desired.",
			promptGuidelines: [
				"Delegate only well-scoped tasks with clear expected output.",
				"Use tool filters and timeout to keep delegated runs bounded.",
			],
			parameters: DELEGATE_PARAMS,
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const started = Date.now();
				const timeoutSeconds = clampTimeout(params.timeoutSeconds);
				const args: string[] = ["--mode", "json", "-p", "--no-session", params.task];
				if (params.model?.trim()) args.unshift("--model", params.model.trim());
				const toolFilter = params.tools?.filter((tool) => tool.trim().length > 0);
				if (toolFilter && toolFilter.length > 0) {
					args.push("--tools", toolFilter.join(","));
				}

				const delegatedCwd = params.cwd?.trim() || ctx.cwd;
				const usage: DelegateUsage = {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					cost: 0,
					turns: 0,
				};
				let finalText = "";
				let model = params.model?.trim();
				let stopReason: string | undefined;
				let errorMessage: string | undefined;
				let stderr = "";
				let timedOut = false;

				const exitCode = await new Promise<number>((resolve) => {
					const proc = spawn("pi", args, {
						cwd: delegatedCwd,
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
						let parsed: StreamEvent | undefined;
						try {
							parsed = JSON.parse(trimmed) as StreamEvent;
						} catch {
							return;
						}
						if (parsed.type !== "message_end" || !parsed.message) return;
						if (parsed.message.role !== "assistant") return;
						usage.turns += 1;
						usage.input += parsed.message.usage?.input ?? 0;
						usage.output += parsed.message.usage?.output ?? 0;
						usage.cacheRead += parsed.message.usage?.cacheRead ?? 0;
						usage.cacheWrite += parsed.message.usage?.cacheWrite ?? 0;
						usage.cost += parsed.message.usage?.cost?.total ?? 0;
						model = model ?? parsed.message.model;
						stopReason = parsed.message.stopReason ?? stopReason;
						errorMessage = parsed.message.errorMessage ?? errorMessage;
						const text = extractAssistantText(parsed);
						if (text) finalText = text;
					};

					proc.stdout.on("data", (chunk) => {
						buffer += chunk.toString();
						const lines = buffer.split("\n");
						buffer = lines.pop() ?? "";
						for (const line of lines) parseLine(line);
					});

					proc.stderr.on("data", (chunk) => {
						stderr += chunk.toString();
					});

					proc.on("error", (error: NodeJS.ErrnoException) => {
						clearTimeout(timeout);
						errorMessage = error.code === "ENOENT" ? "pi executable not found in PATH." : error.message;
						resolve(127);
					});

					proc.on("close", (code) => {
						clearTimeout(timeout);
						if (buffer.trim()) parseLine(buffer);
						resolve(code ?? 0);
					});
				});

				if (timedOut) {
					errorMessage = `Delegated run timed out after ${timeoutSeconds}s.`;
				}

				const durationMs = Date.now() - started;
				const details: DelegateDetails = {
					mode: "single",
					task: params.task,
					cwd: delegatedCwd,
					model,
					toolFilter,
					durationMs,
					exitCode,
					stopReason,
					errorMessage,
					usage,
					stderr: stderr.trim() || undefined,
				};

				if (exitCode !== 0 || errorMessage) {
					const message = errorMessage ?? (stderr.trim() || `Delegate failed with exit code ${exitCode}.`);
					return {
						content: [{ type: "text", text: `Delegate error: ${message}` }],
						details,
						isError: true,
					};
				}

				const outputMode = params.outputMode ?? "final";
				const output = outputMode === "summary" ? summarizeText(finalText || "(no output)") : finalText || "(no output)";
				return {
					content: [{ type: "text", text: output }],
					details,
				};
			},
			renderCall(args, theme) {
				const taskPreview = args.task.length > 72 ? `${args.task.slice(0, 72)}...` : args.task;
				let text = theme.fg("toolTitle", theme.bold("delegate ")) + theme.fg("muted", `"${taskPreview}"`);
				if (args.model) text += ` ${theme.fg("accent", `model=${args.model}`)}`;
				if (args.timeoutSeconds) text += ` ${theme.fg("dim", `${clampTimeout(args.timeoutSeconds)}s`)}`;
				return new Text(text, 0, 0);
			},
			renderResult(result, _options, theme) {
				const details = result.details as DelegateDetails | undefined;
				if (!details) {
					const block = result.content[0];
					return new Text(block?.type === "text" ? block.text : "", 0, 0);
				}
				if (details.exitCode !== 0 || Boolean(details.errorMessage)) {
					const lines = [theme.fg("error", `Delegate failed (exit ${details.exitCode})`)];
					if (details.errorMessage) lines.push(theme.fg("error", details.errorMessage));
					if (details.stderr) lines.push(theme.fg("dim", details.stderr.slice(0, 160)));
					return new Text(lines.join("\n"), 0, 0);
				}
				const lines = [
					`${theme.fg("success", "✓")} ${theme.fg("muted", `delegate done in ${(details.durationMs / 1000).toFixed(1)}s`)}`,
					theme.fg(
						"dim",
						`turns ${details.usage.turns} | input ${details.usage.input} | output ${details.usage.output} | cost $${details.usage.cost.toFixed(4)}`,
					),
				];
				return new Text(lines.join("\n"), 0, 0);
			},
		}),
	);
});

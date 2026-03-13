/**
 * HF-Autoresearch Extension
 *
 * HuggingFace-native autoresearch: fine-tune any HF model on any HF dataset
 * with autonomous hyperparameter loops. The LLM agent reasons about what to
 * try next (not traditional HPO).
 *
 * Tools:
 *   - scaffold_experiment: Set up a new experiment (model + dataset + task)
 *   - run_experiment: Execute a training run with specific hyperparams
 *   - log_experiment: Record results (keep/discard/crash) with metrics
 *
 * UI:
 *   - Widget: run count, best metric, delta %
 *   - Ctrl+Shift+X: expanded dashboard overlay
 *   - /autoresearch: fullscreen dashboard command
 *
 * State: JSONL persistence in project dir (hf-autoresearch.jsonl)
 * Storage: Models/datasets/runs on /Volumes/Expansion/hf-autoresearch/
 *
 * Usage:
 *   pi -e packages/hf-autoresearch/extensions/index.ts
 *   # or place in ~/.pi/agent/extensions/
 */

import { spawn, execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentToolResult, ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { StringEnum } from "@mariozechner/pi-ai";
import { Container, Spacer, Text, truncateToWidth, matchesKey } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Constants
// ============================================================================

const STORAGE_ROOT = "/Volumes/Expansion/hf-autoresearch";
const STATE_FILE = "hf-autoresearch.jsonl";

// Resolve template dir robustly — works with jiti, ESM, and CJS
function resolveTemplateDir(): string {
	// Try import.meta based paths
	try {
		if (import.meta.dirname) return path.resolve(import.meta.dirname, "../templates");
		if (import.meta.url) {
			const dir = path.dirname(fileURLToPath(import.meta.url));
			return path.resolve(dir, "../templates");
		}
	} catch { /* fall through */ }

	// Try __dirname (CJS / jiti)
	try {
		if (typeof __dirname !== "undefined") return path.resolve(__dirname, "../templates");
	} catch { /* fall through */ }

	// Last resort: search known package location
	const knownPath = "/Users/ever/Documents/GitHub/Codi /packages/hf-autoresearch/templates";
	if (fs.existsSync(knownPath)) return knownPath;

	return ".";
}

const TEMPLATE_DIR = resolveTemplateDir();

// ============================================================================
// Python runtime detection
// ============================================================================

let _uvPath: string | null | undefined = undefined;

function findUv(): string | null {
	if (_uvPath !== undefined) return _uvPath;
	try {
		_uvPath = execSync("which uv", { encoding: "utf-8" }).trim();
	} catch {
		_uvPath = null;
	}
	return _uvPath;
}

/**
 * Build spawn command for running train.py.
 * Prefers `uv run` (auto-installs deps via PEP 723 inline metadata).
 * Falls back to plain `python3` if uv isn't available.
 */
function buildTrainCommand(trainPyPath: string, trainArgs: string[]): { cmd: string; args: string[] } {
	const uv = findUv();
	if (uv) {
		// uv run --script: args go directly after script path (no -- separator needed)
		return { cmd: uv, args: ["run", "--script", trainPyPath, ...trainArgs] };
	}
	return { cmd: "python3", args: [trainPyPath, ...trainArgs] };
}

// ============================================================================
// Types
// ============================================================================

interface ExperimentConfig {
	id: string;
	modelName: string;
	datasetName: string;
	datasetConfig?: string;
	task: string;
	metricName: string;
	metricDirection: "higher" | "lower";
	textColumn: string;
	textColumn2?: string;
	labelColumn: string;
	numLabels?: number;
	maxLength: number;
	useLora: boolean;
	loraR?: number;
	loraAlpha?: number;
	loraDropout?: number;
	useQlora: boolean;
	createdAt: number;
	trainPyPath: string;
}

interface ExperimentRun {
	id: string;
	experimentId: string;
	hyperparams: Record<string, string | number>;
	metrics: Record<string, number>;
	primaryMetric: number | null;
	status: "keep" | "discard" | "crash" | "running";
	description: string;
	startedAt: number;
	finishedAt?: number;
	outputDir: string;
	logFile?: string;
}

interface StateEntry {
	type: "config" | "run";
	data: ExperimentConfig | ExperimentRun;
	timestamp: number;
}

interface AutoresearchState {
	configs: Map<string, ExperimentConfig>;
	runs: ExperimentRun[];
}

// ============================================================================
// State Management
// ============================================================================

function getStateFilePath(cwd: string): string {
	return path.join(cwd, STATE_FILE);
}

function loadState(cwd: string): AutoresearchState {
	const state: AutoresearchState = { configs: new Map(), runs: [] };
	const filePath = getStateFilePath(cwd);

	if (!fs.existsSync(filePath)) return state;

	const lines = fs.readFileSync(filePath, "utf-8").split("\n").filter((l) => l.trim());
	for (const line of lines) {
		try {
			const entry: StateEntry = JSON.parse(line);
			if (entry.type === "config") {
				const config = entry.data as ExperimentConfig;
				state.configs.set(config.id, config);
			} else if (entry.type === "run") {
				const run = entry.data as ExperimentRun;
				const idx = state.runs.findIndex((r) => r.id === run.id);
				if (idx >= 0) state.runs[idx] = run;
				else state.runs.push(run);
			}
		} catch {
			// Skip malformed lines
		}
	}
	return state;
}

function appendState(cwd: string, entry: StateEntry): void {
	const filePath = getStateFilePath(cwd);
	fs.appendFileSync(filePath, JSON.stringify(entry) + "\n");
}

function generateId(prefix: string): string {
	return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

// ============================================================================
// Metric helpers
// ============================================================================

function parseMetricLines(output: string): Record<string, number> {
	const metrics: Record<string, number> = {};
	const regex = /^METRIC\s+(\S+)=(\S+)/gm;
	let match: RegExpExecArray | null;
	while ((match = regex.exec(output)) !== null) {
		const val = parseFloat(match[2]);
		if (!isNaN(val)) metrics[match[1]] = val;
	}
	return metrics;
}

function getBestRun(state: AutoresearchState, experimentId: string): ExperimentRun | undefined {
	const config = state.configs.get(experimentId);
	if (!config) return undefined;

	const keptRuns = state.runs.filter(
		(r) => r.experimentId === experimentId && r.status === "keep" && r.primaryMetric !== null,
	);

	if (keptRuns.length === 0) return undefined;

	return keptRuns.reduce((best, run) => {
		if (best.primaryMetric === null) return run;
		if (run.primaryMetric === null) return best;
		if (config.metricDirection === "higher") {
			return run.primaryMetric > best.primaryMetric ? run : best;
		}
		return run.primaryMetric < best.primaryMetric ? run : best;
	});
}

function formatMetric(value: number | null): string {
	if (value === null) return "—";
	if (Math.abs(value) < 0.01 || Math.abs(value) > 9999) return value.toExponential(3);
	return value.toFixed(4);
}

function computeDelta(current: number | null, best: number | null, direction: "higher" | "lower"): string {
	if (current === null || best === null) return "";
	const diff = current - best;
	const pct = best !== 0 ? (diff / Math.abs(best)) * 100 : 0;
	const sign = diff >= 0 ? "+" : "";
	const isGood = direction === "higher" ? diff > 0 : diff < 0;
	return `${sign}${pct.toFixed(1)}%${isGood ? " ▲" : diff === 0 ? " =" : " ▼"}`;
}

// ============================================================================
// Context summary for agent
// ============================================================================

function buildExperimentContext(state: AutoresearchState): string | null {
	const configs = Array.from(state.configs.values());
	if (configs.length === 0 && state.runs.length === 0) return null;

	const lines: string[] = ["## HF Autoresearch State\n"];

	for (const config of configs) {
		const runs = state.runs.filter((r) => r.experimentId === config.id);
		const keptRuns = runs.filter((r) => r.status === "keep");
		const bestRun = getBestRun(state, config.id);
		const runningRuns = runs.filter((r) => r.status === "running");

		lines.push(`### Experiment: ${config.id}`);
		lines.push(`- Model: ${config.modelName} → Dataset: ${config.datasetName} (${config.task})`);
		lines.push(`- Metric: ${config.metricName} (${config.metricDirection} is better)`);
		lines.push(`- Mode: ${config.useLora ? `LoRA (r=${config.loraR}${config.useQlora ? ", QLoRA 4-bit" : ""})` : "full fine-tuning"}`);
		lines.push(`- Runs: ${runs.length} total, ${keptRuns.length} kept`);

		if (bestRun) {
			lines.push(`- **Best:** ${config.metricName}=${formatMetric(bestRun.primaryMetric)} (${bestRun.id})`);
			const hp = Object.entries(bestRun.hyperparams).map(([k, v]) => `${k}=${v}`).join(", ");
			lines.push(`  Hyperparams: ${hp}`);
		}

		if (runningRuns.length > 0) {
			lines.push(`- ⚠️ ${runningRuns.length} run(s) still marked as "running" — may need to be logged as crash`);
		}

		// Show last few kept runs for context
		if (keptRuns.length > 0) {
			lines.push("\nKept runs (most recent first):");
			for (const run of keptRuns.slice(-5).reverse()) {
				const hp = Object.entries(run.hyperparams).map(([k, v]) => `${k}=${v}`).join(", ");
				const delta = bestRun ? computeDelta(run.primaryMetric, bestRun.primaryMetric, config.metricDirection) : "";
				lines.push(`  - ${run.id}: ${config.metricName}=${formatMetric(run.primaryMetric)}${delta ? ` (${delta})` : ""} | ${hp}`);
			}
		}

		lines.push("");
	}

	return lines.join("\n");
}

// ============================================================================
// Extension
// ============================================================================

export default function (pi: ExtensionAPI) {
	let state: AutoresearchState = { configs: new Map(), runs: [] };
	let currentCwd = process.cwd();

	const reloadState = (ctx: ExtensionContext) => {
		currentCwd = ctx.cwd;
		state = loadState(currentCwd);
		updateWidget(ctx);
	};

	// ── Session lifecycle ──────────────────────────────────────────────

	pi.on("session_start", async (_event, ctx) => {
		reloadState(ctx);
		await recoverStaleRuns(ctx);
	});
	pi.on("session_switch", async (_event, ctx) => {
		reloadState(ctx);
		await recoverStaleRuns(ctx);
	});
	pi.on("session_fork", async (_event, ctx) => reloadState(ctx));

	// Detect runs stuck in "running" from a previous session (likely crashed)
	const recoverStaleRuns = async (ctx: ExtensionContext) => {
		const staleRuns = state.runs.filter((r) => r.status === "running");
		if (staleRuns.length === 0) return;

		for (const run of staleRuns) {
			run.status = "crash";
			run.description += " [stale — recovered on session start]";
			run.finishedAt = run.finishedAt || Date.now();
			appendState(ctx.cwd, { type: "run", data: run, timestamp: Date.now() });
		}
		updateWidget(ctx);

		if (ctx.hasUI) {
			ctx.ui.notify(
				`Recovered ${staleRuns.length} stale run(s) from previous session (marked as crash).`,
				"warning",
			);
		}
	};

	// ── Context injection: tell the agent about existing experiments ───

	pi.on("before_agent_start", async (_event, _ctx) => {
		const freshState = loadState(currentCwd);
		const context = buildExperimentContext(freshState);
		if (!context) return;

		return {
			systemPrompt: _event.systemPrompt + "\n\n" + context,
		};
	});

	// ── Widget ─────────────────────────────────────────────────────────

	const updateWidget = (ctx: ExtensionContext) => {
		if (!ctx.hasUI) return;

		const totalRuns = state.runs.length;
		const keptRuns = state.runs.filter((r) => r.status === "keep").length;
		const runningRuns = state.runs.filter((r) => r.status === "running").length;
		const configs = Array.from(state.configs.values());

		if (configs.length === 0 && totalRuns === 0) {
			ctx.ui.setWidget("hf-autoresearch", undefined);
			return;
		}

		ctx.ui.setWidget("hf-autoresearch", (_tui, theme) => {
			const lines: string[] = [];

			let header =
				theme.fg("accent", theme.bold("🔬 autoresearch")) +
				theme.fg("dim", ` ${keptRuns}/${totalRuns} kept`);
			if (runningRuns > 0) {
				header += theme.fg("warning", ` ⏳${runningRuns} running`);
			}
			lines.push(header);

			for (const config of configs) {
				const best = getBestRun(state, config.id);
				const bestVal = best?.primaryMetric ?? null;
				const label = `${config.modelName.split("/").pop()} → ${config.datasetName.split("/").pop()}`;
				let line = theme.fg("muted", `  ${label}: `);
				line += theme.fg(bestVal !== null ? "success" : "dim", `${config.metricName}=${formatMetric(bestVal)}`);
				lines.push(line);
			}

			return new Text(lines.join("\n"), 0, 0);
		});
	};

	// ── Tool: scaffold_experiment ──────────────────────────────────────

	pi.registerTool({
		name: "scaffold_experiment",
		label: "Scaffold Experiment",
		description: [
			"Set up a new HF autoresearch experiment. Provide a HuggingFace model name, dataset name,",
			"and task type. Copies train.py template into the project and creates the experiment config.",
			"Returns the experiment ID and train.py path for subsequent run_experiment calls.",
			"The train.py can be customized before running (edit column names, add preprocessing, etc.).",
		].join(" "),
		promptSnippet: "Set up HF fine-tuning experiment (model + dataset + task → train.py + config)",
		promptGuidelines: [
			"Use scaffold_experiment when the user wants to fine-tune a HuggingFace model on a dataset.",
			"Always inspect the model card and dataset card first to determine task, columns, and labels.",
			"After scaffolding, use run_experiment to execute training runs with different hyperparams.",
			"For NLI/paraphrase datasets with two text columns, set both textColumn and textColumn2.",
		],
		parameters: Type.Object({
			modelName: Type.String({ description: "HuggingFace model name, e.g. 'bert-base-uncased'" }),
			datasetName: Type.String({ description: "HuggingFace dataset name, e.g. 'imdb'" }),
			datasetConfig: Type.Optional(Type.String({ description: "Dataset config/subset name if needed" })),
			task: StringEnum([
				"text-classification",
				"token-classification",
				"question-answering",
				"summarization",
				"translation",
				"causal-lm",
			] as const),
			metricName: Type.Optional(Type.String({ description: "Primary metric to optimize, e.g. 'eval_accuracy'. Default: 'eval_loss'" })),
			metricDirection: Type.Optional(StringEnum(["higher", "lower"] as const)),
			textColumn: Type.Optional(Type.String({ description: "Primary text column name. Default: 'text'" })),
			textColumn2: Type.Optional(Type.String({ description: "Second text column for NLI/paraphrase tasks (e.g. 'sentence2')" })),
			labelColumn: Type.Optional(Type.String({ description: "Label column name. Default: 'label'" })),
			numLabels: Type.Optional(Type.Number({ description: "Number of labels for classification tasks" })),
			maxLength: Type.Optional(Type.Number({ description: "Max sequence length. Default: 512" })),
			useLora: Type.Optional(Type.Boolean({ description: "Use LoRA (PEFT) for parameter-efficient fine-tuning. Recommended for models >1B params." })),
			loraR: Type.Optional(Type.Number({ description: "LoRA rank. Default: 16" })),
			loraAlpha: Type.Optional(Type.Number({ description: "LoRA alpha. Default: 32" })),
			loraDropout: Type.Optional(Type.Number({ description: "LoRA dropout. Default: 0.05" })),
			useQlora: Type.Optional(Type.Boolean({ description: "Use QLoRA (4-bit quantization + LoRA). Linux/CUDA only. Implies useLora." })),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const experimentId = generateId("exp");
			const metricName = params.metricName || "eval_loss";
			const metricDirection = params.metricDirection || (metricName.includes("loss") ? "lower" : "higher");

			// Copy train.py template into project
			const trainPyDest = path.join(ctx.cwd, `train_${experimentId}.py`);
			const templatePath = path.join(TEMPLATE_DIR, "train.py");

			if (fs.existsSync(templatePath)) {
				fs.copyFileSync(templatePath, trainPyDest);
			} else {
				return {
					content: [{ type: "text", text: `Error: train.py template not found at ${templatePath}.\nSearched template dir: ${TEMPLATE_DIR}\nEnsure the hf-autoresearch package is properly installed.` }],
				};
			}

			// Check runtime availability
			const uv = findUv();
			const runtimeNote = uv
				? `Runtime: uv (${uv}) — deps will auto-install on first run`
				: "⚠️ uv not found — train.py requires manual pip install of deps (torch, transformers, datasets, evaluate)";

			const config: ExperimentConfig = {
				id: experimentId,
				modelName: params.modelName,
				datasetName: params.datasetName,
				datasetConfig: params.datasetConfig,
				task: params.task,
				metricName,
				metricDirection,
				textColumn: params.textColumn || "text",
				textColumn2: params.textColumn2,
				labelColumn: params.labelColumn || "label",
				numLabels: params.numLabels,
				maxLength: params.maxLength || 512,
				useLora: params.useLora || params.useQlora || false,
				loraR: params.loraR || 16,
				loraAlpha: params.loraAlpha || 32,
				loraDropout: params.loraDropout || 0.05,
				useQlora: params.useQlora || false,
				createdAt: Date.now(),
				trainPyPath: trainPyDest,
			};

			state.configs.set(config.id, config);
			appendState(ctx.cwd, { type: "config", data: config, timestamp: Date.now() });
			updateWidget(ctx);

			// Ensure storage dirs exist
			const dirs = ["models", "datasets", "runs", "results"].map((d) => path.join(STORAGE_ROOT, d));
			for (const d of dirs) {
				try { fs.mkdirSync(d, { recursive: true }); } catch { /* external drive may be unavailable */ }
			}

			const storageOk = fs.existsSync(path.join(STORAGE_ROOT, "runs"));

			const summary = [
				`Experiment scaffolded: ${experimentId}`,
				`  Model: ${config.modelName}`,
				`  Dataset: ${config.datasetName}${config.datasetConfig ? ` (${config.datasetConfig})` : ""}`,
				`  Task: ${config.task}`,
				`  Metric: ${config.metricName} (${config.metricDirection} is better)`,
				`  Text column: ${config.textColumn}${config.textColumn2 ? ` + ${config.textColumn2}` : ""}`,
				`  Label column: ${config.labelColumn}`,
				config.numLabels ? `  Num labels: ${config.numLabels}` : "",
				`  Max length: ${config.maxLength}`,
				config.useLora ? `  LoRA: r=${config.loraR} alpha=${config.loraAlpha} dropout=${config.loraDropout}${config.useQlora ? " (QLoRA 4-bit)" : ""}` : "  Mode: full fine-tuning",
				`  train.py: ${trainPyDest}`,
				`  ${runtimeNote}`,
				storageOk ? `  Storage: ${STORAGE_ROOT} ✓` : `  ⚠️ Storage ${STORAGE_ROOT} not accessible — external drive mounted?`,
				"",
				"Next: use run_experiment with this experiment ID to start training.",
				config.useLora
					? "Tip: LoRA enabled. Default LR for LoRA is higher (1e-4 to 3e-4). Adjust via hyperparams."
					: "Tip: you can edit train.py for custom preprocessing before the first run.",
			]
				.filter(Boolean)
				.join("\n");

			return {
				content: [{ type: "text", text: summary }],
				details: { config },
			};
		},

		renderCall(args, theme) {
			const model = (args.modelName ?? "").split("/").pop() ?? args.modelName;
			const dataset = (args.datasetName ?? "").split("/").pop() ?? args.datasetName;
			return new Text(
				theme.fg("toolTitle", theme.bold("scaffold ")) +
					theme.fg("accent", `${model}`) +
					theme.fg("dim", " → ") +
					theme.fg("accent", `${dataset}`) +
					theme.fg("muted", ` (${args.task ?? "?"})`),
				0,
				0,
			);
		},

		renderResult(result, _options, theme) {
			const config = (result.details as any)?.config as ExperimentConfig | undefined;
			if (!config) {
				const text = result.content[0];
				return new Text(text?.type === "text" ? text.text : "", 0, 0);
			}
			return new Text(
				theme.fg("success", "✓ ") +
					theme.fg("accent", config.id) +
					theme.fg("dim", ` ${config.modelName} → ${config.datasetName}`),
				0,
				0,
			);
		},
	});

	// ── Tool: run_experiment ───────────────────────────────────────────

	pi.registerTool({
		name: "run_experiment",
		label: "Run Experiment",
		description: [
			"Execute a training run for a scaffolded experiment. Pass the experiment ID and hyperparameters.",
			"Uses `uv run` (auto-installs Python deps) to run train.py with the given hyperparams.",
			"Captures METRIC lines from stdout and streams progress updates.",
			"Returns run ID and parsed metrics. The run starts with status 'running' —",
			"use log_experiment afterward to record the verdict (keep/discard/crash).",
		].join(" "),
		promptSnippet: "Execute HF training run with specific hyperparams, returns metrics",
		promptGuidelines: [
			"After run_experiment completes, analyze the metrics and use log_experiment to record keep/discard.",
			"Then reason about what hyperparams to try next based on the results so far.",
			"Common hyperparams: lr (float), batch_size (int), epochs (int), warmup_ratio (float), weight_decay (float).",
			"For LoRA experiments, typical LR is 1e-4 to 3e-4 (higher than full fine-tuning).",
			"Use max_train_samples/max_eval_samples for quick smoke tests before full runs.",
			"Use early_stopping_patience (e.g. 3) to stop when metric plateaus.",
		],
		parameters: Type.Object({
			experimentId: Type.String({ description: "Experiment ID from scaffold_experiment" }),
			hyperparams: Type.Optional(
				Type.Record(Type.String(), Type.Union([Type.String(), Type.Number()]), {
					description: "Hyperparameters: lr, batch_size, epochs, warmup_ratio, weight_decay, etc.",
				}),
			),
			description: Type.Optional(Type.String({ description: "Description of what this run tests" })),
			timeoutSeconds: Type.Optional(Type.Number({ description: "Timeout in seconds. Default: 3600" })),
		}),

		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			const config = state.configs.get(params.experimentId);
			if (!config) {
				return {
					content: [{ type: "text", text: `Error: experiment ${params.experimentId} not found. Run scaffold_experiment first.` }],
				};
			}

			// Verify train.py exists
			if (!fs.existsSync(config.trainPyPath)) {
				return {
					content: [{ type: "text", text: `Error: train.py not found at ${config.trainPyPath}. Was it deleted? Re-run scaffold_experiment.` }],
				};
			}

			const runId = generateId("run");
			const hp = params.hyperparams || {};
			const outputDir = path.join(STORAGE_ROOT, "runs", runId);
			const logFile = path.join(outputDir, "train.log");

			try {
				fs.mkdirSync(outputDir, { recursive: true });
			} catch (err: any) {
				return {
					content: [{ type: "text", text: `Error: cannot create output dir ${outputDir}. Is /Volumes/Expansion mounted?\n${err.message}` }],
				};
			}

			const run: ExperimentRun = {
				id: runId,
				experimentId: params.experimentId,
				hyperparams: hp,
				metrics: {},
				primaryMetric: null,
				status: "running",
				description: params.description || `Run with ${Object.entries(hp).map(([k, v]) => `${k}=${v}`).join(", ") || "defaults"}`,
				startedAt: Date.now(),
				outputDir,
				logFile,
			};

			state.runs.push(run);
			appendState(ctx.cwd, { type: "run", data: run, timestamp: Date.now() });
			updateWidget(ctx);

			// Build training script arguments
			const trainArgs = [
				"--model", config.modelName,
				"--dataset", config.datasetName,
				"--task", config.task,
				"--max-length", String(config.maxLength),
				"--text-column", config.textColumn,
				"--label-column", config.labelColumn,
				"--output-dir", outputDir,
			];

			if (config.datasetConfig) trainArgs.push("--dataset-config", config.datasetConfig);
			if (config.textColumn2) trainArgs.push("--text-column-2", config.textColumn2);
			if (config.numLabels) trainArgs.push("--num-labels", String(config.numLabels));

			// LoRA/QLoRA flags from experiment config
			if (config.useLora) {
				trainArgs.push("--lora");
				if (config.loraR) trainArgs.push("--lora-r", String(config.loraR));
				if (config.loraAlpha) trainArgs.push("--lora-alpha", String(config.loraAlpha));
				if (config.loraDropout) trainArgs.push("--lora-dropout", String(config.loraDropout));
			}
			if (config.useQlora) trainArgs.push("--qlora");

			// Add hyperparams as CLI flags
			for (const [key, value] of Object.entries(hp)) {
				trainArgs.push(`--${key.replace(/_/g, "-")}`, String(value));
			}

			// Build the spawn command (uv run or python3)
			const { cmd, args: spawnArgs } = buildTrainCommand(config.trainPyPath, trainArgs);

			const timeout = (params.timeoutSeconds || 3600) * 1000;
			let stdout = "";
			let stderr = "";
			let timedOut = false;

			// Stream progress
			onUpdate?.({
				content: [{ type: "text", text: `Starting run ${runId} via ${cmd}...\n${config.modelName} → ${config.datasetName}\nHyperparams: ${JSON.stringify(hp)}` }],
			});

			const exitCode = await new Promise<number>((resolve) => {
				const proc = spawn(cmd, spawnArgs, {
					cwd: ctx.cwd,
					stdio: ["ignore", "pipe", "pipe"],
					env: {
						...process.env,
						HF_AUTORESEARCH_ROOT: STORAGE_ROOT,
						PYTHONUNBUFFERED: "1",
						// Pass HF token for gated models/datasets
						...(process.env.HF_TOKEN ? { HF_TOKEN: process.env.HF_TOKEN } : {}),
						...(process.env.HUGGING_FACE_HUB_TOKEN ? { HUGGING_FACE_HUB_TOKEN: process.env.HUGGING_FACE_HUB_TOKEN } : {}),
						// Ensure uv uses our cache
						UV_CACHE_DIR: path.join(STORAGE_ROOT, "cache", "uv"),
					},
				});

				const timer = setTimeout(() => {
					timedOut = true;
					proc.kill("SIGTERM");
				}, timeout);

				proc.stdout.on("data", (chunk: Buffer) => {
					const text = chunk.toString();
					stdout += text;
					try { fs.appendFileSync(logFile, text); } catch { /* ignore */ }

					// Parse metrics as they come in for progress updates
					const metrics = parseMetricLines(stdout);
					if (Object.keys(metrics).length > 0) {
						const primaryVal = metrics[config.metricName] ?? null;
						onUpdate?.({
							content: [{
								type: "text",
								text: `Run ${runId}: ${Object.entries(metrics).map(([k, v]) => `${k}=${formatMetric(v)}`).join(", ")}`,
							}],
						});
						run.metrics = metrics;
						run.primaryMetric = primaryVal;
					}
				});

				proc.stderr.on("data", (chunk: Buffer) => {
					const text = chunk.toString();
					stderr += text;
					try { fs.appendFileSync(logFile, "[stderr] " + text); } catch { /* ignore */ }
				});

				if (signal) {
					signal.addEventListener("abort", () => {
						proc.kill("SIGTERM");
					}, { once: true });
				}

				proc.on("error", (err: NodeJS.ErrnoException) => {
					clearTimeout(timer);
					stderr += `\nSpawn error: ${err.message}`;
					resolve(127);
				});

				proc.on("close", (code: number | null) => {
					clearTimeout(timer);
					resolve(code ?? 1);
				});
			});

			// Parse final metrics
			const finalMetrics = parseMetricLines(stdout);
			run.metrics = finalMetrics;
			run.primaryMetric = finalMetrics[config.metricName] ?? null;
			run.finishedAt = Date.now();

			if (timedOut) {
				run.status = "crash";
				run.description += " [TIMED OUT]";
			} else if (exitCode !== 0) {
				run.status = "crash";
				run.description += ` [EXIT ${exitCode}]`;
			}
			// If exitCode === 0, leave status as "running" — agent must explicitly log_experiment

			appendState(ctx.cwd, { type: "run", data: run, timestamp: Date.now() });
			updateWidget(ctx);

			const elapsed = ((run.finishedAt - run.startedAt) / 1000).toFixed(1);
			const bestRun = getBestRun(state, config.id);
			const delta = bestRun && run.primaryMetric !== null
				? computeDelta(run.primaryMetric, bestRun.primaryMetric, config.metricDirection)
				: "";

			const metricsStr = Object.entries(run.metrics)
				.map(([k, v]) => `  ${k} = ${formatMetric(v)}`)
				.join("\n");

			const summary = [
				exitCode === 0
					? `Run ${runId} completed successfully in ${elapsed}s`
					: `Run ${runId} FAILED in ${elapsed}s (exit code ${exitCode}${timedOut ? ", timed out" : ""})`,
				`  Experiment: ${config.id}`,
				`  Hyperparams: ${JSON.stringify(hp)}`,
				`  Command: ${cmd} ${spawnArgs.slice(0, 3).join(" ")}...`,
				"",
				"Metrics:",
				metricsStr || "  (no metrics captured)",
				"",
				run.primaryMetric !== null
					? `Primary metric (${config.metricName}): ${formatMetric(run.primaryMetric)}${delta ? ` ${delta}` : ""}`
					: `Primary metric (${config.metricName}): not found in output`,
				bestRun ? `Best so far: ${formatMetric(bestRun.primaryMetric)} (${bestRun.id})` : "",
				"",
				`Output: ${outputDir}`,
				`Log: ${logFile}`,
				"",
				exitCode !== 0
					? `Error output (last 800 chars):\n${stderr.slice(-800)}`
					: "**Next: use log_experiment to record keep/discard, then decide what to try next.**",
			]
				.filter((line) => line !== "")
				.join("\n");

			return {
				content: [{ type: "text", text: summary }],
				details: { run, config },
			};
		},

		renderCall(args, theme) {
			const hp = args.hyperparams || {};
			const hpStr = Object.entries(hp)
				.slice(0, 4)
				.map(([k, v]) => `${k}=${v}`)
				.join(" ");
			return new Text(
				theme.fg("toolTitle", theme.bold("run ")) +
					theme.fg("accent", args.experimentId ?? "?") +
					(hpStr ? " " + theme.fg("dim", hpStr) : ""),
				0,
				0,
			);
		},

		renderResult(result, { expanded, isPartial }, theme) {
			if (isPartial) {
				const text = result.content[0];
				return new Text(theme.fg("warning", text?.type === "text" ? text.text : "Running..."), 0, 0);
			}

			const details = result.details as { run?: ExperimentRun; config?: ExperimentConfig } | undefined;
			if (!details?.run) {
				const text = result.content[0];
				return new Text(text?.type === "text" ? text.text : "", 0, 0);
			}
			const run = details.run;
			const config = details.config;
			const icon =
				run.status === "crash"
					? theme.fg("error", "✗")
					: run.status === "keep"
						? theme.fg("success", "✓")
						: run.status === "discard"
							? theme.fg("warning", "○")
							: theme.fg("muted", "⏳");

			const elapsed = run.finishedAt ? ((run.finishedAt - run.startedAt) / 1000).toFixed(1) + "s" : "...";

			if (!expanded) {
				const primary = config ? `${config.metricName}=${formatMetric(run.primaryMetric)}` : "";
				return new Text(
					`${icon} ${theme.fg("accent", run.id)} ${theme.fg("muted", primary)} ${theme.fg("dim", elapsed)}`,
					0,
					0,
				);
			}

			const lines = [`${icon} ${theme.fg("accent", run.id)} ${theme.fg("dim", elapsed)}`];
			lines.push(theme.fg("muted", `  ${run.description}`));
			for (const [k, v] of Object.entries(run.metrics)) {
				const isPrimary = config && k === config.metricName;
				lines.push(
					`  ${isPrimary ? theme.fg("accent", `${k} = ${formatMetric(v)}`) : theme.fg("dim", `${k} = ${formatMetric(v)}`)}`,
				);
			}
			return new Text(lines.join("\n"), 0, 0);
		},
	});

	// ── Tool: log_experiment ──────────────────────────────────────────

	pi.registerTool({
		name: "log_experiment",
		label: "Log Experiment",
		description: [
			"Record the verdict for a completed run: keep (good result worth building on),",
			"discard (not useful, don't use as reference), or crash (failed to complete).",
			"Updates the run status. After logging, reason about what hyperparams to try next.",
		].join(" "),
		promptSnippet: "Record keep/discard/crash verdict for a training run",
		promptGuidelines: [
			"Always call log_experiment after run_experiment completes.",
			"Use 'keep' for runs that show promising results worth building on.",
			"Use 'discard' for runs that completed but aren't useful (worse metrics, wrong direction).",
			"Use 'crash' for runs that failed (timeout, OOM, errors).",
		],
		parameters: Type.Object({
			runId: Type.String({ description: "Run ID from run_experiment" }),
			status: StringEnum(["keep", "discard", "crash"] as const),
			notes: Type.Optional(Type.String({ description: "Notes about this run's results and reasoning" })),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const run = state.runs.find((r) => r.id === params.runId);
			if (!run) {
				return {
					content: [{ type: "text", text: `Error: run ${params.runId} not found.` }],
				};
			}

			const previousStatus = run.status;
			run.status = params.status;
			if (params.notes) run.description += ` — ${params.notes}`;

			appendState(ctx.cwd, { type: "run", data: run, timestamp: Date.now() });
			updateWidget(ctx);

			const config = state.configs.get(run.experimentId);
			const allRuns = state.runs.filter((r) => r.experimentId === run.experimentId);
			const keptRuns = allRuns.filter((r) => r.status === "keep");
			const bestRun = config ? getBestRun(state, config.id) : undefined;

			const summary = [
				`Run ${run.id}: ${previousStatus} → ${params.status.toUpperCase()}`,
				params.notes ? `Notes: ${params.notes}` : "",
				"",
				`Experiment ${run.experimentId}: ${keptRuns.length}/${allRuns.length} runs kept`,
				bestRun
					? `Best: ${config?.metricName}=${formatMetric(bestRun.primaryMetric)} (${bestRun.id})`
					: "No best run yet.",
				"",
				keptRuns.length > 0 ? "Kept runs:" : "",
				...keptRuns.map((r) => {
					const hp = Object.entries(r.hyperparams)
						.map(([k, v]) => `${k}=${v}`)
						.join(", ");
					const delta = bestRun ? computeDelta(r.primaryMetric, bestRun.primaryMetric, config!.metricDirection) : "";
					return `  ${r.id}: ${config?.metricName}=${formatMetric(r.primaryMetric)}${delta ? ` (${delta})` : ""} | ${hp}`;
				}),
				"",
				"Reason about what to try next based on the pattern of results above.",
			]
				.filter(Boolean)
				.join("\n");

			return {
				content: [{ type: "text", text: summary }],
				details: { run, status: params.status },
			};
		},

		renderCall(args, theme) {
			const statusColor = args.status === "keep" ? "success" : args.status === "crash" ? "error" : "warning";
			return new Text(
				theme.fg("toolTitle", theme.bold("log ")) +
					theme.fg("accent", args.runId ?? "?") +
					" " +
					theme.fg(statusColor, args.status ?? "?"),
				0,
				0,
			);
		},

		renderResult(result, _options, theme) {
			const details = result.details as { status?: string } | undefined;
			const statusColor = details?.status === "keep" ? "success" : details?.status === "crash" ? "error" : "warning";
			const text = result.content[0];
			const firstLine = (text?.type === "text" ? text.text : "").split("\n")[0];
			return new Text(theme.fg(statusColor, firstLine), 0, 0);
		},
	});

	// ── Shortcut: Ctrl+Shift+X for dashboard overlay ──────────────────

	pi.registerShortcut("ctrl+shift+x", {
		description: "Toggle HF Autoresearch dashboard",
		handler: async (ctx) => {
			if (!ctx.hasUI) return;
			const freshState = loadState(currentCwd);

			await ctx.ui.custom<void>((_tui, theme, _kb, done) => {
				return new DashboardComponent(freshState, theme, () => done());
			});
		},
	});

	// ── Command: /autoresearch ──────────────────────────────────────────

	pi.registerCommand("autoresearch", {
		description: "Show HF Autoresearch dashboard",
		handler: async (_args, ctx) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("Dashboard requires interactive mode", "error");
				return;
			}

			const freshState = loadState(ctx.cwd);

			await ctx.ui.custom<void>((_tui, theme, _kb, done) => {
				return new DashboardComponent(freshState, theme, () => done());
			});
		},
	});
}

// ============================================================================
// Dashboard Component
// ============================================================================

class DashboardComponent {
	private state: AutoresearchState;
	private theme: any;
	private onClose: () => void;
	private cachedWidth?: number;
	private cachedLines?: string[];

	constructor(state: AutoresearchState, theme: any, onClose: () => void) {
		this.state = state;
		this.theme = theme;
		this.onClose = onClose;
	}

	handleInput(data: string): void {
		if (matchesKey(data, "escape") || matchesKey(data, "ctrl+c") || matchesKey(data, "ctrl+shift+x")) {
			this.onClose();
		}
	}

	render(width: number): string[] {
		if (this.cachedLines && this.cachedWidth === width) return this.cachedLines;

		const th = this.theme;
		const lines: string[] = [];

		lines.push("");
		const title = th.fg("accent", " 🔬 HF Autoresearch Dashboard ");
		lines.push(th.fg("borderMuted", "─".repeat(3)) + title + th.fg("borderMuted", "─".repeat(Math.max(0, width - 35))));
		lines.push("");

		const configs = Array.from(this.state.configs.values());

		if (configs.length === 0) {
			lines.push(th.fg("dim", "  No experiments yet. Use scaffold_experiment to start."));
			lines.push("");
			lines.push(th.fg("dim", "  Press Escape to close"));
			this.cachedWidth = width;
			this.cachedLines = lines;
			return lines;
		}

		for (const config of configs) {
			lines.push(
				th.fg("accent", th.bold(`  Experiment: ${config.id}`)) +
					th.fg("dim", ` (${config.task})`),
			);
			lines.push(th.fg("muted", `  ${config.modelName} → ${config.datasetName}`));
			lines.push(
				th.fg("dim", `  Metric: ${config.metricName} (${config.metricDirection} is better) | Max length: ${config.maxLength}`),
			);
			lines.push("");

			const runs = this.state.runs.filter((r) => r.experimentId === config.id);
			const keptRuns = runs.filter((r) => r.status === "keep");
			const bestRun = getBestRun(this.state, config.id);

			lines.push(
				th.fg("muted", `  Runs: ${runs.length} total, ${keptRuns.length} kept`) +
					(bestRun
						? th.fg("success", ` | Best: ${config.metricName}=${formatMetric(bestRun.primaryMetric)}`)
						: ""),
			);
			lines.push("");

			// Table header
			const hdr =
				th.fg("muted", "  ") +
				th.fg("dim", padRight("Run", 22)) +
				th.fg("dim", padRight("Status", 10)) +
				th.fg("dim", padRight(config.metricName, 14)) +
				th.fg("dim", padRight("Delta", 12)) +
				th.fg("dim", "Hyperparams");
			lines.push(truncateToWidth(hdr, width));
			lines.push(th.fg("dim", "  " + "─".repeat(Math.max(0, width - 4))));

			for (const run of runs.slice(-20)) {
				const statusColor =
					run.status === "keep" ? "success"
						: run.status === "crash" ? "error"
							: run.status === "running" ? "muted"
								: "warning";
				const statusIcon =
					run.status === "keep" ? "✓"
						: run.status === "crash" ? "✗"
							: run.status === "running" ? "⏳"
								: "○";
				const delta = bestRun ? computeDelta(run.primaryMetric, bestRun.primaryMetric, config.metricDirection) : "";
				const hp = Object.entries(run.hyperparams)
					.slice(0, 5)
					.map(([k, v]) => `${k}=${v}`)
					.join(" ");

				const line =
					"  " +
					th.fg("accent", padRight(run.id, 22)) +
					th.fg(statusColor, padRight(`${statusIcon} ${run.status}`, 10)) +
					th.fg("muted", padRight(formatMetric(run.primaryMetric), 14)) +
					th.fg("dim", padRight(delta, 12)) +
					th.fg("dim", hp);
				lines.push(truncateToWidth(line, width));
			}

			if (runs.length > 20) {
				lines.push(th.fg("dim", `  ... ${runs.length - 20} earlier runs hidden`));
			}

			lines.push("");
			lines.push(th.fg("dim", "  " + "─".repeat(Math.max(0, width - 4))));
			lines.push("");
		}

		lines.push(th.fg("dim", "  Press Escape or Ctrl+Shift+X to close"));
		lines.push("");

		this.cachedWidth = width;
		this.cachedLines = lines;
		return lines;
	}

	invalidate(): void {
		this.cachedWidth = undefined;
		this.cachedLines = undefined;
	}
}

function padRight(str: string, len: number): string {
	return str.length >= len ? str.slice(0, len) : str + " ".repeat(len - str.length);
}

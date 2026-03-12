import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { StringEnum, Text, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

type AutoresearchAction = "setup" | "run" | "record" | "status";
type AutoresearchRecordStatus = "keep" | "discard" | "crash";

interface ParsedMetrics {
	valBpb?: number;
	peakVramMb?: number;
	trainingSeconds?: number;
	totalSeconds?: number;
}

interface ResultsRow {
	commit: string;
	valBpb: number;
	memoryGb: number;
	status: AutoresearchRecordStatus;
	description: string;
}

interface RunExecution {
	exitCode: number;
	timedOut: boolean;
	durationMs: number;
}

interface AutoresearchDetails {
	action: AutoresearchAction;
	cwd: string;
	resultsFile: string;
	logFile: string;
	command?: string;
	timeoutSeconds?: number;
	exitCode?: number;
	timedOut?: boolean;
	durationMs?: number;
	metrics?: ParsedMetrics;
	rowCount: number;
	best?: ResultsRow;
	recorded?: ResultsRow;
	error?: string;
}

interface AutoresearchRequest {
	action: AutoresearchAction;
	cwd?: string;
	resultsFile?: string;
	logFile?: string;
	runCommand?: string;
	timeoutSeconds?: number;
	status?: AutoresearchRecordStatus;
	description?: string;
	commit?: string;
	valBpb?: number;
	memoryGb?: number;
}

interface CommandParseResult {
	request?: AutoresearchRequest;
	usage?: string;
	error?: string;
}

interface ParsedCliArgs {
	positionals: string[];
	options: Record<string, string>;
	error?: string;
}

interface AutoresearchExecutionResult {
	details: AutoresearchDetails;
	text: string;
	isError: boolean;
}

interface AvRunExecution {
	exitCode: number;
	stdout: string;
	stderr: string;
	durationMs: number;
}

interface AutoresearchAvResultsRow {
	runId: string;
	task: string;
	modality: string;
	modelName: string;
	bestValMetric: number;
	status: string;
	notes: string;
}

const RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription";
const DEFAULT_TIMEOUT_SECONDS = 900;
const MAX_TIMEOUT_SECONDS = 60 * 60;
const AUTORESEARCH_ROOT_ENV_VARS = ["CODI_AUTORESEARCH_ROOT", "PI_AUTORESEARCH_ROOT"] as const;
const LAST_CWD_FILENAME = ".codi-autoresearch-last-cwd";
const AUTORESEARCH_AV_ROOT_ENV_VAR = "AUTORESEARCH_AV_ROOT";
const AUTORESEARCH_AV_PACKAGE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../autoresearch-av");
const AUTORESEARCH_USAGE =
	"Usage: /autoresearch setup [cwd] [--cwd <path>] [--results <file>] [--log <file>]\n" +
	"/autoresearch run [--cmd <command>] [--timeout <seconds>] [--cwd <path>] [--results <file>] [--log <file>]\n" +
	"/autoresearch record <keep|discard|crash> [description...] [--commit <hash>] [--val <val_bpb>] [--memory <gb>] [--cwd <path>] [--results <file>] [--log <file>]\n" +
	"/autoresearch status [cwd] [--cwd <path>] [--results <file>] [--log <file>]\n" +
	"/autoresearch av list\n" +
	"/autoresearch av status\n" +
	"/autoresearch av run <vision|audio> <task> [--epochs <n>] [--batch-size <n>] [--root <path>] [--patience <n>] [--random-init] [--dry-run]";

const AUTORESEARCH_PARAMS = Type.Object({
	action: StringEnum(["setup", "run", "record", "status"] as const),
	cwd: Type.Optional(Type.String({ description: "Repo path for the autoresearch run. Defaults to current cwd." })),
	resultsFile: Type.Optional(
		Type.String({
			description: "Path to results TSV (absolute or relative to cwd). Default: results.tsv",
		}),
	),
	logFile: Type.Optional(
		Type.String({
			description: "Path to run log file (absolute or relative to cwd). Default: run.log",
		}),
	),
	runCommand: Type.Optional(
		Type.String({
			description: "Command for action=run. Default: uv run train.py",
		}),
	),
	timeoutSeconds: Type.Optional(
		Type.Number({
			description: `Run timeout in seconds for action=run. Default: ${DEFAULT_TIMEOUT_SECONDS}`,
		}),
	),
	status: Type.Optional(
		StringEnum(["keep", "discard", "crash"] as const),
	),
	description: Type.Optional(
		Type.String({
			description: "Short description for action=record.",
		}),
	),
	commit: Type.Optional(
		Type.String({
			description: "Optional commit hash override for action=record. Default: current git HEAD short hash.",
		}),
	),
	valBpb: Type.Optional(Type.Number({ description: "Optional val_bpb override for action=record." })),
	memoryGb: Type.Optional(Type.Number({ description: "Optional memory_gb override for action=record." })),
});

function resolvePath(cwd: string, value: string | undefined, fallbackFile: string): string {
	const selected = value?.trim();
	if (!selected) {
		return path.join(cwd, fallbackFile);
	}
	return path.isAbsolute(selected) ? selected : path.join(cwd, selected);
}

function getConfiguredAutoresearchRoot(): string | undefined {
	for (const envVar of AUTORESEARCH_ROOT_ENV_VARS) {
		const value = process.env[envVar]?.trim();
		if (value) {
			return path.resolve(value);
		}
	}
	return undefined;
}

function getLastCwdStateFile(root: string): string {
	return path.join(root, LAST_CWD_FILENAME);
}

function readRememberedAutoresearchCwd(root: string): string | undefined {
	const stateFile = getLastCwdStateFile(root);
	if (!fs.existsSync(stateFile)) {
		return undefined;
	}
	try {
		const stored = fs.readFileSync(stateFile, "utf8").trim();
		if (!stored) {
			return undefined;
		}
		const resolved = path.resolve(root, stored);
		if (!fs.existsSync(resolved)) {
			return undefined;
		}
		return resolved;
	} catch {
		return undefined;
	}
}

function rememberAutoresearchCwd(root: string | undefined, cwd: string): void {
	if (!root) {
		return;
	}
	const resolvedRoot = path.resolve(root);
	const resolvedCwd = path.resolve(cwd);
	const relative = path.relative(resolvedRoot, resolvedCwd);
	if (relative.startsWith("..") || path.isAbsolute(relative)) {
		return;
	}
	try {
		fs.mkdirSync(resolvedRoot, { recursive: true });
		fs.writeFileSync(getLastCwdStateFile(resolvedRoot), relative || ".", "utf8");
	} catch {
		// Best-effort only; failure should not break the run.
	}
}

function resolveAutoresearchCwd(requestCwd: string | undefined, fallbackCwd: string): string {
	const selected = requestCwd?.trim();
	if (selected) {
		return path.isAbsolute(selected) ? selected : path.resolve(fallbackCwd, selected);
	}
	const configuredRoot = getConfiguredAutoresearchRoot();
	if (configuredRoot) {
		return readRememberedAutoresearchCwd(configuredRoot) ?? configuredRoot;
	}
	return fallbackCwd;
}

function maybeGetMissingRunTargetError(cwd: string, command: string): string | undefined {
	if (!command.includes("train.py")) {
		return undefined;
	}
	if (fs.existsSync(path.join(cwd, "train.py"))) {
		return undefined;
	}
	return `No train.py found in ${cwd}. Clone an autoresearch repo there or pass --cwd <repo>.`;
}

function clampTimeout(seconds: number | undefined): number {
	if (!seconds || Number.isNaN(seconds) || seconds <= 0) {
		return DEFAULT_TIMEOUT_SECONDS;
	}
	return Math.min(Math.floor(seconds), MAX_TIMEOUT_SECONDS);
}

function ensureResultsFile(resultsFile: string): void {
	const dir = path.dirname(resultsFile);
	fs.mkdirSync(dir, { recursive: true });
	if (!fs.existsSync(resultsFile)) {
		fs.writeFileSync(resultsFile, `${RESULTS_HEADER}\n`, "utf8");
		return;
	}
	const text = fs.readFileSync(resultsFile, "utf8");
	if (!text.trim()) {
		fs.writeFileSync(resultsFile, `${RESULTS_HEADER}\n`, "utf8");
		return;
	}
	const firstLine = text.split(/\r?\n/)[0]?.trim();
	if (firstLine !== RESULTS_HEADER) {
		throw new Error(`results.tsv header mismatch. Expected "${RESULTS_HEADER}".`);
	}
}

function parseMetricsFromLog(logFile: string): ParsedMetrics {
	if (!fs.existsSync(logFile)) {
		return {};
	}
	const text = fs.readFileSync(logFile, "utf8");
	const extract = (pattern: RegExp): number | undefined => {
		const match = text.match(pattern);
		if (!match?.[1]) {
			return undefined;
		}
		const parsed = Number.parseFloat(match[1]);
		return Number.isFinite(parsed) ? parsed : undefined;
	};
	return {
		valBpb: extract(/^val_bpb:\s*([0-9.]+)/m),
		peakVramMb: extract(/^peak_vram_mb:\s*([0-9.]+)/m),
		trainingSeconds: extract(/^training_seconds:\s*([0-9.]+)/m),
		totalSeconds: extract(/^total_seconds:\s*([0-9.]+)/m),
	};
}

function parseResults(resultsFile: string): { rows: ResultsRow[]; best?: ResultsRow } {
	if (!fs.existsSync(resultsFile)) {
		return { rows: [] };
	}
	const text = fs.readFileSync(resultsFile, "utf8");
	const lines = text.split(/\r?\n/).map((line) => line.trimEnd());
	const dataLines = lines.slice(1).filter((line) => line.length > 0);
	const rows: ResultsRow[] = [];
	for (const line of dataLines) {
		const parts = line.split("\t");
		if (parts.length < 5) {
			continue;
		}
		const commit = parts[0] ?? "";
		const valBpb = Number.parseFloat(parts[1] ?? "");
		const memoryGb = Number.parseFloat(parts[2] ?? "");
		const statusRaw = parts[3] ?? "";
		const status = statusRaw === "keep" || statusRaw === "discard" || statusRaw === "crash" ? statusRaw : undefined;
		const description = parts.slice(4).join("\t");
		if (!commit || !Number.isFinite(valBpb) || !Number.isFinite(memoryGb) || !status) {
			continue;
		}
		rows.push({
			commit,
			valBpb,
			memoryGb,
			status,
			description,
		});
	}
	const keepRows = rows.filter((row) => row.status === "keep" && row.valBpb > 0);
	const best = keepRows.reduce<ResultsRow | undefined>((lowest, row) => {
		if (!lowest || row.valBpb < lowest.valBpb) {
			return row;
		}
		return lowest;
	}, undefined);
	return { rows, best };
}

function sanitizeTsvField(value: string): string {
	return value.replace(/[\t\r\n]+/g, " ").trim();
}

function getGitShortCommit(cwd: string): Promise<string> {
	return new Promise((resolve) => {
		const proc = spawn("git", ["rev-parse", "--short", "HEAD"], {
			cwd,
			stdio: ["ignore", "pipe", "ignore"],
			shell: false,
		});
		let output = "";
		proc.stdout.on("data", (chunk) => {
			output += chunk.toString();
		});
		proc.on("close", (code) => {
			if (code === 0) {
				const value = output.trim();
				resolve(value.length > 0 ? value : "unknown");
				return;
			}
			resolve("unknown");
		});
		proc.on("error", () => resolve("unknown"));
	});
}

async function runCommandToLog(cwd: string, command: string, logFile: string, timeoutSeconds: number): Promise<RunExecution> {
	fs.mkdirSync(path.dirname(logFile), { recursive: true });
	const output = fs.createWriteStream(logFile, { flags: "w", encoding: "utf8" });
	const shell = process.env.SHELL?.trim() || "/bin/zsh";
	const startedAt = Date.now();

	return await new Promise<RunExecution>((resolve) => {
		let timedOut = false;
		const child = spawn(shell, ["-lc", command], {
			cwd,
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
			env: process.env,
		});
		const timeout = setTimeout(() => {
			timedOut = true;
			child.kill("SIGTERM");
		}, timeoutSeconds * 1000);

		child.stdout.on("data", (chunk) => {
			output.write(chunk);
		});
		child.stderr.on("data", (chunk) => {
			output.write(chunk);
		});

		child.on("error", (error) => {
			output.write(`\n[autoresearch-tools] spawn error: ${error.message}\n`);
		});

		child.on("close", (code) => {
			clearTimeout(timeout);
			output.end();
			resolve({
				exitCode: code ?? (timedOut ? 124 : 1),
				timedOut,
				durationMs: Date.now() - startedAt,
			});
		});
	});
}

function formatSummary(details: AutoresearchDetails): string {
	const lines: string[] = [];
	lines.push(`action: ${details.action}`);
	lines.push(`rows: ${details.rowCount}`);
	if (details.metrics?.valBpb !== undefined) {
		lines.push(`val_bpb: ${details.metrics.valBpb.toFixed(6)}`);
	}
	if (details.metrics?.peakVramMb !== undefined) {
		lines.push(`peak_vram_mb: ${details.metrics.peakVramMb.toFixed(1)}`);
	}
	if (details.best) {
		lines.push(`best: ${details.best.valBpb.toFixed(6)} (${details.best.commit})`);
	}
	if (details.exitCode !== undefined) {
		lines.push(`exit: ${details.exitCode}${details.timedOut ? " (timeout)" : ""}`);
	}
	return lines.join("\n");
}

function tokenizeCommandInput(input: string): string[] {
	const tokens: string[] = [];
	const pattern = /"((?:\\.|[^"])*)"|'((?:\\.|[^'])*)'|(\S+)/g;
	let match: RegExpExecArray | null;
	while ((match = pattern.exec(input)) !== null) {
		const raw = match[1] ?? match[2] ?? match[3] ?? "";
		const unescaped = raw.replace(/\\(["'])/g, "$1").replace(/\\\\/g, "\\");
		if (unescaped.length > 0) {
			tokens.push(unescaped);
		}
	}
	return tokens;
}

function getAutoresearchAvRoot(): string {
	const configured = process.env[AUTORESEARCH_AV_ROOT_ENV_VAR]?.trim();
	if (configured) {
		return path.resolve(configured);
	}
	return "/Volumes/Expansion/Data/autoresearch-av";
}

function getAutoresearchAvResultsFile(): string {
	return path.join(getAutoresearchAvRoot(), "results", "results.tsv");
}

function parseAutoresearchAvResults(): AutoresearchAvResultsRow[] {
	const resultsFile = getAutoresearchAvResultsFile();
	if (!fs.existsSync(resultsFile)) {
		return [];
	}
	const lines = fs
		.readFileSync(resultsFile, "utf8")
		.split(/\r?\n/)
		.map((line) => line.trimEnd())
		.filter((line) => line.length > 0);
	if (lines.length <= 1) {
		return [];
	}
	const rows: AutoresearchAvResultsRow[] = [];
	for (const line of lines.slice(1)) {
		const parts = line.split("\t");
		if (parts.length < 18) {
			continue;
		}
		const bestValMetric = Number.parseFloat(parts[7] ?? "");
		if (!Number.isFinite(bestValMetric)) {
			continue;
		}
		rows.push({
			runId: parts[0] ?? "",
			task: parts[1] ?? "",
			modality: parts[3] ?? "",
			modelName: parts[4] ?? "",
			bestValMetric,
			status: parts[12] ?? "",
			notes: parts[13] ?? "",
		});
	}
	return rows;
}

function formatAutoresearchAvStatus(): string {
	const rows = parseAutoresearchAvResults();
	if (rows.length === 0) {
		return `No autoresearch-av runs recorded yet.\nResults file: ${getAutoresearchAvResultsFile()}`;
	}

	const sections: string[] = [];
	sections.push(`results: ${rows.length} runs`);
	sections.push(`file: ${getAutoresearchAvResultsFile()}`);
	for (const modality of ["vision", "audio"] as const) {
		const modalityRows = rows.filter((row) => row.modality === modality);
		if (modalityRows.length === 0) {
			continue;
		}
		sections.push("");
		sections.push(modality);
		const bestByTask = new Map<string, AutoresearchAvResultsRow>();
		for (const row of modalityRows) {
			const existing = bestByTask.get(row.task);
			if (!existing || row.bestValMetric > existing.bestValMetric) {
				bestByTask.set(row.task, row);
			}
		}
		for (const task of [...bestByTask.keys()].sort()) {
			const row = bestByTask.get(task);
			if (!row) {
				continue;
			}
			const noteSuffix = row.notes ? ` | ${row.notes}` : "";
			sections.push(`${task}: ${row.bestValMetric.toFixed(4)} | ${row.modelName}${noteSuffix}`);
		}
	}
	return sections.join("\n");
}

function getAutoresearchAvPythonCommand(): { command: string; args: string[] } {
	if (fs.existsSync(path.join(AUTORESEARCH_AV_PACKAGE_DIR, ".venv", "bin", "python"))) {
		return {
			command: path.join(AUTORESEARCH_AV_PACKAGE_DIR, ".venv", "bin", "python"),
			args: [],
		};
	}
	return {
		command: "python3",
		args: [],
	};
}

async function runAutoresearchAvCommand(
	args: string[],
	onUpdate?: (line: string, elapsedMs: number) => void,
): Promise<AvRunExecution> {
	const python = getAutoresearchAvPythonCommand();
	return await new Promise<AvRunExecution>((resolve) => {
		const startedAt = Date.now();
		const child = spawn(python.command, [...python.args, ...args], {
			cwd: AUTORESEARCH_AV_PACKAGE_DIR,
			stdio: ["ignore", "pipe", "pipe"],
			shell: false,
			env: {
				...process.env,
				[AUTORESEARCH_AV_ROOT_ENV_VAR]: getAutoresearchAvRoot(),
			},
		});
		let stdout = "";
		let stderr = "";
		let stdoutBuffer = "";
		let stderrBuffer = "";
		const emitLines = (buffer: string, stream: "stdout" | "stderr"): string => {
			const lines = buffer.split(/\r?\n/);
			const pending = lines.pop() ?? "";
			for (const line of lines) {
				const trimmed = line.trim();
				if (trimmed && onUpdate) {
					onUpdate(`${stream}: ${trimmed}`, Date.now() - startedAt);
				}
			}
			return pending;
		};
		child.stdout.on("data", (chunk) => {
			const text = chunk.toString();
			stdout += text;
			stdoutBuffer += text;
			stdoutBuffer = emitLines(stdoutBuffer, "stdout");
		});
		child.stderr.on("data", (chunk) => {
			const text = chunk.toString();
			stderr += text;
			stderrBuffer += text;
			stderrBuffer = emitLines(stderrBuffer, "stderr");
		});
		child.on("close", (code) => {
			if (stdoutBuffer.trim() && onUpdate) {
				onUpdate(`stdout: ${stdoutBuffer.trim()}`, Date.now() - startedAt);
			}
			if (stderrBuffer.trim() && onUpdate) {
				onUpdate(`stderr: ${stderrBuffer.trim()}`, Date.now() - startedAt);
			}
			resolve({
				exitCode: code ?? 1,
				stdout: stdout.trim(),
				stderr: stderr.trim(),
				durationMs: Date.now() - startedAt,
			});
		});
		child.on("error", (error) => {
			resolve({
				exitCode: 1,
				stdout,
				stderr: error.message,
				durationMs: Date.now() - startedAt,
			});
		});
	});
}

async function executeAutoresearchAvCommand(
	args: string,
	onUpdate?: (line: string, elapsedMs: number) => void,
): Promise<{ text: string; isError: boolean }> {
	const tokens = tokenizeCommandInput(args.trim());
	if (tokens.length === 0) {
		return {
			text:
				"Usage: /autoresearch av list\n" +
				"/autoresearch av status\n" +
				"/autoresearch av run <vision|audio> <task> [--epochs <n>] [--batch-size <n>] [--root <path>] [--patience <n>] [--random-init] [--dry-run]",
			isError: false,
		};
	}

	const action = tokens[0]?.toLowerCase();
	if (action === "status") {
		return {
			text: formatAutoresearchAvStatus(),
			isError: false,
		};
	}
	if (action === "list") {
		const vision = await runAutoresearchAvCommand(["train_vision.py", "--list-tasks"]);
		const audio = await runAutoresearchAvCommand(["train_audio.py", "--list-tasks"]);
		if (vision.exitCode !== 0 || audio.exitCode !== 0) {
			return {
				text: `Error: ${vision.stderr || audio.stderr || "Unable to list autoresearch-av tasks."}`,
				isError: true,
			};
		}
		return {
			text: `vision\n${vision.stdout}\n\naudio\n${audio.stdout}`,
			isError: false,
		};
	}

	if (action !== "run") {
		return {
			text:
				"Unknown autoresearch av action.\n" +
				"Usage: /autoresearch av list\n" +
				"/autoresearch av status\n" +
				"/autoresearch av run <vision|audio> <task> [--epochs <n>] [--batch-size <n>] [--root <path>] [--patience <n>] [--random-init] [--dry-run]",
			isError: true,
		};
	}

	const modality = tokens[1]?.toLowerCase();
	const task = tokens[2];
	if ((modality !== "vision" && modality !== "audio") || !task) {
		return {
			text: "Usage: /autoresearch av run <vision|audio> <task> [--epochs <n>] [--batch-size <n>] [--root <path>] [--patience <n>] [--random-init] [--dry-run]",
			isError: true,
		};
	}

	const parsedCli = parseCliArgs(tokens.slice(3));
	if (parsedCli.error || parsedCli.positionals.length > 0) {
		return {
			text: `${parsedCli.error ?? "Unexpected positional arguments."}\nUsage: /autoresearch av run <vision|audio> <task> [--epochs <n>] [--batch-size <n>] [--root <path>] [--patience <n>] [--random-init] [--dry-run]`,
			isError: true,
		};
	}

	const options = parsedCli.options;
	const script = modality === "vision" ? "train_vision.py" : "train_audio.py";
	const commandArgs = [script, "--task", task];
	if (options.epochs) commandArgs.push("--epochs", options.epochs);
	if (options["batch-size"]) commandArgs.push("--batch-size", options["batch-size"]);
	if (options.root) commandArgs.push("--root", options.root);
	if (options["max-train-samples"]) commandArgs.push("--max-train-samples", options["max-train-samples"]);
	if (options["max-val-samples"]) commandArgs.push("--max-val-samples", options["max-val-samples"]);
	if (options["image-size"] && modality === "vision") commandArgs.push("--image-size", options["image-size"]);
	if (options["sample-rate"] && modality === "audio") commandArgs.push("--sample-rate", options["sample-rate"]);
	if (options["clip-seconds"] && modality === "audio") commandArgs.push("--clip-seconds", options["clip-seconds"]);
	if ("dry-run" in options) commandArgs.push("--dry-run");
	if ("pretrained" in options && modality === "vision") commandArgs.push("--pretrained");
	if ("random-init" in options && modality === "vision") commandArgs.push("--random-init");

	if (options.lr) commandArgs.push("--lr", options.lr);
	if (options["weight-decay"]) commandArgs.push("--weight-decay", options["weight-decay"]);
	if (options["num-workers"]) commandArgs.push("--num-workers", options["num-workers"]);
	if (options.patience) commandArgs.push("--patience", options.patience);
	const result = await runAutoresearchAvCommand(commandArgs, onUpdate);
	if (result.exitCode !== 0) {
		return {
			text: `Error: ${result.stderr || result.stdout || `autoresearch-av run failed with exit ${result.exitCode}.`}`,
			isError: true,
		};
	}
	return {
		text: `${result.stdout || "autoresearch-av run completed."}\nDuration: ${(result.durationMs / 1000).toFixed(1)}s`,
		isError: false,
	};
}

function parseCliArgs(tokens: string[]): ParsedCliArgs {
	const positionals: string[] = [];
	const options: Record<string, string> = {};
	for (let index = 0; index < tokens.length; index += 1) {
		const token = tokens[index] ?? "";
		if (token === "--dry-run" || token === "--pretrained" || token === "--random-init") {
			options[token.slice(2).trim().toLowerCase()] = "true";
			continue;
		}
		if (!token.startsWith("--")) {
			positionals.push(token);
			continue;
		}
		const key = token.slice(2).trim().toLowerCase();
		if (!key) {
			return { positionals, options, error: `Invalid option "${token}".` };
		}
		const value = tokens[index + 1];
		if (!value || value.startsWith("--")) {
			return { positionals, options, error: `Option --${key} requires a value.` };
		}
		options[key] = value;
		index += 1;
	}
	return { positionals, options };
}

function parseFiniteNumber(value: string, label: string): { value?: number; error?: string } {
	const parsed = Number.parseFloat(value);
	if (!Number.isFinite(parsed)) {
		return { error: `${label} must be a number.` };
	}
	return { value: parsed };
}

function parseCommandArgs(args: string): CommandParseResult {
	const tokens = tokenizeCommandInput(args.trim());
	if (tokens.length === 0 || tokens[0] === "help" || tokens[0] === "--help") {
		return { usage: AUTORESEARCH_USAGE };
	}

	const actionToken = tokens[0]?.toLowerCase() ?? "";
	if (actionToken !== "setup" && actionToken !== "run" && actionToken !== "record" && actionToken !== "status") {
		return { error: `Unknown action "${actionToken}".\n${AUTORESEARCH_USAGE}` };
	}

	const parsedCli = parseCliArgs(tokens.slice(1));
	if (parsedCli.error) {
		return { error: `${parsedCli.error}\n${AUTORESEARCH_USAGE}` };
	}

	const request: AutoresearchRequest = {
		action: actionToken,
	};
	const options = parsedCli.options;
	for (const [key, value] of Object.entries(options)) {
		switch (key) {
			case "cwd":
				request.cwd = value;
				break;
			case "results":
			case "results-file":
				request.resultsFile = value;
				break;
			case "log":
			case "log-file":
				request.logFile = value;
				break;
			case "cmd":
			case "command":
				request.runCommand = value;
				break;
			case "timeout": {
				const numeric = parseFiniteNumber(value, "--timeout");
				if (numeric.error) return { error: `${numeric.error}\n${AUTORESEARCH_USAGE}` };
				request.timeoutSeconds = numeric.value;
				break;
			}
			case "commit":
				request.commit = value;
				break;
			case "val":
			case "val-bpb": {
				const numeric = parseFiniteNumber(value, "--val");
				if (numeric.error) return { error: `${numeric.error}\n${AUTORESEARCH_USAGE}` };
				request.valBpb = numeric.value;
				break;
			}
			case "memory":
			case "memory-gb": {
				const numeric = parseFiniteNumber(value, "--memory");
				if (numeric.error) return { error: `${numeric.error}\n${AUTORESEARCH_USAGE}` };
				request.memoryGb = numeric.value;
				break;
			}
			default:
				return { error: `Unknown option "--${key}".\n${AUTORESEARCH_USAGE}` };
		}
	}

	if (request.action === "record") {
		const statusToken = parsedCli.positionals[0]?.toLowerCase();
		if (statusToken !== "keep" && statusToken !== "discard" && statusToken !== "crash") {
			return { error: `record requires <keep|discard|crash>.\n${AUTORESEARCH_USAGE}` };
		}
		request.status = statusToken;
		const description = parsedCli.positionals.slice(1).join(" ").trim();
		if (description) {
			request.description = description;
		}
		return { request };
	}

	if (request.action === "run") {
		if (parsedCli.positionals.length > 0) {
			if (request.runCommand) {
				return { error: `Provide run command either as positional text or --cmd, not both.\n${AUTORESEARCH_USAGE}` };
			}
			request.runCommand = parsedCli.positionals.join(" ");
		}
		return { request };
	}

	if (parsedCli.positionals.length > 1) {
		return { error: `Unexpected positional arguments for "${request.action}".\n${AUTORESEARCH_USAGE}` };
	}
	if (parsedCli.positionals.length === 1) {
		if (request.cwd) {
			return { error: `Provide cwd either as positional text or --cwd, not both.\n${AUTORESEARCH_USAGE}` };
		}
		request.cwd = parsedCli.positionals[0];
	}

	return { request };
}

async function executeAutoresearchRequest(request: AutoresearchRequest, fallbackCwd: string): Promise<AutoresearchExecutionResult> {
	const cwd = resolveAutoresearchCwd(request.cwd, fallbackCwd);
	const configuredRoot = getConfiguredAutoresearchRoot();
	const resultsFile = resolvePath(cwd, request.resultsFile, "results.tsv");
	const logFile = resolvePath(cwd, request.logFile, "run.log");
	const action = request.action;
	rememberAutoresearchCwd(configuredRoot, cwd);

	let rows: ResultsRow[] = [];
	let best: ResultsRow | undefined;
	try {
		ensureResultsFile(resultsFile);
		const parsed = parseResults(resultsFile);
		rows = parsed.rows;
		best = parsed.best;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		const details: AutoresearchDetails = {
			action,
			cwd,
			resultsFile,
			logFile,
			rowCount: rows.length,
			best,
			error: message,
		};
		return {
			text: `Error: ${message}`,
			details,
			isError: true,
		};
	}

	if (action === "setup" || action === "status") {
		const details: AutoresearchDetails = {
			action,
			cwd,
			resultsFile,
			logFile,
			rowCount: rows.length,
			best,
		};
		return {
			text: formatSummary(details),
			details,
			isError: false,
		};
	}

	if (action === "run") {
		const runCommand = request.runCommand?.trim() || "uv run train.py";
		const timeoutSeconds = clampTimeout(request.timeoutSeconds);
		const missingTargetError = maybeGetMissingRunTargetError(cwd, runCommand);
		if (missingTargetError) {
			const details: AutoresearchDetails = {
				action,
				cwd,
				resultsFile,
				logFile,
				command: runCommand,
				timeoutSeconds,
				rowCount: rows.length,
				best,
				error: missingTargetError,
			};
			return {
				text: `Error: ${missingTargetError}`,
				details,
				isError: true,
			};
		}
		const run = await runCommandToLog(cwd, runCommand, logFile, timeoutSeconds);
		const metrics = parseMetricsFromLog(logFile);
		const nextParsed = parseResults(resultsFile);
		const details: AutoresearchDetails = {
			action,
			cwd,
			resultsFile,
			logFile,
			command: runCommand,
			timeoutSeconds,
			exitCode: run.exitCode,
			timedOut: run.timedOut,
			durationMs: run.durationMs,
			metrics,
			rowCount: nextParsed.rows.length,
			best: nextParsed.best,
		};
		if (run.exitCode !== 0) {
			return {
				text: `Run failed with exit ${run.exitCode}. Check ${logFile}.`,
				details,
				isError: true,
			};
		}
		return {
			text: formatSummary(details),
			details,
			isError: false,
		};
	}

	const recordStatus = request.status;
	if (!recordStatus) {
		const details: AutoresearchDetails = {
			action,
			cwd,
			resultsFile,
			logFile,
			rowCount: rows.length,
			best,
			error: "status is required for action=record.",
		};
		return {
			text: `Error: ${details.error}`,
			details,
			isError: true,
		};
	}

	const metrics = parseMetricsFromLog(logFile);
	const commit = sanitizeTsvField(request.commit?.trim() || (await getGitShortCommit(cwd)));
	const description = sanitizeTsvField(request.description?.trim() || "experiment");
	const valBpb = request.valBpb ?? metrics.valBpb ?? (recordStatus === "crash" ? 0 : undefined);
	if (valBpb === undefined || !Number.isFinite(valBpb)) {
		const details: AutoresearchDetails = {
			action,
			cwd,
			resultsFile,
			logFile,
			rowCount: rows.length,
			best,
			metrics,
			error: "Unable to resolve val_bpb. Provide valBpb explicitly or run after a valid log.",
		};
		return {
			text: `Error: ${details.error}`,
			details,
			isError: true,
		};
	}

	const memoryGb = request.memoryGb ?? (metrics.peakVramMb !== undefined ? metrics.peakVramMb / 1024 : recordStatus === "crash" ? 0 : undefined);
	if (memoryGb === undefined || !Number.isFinite(memoryGb)) {
		const details: AutoresearchDetails = {
			action,
			cwd,
			resultsFile,
			logFile,
			rowCount: rows.length,
			best,
			metrics,
			error: "Unable to resolve memory_gb. Provide memoryGb explicitly or ensure peak_vram_mb exists in log.",
		};
		return {
			text: `Error: ${details.error}`,
			details,
			isError: true,
		};
	}

	const row: ResultsRow = {
		commit,
		valBpb,
		memoryGb,
		status: recordStatus,
		description,
	};

	const rowLine = `${row.commit}\t${row.valBpb.toFixed(6)}\t${row.memoryGb.toFixed(1)}\t${row.status}\t${row.description}\n`;
	fs.appendFileSync(resultsFile, rowLine, "utf8");
	const nextParsed = parseResults(resultsFile);
	const details: AutoresearchDetails = {
		action,
		cwd,
		resultsFile,
		logFile,
		metrics,
		rowCount: nextParsed.rows.length,
		best: nextParsed.best,
		recorded: row,
	};
	return {
		text: formatSummary(details),
		details,
		isError: false,
	};
}

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool<typeof AUTORESEARCH_PARAMS, AutoresearchDetails>({
			name: "autoresearch",
			label: "Autoresearch",
			description:
				"Run Karpathy-style autoresearch loops with setup, bounded experiment execution, TSV recording, and best-result inspection.",
			promptSnippet:
				"Use setup/run/record/status actions to run autonomous training experiments and track results in results.tsv.",
			promptGuidelines: [
				"Use setup first to validate results.tsv shape.",
				"Use run with bounded timeout and inspect parsed metrics from run.log.",
				"Use record after each run with keep/discard/crash and a short description.",
			],
			parameters: AUTORESEARCH_PARAMS,
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const result = await executeAutoresearchRequest(
					{
						action: params.action,
						cwd: params.cwd,
						resultsFile: params.resultsFile,
						logFile: params.logFile,
						runCommand: params.runCommand,
						timeoutSeconds: params.timeoutSeconds,
						status: params.status,
						description: params.description,
						commit: params.commit,
						valBpb: params.valBpb,
						memoryGb: params.memoryGb,
					},
					ctx.cwd,
				);
				return {
					content: [{ type: "text", text: result.text }],
					details: result.details,
					isError: result.isError,
				};
			},
			renderCall(args, theme) {
				const textParts: string[] = [theme.fg("toolTitle", theme.bold("autoresearch ")), theme.fg("muted", args.action)];
				if (args.status) {
					textParts.push(` ${theme.fg("accent", args.status)}`);
				}
				if (args.runCommand) {
					const trimmed = args.runCommand.length > 48 ? `${args.runCommand.slice(0, 48)}...` : args.runCommand;
					textParts.push(` ${theme.fg("dim", `"${trimmed}"`)}`);
				}
				return new Text(textParts.join(""), 0, 0);
			},
			renderResult(result, _options, theme) {
				const details = result.details as AutoresearchDetails | undefined;
				if (!details) {
					const text = result.content[0];
					return new Text(text?.type === "text" ? text.text : "", 0, 0);
				}
				if (details.error) {
					return new Text(theme.fg("error", `Error: ${details.error}`), 0, 0);
				}
				const lines: string[] = [];
				lines.push(`${theme.fg("success", "✓")} ${theme.fg("muted", `${details.action} complete`)}`);
				if (details.metrics?.valBpb !== undefined) {
					lines.push(theme.fg("dim", `val_bpb ${details.metrics.valBpb.toFixed(6)}`));
				}
				if (details.metrics?.peakVramMb !== undefined) {
					lines.push(theme.fg("dim", `peak_vram_mb ${details.metrics.peakVramMb.toFixed(1)}`));
				}
				if (details.best) {
					lines.push(theme.fg("dim", `best ${details.best.valBpb.toFixed(6)} @ ${details.best.commit}`));
				}
				lines.push(theme.fg("dim", `rows ${details.rowCount}`));
				return new Text(lines.join("\n"), 0, 0);
			},
		}),
	);

	pi.registerCommand("autoresearch", {
		description: "Run autoresearch setup/run/record/status without calling the tool directly.",
		handler: async (args, ctx) => {
			const trimmed = args.trim();
			if (trimmed === "av" || trimmed.startsWith("av ")) {
				const avArgs = trimmed === "av" ? "" : trimmed.slice(3);
				const statusKey = "autoresearch-av";
				const runMatch = avArgs.match(/^run\s+(vision|audio)\s+([^\s]+)/);
				if (runMatch) {
					ctx.ui.setStatus(statusKey, `running ${runMatch[1]}:${runMatch[2]}...`);
				}
				const result = await executeAutoresearchAvCommand(avArgs, (line, elapsedMs) => {
					const seconds = Math.max(1, Math.floor(elapsedMs / 1000));
					const compact = line.length > 96 ? `${line.slice(0, 96)}...` : line;
					ctx.ui.setStatus(statusKey, `${seconds}s ${compact}`);
				});
				ctx.ui.setStatus(statusKey, undefined);
				ctx.ui.notify(result.text, result.isError ? "error" : "info");
				return;
			}
			const parsed = parseCommandArgs(args);
			if (parsed.usage) {
				ctx.ui.notify(parsed.usage, "info");
				return;
			}
			if (parsed.error || !parsed.request) {
				ctx.ui.notify(parsed.error ?? AUTORESEARCH_USAGE, "error");
				return;
			}
			const result = await executeAutoresearchRequest(parsed.request, ctx.cwd);
			ctx.ui.notify(result.text, result.isError ? "error" : "info");
		},
		getArgumentCompletions: (argumentPrefix) => {
			const prefix = argumentPrefix.trim().toLowerCase();
			if (!prefix) {
				return [
					{ value: "setup", label: "setup", description: "Create/validate results.tsv" },
					{ value: "run", label: "run", description: "Run experiment command into run.log" },
					{ value: "record keep", label: "record keep", description: "Append keep row to results.tsv" },
					{ value: "status", label: "status", description: "Show row count and best result" },
					{ value: "av list", label: "av list", description: "List autoresearch-av tasks" },
					{ value: "av status", label: "av status", description: "Show best autoresearch-av results by task" },
					{ value: "av run vision beans", label: "av run", description: "Run autoresearch-av vision/audio baselines" },
				];
			}
			if ("av".startsWith(prefix)) {
				return [{ value: "av ", label: "av", description: "Dispatch into autoresearch-av baselines" }];
			}
			if (prefix.startsWith("av ")) {
				return [
					{ value: "av list", label: "list", description: "List vision/audio tasks" },
					{ value: "av status", label: "status", description: "Show best runs by task" },
					{ value: "av run vision beans", label: "run vision", description: "Run a vision baseline" },
					{ value: "av run audio speech_commands", label: "run audio", description: "Run an audio baseline" },
				];
			}
			if ("record".startsWith(prefix)) {
				return [{ value: "record ", label: "record", description: "record <keep|discard|crash> [description]" }];
			}
			if (prefix.startsWith("record ")) {
				return [
					{ value: "record keep ", label: "keep", description: "Mark run as improved" },
					{ value: "record discard ", label: "discard", description: "Mark run as non-improving" },
					{ value: "record crash ", label: "crash", description: "Mark run as crashed" },
				];
			}
			return null;
		},
	});
});

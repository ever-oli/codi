/**
 * Export-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import { spawn, spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { Component, EditorComponent, TUI } from "@mariozechner/pi-tui";
import { getShareViewerUrl } from "../../../config.js";
import type { AgentSession } from "../../../core/agent-session.js";
import { copyToClipboard } from "../../../utils/clipboard.js";
import { BorderedLoader } from "../components/bordered-loader.js";
import { theme } from "../theme/theme.js";

export interface ExportCommandContext {
	readonly session: AgentSession;
	readonly ui: TUI;
	readonly editor: EditorComponent;
	readonly editorContainer: { addChild(child: Component): void; clear(): void };

	showStatus(message: string): void;
	showError(message: string): void;
}

export async function handleExportCommand(ctx: ExportCommandContext, text: string): Promise<void> {
	const parts = text.split(/\s+/);
	const outputPath = parts.length > 1 ? parts[1] : undefined;

	try {
		const filePath = await ctx.session.exportToHtml(outputPath);
		ctx.showStatus(`Session exported to: ${filePath}`);
	} catch (error: unknown) {
		ctx.showError(`Failed to export session: ${error instanceof Error ? error.message : "Unknown error"}`);
	}
}

export async function handleShareCommand(ctx: ExportCommandContext): Promise<void> {
	// Check if gh is available and logged in
	try {
		const authResult = spawnSync("gh", ["auth", "status"], { encoding: "utf-8" });
		if (authResult.status !== 0) {
			ctx.showError("GitHub CLI is not logged in. Run 'gh auth login' first.");
			return;
		}
	} catch {
		ctx.showError("GitHub CLI (gh) is not installed. Install it from https://cli.github.com/");
		return;
	}

	// Export to a temp file
	const tmpFile = path.join(os.tmpdir(), "session.html");
	try {
		await ctx.session.exportToHtml(tmpFile);
	} catch (error: unknown) {
		ctx.showError(`Failed to export session: ${error instanceof Error ? error.message : "Unknown error"}`);
		return;
	}

	// Show cancellable loader, replacing the editor
	const loader = new BorderedLoader(ctx.ui, theme, "Creating gist...");
	ctx.editorContainer.clear();
	ctx.editorContainer.addChild(loader);
	ctx.ui.setFocus(loader);
	ctx.ui.requestRender();

	const restoreEditor = () => {
		loader.dispose();
		ctx.editorContainer.clear();
		ctx.editorContainer.addChild(ctx.editor);
		ctx.ui.setFocus(ctx.editor);
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
		ctx.showStatus("Share cancelled");
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
			ctx.showError(`Failed to create gist: ${errorMsg}`);
			return;
		}

		// Extract gist ID from the URL returned by gh
		const gistUrl = result.stdout?.trim();
		const gistId = gistUrl?.split("/").pop();
		if (!gistId) {
			ctx.showError("Failed to parse gist ID from gh output");
			return;
		}

		// Create the preview URL
		const previewUrl = getShareViewerUrl(gistId);
		ctx.showStatus(`Share URL: ${previewUrl}\nGist: ${gistUrl}`);
	} catch (error: unknown) {
		if (!loader.signal.aborted) {
			restoreEditor();
			ctx.showError(`Failed to create gist: ${error instanceof Error ? error.message : "Unknown error"}`);
		}
	}
}

export function handleCopyCommand(ctx: ExportCommandContext): void {
	const text = ctx.session.getLastAssistantText();
	if (!text) {
		ctx.showError("No agent messages to copy yet.");
		return;
	}

	try {
		copyToClipboard(text);
		ctx.showStatus("Copied last agent message to clipboard");
	} catch (error) {
		ctx.showError(error instanceof Error ? error.message : String(error));
	}
}

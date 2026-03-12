/**
 * Miscellaneous slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { MarkdownTheme } from "@mariozechner/pi-tui";
import { type Component, DynamicBorder, Markdown, Spacer, Text, visibleWidth } from "@mariozechner/pi-tui";
import { getDebugLogPath } from "../../../config.js";
import type { AgentSession } from "../../../core/agent-session.js";
import type { AppAction, KeybindingsManager } from "../../../core/keybindings.js";
import type { SettingsManager } from "../../../core/settings-manager.js";
import { getChangelogPath, parseChangelog } from "../../../utils/changelog.js";
import { theme } from "../theme/theme.js";

export interface MiscCommandContext {
	readonly session: AgentSession;
	readonly settingsManager: SettingsManager;
	readonly keybindings: KeybindingsManager;
	readonly ui: {
		terminal: { columns: number; rows: number };
		render(width: number): string[];
		requestRender(force?: boolean): void;
	};
	readonly chatContainer: { addChild(child: Component): void };

	showStatus(message: string): void;
	showWarning(message: string): void;
	showError(message: string): void;
	showLoadedResources(options?: {
		extensionPaths?: string[];
		listingMode?: "none" | "summary" | "summaryPreview" | "full";
		issuesOnly?: boolean;
		showDiagnostics?: boolean;
	}): void;
	getMarkdownThemeWithSettings(): MarkdownTheme;
	capitalizeKey(key: string): string;
	getAppKeyDisplay(action: AppAction): string;
	getEditorKeyDisplay(action: string): string;
}

export function handleResourcesCommand(ctx: MiscCommandContext, text: string): void {
	const argText = text
		.replace(/^\/resources\s*/, "")
		.trim()
		.toLowerCase();
	if (!argText || argText === "summary") {
		ctx.showLoadedResources({
			extensionPaths: ctx.session.extensionRunner?.getExtensionPaths() ?? [],
			listingMode: "summary",
			showDiagnostics: true,
		});
		return;
	}
	if (argText === "issues") {
		ctx.showLoadedResources({
			extensionPaths: ctx.session.extensionRunner?.getExtensionPaths() ?? [],
			issuesOnly: true,
			showDiagnostics: true,
		});
		return;
	}
	if (argText === "full") {
		ctx.showLoadedResources({
			extensionPaths: ctx.session.extensionRunner?.getExtensionPaths() ?? [],
			listingMode: "full",
			showDiagnostics: true,
		});
		return;
	}
	ctx.showWarning("Usage: /resources [summary|issues|full]");
}

export function handleChangelogCommand(ctx: MiscCommandContext): void {
	const changelogPath = getChangelogPath();
	const allEntries = parseChangelog(changelogPath);

	const changelogMarkdown =
		allEntries.length > 0
			? allEntries
					.reverse()
					.map((e) => e.content)
					.join("\n\n")
			: "No changelog entries found.";

	ctx.chatContainer.addChild(new Spacer(1));
	ctx.chatContainer.addChild(new DynamicBorder());
	ctx.chatContainer.addChild(new Text(theme.bold(theme.fg("accent", "What's New")), 1, 0));
	ctx.chatContainer.addChild(new Spacer(1));
	ctx.chatContainer.addChild(new Markdown(changelogMarkdown, 1, 1, ctx.getMarkdownThemeWithSettings()));
	ctx.chatContainer.addChild(new DynamicBorder());
	ctx.ui.requestRender();
}

export function handleHotkeysCommand(ctx: MiscCommandContext): void {
	// Navigation keybindings
	const cursorWordLeft = ctx.getEditorKeyDisplay("cursorWordLeft");
	const cursorWordRight = ctx.getEditorKeyDisplay("cursorWordRight");
	const cursorLineStart = ctx.getEditorKeyDisplay("cursorLineStart");
	const cursorLineEnd = ctx.getEditorKeyDisplay("cursorLineEnd");
	const jumpForward = ctx.getEditorKeyDisplay("jumpForward");
	const jumpBackward = ctx.getEditorKeyDisplay("jumpBackward");
	const pageUp = ctx.getEditorKeyDisplay("pageUp");
	const pageDown = ctx.getEditorKeyDisplay("pageDown");

	// Editing keybindings
	const submit = ctx.getEditorKeyDisplay("submit");
	const newLine = ctx.getEditorKeyDisplay("newLine");
	const deleteWordBackward = ctx.getEditorKeyDisplay("deleteWordBackward");
	const deleteWordForward = ctx.getEditorKeyDisplay("deleteWordForward");
	const deleteToLineStart = ctx.getEditorKeyDisplay("deleteToLineStart");
	const deleteToLineEnd = ctx.getEditorKeyDisplay("deleteToLineEnd");
	const yank = ctx.getEditorKeyDisplay("yank");
	const yankPop = ctx.getEditorKeyDisplay("yankPop");
	const undo = ctx.getEditorKeyDisplay("undo");
	const tab = ctx.getEditorKeyDisplay("tab");

	// App keybindings
	const interrupt = ctx.getAppKeyDisplay("interrupt");
	const clear = ctx.getAppKeyDisplay("clear");
	const exit = ctx.getAppKeyDisplay("exit");
	const suspend = ctx.getAppKeyDisplay("suspend");
	const cycleThinkingLevel = ctx.getAppKeyDisplay("cycleThinkingLevel");
	const cycleModelForward = ctx.getAppKeyDisplay("cycleModelForward");
	const selectModel = ctx.getAppKeyDisplay("selectModel");
	const expandTools = ctx.getAppKeyDisplay("expandTools");
	const toggleThinking = ctx.getAppKeyDisplay("toggleThinking");
	const externalEditor = ctx.getAppKeyDisplay("externalEditor");
	const followUp = ctx.getAppKeyDisplay("followUp");
	const dequeue = ctx.getAppKeyDisplay("dequeue");

	let hotkeys = `
**Navigation**
| Key | Action |
|-----|--------|
| \`Arrow keys\` | Move cursor / browse history (Up when empty) |
| \`${cursorWordLeft}\` / \`${cursorWordRight}\` | Move by word |
| \`${cursorLineStart}\` | Start of line |
| \`${cursorLineEnd}\` | End of line |
| \`${jumpForward}\` | Jump forward to character |
| \`${jumpBackward}\` | Jump backward to character |
| \`${pageUp}\` / \`${pageDown}\` | Scroll by page |

**Editing**
| Key | Action |
|-----|--------|
| \`${submit}\` | Send message |
| \`${newLine}\` | New line${process.platform === "win32" ? " (Ctrl+Enter on Windows Terminal)" : ""} |
| \`${deleteWordBackward}\` | Delete word backwards |
| \`${deleteWordForward}\` | Delete word forwards |
| \`${deleteToLineStart}\` | Delete to start of line |
| \`${deleteToLineEnd}\` | Delete to end of line |
| \`${yank}\` | Paste the most-recently-deleted text |
| \`${yankPop}\` | Cycle through the deleted text after pasting |
| \`${undo}\` | Undo |

**Other**
| Key | Action |
|-----|--------|
| \`${tab}\` | Path completion / accept autocomplete |
| \`${interrupt}\` | Cancel autocomplete / abort streaming |
| \`${clear}\` | Clear editor (first) / exit (second) |
| \`${exit}\` | Exit (when editor is empty) |
| \`${suspend}\` | Suspend to background |
| \`${cycleThinkingLevel}\` | Cycle thinking level |
| \`${cycleModelForward}\` | Cycle models |
| \`${selectModel}\` | Open model selector |
| \`${expandTools}\` | Toggle tool output expansion |
| \`${toggleThinking}\` | Toggle thinking block visibility |
| \`${externalEditor}\` | Edit message in external editor |
| \`${followUp}\` | Queue follow-up message |
| \`${dequeue}\` | Restore queued messages |
| \`Ctrl+V\` | Paste image from clipboard |
| \`/\` | Slash commands |
| \`!\` | Run bash command |
| \`!!\` | Run bash command (excluded from context) |
`;

	// Add extension-registered shortcuts
	const extensionRunner = ctx.session.extensionRunner;
	if (extensionRunner) {
		const shortcuts = extensionRunner.getShortcuts(ctx.keybindings.getEffectiveConfig());
		if (shortcuts.size > 0) {
			hotkeys += `
**Extensions**
| Key | Action |
|-----|--------|
`;
			for (const [key, shortcut] of shortcuts) {
				const description = shortcut.description ?? shortcut.extensionPath;
				const keyDisplay = key.replace(/\b\w/g, (c) => c.toUpperCase());
				hotkeys += `| \`${keyDisplay}\` | ${description} |\n`;
			}
		}
	}

	ctx.chatContainer.addChild(new Spacer(1));
	ctx.chatContainer.addChild(new DynamicBorder());
	ctx.chatContainer.addChild(new Text(theme.bold(theme.fg("accent", "Keyboard Shortcuts")), 1, 0));
	ctx.chatContainer.addChild(new Spacer(1));
	ctx.chatContainer.addChild(new Markdown(hotkeys.trim(), 1, 1, ctx.getMarkdownThemeWithSettings()));
	ctx.chatContainer.addChild(new DynamicBorder());
	ctx.ui.requestRender();
}

export function handleDebugCommand(ctx: MiscCommandContext): void {
	const width = ctx.ui.terminal.columns;
	const height = ctx.ui.terminal.rows;
	const allLines = ctx.ui.render(width);

	const debugLogPath = getDebugLogPath();
	const debugData = [
		`Debug output at ${new Date().toISOString()}`,
		`Terminal: ${width}x${height}`,
		`Total lines: ${allLines.length}`,
		"",
		"=== All rendered lines with visible widths ===",
		...allLines.map((line, idx) => {
			const vw = visibleWidth(line);
			const escaped = JSON.stringify(line);
			return `[${idx}] (w=${vw}) ${escaped}`;
		}),
		"",
		"=== Agent messages (JSONL) ===",
		...ctx.session.messages.map((msg) => JSON.stringify(msg)),
		"",
	].join("\n");

	fs.mkdirSync(path.dirname(debugLogPath), { recursive: true });
	fs.writeFileSync(debugLogPath, debugData);

	ctx.chatContainer.addChild(new Spacer(1));
	ctx.chatContainer.addChild(
		new Text(`${theme.fg("accent", "✓ Debug log written")}\n${theme.fg("muted", debugLogPath)}`, 1, 1),
	);
	ctx.ui.requestRender();
}

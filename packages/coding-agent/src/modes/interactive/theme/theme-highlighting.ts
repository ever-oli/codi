import type { MarkdownTheme } from "@mariozechner/pi-tui";
import chalk from "chalk";
import { highlight, supportsLanguage } from "cli-highlight";

interface HighlightThemeAdapter {
	fg(color: string, text: string): string;
	bold(text: string): string;
	italic(text: string): string;
	underline(text: string): string;
}

type CliHighlightTheme = Record<string, (s: string) => string>;

let cachedHighlightThemeFor: HighlightThemeAdapter | undefined;
let cachedCliHighlightTheme: CliHighlightTheme | undefined;

function buildCliHighlightTheme(theme: HighlightThemeAdapter): CliHighlightTheme {
	return {
		keyword: (text: string) => theme.fg("syntaxKeyword", text),
		built_in: (text: string) => theme.fg("syntaxType", text),
		literal: (text: string) => theme.fg("syntaxNumber", text),
		number: (text: string) => theme.fg("syntaxNumber", text),
		string: (text: string) => theme.fg("syntaxString", text),
		comment: (text: string) => theme.fg("syntaxComment", text),
		function: (text: string) => theme.fg("syntaxFunction", text),
		title: (text: string) => theme.fg("syntaxFunction", text),
		class: (text: string) => theme.fg("syntaxType", text),
		type: (text: string) => theme.fg("syntaxType", text),
		attr: (text: string) => theme.fg("syntaxVariable", text),
		variable: (text: string) => theme.fg("syntaxVariable", text),
		params: (text: string) => theme.fg("syntaxVariable", text),
		operator: (text: string) => theme.fg("syntaxOperator", text),
		punctuation: (text: string) => theme.fg("syntaxPunctuation", text),
	};
}

function getCliHighlightTheme(theme: HighlightThemeAdapter): CliHighlightTheme {
	if (cachedHighlightThemeFor !== theme || !cachedCliHighlightTheme) {
		cachedHighlightThemeFor = theme;
		cachedCliHighlightTheme = buildCliHighlightTheme(theme);
	}
	return cachedCliHighlightTheme;
}

function normalizeLanguage(lang?: string): string | undefined {
	return lang && supportsLanguage(lang) ? lang : undefined;
}

export function highlightCodeWithTheme(
	theme: HighlightThemeAdapter,
	code: string,
	lang?: string,
	options?: { fallback?: (line: string) => string },
): string[] {
	try {
		return highlight(code, {
			language: normalizeLanguage(lang),
			ignoreIllegals: true,
			theme: getCliHighlightTheme(theme),
		}).split("\n");
	} catch {
		if (!options?.fallback) {
			return code.split("\n");
		}
		return code.split("\n").map((line) => options.fallback!(line));
	}
}

export function getLanguageFromPath(filePath: string): string | undefined {
	const ext = filePath.split(".").pop()?.toLowerCase();
	if (!ext) return undefined;

	const extToLang: Record<string, string> = {
		ts: "typescript",
		tsx: "typescript",
		js: "javascript",
		jsx: "javascript",
		mjs: "javascript",
		cjs: "javascript",
		py: "python",
		rb: "ruby",
		rs: "rust",
		go: "go",
		java: "java",
		kt: "kotlin",
		swift: "swift",
		c: "c",
		h: "c",
		cpp: "cpp",
		cc: "cpp",
		cxx: "cpp",
		hpp: "cpp",
		cs: "csharp",
		php: "php",
		sh: "bash",
		bash: "bash",
		zsh: "bash",
		fish: "fish",
		ps1: "powershell",
		sql: "sql",
		html: "html",
		htm: "html",
		css: "css",
		scss: "scss",
		sass: "sass",
		less: "less",
		json: "json",
		yaml: "yaml",
		yml: "yaml",
		toml: "toml",
		xml: "xml",
		md: "markdown",
		markdown: "markdown",
		dockerfile: "dockerfile",
		makefile: "makefile",
		cmake: "cmake",
		lua: "lua",
		perl: "perl",
		r: "r",
		scala: "scala",
		clj: "clojure",
		ex: "elixir",
		exs: "elixir",
		erl: "erlang",
		hs: "haskell",
		ml: "ocaml",
		vim: "vim",
		graphql: "graphql",
		proto: "protobuf",
		tf: "hcl",
		hcl: "hcl",
	};

	return extToLang[ext];
}

export function createMarkdownTheme(theme: HighlightThemeAdapter): MarkdownTheme {
	return {
		heading: (text: string) => theme.fg("mdHeading", text),
		link: (text: string) => theme.fg("mdLink", text),
		linkUrl: (text: string) => theme.fg("mdLinkUrl", text),
		code: (text: string) => theme.fg("mdCode", text),
		codeBlock: (text: string) => theme.fg("mdCodeBlock", text),
		codeBlockBorder: (text: string) => theme.fg("mdCodeBlockBorder", text),
		quote: (text: string) => theme.fg("mdQuote", text),
		quoteBorder: (text: string) => theme.fg("mdQuoteBorder", text),
		hr: (text: string) => theme.fg("mdHr", text),
		listBullet: (text: string) => theme.fg("mdListBullet", text),
		bold: (text: string) => theme.bold(text),
		italic: (text: string) => theme.italic(text),
		underline: (text: string) => theme.underline(text),
		strikethrough: (text: string) => chalk.strikethrough(text),
		highlightCode: (code: string, lang?: string) =>
			highlightCodeWithTheme(theme, code, lang, {
				fallback: (line) => theme.fg("mdCodeBlock", line),
			}),
	};
}

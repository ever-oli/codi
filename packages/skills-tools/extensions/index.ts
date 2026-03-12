import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { StringEnum, Text, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

type SkillScope = "user" | "project";
type SkillsAction = "list" | "read" | "write" | "remove";

interface SkillRecord {
	name: string;
	scope: SkillScope;
	relativePath: string;
	filePath: string;
	size: number;
}

interface SkillsToolDetails {
	action: SkillsAction;
	scope: SkillScope;
	root: string;
	skills: SkillRecord[];
	content?: string;
	error?: string;
}

const SKILLS_PARAMS = Type.Object({
	action: StringEnum(["list", "read", "write", "remove"] as const),
	scope: Type.Optional(StringEnum(["user", "project"] as const)),
	name: Type.Optional(Type.String({ description: "Skill name (directory name) for read/write/remove." })),
	query: Type.Optional(Type.String({ description: "Filter skills by name/path for list." })),
	content: Type.Optional(Type.String({ description: "SKILL.md content for write." })),
	overwrite: Type.Optional(Type.Boolean({ description: "Allow overwriting existing SKILL.md on write." })),
	confirm: Type.Optional(Type.Boolean({ description: "Required true for remove action." })),
	limit: Type.Optional(Type.Number({ description: "Maximum number of list results. Default: 50." })),
	maxChars: Type.Optional(Type.Number({ description: "Maximum chars returned for read. Default: 12000." })),
});

const SKILL_NAME_RE = /^[a-z0-9](?:[a-z0-9-]{0,63})$/;

function resolveAgentDir(): string {
	const explicit = process.env.PI_CODING_AGENT_DIR?.trim();
	if (explicit) {
		if (explicit === "~") return os.homedir();
		if (explicit.startsWith("~/")) return path.join(os.homedir(), explicit.slice(2));
		return explicit;
	}
	return path.join(os.homedir(), ".pi", "agent");
}

function userSkillsRoot(): string {
	return path.join(resolveAgentDir(), "skills");
}

function projectSkillsRoot(cwd: string): string {
	return path.join(cwd, ".pi", "skills");
}

function ensureInsideRoot(root: string, candidate: string): boolean {
	const resolvedRoot = path.resolve(root);
	const resolvedCandidate = path.resolve(candidate);
	return resolvedCandidate === resolvedRoot || resolvedCandidate.startsWith(`${resolvedRoot}${path.sep}`);
}

function listSkillRecords(root: string, scope: SkillScope): SkillRecord[] {
	if (!fs.existsSync(root)) return [];
	const records: SkillRecord[] = [];
	const stack: string[] = [root];

	while (stack.length > 0) {
		const dir = stack.pop();
		if (!dir) continue;
		let entries: fs.Dirent[] = [];
		try {
			entries = fs.readdirSync(dir, { withFileTypes: true });
		} catch {
			continue;
		}
		for (const entry of entries) {
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				stack.push(full);
				continue;
			}
			if (!entry.isFile() || entry.name !== "SKILL.md") continue;
			let size = 0;
			try {
				size = fs.statSync(full).size;
			} catch {
				size = 0;
			}
			const skillDir = path.dirname(full);
			const relativePath = path.relative(root, skillDir);
			const name = path.basename(skillDir);
			records.push({ name, scope, relativePath, filePath: full, size });
		}
	}

	return records.sort((a, b) => a.relativePath.localeCompare(b.relativePath));
}

function findSkill(records: SkillRecord[], name: string): { skill?: SkillRecord; error?: string } {
	const needle = name.trim();
	if (!needle) return { error: "name is required." };
	const exactPath = records.filter((record) => record.relativePath === needle || record.name === needle);
	if (exactPath.length === 1) return { skill: exactPath[0] };
	if (exactPath.length > 1) {
		return {
			error: `Multiple skills matched "${needle}". Use a relative path. Matches: ${exactPath.map((skill) => skill.relativePath).join(", ")}`,
		};
	}
	return { error: `Skill "${needle}" not found.` };
}

function truncateText(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	return `${text.slice(0, maxChars)}\n\n...[truncated]`;
}

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool<typeof SKILLS_PARAMS, SkillsToolDetails>({
			name: "skills",
			label: "Skills",
			description:
				"Inspect and manage skill files under user/project skill roots. Actions: list, read, write, remove.",
			promptSnippet:
				"Manage SKILL.md files in user/project roots (discover, inspect, create, and remove skills).",
			parameters: SKILLS_PARAMS,
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				const action = params.action;
				const scope: SkillScope = params.scope ?? "user";
				const root = scope === "user" ? userSkillsRoot() : projectSkillsRoot(ctx.cwd);
				const records = listSkillRecords(root, scope);

				if (action === "list") {
					const query = params.query?.trim().toLowerCase();
					const limit = Math.max(1, Math.min(Math.floor(params.limit ?? 50), 200));
					const filtered = query
						? records.filter((record) => {
								const haystack = `${record.name} ${record.relativePath}`.toLowerCase();
								return haystack.includes(query);
							})
						: records;
					const visible = filtered.slice(0, limit);
					const text =
						visible.length === 0
							? "No skills found."
							: visible.map((record, index) => `${index + 1}. ${record.relativePath} (${record.size} bytes)`).join("\n");
					return {
						content: [{ type: "text", text }],
						details: { action, scope, root, skills: visible },
					};
				}

				if (action === "read") {
					if (!params.name?.trim()) {
						const details: SkillsToolDetails = { action, scope, root, skills: records, error: "name is required for read." };
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const match = findSkill(records, params.name);
					if (!match.skill) {
						const details: SkillsToolDetails = { action, scope, root, skills: records, error: match.error };
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					let raw = "";
					try {
						raw = fs.readFileSync(match.skill.filePath, "utf8");
					} catch {
						const details: SkillsToolDetails = {
							action,
							scope,
							root,
							skills: records,
							error: `Failed to read ${match.skill.relativePath}.`,
						};
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const maxChars = Math.max(200, Math.min(Math.floor(params.maxChars ?? 12000), 50000));
					const content = truncateText(raw, maxChars);
					return {
						content: [{ type: "text", text: content }],
						details: { action, scope, root, skills: [match.skill], content },
					};
				}

				if (action === "write") {
					const name = params.name?.trim();
					const content = params.content;
					if (!name) {
						const details: SkillsToolDetails = { action, scope, root, skills: records, error: "name is required for write." };
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					if (!SKILL_NAME_RE.test(name)) {
						const details: SkillsToolDetails = {
							action,
							scope,
							root,
							skills: records,
							error: "name must match ^[a-z0-9](?:[a-z0-9-]{0,63})$.",
						};
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					if (!content?.trim()) {
						const details: SkillsToolDetails = { action, scope, root, skills: records, error: "content is required for write." };
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}

					const skillDir = path.join(root, name);
					const skillFile = path.join(skillDir, "SKILL.md");
					if (!ensureInsideRoot(root, skillFile)) {
						const details: SkillsToolDetails = { action, scope, root, skills: records, error: "Invalid path outside skill root." };
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					const exists = fs.existsSync(skillFile);
					if (exists && !params.overwrite) {
						const details: SkillsToolDetails = {
							action,
							scope,
							root,
							skills: records,
							error: `Skill "${name}" already exists. Set overwrite=true to replace it.`,
						};
						return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
					}
					fs.mkdirSync(skillDir, { recursive: true });
					fs.writeFileSync(skillFile, content, "utf8");
					const nextRecords = listSkillRecords(root, scope);
					const written = findSkill(nextRecords, name).skill;
					return {
						content: [{ type: "text", text: `Wrote skill "${name}" to ${scope} skills.` }],
						details: { action, scope, root, skills: written ? [written] : [] },
					};
				}

				if (!params.name?.trim()) {
					const details: SkillsToolDetails = { action, scope, root, skills: records, error: "name is required for remove." };
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				if (!params.confirm) {
					const details: SkillsToolDetails = {
						action,
						scope,
						root,
						skills: records,
						error: "remove requires confirm=true.",
					};
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				const match = findSkill(records, params.name);
				if (!match.skill) {
					const details: SkillsToolDetails = { action, scope, root, skills: records, error: match.error };
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				const skillDir = path.dirname(match.skill.filePath);
				if (!ensureInsideRoot(root, skillDir)) {
					const details: SkillsToolDetails = { action, scope, root, skills: records, error: "Invalid path outside skill root." };
					return { content: [{ type: "text", text: `Error: ${details.error}` }], details, isError: true };
				}
				fs.rmSync(skillDir, { recursive: true, force: true });
				const nextRecords = listSkillRecords(root, scope);
				return {
					content: [{ type: "text", text: `Removed skill "${match.skill.relativePath}".` }],
					details: { action, scope, root, skills: nextRecords },
				};
			},
			renderCall(args, theme) {
				let text = theme.fg("toolTitle", theme.bold("skills ")) + theme.fg("muted", args.action);
				text += ` ${theme.fg("accent", args.scope ?? "user")}`;
				if (args.name) text += ` ${theme.fg("dim", args.name)}`;
				if (args.query) text += ` ${theme.fg("dim", `query="${args.query}"`)}`;
				return new Text(text, 0, 0);
			},
			renderResult(result, _options, theme) {
				const details = result.details as SkillsToolDetails | undefined;
				if (!details) {
					const block = result.content[0];
					return new Text(block?.type === "text" ? block.text : "", 0, 0);
				}
				if (details.error) {
					return new Text(theme.fg("error", `Error: ${details.error}`), 0, 0);
				}
				const lines = [
					`${theme.fg("success", "✓")} ${theme.fg("muted", `${details.skills.length} skill(s) in ${details.scope}`)}`,
				];
				for (const skill of details.skills.slice(0, 6)) {
					lines.push(theme.fg("dim", `- ${skill.relativePath}`));
				}
				if (details.skills.length > 6) {
					lines.push(theme.fg("dim", `... ${details.skills.length - 6} more`));
				}
				return new Text(lines.join("\n"), 0, 0);
			},
		}),
	);
});

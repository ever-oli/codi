import { StringEnum, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";
import type { CustomEntry, CustomMessageEntry, SessionEntry, SessionMessageEntry } from "@mariozechner/pi-extension-sdk";

interface SearchHit {
	id: string;
	entryType: SessionEntry["type"];
	label: string;
	text: string;
}

const SEARCH_PARAMS = Type.Object({
	query: Type.String({ description: "Text to search for in the session." }),
	scope: Type.Optional(StringEnum(["branch", "all"] as const)),
	limit: Type.Optional(Type.Number({ description: "Maximum number of results to return. Default: 8." })),
});

function stringifyUnknown(value: unknown): string {
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
}

function extractEntryText(entry: SessionEntry): { label: string; text: string } | undefined {
	switch (entry.type) {
		case "message": {
			const message = (entry as SessionMessageEntry).message;
			if (!("content" in message)) {
				return { label: `message:${message.role}`, text: stringifyUnknown(message) };
			}
			if (typeof message.content === "string") {
				return { label: `message:${message.role}`, text: message.content };
			}
			const text = message.content
				.map((block: (typeof message.content)[number]) => {
					if (block.type === "text") return block.text;
					if (block.type === "toolCall") return `${block.name} ${stringifyUnknown(block.arguments)}`;
					if (block.type === "thinking") return block.thinking;
					return "";
				})
				.join(" ");
			return { label: `message:${message.role}`, text };
		}
		case "custom_message": {
			const custom = entry as CustomMessageEntry;
			const content =
				typeof custom.content === "string"
					? custom.content
					: custom.content
							.map((block) => (block.type === "text" ? block.text : block.type === "image" ? block.mimeType : ""))
							.join(" ");
			return { label: `custom_message:${custom.customType}`, text: content };
		}
		case "custom": {
			const custom = entry as CustomEntry;
			return { label: `custom:${custom.customType}`, text: stringifyUnknown(custom.data) };
		}
		case "branch_summary":
			return { label: "branch_summary", text: entry.summary };
		case "compaction":
			return { label: "compaction", text: entry.summary };
		case "session_info":
			return { label: "session_info", text: entry.name ?? "" };
		case "workflow_artifact":
			return { label: `workflow_artifact:${entry.artifact.type}`, text: stringifyUnknown(entry.artifact) };
		case "workflow_verification":
			return { label: `workflow_verification:${entry.record.status}`, text: stringifyUnknown(entry.record) };
		default:
			return undefined;
	}
}

function buildExcerpt(text: string, query: string): string {
	const normalizedText = text.replace(/\s+/g, " ").trim();
	const normalizedQuery = query.trim().toLowerCase();
	const index = normalizedText.toLowerCase().indexOf(normalizedQuery);
	if (index < 0) {
		return normalizedText.slice(0, 220);
	}
	const start = Math.max(index - 80, 0);
	const end = Math.min(index + normalizedQuery.length + 120, normalizedText.length);
	return normalizedText.slice(start, end);
}

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool({
		name: "session_search",
		label: "Session Search",
		description: "Search the current session history, including messages, custom entries, and workflow records.",
		parameters: SEARCH_PARAMS,
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const scope = params.scope ?? "branch";
			const entries = scope === "all" ? ctx.sessionManager.getEntries() : ctx.sessionManager.getBranch();
			const limit = Math.min(Math.max(params.limit ?? 8, 1), 20);
			const hits: SearchHit[] = [];

			for (const entry of entries) {
				const extracted = extractEntryText(entry);
				if (!extracted) continue;
				if (!extracted.text.toLowerCase().includes(params.query.toLowerCase())) continue;
				hits.push({
					id: entry.id,
					entryType: entry.type,
					label: extracted.label,
					text: buildExcerpt(extracted.text, params.query),
				});
			}

			const text =
				hits.length === 0
					? `No session hits found for: ${params.query}`
					: hits
							.slice(0, limit)
							.map((hit, index) => `${index + 1}. ${hit.label} (${hit.id})\n${hit.text}`)
							.join("\n\n");

			return {
				content: [{ type: "text", text }],
				details: { query: params.query, scope, hitCount: hits.length, hits: hits.slice(0, limit) },
			};
		},
		}),
	);
});

import { appendFileSync, writeFileSync } from "fs";
import type { FileEntry, SessionEntry } from "./session-manager.js";

export interface SessionIndexState {
	byId: Map<string, SessionEntry>;
	labelsById: Map<string, string>;
	leafId: string | null;
}

export function buildSessionIndex(fileEntries: FileEntry[]): SessionIndexState {
	const byId = new Map<string, SessionEntry>();
	const labelsById = new Map<string, string>();
	let leafId: string | null = null;

	for (const entry of fileEntries) {
		if (entry.type === "session") continue;
		byId.set(entry.id, entry);
		leafId = entry.id;
		if (entry.type === "label") {
			if (entry.label) {
				labelsById.set(entry.targetId, entry.label);
			} else {
				labelsById.delete(entry.targetId);
			}
		}
	}

	return { byId, labelsById, leafId };
}

export function rewriteSessionFile(
	fileEntries: FileEntry[],
	options: { persist: boolean; sessionFile?: string },
): void {
	if (!options.persist || !options.sessionFile) return;
	const content = `${fileEntries.map((entry) => JSON.stringify(entry)).join("\n")}\n`;
	writeFileSync(options.sessionFile, content);
}

export function persistSessionEntry(
	entry: SessionEntry,
	options: {
		fileEntries: FileEntry[];
		persist: boolean;
		sessionFile?: string;
		flushed: boolean;
	},
): boolean {
	if (!options.persist || !options.sessionFile) {
		return options.flushed;
	}

	const hasAssistant = options.fileEntries.some(
		(candidate) => candidate.type === "message" && candidate.message.role === "assistant",
	);
	if (!hasAssistant) {
		return false;
	}

	if (!options.flushed) {
		for (const fileEntry of options.fileEntries) {
			appendFileSync(options.sessionFile, `${JSON.stringify(fileEntry)}\n`);
		}
		return true;
	}

	appendFileSync(options.sessionFile, `${JSON.stringify(entry)}\n`);
	return true;
}

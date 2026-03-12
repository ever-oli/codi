export type SlashCommandSource = "extension" | "prompt" | "skill";

export type SlashCommandLocation = "user" | "project" | "path";

export interface SlashCommandInfo {
	name: string;
	description?: string;
	source: SlashCommandSource;
	location?: SlashCommandLocation;
	path?: string;
}

export interface BuiltinSlashCommand {
	name: string;
	description: string;
}

export const BUILTIN_SLASH_COMMANDS: ReadonlyArray<BuiltinSlashCommand> = [
	{ name: "settings", description: "Open settings menu" },
	{ name: "model", description: "Select model (opens selector UI)" },
	{ name: "models", description: "Inspect and set role-based model profile (main/task/compact/quick)" },
	{ name: "scoped-models", description: "Enable/disable models for Ctrl+P cycling" },
	{ name: "export", description: "Export session to HTML file" },
	{ name: "share", description: "Share session as a secret GitHub gist" },
	{ name: "copy", description: "Copy last agent message to clipboard" },
	{ name: "name", description: "Set session display name" },
	{ name: "session", description: "Show session info and stats" },
	{ name: "plan", description: "Inspect, update, or split the workflow plan" },
	{ name: "phase", description: "Show or advance workflow phase" },
	{ name: "task", description: "Show or mutate workflow tasks" },
	{ name: "verify", description: "Record explicit verification for the active workflow task" },
	{ name: "workflow", description: "Show workflow summary (phase, task, verification, completion)" },
	{ name: "events", description: "Inspect runtime/session events, filter, tail, and prune" },
	{ name: "queue", description: "Inspect/retry delivery queue and dead-letter messages" },
	{ name: "lanes", description: "Inspect and tune named-lane concurrency limits" },
	{ name: "packages", description: "Install/update/remove/manage packages and resource enablement" },
	{ name: "mailbox", description: "Inspect inbox/outbox and manage delegation handoff envelopes" },
	{ name: "delegated", description: "Inspect and update delegated task progress" },
	{ name: "heartbeat", description: "Inspect and control core heartbeat + cron orchestration" },
	{ name: "ops", description: "Unified runtime operations dashboard and command umbrella" },
	{ name: "resources", description: "Show startup resources (summary/issues/full)" },
	{ name: "changelog", description: "Show changelog entries" },
	{ name: "hotkeys", description: "Show all keyboard shortcuts" },
	{ name: "fork", description: "Create a new fork from a previous message" },
	{ name: "tree", description: "Navigate session tree (switch branches)" },
	{ name: "login", description: "Login with OAuth provider" },
	{ name: "logout", description: "Logout from OAuth provider" },
	{ name: "new", description: "Start a new session" },
	{ name: "compact", description: "Manually compact the session context" },
	{ name: "resume", description: "Resume a different session" },
	{ name: "reload", description: "Reload extensions, skills, prompts, and themes" },
	{ name: "quit", description: "Quit pi" },
];

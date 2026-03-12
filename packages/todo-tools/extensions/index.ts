import {
	StringEnum,
	Text,
	Type,
	defineExtension,
	defineTool,
	matchesKey,
	truncateToWidth,
} from "@mariozechner/pi-extension-sdk";
import type { ExtensionContext, Theme } from "@mariozechner/pi-extension-sdk";

interface Todo {
	id: number;
	text: string;
	done: boolean;
}

interface TodoDetails {
	action: "list" | "add" | "toggle" | "clear";
	todos: Todo[];
	nextId: number;
	error?: string;
}

const TODO_PARAMS = Type.Object({
	action: StringEnum(["list", "add", "toggle", "clear"] as const),
	text: Type.Optional(Type.String({ description: "Todo text for add." })),
	id: Type.Optional(Type.Number({ description: "Todo ID for toggle." })),
});

class TodoListComponent {
	private todos: Todo[];
	private theme: Theme;
	private onClose: () => void;
	private cachedWidth?: number;
	private cachedLines?: string[];

	constructor(todos: Todo[], theme: Theme, onClose: () => void) {
		this.todos = todos;
		this.theme = theme;
		this.onClose = onClose;
	}

	handleInput(data: string): void {
		if (matchesKey(data, "escape") || matchesKey(data, "ctrl+c")) {
			this.onClose();
		}
	}

	render(width: number): string[] {
		if (this.cachedLines && this.cachedWidth === width) {
			return this.cachedLines;
		}

		const lines: string[] = [];
		const header =
			this.theme.fg("borderMuted", "─".repeat(3)) +
			this.theme.fg("accent", " Todos ") +
			this.theme.fg("borderMuted", "─".repeat(Math.max(0, width - 10)));
		lines.push("");
		lines.push(truncateToWidth(header, width));
		lines.push("");

		if (this.todos.length === 0) {
			lines.push(truncateToWidth(`  ${this.theme.fg("dim", "No todos yet.")}`, width));
		} else {
			const done = this.todos.filter((todo) => todo.done).length;
			lines.push(truncateToWidth(`  ${this.theme.fg("muted", `${done}/${this.todos.length} completed`)}`, width));
			lines.push("");
			for (const todo of this.todos) {
				const check = todo.done ? this.theme.fg("success", "✓") : this.theme.fg("dim", "○");
				const id = this.theme.fg("accent", `#${todo.id}`);
				const text = todo.done ? this.theme.fg("dim", todo.text) : this.theme.fg("text", todo.text);
				lines.push(truncateToWidth(`  ${check} ${id} ${text}`, width));
			}
		}

		lines.push("");
		lines.push(truncateToWidth(`  ${this.theme.fg("dim", "Press Escape to close")}`, width));
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

export default defineExtension((pi) => {
	let todos: Todo[] = [];
	let nextId = 1;

	const reconstructState = (ctx: ExtensionContext) => {
		todos = [];
		nextId = 1;
		for (const entry of ctx.sessionManager.getBranch()) {
			if (entry.type !== "message") continue;
			const message = entry.message;
			if (message.role !== "toolResult" || message.toolName !== "todo") continue;
			const details = message.details as TodoDetails | undefined;
			if (details) {
				todos = details.todos;
				nextId = details.nextId;
			}
		}
	};

	pi.on("session_start", (_event, ctx) => reconstructState(ctx));
	pi.on("session_switch", (_event, ctx) => reconstructState(ctx));
	pi.on("session_fork", (_event, ctx) => reconstructState(ctx));
	pi.on("session_tree", (_event, ctx) => reconstructState(ctx));

	pi.registerTool(
		defineTool<typeof TODO_PARAMS, TodoDetails>({
		name: "todo",
		label: "Todo",
		description: "Manage a simple todo list that branches with the current session.",
		parameters: TODO_PARAMS,
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			switch (params.action) {
				case "list":
					return {
						content: [
							{
								type: "text",
								text: todos.length > 0 ? todos.map((todo) => `[${todo.done ? "x" : " "}] #${todo.id}: ${todo.text}`).join("\n") : "No todos",
							},
						],
						details: { action: "list", todos: [...todos], nextId } satisfies TodoDetails,
					};
				case "add": {
					if (!params.text?.trim()) {
						return {
							content: [{ type: "text", text: "Error: text required for add" }],
							details: { action: "add", todos: [...todos], nextId, error: "text required" } satisfies TodoDetails,
						};
					}
					const newTodo: Todo = { id: nextId++, text: params.text, done: false };
					todos.push(newTodo);
					return {
						content: [{ type: "text", text: `Added todo #${newTodo.id}: ${newTodo.text}` }],
						details: { action: "add", todos: [...todos], nextId } satisfies TodoDetails,
					};
				}
				case "toggle": {
					if (params.id === undefined) {
						return {
							content: [{ type: "text", text: "Error: id required for toggle" }],
							details: { action: "toggle", todos: [...todos], nextId, error: "id required" } satisfies TodoDetails,
						};
					}
					const todo = todos.find((item) => item.id === params.id);
					if (!todo) {
						return {
							content: [{ type: "text", text: `Todo #${params.id} not found` }],
							details: {
								action: "toggle",
								todos: [...todos],
								nextId,
								error: `#${params.id} not found`,
							} satisfies TodoDetails,
						};
					}
					todo.done = !todo.done;
					return {
						content: [{ type: "text", text: `Todo #${todo.id} ${todo.done ? "completed" : "uncompleted"}` }],
						details: { action: "toggle", todos: [...todos], nextId } satisfies TodoDetails,
					};
				}
				case "clear": {
					const count = todos.length;
					todos = [];
					nextId = 1;
					return {
						content: [{ type: "text", text: `Cleared ${count} todos` }],
						details: { action: "clear", todos: [], nextId: 1 } satisfies TodoDetails,
					};
				}
			}
		},
		renderCall(args, theme) {
			let text = theme.fg("toolTitle", theme.bold("todo ")) + theme.fg("muted", args.action);
			if (args.text) {
				text += ` ${theme.fg("dim", `"${args.text}"`)}`;
			}
			if (args.id !== undefined) {
				text += ` ${theme.fg("accent", `#${args.id}`)}`;
			}
			return new Text(text, 0, 0);
		},
		renderResult(result, { expanded }, theme) {
			const details = result.details as TodoDetails | undefined;
			if (!details) {
				const text = result.content[0];
				return new Text(text?.type === "text" ? text.text : "", 0, 0);
			}
			if (details.error) {
				return new Text(theme.fg("error", `Error: ${details.error}`), 0, 0);
			}
			if (details.action === "list") {
				if (details.todos.length === 0) {
					return new Text(theme.fg("dim", "No todos"), 0, 0);
				}
				const visible = expanded ? details.todos : details.todos.slice(0, 5);
				const lines = visible.map((todo) => {
					const check = todo.done ? theme.fg("success", "✓") : theme.fg("dim", "○");
					const text = todo.done ? theme.fg("dim", todo.text) : theme.fg("muted", todo.text);
					return `${check} ${theme.fg("accent", `#${todo.id}`)} ${text}`;
				});
				if (!expanded && details.todos.length > visible.length) {
					lines.push(theme.fg("dim", `... ${details.todos.length - visible.length} more`));
				}
				return new Text(lines.join("\n"), 0, 0);
			}
			const summary = result.content[0];
			return new Text(
				theme.fg("success", "✓ ") + theme.fg("muted", summary?.type === "text" ? summary.text : "Todo updated"),
				0,
				0,
			);
		},
		}),
	);

	pi.registerCommand("todos", {
		description: "Show all todos on the current branch",
		handler: async (_args, ctx) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("/todos requires interactive mode", "error");
				return;
			}
			await ctx.ui.custom<void>((_tui, theme, _kb, done) => new TodoListComponent(todos, theme, () => done()));
		},
	});
});

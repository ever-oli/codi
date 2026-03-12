import { StringEnum, Text, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

const CLARIFY_PARAMS = Type.Object({
	mode: Type.Optional(StringEnum(["text", "select", "confirm"] as const)),
	question: Type.String({ description: "The clarification question to ask the user." }),
	reason: Type.Optional(Type.String({ description: "Optional reason for asking the clarification." })),
	placeholder: Type.Optional(Type.String({ description: "Placeholder text for text input mode." })),
	options: Type.Optional(
		Type.Array(Type.String({ description: "Selectable option" }), {
			description: "Options for select mode.",
		}),
	),
});

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool({
		name: "clarify",
		label: "Clarify",
		description:
			"Ask the user a structured clarification question. Use when requirements are ambiguous and a direct answer is needed before continuing.",
		promptSnippet: "Ask the user a structured clarification question through Pi's UI.",
		parameters: CLARIFY_PARAMS,
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			if (!ctx.hasUI) {
				return {
					content: [{ type: "text", text: `Clarification needed: ${params.question}` }],
					details: { mode: params.mode ?? "text", answered: false, reason: "ui_unavailable" },
				};
			}

			const mode = params.mode ?? (params.options && params.options.length > 0 ? "select" : "text");
			let answerText: string | undefined;

			if (mode === "select") {
				const options = params.options?.filter((option) => option.trim().length > 0) ?? [];
				if (options.length === 0) {
					return {
						content: [{ type: "text", text: "Clarify tool requires at least one option in select mode." }],
						details: { mode, answered: false, reason: "missing_options" },
						isError: true,
					};
				}
				answerText = await ctx.ui.select(params.question, options);
			} else if (mode === "confirm") {
				const confirmed = await ctx.ui.confirm("Clarification", params.question);
				answerText = confirmed ? "yes" : "no";
			} else {
				answerText = await ctx.ui.input("Clarification", params.placeholder ?? params.question);
			}

			const lines = [params.question];
			if (params.reason) {
				lines.push(`Reason: ${params.reason}`);
			}
			lines.push(`Answer: ${answerText ?? "(cancelled)"}`);

			return {
				content: [{ type: "text", text: lines.join("\n") }],
				details: {
					mode,
					question: params.question,
					reason: params.reason,
					answer: answerText,
					answered: Boolean(answerText),
				},
			};
		},
		renderCall(args, theme) {
			return new Text(
				theme.fg("toolTitle", theme.bold("clarify ")) + theme.fg("muted", `${args.mode ?? "text"}: ${args.question}`),
				0,
				0,
			);
		},
		}),
	);
});

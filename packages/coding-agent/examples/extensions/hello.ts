/**
 * Hello Tool - Minimal custom tool example
 */

import { defineExtension, defineTool, Type, textToolResult } from "@mariozechner/pi-extension-sdk";

const HELLO_PARAMS = Type.Object({
	name: Type.String({ description: "Name to greet" }),
});

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool({
			name: "hello",
			label: "Hello",
			description: "A simple greeting tool",
			parameters: HELLO_PARAMS,

			async execute(_toolCallId, params) {
				return textToolResult(`Hello, ${params.name}!`, { greeted: params.name });
			},
		}),
	);
});

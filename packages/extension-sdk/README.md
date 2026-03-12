# @mariozechner/pi-extension-sdk

Narrow SDK for authoring Pi extensions and installable tool packages without importing the full `@mariozechner/pi-coding-agent` surface directly.

## Install

```bash
npm install @mariozechner/pi-extension-sdk
```

## Usage

```ts
import { Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

const HELLO_PARAMS = Type.Object({
	name: Type.String({ description: "Name to greet." }),
});

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool({
			name: "hello",
			label: "Hello",
			description: "Greet someone by name.",
			parameters: HELLO_PARAMS,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `Hello, ${params.name}!` }],
				};
			},
		}),
	);
});
```

## Notes

- Re-exports the common extension authoring surface from `@mariozechner/pi-coding-agent`
- Re-exports `Type` from `@sinclair/typebox`
- Re-exports common helpers from `@mariozechner/pi-ai` and `@mariozechner/pi-tui`
- Intended for Pi extensions and Pi packages loaded through Pi's TypeScript runtime

## Reference examples

- Minimal tool: [`packages/coding-agent/examples/extensions/hello.ts`](../coding-agent/examples/extensions/hello.ts)
- Interactive tool UI: [`packages/coding-agent/examples/extensions/question.ts`](../coding-agent/examples/extensions/question.ts)

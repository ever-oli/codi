import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { StringEnum, Type, defineExtension, defineTool } from "@mariozechner/pi-extension-sdk";

const PROVIDER = "google-antigravity";
const DEFAULT_MODEL = "gemini-3-pro-image";
const DEFAULT_ASPECT_RATIO = "1:1";
const DEFAULT_SAVE_MODE = "none";
const DEFAULT_ANTIGRAVITY_VERSION = "1.18.3";
const ANTIGRAVITY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"] as const;
const SAVE_MODES = ["none", "project", "global", "custom"] as const;

const ANTIGRAVITY_HEADERS = {
	"User-Agent": `antigravity/${process.env.PI_AI_ANTIGRAVITY_VERSION || DEFAULT_ANTIGRAVITY_VERSION} darwin/arm64`,
	"X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
	"Client-Metadata": JSON.stringify({
		ideType: "IDE_UNSPECIFIED",
		platform: "PLATFORM_UNSPECIFIED",
		pluginType: "GEMINI",
	}),
};

const IMAGE_PARAMS = Type.Object({
	prompt: Type.String({ description: "Image description." }),
	model: Type.Optional(Type.String({ description: "Image model id. Default: gemini-3-pro-image." })),
	aspectRatio: Type.Optional(StringEnum(ASPECT_RATIOS)),
	save: Type.Optional(StringEnum(SAVE_MODES)),
	saveDir: Type.Optional(Type.String({ description: "Directory to save image when save=custom." })),
});

interface ExtensionConfig {
	save?: (typeof SAVE_MODES)[number];
	saveDir?: string;
}

function parseOAuthCredentials(raw: string): { accessToken: string; projectId: string } {
	const parsed = JSON.parse(raw) as { token?: string; projectId?: string };
	if (!parsed.token || !parsed.projectId) {
		throw new Error("Missing token or projectId in Google Antigravity credentials.");
	}
	return { accessToken: parsed.token, projectId: parsed.projectId };
}

function readConfigFile(path: string): ExtensionConfig {
	if (!existsSync(path)) {
		return {};
	}
	try {
		return (JSON.parse(readFileSync(path, "utf8")) as ExtensionConfig) ?? {};
	} catch {
		return {};
	}
}

function loadConfig(cwd: string): ExtensionConfig {
	const globalConfig = readConfigFile(join(homedir(), ".pi", "agent", "extensions", "image-tools.json"));
	const projectConfig = readConfigFile(join(cwd, ".pi", "extensions", "image-tools.json"));
	return { ...globalConfig, ...projectConfig };
}

function imageExtension(mimeType: string): string {
	const lower = mimeType.toLowerCase();
	if (lower.includes("jpeg") || lower.includes("jpg")) return "jpg";
	if (lower.includes("gif")) return "gif";
	if (lower.includes("webp")) return "webp";
	return "png";
}

async function saveImage(base64Data: string, mimeType: string, outputDir: string): Promise<string> {
	await mkdir(outputDir, { recursive: true });
	const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
	const filename = `image-${timestamp}-${randomUUID().slice(0, 8)}.${imageExtension(mimeType)}`;
	const filePath = join(outputDir, filename);
	await writeFile(filePath, Buffer.from(base64Data, "base64"));
	return filePath;
}

function resolveSaveConfig(params: { save?: string; saveDir?: string }, cwd: string): { mode: string; outputDir?: string } {
	const config = loadConfig(cwd);
	const mode = params.save || process.env.PI_IMAGE_SAVE_MODE || config.save || DEFAULT_SAVE_MODE;
	if (mode === "project") return { mode, outputDir: join(cwd, ".pi", "generated-images") };
	if (mode === "global") return { mode, outputDir: join(homedir(), ".pi", "agent", "generated-images") };
	if (mode === "custom") {
		const outputDir = params.saveDir || process.env.PI_IMAGE_SAVE_DIR || config.saveDir;
		if (!outputDir) throw new Error("save=custom requires saveDir or PI_IMAGE_SAVE_DIR.");
		return { mode, outputDir };
	}
	return { mode };
}

async function parseSseForImage(
	response: Response,
	signal?: AbortSignal,
): Promise<{ image: { data: string; mimeType: string }; text: string[] }> {
	if (!response.body) {
		throw new Error("No response body");
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	const textParts: string[] = [];

	try {
		while (true) {
			if (signal?.aborted) {
				throw new Error("Request was aborted");
			}
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split("\n");
			buffer = lines.pop() || "";

			for (const line of lines) {
				if (!line.startsWith("data:")) continue;
				const raw = line.slice(5).trim();
				if (!raw) continue;
				const chunk = JSON.parse(raw) as {
					response?: {
						candidates?: Array<{
							content?: {
								parts?: Array<{ text?: string; inlineData?: { mimeType?: string; data?: string } }>;
							};
						}>;
					};
				};
				for (const candidate of chunk.response?.candidates ?? []) {
					for (const part of candidate.content?.parts ?? []) {
						if (part.text) {
							textParts.push(part.text);
						}
						if (part.inlineData?.data) {
							await reader.cancel();
							return {
								image: {
									data: part.inlineData.data,
									mimeType: part.inlineData.mimeType || "image/png",
								},
								text: textParts,
							};
						}
					}
				}
			}
		}
	} finally {
		reader.releaseLock();
	}

	throw new Error("No image data returned by the model");
}

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool({
		name: "image_generate",
		label: "Image Generate",
		description:
			"Generate an image via Google Antigravity image models. Returns the image inline and can optionally save to disk.",
		parameters: IMAGE_PARAMS,
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER);
			if (!apiKey) {
				throw new Error("Missing Google Antigravity OAuth credentials. Run /login for google-antigravity.");
			}
			const { accessToken, projectId } = parseOAuthCredentials(apiKey);
			const model = params.model || DEFAULT_MODEL;
			const aspectRatio = params.aspectRatio || DEFAULT_ASPECT_RATIO;
			const requestBody = {
				project: projectId,
				model,
				request: {
					contents: [{ role: "user", parts: [{ text: params.prompt }] }],
					systemInstruction: {
						parts: [{ text: "You are an AI image generator. Create a high-quality image matching the prompt." }],
					},
					generationConfig: {
						imageConfig: { aspectRatio },
						candidateCount: 1,
					},
				},
				requestType: "agent",
				requestId: `agent-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
				userAgent: "antigravity",
			};

			onUpdate?.({
				content: [{ type: "text", text: `Requesting image from ${PROVIDER}/${model}...` }],
				details: { provider: PROVIDER, model, aspectRatio },
			});

			const response = await fetch(`${ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse`, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${accessToken}`,
					"Content-Type": "application/json",
					Accept: "text/event-stream",
					...ANTIGRAVITY_HEADERS,
				},
				body: JSON.stringify(requestBody),
				signal,
			});
			if (!response.ok) {
				throw new Error(`Image request failed (${response.status}): ${await response.text()}`);
			}

			const parsed = await parseSseForImage(response, signal);
			const saveConfig = resolveSaveConfig(params, ctx.cwd);
			let savedPath: string | undefined;
			if (saveConfig.mode !== "none" && saveConfig.outputDir) {
				savedPath = await saveImage(parsed.image.data, parsed.image.mimeType, saveConfig.outputDir);
			}

			const summary = [`Generated image via ${PROVIDER}/${model}.`, `Aspect ratio: ${aspectRatio}.`];
			if (savedPath) {
				summary.push(`Saved image to: ${savedPath}`);
			}
			if (parsed.text.length > 0) {
				summary.push(`Model notes: ${parsed.text.join(" ")}`);
			}

			return {
				content: [
					{ type: "text", text: summary.join(" ") },
					{ type: "image", data: parsed.image.data, mimeType: parsed.image.mimeType },
				],
				details: { provider: PROVIDER, model, aspectRatio, saveMode: saveConfig.mode, savedPath },
			};
		},
		}),
	);
});

import { DEFAULT_MAX_BYTES, Type, defineExtension, defineTool, truncateHead } from "@mariozechner/pi-extension-sdk";

interface SearchResult {
	title: string;
	url: string;
	snippet: string;
}

const SEARCH_PARAMS = Type.Object({
	query: Type.String({ description: "Search query to run on the web." }),
	maxResults: Type.Optional(Type.Number({ description: "Maximum number of results to return. Default: 5." })),
});

const EXTRACT_PARAMS = Type.Object({
	url: Type.String({ description: "URL to fetch and extract." }),
	maxChars: Type.Optional(Type.Number({ description: "Maximum extracted text characters. Default: 12000." })),
});

function decodeEntities(text: string): string {
	return text
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'");
}

function normalizeWhitespace(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

function stripHtml(html: string): string {
	return normalizeWhitespace(
		decodeEntities(
			html
				.replace(/<script[\s\S]*?<\/script>/gi, " ")
				.replace(/<style[\s\S]*?<\/style>/gi, " ")
				.replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
				.replace(/<[^>]+>/g, " "),
		),
	);
}

function extractTitle(html: string): string | undefined {
	const match = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
	return match ? normalizeWhitespace(decodeEntities(match[1])) : undefined;
}

function extractDescription(html: string): string | undefined {
	const metaMatch = html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([\s\S]*?)["'][^>]*>/i);
	if (metaMatch) {
		return normalizeWhitespace(decodeEntities(metaMatch[1]));
	}
	const propertyMatch = html.match(/<meta[^>]+property=["']og:description["'][^>]+content=["']([\s\S]*?)["'][^>]*>/i);
	return propertyMatch ? normalizeWhitespace(decodeEntities(propertyMatch[1])) : undefined;
}

function parseSearchResults(html: string, maxResults: number): SearchResult[] {
	const results: SearchResult[] = [];
	const linkRegex =
		/<a[^>]+(?:class=["'][^"']*(?:result__a|result-link)[^"']*["'][^>]*|rel=["']nofollow["'][^>]*)href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi;

	for (const match of html.matchAll(linkRegex)) {
		if (results.length >= maxResults) break;
		const url = decodeEntities(match[1]);
		const title = normalizeWhitespace(stripHtml(match[2]));
		if (!url || !title) continue;

		const snippetStart = match.index ?? 0;
		const snippetSource = html.slice(snippetStart, snippetStart + 600);
		const snippet = normalizeWhitespace(
			stripHtml(
				snippetSource
					.replace(/<a[\s\S]*?<\/a>/gi, " ")
					.replace(/<span[^>]+class=["'][^"']*(?:result__snippet|result-snippet)[^"']*["'][^>]*>([\s\S]*?)<\/span>/gi, "$1"),
			),
		).slice(0, 240);

		if (results.some((result) => result.url === url)) continue;
		results.push({ title, url, snippet });
	}

	return results;
}

function formatSearchResults(query: string, results: SearchResult[]): string {
	if (results.length === 0) {
		return `No search results found for: ${query}`;
	}

	return results
		.map((result, index) => {
			const parts = [`${index + 1}. ${result.title}`, result.url];
			if (result.snippet) {
				parts.push(result.snippet);
			}
			return parts.join("\n");
		})
		.join("\n\n");
}

export default defineExtension((pi) => {
	pi.registerTool(
		defineTool({
		name: "quick_web_search",
		label: "Web Search",
		description: "Search the web and return a compact set of relevant results with URLs and snippets.",
		promptSnippet: "Search the public web for URLs and summaries.",
		parameters: SEARCH_PARAMS,
		async execute(_toolCallId, params) {
			const maxResults = Math.min(Math.max(params.maxResults ?? 5, 1), 10);
			const url = new URL("https://duckduckgo.com/html/");
			url.searchParams.set("q", params.query);

			const response = await fetch(url, {
				headers: {
					"User-Agent": "pi-web-tools/0.55.4",
				},
			});
			if (!response.ok) {
				throw new Error(`Search request failed (${response.status})`);
			}

			const html = await response.text();
			const results = parseSearchResults(html, maxResults);

			return {
				content: [{ type: "text", text: formatSearchResults(params.query, results) }],
				details: { query: params.query, resultCount: results.length, results },
			};
		},
		}),
	);

	pi.registerTool(
		defineTool({
		name: "quick_web_extract",
		label: "Web Extract",
		description: "Fetch a URL and extract a compact readable text summary from the page.",
		promptSnippet: "Extract readable text and metadata from a URL.",
		parameters: EXTRACT_PARAMS,
		async execute(_toolCallId, params) {
			const response = await fetch(params.url, {
				headers: {
					"User-Agent": "pi-web-tools/0.55.4",
				},
			});
			if (!response.ok) {
				throw new Error(`Extract request failed (${response.status})`);
			}

			const html = await response.text();
			const title = extractTitle(html);
			const description = extractDescription(html);
			const extracted = stripHtml(html);
			const maxChars = Math.min(Math.max(params.maxChars ?? 12000, 1000), DEFAULT_MAX_BYTES);
			const truncated = truncateHead(extracted, { maxBytes: maxChars, maxLines: 400 });
			const headerLines = [title ? `Title: ${title}` : undefined, description ? `Description: ${description}` : undefined]
				.filter((line): line is string => Boolean(line))
				.join("\n");
			const body = headerLines ? `${headerLines}\n\n${truncated.content}` : truncated.content;

			return {
				content: [{ type: "text", text: body }],
				details: {
					url: params.url,
					title,
					description,
					truncated: truncated.truncated,
					outputBytes: truncated.outputBytes,
					totalBytes: truncated.totalBytes,
				},
			};
		},
		}),
	);
});

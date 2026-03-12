/**
 * Prompt injection scanner for context files (AGENTS.md, CLAUDE.md, etc.).
 *
 * Scans content loaded from project context files for patterns that indicate
 * prompt injection attempts — instructions to ignore prior context, exfiltrate
 * secrets, hide information from the user, or execute arbitrary code.
 *
 * When a threat is detected, the content is replaced with a warning message
 * so the LLM never sees the injected instructions.
 *
 * Inspired by NousResearch/hermes-agent's prompt_builder.py scanner,
 * ported to TypeScript with patterns tuned for the pi ecosystem.
 */

export interface ScanResult {
	/** Whether any threats were detected */
	blocked: boolean;
	/** List of threat IDs found */
	threats: string[];
	/** The sanitized content (original if clean, warning message if blocked) */
	content: string;
}

// ── Threat pattern definitions ─────────────────────────────────────────────

interface ThreatPattern {
	pattern: RegExp;
	id: string;
}

const THREAT_PATTERNS: ThreatPattern[] = [
	// ── Direct instruction override ────────────────────────────────────────
	{ pattern: /ignore\s+(previous|all|above|prior)\s+instructions/i, id: "prompt_injection" },
	{ pattern: /do\s+not\s+tell\s+the\s+user/i, id: "deception_hide" },
	{ pattern: /system\s+prompt\s+override/i, id: "sys_prompt_override" },
	{
		pattern: /disregard\s+(your|all|any)\s+(instructions|rules|guidelines)/i,
		id: "disregard_rules",
	},
	{
		pattern: /act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|limits|rules)/i,
		id: "bypass_restrictions",
	},

	// ── Hidden content injection ───────────────────────────────────────────
	{
		pattern: /<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->/i,
		id: "html_comment_injection",
	},
	{ pattern: /<\s*div\s+style\s*=\s*["'].*display\s*:\s*none/i, id: "hidden_div" },

	// ── Code execution attempts ────────────────────────────────────────────
	{
		pattern: /translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)/i,
		id: "translate_execute",
	},

	// ── Secret exfiltration ────────────────────────────────────────────────
	{
		pattern: /curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)/i,
		id: "exfil_curl",
	},
	{ pattern: /cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)/i, id: "read_secrets" },

	// ── Persona hijacking ──────────────────────────────────────────────────
	{
		pattern: /you\s+are\s+now\s+(a|an)\s+(?:unrestricted|uncensored|jailbroken)/i,
		id: "persona_hijack",
	},
	{
		pattern: /forget\s+(everything|all|what)\s+(you|about)\s+(know|were|learned)/i,
		id: "memory_wipe",
	},

	// ── Encoded payload injection ──────────────────────────────────────────
	{
		// Base64-encoded shell/eval commands: echo <base64> | base64 -d | sh
		pattern: /base64\s+(?:-d|--decode)\s*\|\s*(?:sh|bash|zsh|eval|python|node)/i,
		id: "base64_exec",
	},
	{
		// data: URI with script or dangerous MIME types
		pattern: /data:\s*(?:text\/(?:html|javascript)|application\/(?:javascript|x-sh))/i,
		id: "data_uri_injection",
	},

	// ── Exfiltration via outbound requests ─────────────────────────────────
	{
		// Markdown link/image exfiltration: ![](https://evil.com/?data=LEAKED)
		pattern: /!\[[^\]]*\]\(\s*https?:\/\/[^)]*\$\{?\w*(KEY|TOKEN|SECRET|API)/i,
		id: "markdown_exfil",
	},
	{
		// wget/fetch exfiltration
		pattern: /(?:wget|fetch)\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)/i,
		id: "exfil_fetch",
	},

	// ── Multi-line override attempts ───────────────────────────────────────
	{
		// "From now on" / "Going forward" instruction override
		pattern: /(?:from\s+now\s+on|going\s+forward),?\s+(?:you\s+(?:must|should|will)|always|never)/i,
		id: "instruction_override",
	},
	{
		// "New instructions:" / "Updated system prompt:" style overrides
		pattern: /(?:new|updated|revised|replacement)\s+(?:system\s+)?(?:instructions|prompt|directive)s?\s*:/i,
		id: "new_instructions",
	},
];

// Unicode characters used to hide content from visual inspection
const INVISIBLE_CHARS = new Set([
	"\u200b", // Zero-width space
	"\u200c", // Zero-width non-joiner
	"\u200d", // Zero-width joiner
	"\u2060", // Word joiner
	"\ufeff", // BOM / zero-width no-break space
	"\u202a", // Left-to-right embedding
	"\u202b", // Right-to-left embedding
	"\u202c", // Pop directional formatting
	"\u202d", // Left-to-right override
	"\u202e", // Right-to-left override
]);

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Scan a context file's content for prompt injection patterns.
 *
 * @param content - The raw file content to scan
 * @param filename - The filename (for the warning message)
 * @returns ScanResult with blocked status, threat IDs, and sanitized content
 */
export function scanContextContent(content: string, filename: string): ScanResult {
	const threats: string[] = [];

	// Check for invisible Unicode characters
	for (const char of content) {
		if (INVISIBLE_CHARS.has(char)) {
			threats.push(`invisible_unicode_U+${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`);
			break; // One finding is enough
		}
	}

	// Check threat patterns
	for (const { pattern, id } of THREAT_PATTERNS) {
		if (pattern.test(content)) {
			threats.push(id);
		}
	}

	if (threats.length > 0) {
		return {
			blocked: true,
			threats,
			content: `[BLOCKED: ${filename} contained potential prompt injection (${threats.join(", ")}). Content not loaded for safety.]`,
		};
	}

	return {
		blocked: false,
		threats: [],
		content,
	};
}

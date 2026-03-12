/**
 * Regex-based secret redaction for tool output.
 *
 * Applies pattern matching to mask API keys, tokens, and credentials
 * before they reach the LLM context. Short tokens (< 18 chars) are
 * fully masked. Longer tokens preserve the first 6 and last 4
 * characters for debuggability.
 *
 * Also detects loaded environment variable values for secret-named env
 * vars (e.g. ANTHROPIC_API_KEY) and redacts their literal values even
 * if they don't match a known prefix pattern.
 *
 * Inspired by NousResearch/hermes-agent's redact.py, ported to
 * TypeScript with patterns tuned for the pi ecosystem.
 */

// ── Known API key prefix patterns ──────────────────────────────────────────
// Each pattern matches a well-known prefix followed by contiguous token chars.
const PREFIX_PATTERNS = [
	/sk-[A-Za-z0-9_-]{10,}/g, // OpenAI / OpenRouter / Anthropic (sk-ant-*)
	/ghp_[A-Za-z0-9]{10,}/g, // GitHub PAT (classic)
	/github_pat_[A-Za-z0-9_]{10,}/g, // GitHub PAT (fine-grained)
	/ghu_[A-Za-z0-9]{10,}/g, // GitHub user-to-server token
	/ghs_[A-Za-z0-9]{10,}/g, // GitHub server-to-server token
	/xox[baprs]-[A-Za-z0-9-]{10,}/g, // Slack tokens
	/AIza[A-Za-z0-9_-]{30,}/g, // Google API keys
	/pplx-[A-Za-z0-9]{10,}/g, // Perplexity
	/fal_[A-Za-z0-9_-]{10,}/g, // Fal.ai
	/fc-[A-Za-z0-9]{10,}/g, // Firecrawl
	/bb_live_[A-Za-z0-9_-]{10,}/g, // BrowserBase
	/AKIA[A-Z0-9]{16}/g, // AWS Access Key ID
	/sk_live_[A-Za-z0-9]{10,}/g, // Stripe secret key (live)
	/sk_test_[A-Za-z0-9]{10,}/g, // Stripe secret key (test)
	/rk_live_[A-Za-z0-9]{10,}/g, // Stripe restricted key
	/SG\.[A-Za-z0-9_-]{10,}/g, // SendGrid API key
	/hf_[A-Za-z0-9]{10,}/g, // HuggingFace token
	/r8_[A-Za-z0-9]{10,}/g, // Replicate API token
	/npm_[A-Za-z0-9]{10,}/g, // npm access token
	/pypi-[A-Za-z0-9_-]{10,}/g, // PyPI API token
	/dop_v1_[A-Za-z0-9]{10,}/g, // DigitalOcean PAT
	/doo_v1_[A-Za-z0-9]{10,}/g, // DigitalOcean OAuth token
	/am_[A-Za-z0-9_-]{10,}/g, // AgentMail API key
	/whsec_[A-Za-z0-9_-]{10,}/g, // Webhook secrets
	/glpat-[A-Za-z0-9_-]{10,}/g, // GitLab PAT
	/gAAAA[A-Za-z0-9_=-]{20,}/g, // Codex encrypted tokens
	/vercel_[A-Za-z0-9_-]{10,}/g, // Vercel API tokens
	/sbp_[A-Za-z0-9]{10,}/g, // Supabase project tokens
	/eyJhbGciOi[A-Za-z0-9_=-]{20,}\.[A-Za-z0-9_=-]{10,}/g, // JWT tokens (Base64-encoded header)
	/AC[a-f0-9]{32}/g, // Twilio Account SID
	/SK[a-f0-9]{32}/g, // Twilio API Key SID
	/clerk_[A-Za-z0-9_-]{10,}/g, // Clerk API keys
	/nk_[A-Za-z0-9_-]{10,}/g, // Neon API keys
	/age1[a-z0-9]{58}/g, // age encryption public keys
];

// ── Structural patterns ────────────────────────────────────────────────────

// ENV assignment: KEY=value where KEY contains a secret-like name
const SECRET_ENV_NAMES =
	/([A-Z_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z_]*)\s*=\s*(['"]?)(\S+)\2/gi;

// JSON fields: "apiKey": "value", "token": "value", etc.
const JSON_FIELD =
	/("(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer)")\s*:\s*"([^"]+)"/gi;

// Authorization headers
const AUTH_HEADER = /(Authorization:\s*Bearer\s+)(\S+)/gi;

// Private key blocks
const PRIVATE_KEY = /-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----/g;

// Database connection string passwords: protocol://user:PASSWORD@host
const DB_CONNSTR = /((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp):\/\/[^:]+:)([^@]+)(@)/gi;

// ── Mask helper ────────────────────────────────────────────────────────────

function maskToken(token: string): string {
	if (token.length < 18) {
		return "***";
	}
	return `${token.slice(0, 6)}...${token.slice(-4)}`;
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Redact sensitive secrets from a block of text.
 *
 * Safe to call on any string — non-matching text passes through unchanged.
 * Disable by setting PI_REDACT_SECRETS=0 in the environment.
 *
 * @returns The text with secrets replaced by masked versions.
 */
export function redactSecrets(text: string): string {
	if (!text) return text;

	// Allow opt-out via environment variable
	const envVal = process.env.PI_REDACT_SECRETS;
	if (envVal && ["0", "false", "no", "off"].includes(envVal.toLowerCase())) {
		return text;
	}

	let result = text;

	// Known prefix patterns
	for (const pattern of PREFIX_PATTERNS) {
		// Reset lastIndex since we reuse the regex
		pattern.lastIndex = 0;
		result = result.replace(pattern, (match) => maskToken(match));
	}

	// ENV assignments: OPENAI_API_KEY=sk-abc...
	result = result.replace(SECRET_ENV_NAMES, (_match, name, quote, value) => {
		return `${name}=${quote}${maskToken(value)}${quote}`;
	});

	// JSON fields: "apiKey": "value"
	result = result.replace(JSON_FIELD, (_match, key, value) => {
		return `${key}: "${maskToken(value)}"`;
	});

	// Authorization headers
	result = result.replace(AUTH_HEADER, (_match, prefix, token) => {
		return `${prefix}${maskToken(token)}`;
	});

	// Private key blocks
	result = result.replace(PRIVATE_KEY, "[REDACTED PRIVATE KEY]");

	// Database connection string passwords
	result = result.replace(DB_CONNSTR, (_match, prefix, _password, suffix) => {
		return `${prefix}***${suffix}`;
	});

	// Runtime env var value redaction: detect loaded env vars with secret-like
	// names and redact their literal values even if they don't match a prefix pattern.
	// This catches custom/self-hosted tokens that have no known prefix.
	const envSecrets = getEnvSecretValues();
	for (const { value, masked } of envSecrets) {
		if (result.includes(value)) {
			result = result.replaceAll(value, masked);
		}
	}

	return result;
}

// ── Runtime env var secret detection ───────────────────────────────────────

/** Regex matching environment variable names that likely contain secrets. */
const SECRET_ENV_NAME_RE = /(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|PRIVATE_KEY)$/i;

/** Minimum length for an env var value to be considered a secret worth redacting. */
const MIN_SECRET_VALUE_LENGTH = 8;

/** Cache of env var secret values, refreshed periodically. */
let _envSecretsCache: Array<{ value: string; masked: string }> | null = null;
let _envSecretsCacheTime = 0;
const ENV_SECRETS_CACHE_TTL_MS = 10_000; // 10 seconds

/**
 * Scan process.env for secret-named variables and return their values + masks.
 * Results are cached for 10 seconds to avoid scanning on every call.
 */
function getEnvSecretValues(): Array<{ value: string; masked: string }> {
	const now = Date.now();
	if (_envSecretsCache && now - _envSecretsCacheTime < ENV_SECRETS_CACHE_TTL_MS) {
		return _envSecretsCache;
	}

	const secrets: Array<{ value: string; masked: string }> = [];
	for (const [name, value] of Object.entries(process.env)) {
		if (!value || value.length < MIN_SECRET_VALUE_LENGTH) continue;
		if (!SECRET_ENV_NAME_RE.test(name)) continue;
		secrets.push({ value, masked: maskToken(value) });
	}

	// Sort by value length descending so longer values are replaced first
	// (prevents partial replacements when one value is a substring of another)
	secrets.sort((a, b) => b.value.length - a.value.length);

	_envSecretsCache = secrets;
	_envSecretsCacheTime = now;
	return secrets;
}

/**
 * Invalidate the env secrets cache. Call this if environment variables
 * are modified at runtime and you want immediate redaction of new values.
 */
export function invalidateEnvSecretsCache(): void {
	_envSecretsCache = null;
	_envSecretsCacheTime = 0;
}

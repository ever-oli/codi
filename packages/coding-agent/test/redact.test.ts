import { afterEach, describe, expect, test } from "vitest";
import { invalidateEnvSecretsCache, redactSecrets } from "../src/core/tools/redact.js";

describe("redactSecrets", () => {
	const originalEnv = process.env.PI_REDACT_SECRETS;

	afterEach(() => {
		if (originalEnv === undefined) {
			delete process.env.PI_REDACT_SECRETS;
		} else {
			process.env.PI_REDACT_SECRETS = originalEnv;
		}
	});

	describe("known prefix patterns", () => {
		test("masks OpenAI API keys", () => {
			const input = "export OPENAI_API_KEY=sk-proj-abcdefghij1234567890abcdefghij";
			const result = redactSecrets(input);
			expect(result).not.toContain("abcdefghij1234567890");
			expect(result).toContain("***"); // short enough or masked
		});

		test("masks GitHub PAT (classic)", () => {
			const result = redactSecrets("token: ghp_1234567890abcdefghij1234567890abcd");
			expect(result).not.toContain("1234567890abcdefghij");
			expect(result).toContain("...");
		});

		test("masks GitHub PAT (fine-grained)", () => {
			const result = redactSecrets("github_pat_abcdefghij1234567890_abcdefghij1234567890");
			expect(result).not.toContain("1234567890");
		});

		test("masks HuggingFace tokens", () => {
			const result = redactSecrets("hf_abcdefghij1234567890");
			expect(result).not.toContain("abcdefghij1234567890");
		});

		test("masks Slack tokens", () => {
			const fakeToken = "xoxb-" + "1".repeat(10) + "-" + "a".repeat(10);
			const result = redactSecrets(fakeToken);
			expect(result).not.toContain("1".repeat(10));
		});

		test("masks AWS access key IDs", () => {
			const result = redactSecrets("AKIAIOSFODNN7EXAMPLE");
			expect(result).not.toBe("AKIAIOSFODNN7EXAMPLE");
			expect(result).toContain("...");
		});

		test("masks Stripe keys", () => {
			const result = redactSecrets("sk_live_abcdefghij1234567890");
			expect(result).not.toContain("abcdefghij1234567890");
		});

		test("masks SendGrid keys", () => {
			const result = redactSecrets("SG.abcdefghij_1234567890");
			expect(result).not.toContain("abcdefghij_1234567890");
		});

		test("masks Google API keys", () => {
			const result = redactSecrets("AIzaSyCdefghijklmnopqrstuvwxyz1234567890");
			expect(result).not.toContain("defghijklmnopqrstuvwxyz1234567890");
		});
	});

	describe("structural patterns", () => {
		test("masks ENV assignments with secret-like names", () => {
			const result = redactSecrets("MY_API_KEY=some-secret-value-here");
			expect(result).toContain("MY_API_KEY=");
			expect(result).not.toContain("some-secret-value-here");
		});

		test("masks quoted ENV assignments", () => {
			const result = redactSecrets('DATABASE_PASSWORD="super_secret_123"');
			expect(result).toContain("DATABASE_PASSWORD=");
			expect(result).not.toContain("super_secret_123");
		});

		test("masks JSON secret fields", () => {
			const result = redactSecrets('{"apiKey": "my-super-secret-api-key-12345"}');
			expect(result).toContain('"apiKey"');
			expect(result).not.toContain("my-super-secret-api-key-12345");
		});

		test("masks Authorization headers", () => {
			const result = redactSecrets("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.long.token");
			expect(result).toContain("Authorization: Bearer ");
			expect(result).not.toContain("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.long.token");
		});

		test("masks private key blocks", () => {
			const input = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWzF8F2dN...
-----END RSA PRIVATE KEY-----`;
			const result = redactSecrets(input);
			expect(result).toBe("[REDACTED PRIVATE KEY]");
		});

		test("masks database connection string passwords", () => {
			const result = redactSecrets("postgres://admin:supersecretpw@db.example.com:5432/mydb");
			expect(result).toContain("postgres://admin:");
			expect(result).toContain("***");
			expect(result).toContain("@db.example.com:5432/mydb");
			expect(result).not.toContain("supersecretpw");
		});

		test("masks MongoDB connection strings", () => {
			const result = redactSecrets("mongodb+srv://user:pass123@cluster.example.net/db");
			expect(result).not.toContain("pass123");
			expect(result).toContain("***");
		});
	});

	describe("passthrough", () => {
		test("does not modify normal text", () => {
			const input = "Hello world, this is a normal message with no secrets.";
			expect(redactSecrets(input)).toBe(input);
		});

		test("does not modify normal code", () => {
			const input = `function main() {
  const x = 42;
  console.log("hello");
  return x;
}`;
			expect(redactSecrets(input)).toBe(input);
		});

		test("handles empty string", () => {
			expect(redactSecrets("")).toBe("");
		});

		test("handles null-ish values", () => {
			expect(redactSecrets(null as any)).toBe(null);
			expect(redactSecrets(undefined as any)).toBe(undefined);
		});
	});

	describe("opt-out", () => {
		test("respects PI_REDACT_SECRETS=0", () => {
			process.env.PI_REDACT_SECRETS = "0";
			const input = "sk-proj-abcdefghij1234567890abcdefghij";
			expect(redactSecrets(input)).toBe(input);
		});

		test("respects PI_REDACT_SECRETS=false", () => {
			process.env.PI_REDACT_SECRETS = "false";
			const input = "ghp_1234567890abcdefghij1234567890abcd";
			expect(redactSecrets(input)).toBe(input);
		});
	});

	describe("mask formatting", () => {
		test("short tokens are fully masked", () => {
			// Token < 18 chars should be "***"
			const result = redactSecrets("hf_short12345");
			expect(result).toBe("***");
		});

		test("long tokens preserve prefix and suffix", () => {
			// Token >= 18 chars: first 6 + ... + last 4
			const token = "sk-abcdefghijklmnopqrstuvwxyz";
			const result = redactSecrets(token);
			expect(result).toMatch(/^sk-abc\.\.\.wxyz$/);
		});
	});

	describe("multiple secrets in one text", () => {
		test("redacts all secrets in a multi-line block", () => {
			const input = `# .env
OPENAI_API_KEY=sk-proj-abcdefghij1234567890abcdefghij
DATABASE_URL=postgres://admin:mysecret@localhost:5432/db
GITHUB_TOKEN=ghp_1234567890abcdefghij1234567890abcd`;
			const result = redactSecrets(input);
			expect(result).not.toContain("abcdefghij1234567890abcdefghij");
			expect(result).not.toContain("mysecret");
			expect(result).not.toContain("1234567890abcdefghij1234567890abcd");
		});
	});

	describe("new prefix patterns", () => {
		test("masks Codex encrypted tokens", () => {
			const result = redactSecrets("gAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
			expect(result).not.toContain("BCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
			expect(result).toContain("...");
		});

		test("masks Vercel API tokens", () => {
			const result = redactSecrets("vercel_abcdefghij1234567890");
			expect(result).not.toContain("abcdefghij1234567890");
		});

		test("masks Supabase project tokens", () => {
			const result = redactSecrets("sbp_abcdefghij1234567890");
			expect(result).not.toContain("abcdefghij1234567890");
		});

		test("masks Clerk API keys", () => {
			const result = redactSecrets("clerk_abcdefghij1234567890");
			expect(result).not.toContain("abcdefghij1234567890");
		});

		test("masks JWT tokens", () => {
			const result = redactSecrets("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0");
			expect(result).not.toContain("eyJzdWIiOiIxMjM0NTY3ODkwIn0");
			expect(result).toContain("...");
		});

		test("masks DigitalOcean OAuth tokens", () => {
			const result = redactSecrets("doo_v1_abcdefghij1234567890");
			expect(result).not.toContain("abcdefghij1234567890");
		});

		test("masks Neon API keys", () => {
			const result = redactSecrets("nk_abcdefghij1234567890xyz");
			expect(result).not.toContain("abcdefghij1234567890xyz");
		});
	});

	describe("runtime env var value redaction", () => {
		const savedKey = process.env.TEST_SECRET_API_KEY;

		afterEach(() => {
			if (savedKey === undefined) {
				delete process.env.TEST_SECRET_API_KEY;
			} else {
				process.env.TEST_SECRET_API_KEY = savedKey;
			}
		});

		test("redacts literal env var values for secret-named vars", () => {
			process.env.TEST_SECRET_API_KEY = "my-custom-nonstandard-secret-value-42";
			invalidateEnvSecretsCache();

			const result = redactSecrets("The key is my-custom-nonstandard-secret-value-42 in the config");
			expect(result).not.toContain("my-custom-nonstandard-secret-value-42");
			expect(result).toContain("...");
		});

		test("does not redact short env var values", () => {
			process.env.TEST_SECRET_API_KEY = "short";
			invalidateEnvSecretsCache();

			const result = redactSecrets("The value is short");
			expect(result).toBe("The value is short");
		});

		test("does not redact non-secret-named env vars", () => {
			process.env.HOME = "/Users/test";
			invalidateEnvSecretsCache();

			const result = redactSecrets("Home dir is /Users/test");
			expect(result).toBe("Home dir is /Users/test");
		});
	});
});

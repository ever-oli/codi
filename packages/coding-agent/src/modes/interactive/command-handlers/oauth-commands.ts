/**
 * OAuth-related slash command handlers.
 * Extracted from InteractiveMode to reduce class size.
 */

import { getOAuthProviders } from "@mariozechner/pi-ai/oauth";
import type { AgentSession } from "../../../core/agent-session.js";

export interface OAuthCommandContext {
	readonly session: AgentSession;

	showStatus(message: string): void;
	showError(message: string): void;
	updateAvailableProviderCount(): Promise<void>;
}

export async function handleLogout(ctx: OAuthCommandContext, providerId: string): Promise<void> {
	const providerInfo = getOAuthProviders().find((p) => p.id === providerId);
	const providerName = providerInfo?.name || providerId;

	try {
		ctx.session.modelRegistry.authStorage.logout(providerId);
		ctx.session.modelRegistry.refresh();
		await ctx.updateAvailableProviderCount();
		ctx.showStatus(`Logged out of ${providerName}`);
	} catch (error: unknown) {
		ctx.showError(`Logout failed: ${error instanceof Error ? error.message : String(error)}`);
	}
}

export function getLoggedInOAuthProviders(ctx: OAuthCommandContext): string[] {
	const providers = ctx.session.modelRegistry.authStorage.list();
	return providers.filter((p) => ctx.session.modelRegistry.authStorage.get(p)?.type === "oauth");
}

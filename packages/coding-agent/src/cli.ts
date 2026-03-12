#!/usr/bin/env node
/**
 * CLI entry point for the refactored coding agent.
 * Uses main.ts with AgentSession and new mode modules.
 *
 * Test with: npx tsx src/cli-new.ts [args...]
 */
import { APP_NAME } from "./config.js";
import { main } from "./main.js";

process.title = APP_NAME;

const launchCwd = process.env.PI_LAUNCH_CWD;
if (launchCwd) {
	try {
		process.chdir(launchCwd);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		console.error(`Failed to switch to launch cwd "${launchCwd}": ${message}`);
		process.exit(1);
	}
}

main(process.argv.slice(2));

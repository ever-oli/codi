#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";

const require = createRequire(import.meta.url);
const packageJsonPath = require.resolve("pi-acp/package.json");
const packageJson = require(packageJsonPath);

const binEntry =
	typeof packageJson.bin === "string"
		? packageJson.bin
		: packageJson.bin?.["pi-acp"] ?? packageJson.bin?.pi_acp ?? packageJson.bin?.default;

if (typeof binEntry !== "string") {
	console.error("Unable to resolve pi-acp bin entry.");
	process.exit(1);
}

const entryPath = path.resolve(path.dirname(packageJsonPath), binEntry);
const child = spawn(process.execPath, [entryPath, ...process.argv.slice(2)], {
	stdio: "inherit",
	env: process.env,
});

child.on("exit", (code, signal) => {
	if (signal) {
		process.kill(process.pid, signal);
		return;
	}
	process.exit(code ?? 1);
});

child.on("error", (error) => {
	console.error(error.message);
	process.exit(1);
});

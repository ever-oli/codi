/**
 * Skill Evolver Extension
 *
 * Inspired by EvoSkill (arxiv.org/abs/2603.02766), this extension implements
 * self-evolving agent skills through failure analysis:
 *
 * 1. Tracks session outcomes via agent events
 * 2. Analyzes failures to identify missing capabilities
 * 3. Proposes new skills based on failure patterns
 * 4. Validates skills by tracking their usage outcomes
 * 5. Maintains a Pareto frontier of effective skills
 *
 * Usage:
 *   pi --extension skill-evolver.ts
 *
 * Commands:
 *   /evolve analyze     - Analyze recent session failures and propose skills
 *   /evolve propose     - Generate a skill proposal for recent failures
 *   /evolve apply       - Apply the most recent skill proposal
 *   /evolve list        - List evolved skills and their metrics
 *   /evolve prune       - Remove skills that don't improve outcomes
 *   /evolve status      - Show evolution status and metrics
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI, AgentMessage } from "@mariozechner/pi-coding-agent";

// ============================================================================
// Types
// ============================================================================

interface SessionOutcome {
	sessionId: string;
	timestamp: number;
	goal: string;
	success: boolean;
	failureMode?: FailureMode;
	toolsUsed: string[];
	errorCount: number;
	turns: number;
	// Workflow-specific (stored via custom entries)
	workflowPhase?: string;
	taskStatus?: string;
}

type FailureMode =
	| "tool_error"           // Tool execution failed
	| "wrong_tool"           // Agent chose wrong tool
	| "missing_capability"   // Agent couldn't find right approach
	| "context_loss"         // Lost context, repeated work
	| "infinite_loop"        // Stuck in loop
	| "user_abort"           // User cancelled
	| "verification_failed"  // Task verification failed
	| "timeout"              // Task took too long
	| "unknown";

interface SkillProposal {
	name: string;
	description: string;
	content: string;           // SKILL.md content
	basedOnSessions: number;   // Number of sessions analyzed
	confidence: number;        // 0-1 how confident we are this will help
	expectedImprovement: string;
	createdAt: string;
}

interface SkillMetrics {
	skillPath: string;
	skillName: string;
	firstSeen: number;
	lastUsed: number;
	tasksAttempted: number;
	tasksImproved: number;
	tasksDegraded: number;
	tasksNoEffect: number;
	averageImprovement: number; // -1 to 1
	isEvolved: boolean;         // Created by evolution vs hand-crafted
}

interface EvolutionState {
	outcomes: SessionOutcome[];
	metrics: Record<string, SkillMetrics>;
	evolutionRuns: number;
}

// ============================================================================
// Constants
// ============================================================================

const EVOLVED_SKILLS_DIR = ".pi/skills/evolved";
const STATE_FILE = ".pi/skill-evolution/state.json";
const PROPOSALS_DIR = ".pi/skill-evolution/proposals";
const MAX_OUTCOMES = 500;
const AUTO_EVOLVE_THRESHOLD = 5;

// ============================================================================
// State Management
// ============================================================================

let state: EvolutionState = {
	outcomes: [],
	metrics: {},
	evolutionRuns: 0,
};

let pi: ExtensionAPI;
let cwd: string;

function ensureDirs() {
	const dirs = [
		path.join(cwd, EVOLVED_SKILLS_DIR),
		path.join(cwd, ".pi/skill-evolution"),
		path.join(cwd, PROPOSALS_DIR),
	];
	for (const dir of dirs) {
		fs.mkdirSync(dir, { recursive: true });
	}
}

function loadState() {
	ensureDirs();
	const statePath = path.join(cwd, STATE_FILE);
	if (fs.existsSync(statePath)) {
		try {
			state = JSON.parse(fs.readFileSync(statePath, "utf-8"));
		} catch {}
	}
}

function saveState() {
	ensureDirs();
	const statePath = path.join(cwd, STATE_FILE);
	// Prune old outcomes
	state.outcomes = state.outcomes.slice(-MAX_OUTCOMES);
	fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
}

// ============================================================================
// Outcome Tracking
// ============================================================================

let sessionStart = Date.now();
let sessionTools: Set<string> = new Set();
let sessionErrors = 0;
let sessionTurns = 0;

function trackSessionOutcome(success: boolean, failureMode?: FailureMode) {
	const outcome: SessionOutcome = {
		sessionId: `session-${Date.now()}`,
		timestamp: Date.now(),
		goal: "Session tracked",
		success,
		failureMode,
		toolsUsed: Array.from(sessionTools),
		errorCount: sessionErrors,
		turns: sessionTurns,
	};
	
	state.outcomes.push(outcome);
	saveState();
	
	// Reset tracking
	sessionStart = Date.now();
	sessionTools = new Set();
	sessionErrors = 0;
	sessionTurns = 0;
}

// ============================================================================
// Failure Analysis
// ============================================================================

interface FailureAnalysis {
	totalSessions: number;
	failedSessions: number;
	failureModes: Record<FailureMode, number>;
	commonTools: Array<{ tool: string; count: number }>;
	summary: string;
}

function analyzeFailures(outcomes: SessionOutcome[]): FailureAnalysis {
	const failures = outcomes.filter(o => !o.success);
	
	// Count failure modes
	const failureModes: Record<string, number> = {};
	for (const f of failures) {
		const mode = f.failureMode || "unknown";
		failureModes[mode] = (failureModes[mode] || 0) + 1;
	}
	
	// Count tool usage in failures
	const toolCounts: Record<string, number> = {};
	for (const f of failures) {
		for (const t of f.toolsUsed) {
			toolCounts[t] = (toolCounts[t] || 0) + 1;
		}
	}
	
	const commonTools = Object.entries(toolCounts)
		.sort((a, b) => b[1] - a[1])
		.slice(0, 10)
		.map(([tool, count]) => ({ tool, count }));
	
	const summary = [
		`Analyzed ${outcomes.length} sessions (${failures.length} failed)`,
		`Failure rate: ${outcomes.length > 0 ? ((failures.length / outcomes.length) * 100).toFixed(1) : 0}%`,
		"",
		"Failure modes:",
		...Object.entries(failureModes).map(([mode, count]) => `  ${mode}: ${count}`),
		"",
		"Most common tools in failures:",
		...commonTools.slice(0, 5).map(t => `  ${t.tool}: ${t.count}x`),
	].join("\n");
	
	return {
		totalSessions: outcomes.length,
		failedSessions: failures.length,
		failureModes: failureModes as Record<FailureMode, number>,
		commonTools,
		summary,
	};
}

// ============================================================================
// Skill Proposal Generation
// ============================================================================

async function generateSkillProposal(analysis: FailureAnalysis): Promise<SkillProposal | null> {
	if (analysis.failedSessions < 2) {
		return null;
	}
	
	// Identify the most common failure mode
	const topFailureMode = Object.entries(analysis.failureModes)
		.sort((a, b) => b[1] - a[1])[0];
	
	if (!topFailureMode) return null;
	
	const [mode, count] = topFailureMode;
	const topTools = analysis.commonTools.slice(0, 3).map(t => t.tool).join(", ");
	
	// Generate skill based on failure mode
	let skillName: string;
	let skillDescription: string;
	let skillContent: string;
	
	switch (mode as FailureMode) {
		case "tool_error":
			skillName = "error-recovery";
			skillDescription = "Recover gracefully from tool errors";
			skillContent = generateErrorRecoverySkill(topTools);
			break;
		case "missing_capability":
			skillName = "problem-decomposition";
			skillDescription = "Break down complex problems into smaller steps";
			skillContent = generateProblemDecompositionSkill();
			break;
		case "infinite_loop":
			skillName = "loop-detection";
			skillDescription = "Detect and escape infinite loops";
			skillContent = generateLoopDetectionSkill();
			break;
		default:
			skillName = `evolved-${mode}`;
			skillDescription = `Address ${mode} failures`;
			skillContent = generateGenericSkill(mode, topTools);
	}
	
	const proposal: SkillProposal = {
		name: skillName,
		description: skillDescription,
		content: skillContent,
		basedOnSessions: analysis.failedSessions,
		confidence: Math.min(0.9, 0.3 + (count / analysis.totalSessions)),
		expectedImprovement: `Addresses ${count} ${mode} failures from ${analysis.failedSessions} sessions`,
		createdAt: new Date().toISOString(),
	};
	
	// Save proposal
	ensureDirs();
	const proposalPath = path.join(cwd, PROPOSALS_DIR, `${skillName}.json`);
	fs.writeFileSync(proposalPath, JSON.stringify(proposal, null, 2));
	
	return proposal;
}

// ============================================================================
// Skill Content Generators
// ============================================================================

function generateErrorRecoverySkill(tools: string): string {
	return `---
name: error-recovery
description: Recover gracefully from tool execution errors
---

# Error Recovery

## When to Use
When a tool call fails with an error, especially with: ${tools}

## Approach

1. **Read the error carefully** - Error messages often contain the solution
2. **Check prerequisites** - Does the file exist? Is the path correct? Are permissions set?
3. **Try alternatives** - If one approach fails, try a different tool or method
4. **Simplify** - Break the failing operation into smaller steps
5. **Ask for context** - Use \`ls\`, \`read\`, or \`find\` to understand the current state before retrying

## Key Patterns

\`\`\`typescript
// Instead of retrying the same failing command:
// 1. Check what exists
ls directory/
// 2. Read relevant files
read path/to/file
// 3. Try with corrected parameters
bash corrected-command
\`\`\`

## Common Pitfalls
- Retrying the exact same command without changes
- Not checking if files/directories exist before operating on them
- Ignoring error message details
`;
}

function generateProblemDecompositionSkill(): string {
	return `---
name: problem-decomposition
description: Break complex problems into smaller, verifiable steps
---

# Problem Decomposition

## When to Use
When facing a complex task that involves multiple steps or when you're not making progress.

## Approach

1. **State the goal clearly** - What exactly needs to be accomplished?
2. **Identify subtasks** - What are the independent pieces?
3. **Order by dependency** - What must be done first?
4. **Verify each step** - Test before moving on
5. **Document progress** - Use notes to track what's been done

## Key Patterns

\`\`\`markdown
## Task: [Main Goal]

### Subtasks:
1. [ ] First independent task
2. [ ] Second task (depends on #1)
3. [ ] Third task (depends on #2)

### Verification:
- After step 1: Check that X works
- After step 2: Verify Y is correct
\`\`\`

## Common Pitfalls
- Trying to do everything at once
- Not verifying intermediate results
- Losing track of what's been completed
`;
}

function generateLoopDetectionSkill(): string {
	return `---
name: loop-detection
description: Detect and escape infinite loops in problem-solving
---

# Loop Detection

## When to Use
When you notice you're repeating the same actions or getting the same errors.

## Approach

1. **Track your actions** - Note what you've tried
2. **Detect repetition** - If you've done the same thing 2-3 times, STOP
3. **Change strategy** - Try a fundamentally different approach
4. **Seek help** - Ask the user for clarification or try a different tool
5. **Simplify** - Maybe the task needs to be broken down further

## Warning Signs
- Same tool call with same arguments repeated
- Same error message appearing multiple times
- No progress after 3+ turns on the same subtask

## Escape Strategies

1. **Switch tools** - If bash fails, try reading files directly
2. **Go up a level** - Instead of fixing details, rethink the approach
3. **Ask the user** - "I've tried X and Y without success. How would you like me to proceed?"
4. **Skip and return** - Mark the step as blocked and continue with what you can do

## Common Pitfalls
- Assuming the 4th retry will work when the first 3 didn't
- Not noticing you're in a loop
- Making the same "small change" repeatedly
`;
}

function generateGenericSkill(mode: string, tools: string): string {
	return `---
name: evolved-${mode}
description: Address ${mode} failures based on session analysis
---

# Evolved Skill: ${mode}

## When to Use
When encountering ${mode} patterns, especially with tools: ${tools}

## Approach

1. **Recognize the pattern** - This failure mode has been seen in ${mode} situations
2. **Pause and assess** - Before continuing, understand why this is happening
3. **Try alternatives** - Don't repeat the same approach
4. **Track progress** - Note what you've tried

## Key Patterns
- Monitor for ${mode} conditions
- Have backup approaches ready
- Document what works and what doesn't

## Common Pitfalls
- Continuing without reflection when this pattern appears
- Not learning from previous attempts
`;
}

// ============================================================================
// Skill Materialization
// ============================================================================

function applySkill(proposal: SkillProposal): string {
	ensureDirs();
	
	const skillDir = path.join(cwd, EVOLVED_SKILLS_DIR, proposal.name);
	fs.mkdirSync(skillDir, { recursive: true });
	
	const skillPath = path.join(skillDir, "SKILL.md");
	fs.writeFileSync(skillPath, proposal.content);
	
	// Save metadata
	const metaPath = path.join(skillDir, ".evolution-meta.json");
	fs.writeFileSync(metaPath, JSON.stringify({
		createdAt: proposal.createdAt,
		basedOnSessions: proposal.basedOnSessions,
		confidence: proposal.confidence,
		expectedImprovement: proposal.expectedImprovement,
	}, null, 2));
	
	// Initialize metrics
	state.metrics[skillPath] = {
		skillPath,
		skillName: proposal.name,
		firstSeen: Date.now(),
		lastUsed: 0,
		tasksAttempted: 0,
		tasksImproved: 0,
		tasksDegraded: 0,
		tasksNoEffect: 0,
		averageImprovement: 0,
		isEvolved: true,
	};
	
	saveState();
	return skillPath;
}

// ============================================================================
// Skill Pruning (Pareto Frontier)
// ============================================================================

function getEvolvedSkills(): SkillMetrics[] {
	return Object.values(state.metrics).filter(m => m.isEvolved);
}

function getEffectiveSkills(): SkillMetrics[] {
	return getEvolvedSkills()
		.filter(m => m.averageImprovement > 0)
		.sort((a, b) => b.averageImprovement - a.averageImprovement);
}

function pruneSkills(): string[] {
	const pruned: string[] = [];
	
	for (const [path, metrics] of Object.entries(state.metrics)) {
		// Prune evolved skills that have been tried and don't help
		if (metrics.isEvolved && 
			metrics.tasksAttempted >= 3 && 
			metrics.averageImprovement <= 0) {
			
			try {
				const skillDir = path.dirname(metrics.skillPath);
				fs.rmSync(skillDir, { recursive: true, force: true });
				delete state.metrics[metrics.skillPath];
				pruned.push(metrics.skillName);
			} catch {}
		}
	}
	
	saveState();
	return pruned;
}

// ============================================================================
// Extension Registration
// ============================================================================

export default function skillEvolverExtension(extensionAPI: ExtensionAPI) {
	pi = extensionAPI;
	// cwd will be set from command handler context
	cwd = process.cwd();
	
	// Load persisted state
	loadState();
	
	// Track tool usage via events
	pi.on("tool_execution_end", async (event) => {
		sessionTools.add(event.toolName);
		if (event.isError) {
			sessionErrors++;
		}
	});
	
	// Track turns
	pi.on("turn_end", async () => {
		sessionTurns++;
	});
	
	// Track session end
	pi.on("session_shutdown", async () => {
		// Determine success based on error count
		const success = sessionErrors === 0;
		const failureMode: FailureMode | undefined = success 
			? undefined 
			: sessionErrors > 3 ? "tool_error" : "unknown";
		
		trackSessionOutcome(success, failureMode);
	});
	
	// Register commands
	pi.registerCommand("evolve", {
		description: "Skill evolution commands (analyze, propose, apply, list, prune, status)",
		handler: async (args, ctx) => {
			const subcommand = args.trim().split(/\s+/)[0] || "status";
	
			switch (subcommand) {
				case "analyze": {
					const analysis = analyzeFailures(state.outcomes);
					ctx.ui.showPanel("Failure Analysis", analysis.summary);
					break;
				}
	
				case "propose": {
					if (state.outcomes.length < 3) {
						ctx.ui.notify("Need at least 3 sessions tracked to propose skills", "warning");
						return;
					}
					
					ctx.ui.notify("Analyzing failures and generating skill proposal...", "info");
					const analysis = analyzeFailures(state.outcomes);
					const proposal = await generateSkillProposal(analysis);
	
					if (!proposal) {
						ctx.ui.notify("Not enough failure data to generate a proposal", "warning");
						return;
					}
	
					const lines = [
						`=== Skill Proposal: ${proposal.name} ===`,
						"",
						`Description: ${proposal.description}`,
						`Confidence: ${(proposal.confidence * 100).toFixed(0)}%`,
						`Based on: ${proposal.basedOnSessions} sessions`,
						`Expected: ${proposal.expectedImprovement}`,
						"",
						"Use '/evolve apply' to install this skill.",
					];
	
					ctx.ui.showPanel("Skill Proposal", lines.join("\n"));
					break;
				}
	
				case "apply": {
					// Find latest proposal
					const proposalsDir = path.join(cwd, PROPOSALS_DIR);
					if (!fs.existsSync(proposalsDir)) {
						ctx.ui.notify("No proposals directory found", "warning");
						return;
					}
					
					const proposals = fs.readdirSync(proposalsDir)
						.filter(f => f.endsWith(".json"))
						.sort();
	
					if (proposals.length === 0) {
						ctx.ui.notify("No proposals to apply. Use '/evolve propose' first.", "warning");
						return;
					}
	
					const latestFile = proposals[proposals.length - 1];
					const proposalPath = path.join(proposalsDir, latestFile);
					const proposal: SkillProposal = JSON.parse(fs.readFileSync(proposalPath, "utf-8"));
	
					const skillPath = applySkill(proposal);
					ctx.ui.notify(`Skill installed: ${proposal.name}`, "success");
	
					// Trigger reload to pick up new skill
					await ctx.reload();
					break;
				}
	
				case "list": {
					const evolved = getEvolvedSkills();
					const effective = getEffectiveSkills();
	
					const lines = [
						"=== Evolved Skills ===",
						"",
						`Total: ${evolved.length} evolved, ${effective.length} effective`,
						"",
					];
	
					if (effective.length > 0) {
						lines.push("=== Top Skills ===");
						for (const m of effective.slice(0, 10)) {
							lines.push(`  ${m.skillName}: ${(m.averageImprovement * 100).toFixed(0)}% improvement (${m.tasksAttempted} tasks)`);
						}
						lines.push("");
					}
	
					if (evolved.length > 0) {
						lines.push("=== All Evolved ===");
						for (const m of evolved) {
							const status = m.averageImprovement > 0 ? "✓" : m.averageImprovement < 0 ? "✗" : "?";
							lines.push(`  ${status} ${m.skillName}`);
						}
					}
	
					ctx.ui.showPanel("Evolved Skills", lines.join("\n"));
					break;
				}
	
				case "prune": {
					const pruned = pruneSkills();
					if (pruned.length === 0) {
						ctx.ui.notify("No skills needed pruning", "info");
					} else {
						ctx.ui.notify(`Pruned ${pruned.length} skills: ${pruned.join(", ")}`, "success");
					}
					break;
				}
	
				case "status": {
					const recentOutcomes = state.outcomes.slice(-20);
					const successRate = recentOutcomes.length > 0
						? recentOutcomes.filter(o => o.success).length / recentOutcomes.length
						: 0;
	
					const lines = [
						"=== Skill Evolution Status ===",
						"",
						`Sessions tracked: ${state.outcomes.length}`,
						`Success rate: ${(successRate * 100).toFixed(1)}% (last 20)`,
						`Evolved skills: ${getEvolvedSkills().length}`,
						`Effective skills: ${getEffectiveSkills().length}`,
						`Evolution runs: ${state.evolutionRuns}`,
						"",
						"Commands: analyze, propose, apply, list, prune",
					];
	
					ctx.ui.showPanel("Skill Evolution", lines.join("\n"));
					break;
				}
	
				default:
					ctx.ui.notify(`Unknown: ${subcommand}. Use: analyze, propose, apply, list, prune, status`, "warning");
			}
		},
	});
}

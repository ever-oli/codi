export type TaskStatus = "pending" | "ready" | "in_progress" | "blocked" | "done" | "waived";

export interface TaskNode {
	id: string;
	goal: string;
	status: TaskStatus;
	dependencies: string[];
	acceptanceCriteria: string[];
	notes: string[];
}

export interface TaskGraph {
	tasks: Record<string, TaskNode>;
	taskOrder: string[];
	activeTaskId?: string;
	createdAt: string;
	updatedAt: string;
}

export interface CreateTaskNodeInput {
	id: string;
	goal: string;
	status?: TaskStatus;
	dependencies?: string[];
	acceptanceCriteria?: string[];
	notes?: string[];
}

export interface UpdateTaskNodeInput {
	goal?: string;
	status?: TaskStatus;
	dependencies?: string[];
	acceptanceCriteria?: string[];
	notes?: string[];
}

const MIN_SPLIT_TASKS = 2;
const MAX_SPLIT_TASKS = 5;

function normalizeGoalFragment(value: string): string {
	return value
		.toLowerCase()
		.replace(/[`"'()[\]{}]/g, " ")
		.replace(/[^\w\s-]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function toSentenceCase(value: string): string {
	if (value.length === 0) {
		return value;
	}
	return `${value[0]!.toUpperCase()}${value.slice(1)}`;
}

function cleanGoalFragment(value: string): string | undefined {
	const normalized = value
		.replace(/\r/g, " ")
		.replace(/\s+/g, " ")
		.replace(/^[-*\d.)\s]+/, "")
		.replace(/^(and then|then|also)\s+/i, "")
		.replace(/\s+/g, " ")
		.trim();
	if (normalized.length < 6) {
		return undefined;
	}
	if (!/[a-z0-9]/i.test(normalized)) {
		return undefined;
	}
	const lowSignalPatterns = [/^please$/i, /^thanks?$/i, /^maybe$/i, /^if possible$/i, /^for now$/i];
	if (lowSignalPatterns.some((pattern) => pattern.test(normalized))) {
		return undefined;
	}
	return normalized;
}

function createDefaultTaskGoals(goal: string): string[] {
	const normalizedGoal = goal.trim();
	return [
		`Inspect and shape the work needed for: ${normalizedGoal}`,
		`Implement the scoped changes for: ${normalizedGoal}`,
		`Verify the changes for: ${normalizedGoal}`,
	];
}

function deriveAcceptanceCriteria(goal: string, isFinalTask: boolean): string[] {
	const criteria = [`Complete: ${goal}`];
	if (isFinalTask) {
		criteria.push("Collect verification evidence that shows the goal is complete or intentionally waived.");
	}
	return criteria;
}

function findMatchingExistingTask(
	existingTasks: TaskNode[],
	goal: string,
	usedTaskIds: Set<string>,
): TaskNode | undefined {
	const normalizedGoal = normalizeGoalFragment(goal);
	for (const task of existingTasks) {
		if (usedTaskIds.has(task.id)) {
			continue;
		}
		const normalizedTaskGoal = normalizeGoalFragment(task.goal);
		if (normalizedTaskGoal === normalizedGoal) {
			return task;
		}
	}
	for (const task of existingTasks) {
		if (usedTaskIds.has(task.id)) {
			continue;
		}
		const normalizedTaskGoal = normalizeGoalFragment(task.goal);
		if (normalizedTaskGoal.includes(normalizedGoal) || normalizedGoal.includes(normalizedTaskGoal)) {
			return task;
		}
	}
	return undefined;
}

function findFirstOpenTaskIndex(tasks: TaskNode[]): number {
	return tasks.findIndex((task) => task.status !== "done" && task.status !== "waived" && task.status !== "blocked");
}

function ensureRunnableStatuses(tasks: TaskNode[]): TaskNode[] {
	if (tasks.length === 0) {
		return tasks;
	}
	const hasActiveTask = tasks.some((task) => task.status === "in_progress" || task.status === "ready");
	if (hasActiveTask) {
		return tasks;
	}
	const firstOpenTaskIndex = findFirstOpenTaskIndex(tasks);
	if (firstOpenTaskIndex === -1) {
		return tasks;
	}
	return tasks.map((task, index) =>
		index === firstOpenTaskIndex
			? {
					...task,
					status: "ready",
				}
			: task,
	);
}

function createGeneratedTaskId(index: number, usedTaskIds: Set<string>, existingTasks: TaskNode[]): string {
	let counter = index + 1;
	let taskId = `task-${counter}`;
	const existingTaskIds = new Set(existingTasks.map((task) => task.id));
	while (usedTaskIds.has(taskId) || existingTaskIds.has(taskId)) {
		counter += 1;
		taskId = `task-${counter}`;
	}
	usedTaskIds.add(taskId);
	return taskId;
}

function isCompletedStatus(status: TaskStatus): boolean {
	return status === "done" || status === "waived";
}

function getFirstSchedulablePendingTaskId(graph: TaskGraph): string | undefined {
	return graph.taskOrder.find((taskId) => {
		const task = graph.tasks[taskId];
		if (!task || task.status !== "pending") {
			return false;
		}
		return task.dependencies.every((dependencyId) => {
			const dependency = graph.tasks[dependencyId];
			return dependency !== undefined && isCompletedStatus(dependency.status);
		});
	});
}

function promoteNextSchedulableTask(graph: TaskGraph): TaskGraph {
	const hasRunnableTask = graph.taskOrder.some((taskId) => {
		const task = graph.tasks[taskId];
		return task !== undefined && (task.status === "ready" || task.status === "in_progress");
	});
	if (hasRunnableTask) {
		return graph;
	}

	const nextTaskId = getFirstSchedulablePendingTaskId(graph);
	if (!nextTaskId) {
		return {
			...graph,
			activeTaskId: undefined,
		};
	}

	const nextTask = graph.tasks[nextTaskId];
	if (!nextTask) {
		return graph;
	}

	return {
		...graph,
		tasks: {
			...graph.tasks,
			[nextTaskId]: {
				...nextTask,
				status: "ready",
			},
		},
		activeTaskId: nextTaskId,
	};
}

export function createTaskNode(input: CreateTaskNodeInput): TaskNode {
	return {
		id: input.id,
		goal: input.goal,
		status: input.status ?? "pending",
		dependencies: input.dependencies ?? [],
		acceptanceCriteria: input.acceptanceCriteria ?? [],
		notes: input.notes ?? [],
	};
}

export function createTaskGraph(tasks: TaskNode[], timestamp = new Date().toISOString()): TaskGraph {
	const taskMap: Record<string, TaskNode> = {};
	for (const task of tasks) {
		taskMap[task.id] = task;
	}

	return {
		tasks: taskMap,
		taskOrder: tasks.map((task) => task.id),
		activeTaskId: tasks.find((task) => task.status === "in_progress" || task.status === "ready")?.id,
		createdAt: timestamp,
		updatedAt: timestamp,
	};
}

export function withTask(graph: TaskGraph, task: TaskNode, timestamp = new Date().toISOString()): TaskGraph {
	const exists = task.id in graph.tasks;
	return {
		...graph,
		tasks: {
			...graph.tasks,
			[task.id]: task,
		},
		taskOrder: exists ? graph.taskOrder : [...graph.taskOrder, task.id],
		activeTaskId: graph.activeTaskId ?? task.id,
		updatedAt: timestamp,
	};
}

export function setTaskStatus(
	graph: TaskGraph,
	taskId: string,
	status: TaskStatus,
	timestamp = new Date().toISOString(),
): TaskGraph {
	const task = graph.tasks[taskId];
	if (!task) {
		throw new Error(`Task ${taskId} not found`);
	}

	const nextGraph: TaskGraph = {
		...graph,
		tasks: {
			...graph.tasks,
			[taskId]: {
				...task,
				status,
			},
		},
		activeTaskId:
			status === "in_progress" || status === "ready"
				? taskId
				: graph.activeTaskId === taskId
					? undefined
					: graph.activeTaskId,
		updatedAt: timestamp,
	};
	return promoteNextSchedulableTask(nextGraph);
}

export function updateTask(
	graph: TaskGraph,
	taskId: string,
	updates: UpdateTaskNodeInput,
	timestamp = new Date().toISOString(),
): TaskGraph {
	const task = graph.tasks[taskId];
	if (!task) {
		throw new Error(`Task ${taskId} not found`);
	}

	const nextTask: TaskNode = {
		...task,
		...updates,
	};

	return {
		...graph,
		tasks: {
			...graph.tasks,
			[taskId]: nextTask,
		},
		updatedAt: timestamp,
	};
}

export function setActiveTask(
	graph: TaskGraph,
	taskId: string | undefined,
	timestamp = new Date().toISOString(),
): TaskGraph {
	if (taskId !== undefined && !graph.tasks[taskId]) {
		throw new Error(`Task ${taskId} not found`);
	}

	return {
		...graph,
		activeTaskId: taskId,
		updatedAt: timestamp,
	};
}

export function getReadyTasks(graph: TaskGraph): TaskNode[] {
	return graph.taskOrder
		.map((taskId) => graph.tasks[taskId])
		.filter((task): task is TaskNode => task !== undefined)
		.filter((task) => task.status === "ready");
}

export function areTaskDependenciesSatisfied(graph: TaskGraph, taskId: string): boolean {
	const task = graph.tasks[taskId];
	if (!task) {
		throw new Error(`Task ${taskId} not found`);
	}
	return task.dependencies.every((dependencyId) => {
		const dependency = graph.tasks[dependencyId];
		return dependency !== undefined && isCompletedStatus(dependency.status);
	});
}

export function getSchedulableTasks(graph: TaskGraph): TaskNode[] {
	return graph.taskOrder
		.map((taskId) => graph.tasks[taskId])
		.filter((task): task is TaskNode => task !== undefined)
		.filter((task) => task.status === "pending" || task.status === "ready" || task.status === "in_progress")
		.filter((task) => areTaskDependenciesSatisfied(graph, task.id));
}

export function isSingleGoalTaskGraph(graph: TaskGraph, goal: string): boolean {
	if (graph.taskOrder.length !== 1) {
		return false;
	}
	const [taskId] = graph.taskOrder;
	if (!taskId) {
		return false;
	}
	const task = graph.tasks[taskId];
	if (!task) {
		return false;
	}
	return normalizeGoalFragment(task.goal) === normalizeGoalFragment(goal);
}

export function splitGoalIntoTaskGoals(goal: string): string[] {
	const normalizedGoal = goal.trim();
	if (normalizedGoal.length === 0) {
		return createDefaultTaskGoals("the current workflow goal");
	}

	const segmentedGoal = normalizedGoal
		.replace(/\r?\n+/g, "\n")
		.replace(/\n\d+[.)]\s+/g, "|")
		.replace(/\n[-*]\s+/g, "|")
		.replace(/;\s+/g, "|")
		.replace(/\.\s+/g, "|")
		.replace(/,\s+(and then|then|also)\s+/gi, "|")
		.replace(/\s+(and then|then|also)\s+/gi, "|");

	const rawFragments = segmentedGoal
		.split("|")
		.map((fragment) => cleanGoalFragment(fragment))
		.filter((fragment): fragment is string => fragment !== undefined);

	const mergedFragments: string[] = [];
	for (const fragment of rawFragments) {
		if (fragment.length < 18 && mergedFragments.length > 0) {
			const previous = mergedFragments.pop();
			if (previous) {
				mergedFragments.push(`${previous} ${fragment}`);
			}
			continue;
		}
		mergedFragments.push(toSentenceCase(fragment));
		if (mergedFragments.length >= MAX_SPLIT_TASKS) {
			break;
		}
	}

	if (mergedFragments.length < MIN_SPLIT_TASKS) {
		return createDefaultTaskGoals(normalizedGoal);
	}

	return mergedFragments.slice(0, MAX_SPLIT_TASKS);
}

export function createTaskGraphFromGoal(
	goal: string,
	options?: {
		existingGraph?: TaskGraph;
		timestamp?: string;
	},
): TaskGraph {
	const timestamp = options?.timestamp ?? new Date().toISOString();
	const taskGoals = splitGoalIntoTaskGoals(goal);
	const existingGraph = options?.existingGraph;
	const existingTasks =
		existingGraph?.taskOrder
			.map((taskId) => existingGraph.tasks[taskId])
			.filter((task): task is TaskNode => task !== undefined) ?? [];
	const preserveExisting = existingGraph !== undefined && !isSingleGoalTaskGraph(existingGraph, goal);
	const usedTaskIds = new Set<string>();

	const generatedTasks = taskGoals.map((taskGoal, index) => {
		const matchingTask = preserveExisting
			? findMatchingExistingTask(existingTasks, taskGoal, usedTaskIds)
			: undefined;
		if (matchingTask) {
			usedTaskIds.add(matchingTask.id);
			return createTaskNode({
				id: matchingTask.id,
				goal: matchingTask.goal,
				status: matchingTask.status,
				dependencies: matchingTask.dependencies,
				acceptanceCriteria:
					matchingTask.acceptanceCriteria.length > 0
						? matchingTask.acceptanceCriteria
						: deriveAcceptanceCriteria(taskGoal, index === taskGoals.length - 1),
				notes: [...matchingTask.notes, "Preserved while refining the task graph from the workflow goal."],
			});
		}

		const taskId = createGeneratedTaskId(index, usedTaskIds, existingTasks);
		return createTaskNode({
			id: taskId,
			goal: taskGoal,
			status: index === 0 ? "ready" : "pending",
			dependencies: index === 0 ? [] : [`task-${index}`],
			acceptanceCriteria: deriveAcceptanceCriteria(taskGoal, index === taskGoals.length - 1),
			notes: ["Generated from deterministic workflow goal splitting."],
		});
	});

	const tasksWithDependencies = generatedTasks.map((task, index, tasks) => ({
		...task,
		dependencies: index === 0 ? [] : [tasks[index - 1]!.id],
	}));
	const runnableTasks = ensureRunnableStatuses(tasksWithDependencies);
	const graph = createTaskGraph(runnableTasks, timestamp);
	const activeTask = runnableTasks.find((task) => task.status === "in_progress" || task.status === "ready");
	return {
		...graph,
		activeTaskId: activeTask?.id ?? graph.activeTaskId,
	};
}

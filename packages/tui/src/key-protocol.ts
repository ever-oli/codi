import {
	ARROW_CODEPOINTS,
	CODEPOINTS,
	FUNCTIONAL_CODEPOINTS,
	LOCK_MASK,
	MODIFIERS,
	SYMBOL_KEYS,
} from "./key-constants.js";

export type KeyEventType = "press" | "repeat" | "release";

export interface ParsedKittySequence {
	codepoint: number;
	shiftedKey?: number;
	baseLayoutKey?: number;
	modifier: number;
	eventType: KeyEventType;
}

export function isKeyRelease(data: string): boolean {
	if (data.includes("\x1b[200~")) {
		return false;
	}

	return (
		data.includes(":3u") ||
		data.includes(":3~") ||
		data.includes(":3A") ||
		data.includes(":3B") ||
		data.includes(":3C") ||
		data.includes(":3D") ||
		data.includes(":3H") ||
		data.includes(":3F")
	);
}

export function isKeyRepeat(data: string): boolean {
	if (data.includes("\x1b[200~")) {
		return false;
	}

	return (
		data.includes(":2u") ||
		data.includes(":2~") ||
		data.includes(":2A") ||
		data.includes(":2B") ||
		data.includes(":2C") ||
		data.includes(":2D") ||
		data.includes(":2H") ||
		data.includes(":2F")
	);
}

export function parseKittySequence(data: string): ParsedKittySequence | null {
	const csiUMatch = data.match(/^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$/);
	if (csiUMatch) {
		const codepoint = parseInt(csiUMatch[1]!, 10);
		const shiftedKey = csiUMatch[2] && csiUMatch[2].length > 0 ? parseInt(csiUMatch[2], 10) : undefined;
		const baseLayoutKey = csiUMatch[3] ? parseInt(csiUMatch[3], 10) : undefined;
		const modValue = csiUMatch[4] ? parseInt(csiUMatch[4], 10) : 1;
		return {
			codepoint,
			shiftedKey,
			baseLayoutKey,
			modifier: modValue - 1,
			eventType: parseEventType(csiUMatch[5]),
		};
	}

	const arrowMatch = data.match(/^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$/);
	if (arrowMatch) {
		const modValue = parseInt(arrowMatch[1]!, 10);
		const arrowCodes: Record<string, number> = {
			A: ARROW_CODEPOINTS.up,
			B: ARROW_CODEPOINTS.down,
			C: ARROW_CODEPOINTS.right,
			D: ARROW_CODEPOINTS.left,
		};
		return {
			codepoint: arrowCodes[arrowMatch[3]!]!,
			modifier: modValue - 1,
			eventType: parseEventType(arrowMatch[2]),
		};
	}

	const funcMatch = data.match(/^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$/);
	if (funcMatch) {
		const keyNum = parseInt(funcMatch[1]!, 10);
		const modValue = funcMatch[2] ? parseInt(funcMatch[2], 10) : 1;
		const funcCodes: Record<number, number> = {
			2: FUNCTIONAL_CODEPOINTS.insert,
			3: FUNCTIONAL_CODEPOINTS.delete,
			5: FUNCTIONAL_CODEPOINTS.pageUp,
			6: FUNCTIONAL_CODEPOINTS.pageDown,
			7: FUNCTIONAL_CODEPOINTS.home,
			8: FUNCTIONAL_CODEPOINTS.end,
		};
		const codepoint = funcCodes[keyNum];
		if (codepoint !== undefined) {
			return {
				codepoint,
				modifier: modValue - 1,
				eventType: parseEventType(funcMatch[3]),
			};
		}
	}

	const homeEndMatch = data.match(/^\x1b\[1;(\d+)(?::(\d+))?([HF])$/);
	if (homeEndMatch) {
		const modValue = parseInt(homeEndMatch[1]!, 10);
		return {
			codepoint: homeEndMatch[3] === "H" ? FUNCTIONAL_CODEPOINTS.home : FUNCTIONAL_CODEPOINTS.end,
			modifier: modValue - 1,
			eventType: parseEventType(homeEndMatch[2]),
		};
	}

	return null;
}

export function matchesKittySequence(data: string, expectedCodepoint: number, expectedModifier: number): boolean {
	const parsed = parseKittySequence(data);
	if (!parsed) return false;
	const actualMod = parsed.modifier & ~LOCK_MASK;
	const normalizedExpectedModifier = expectedModifier & ~LOCK_MASK;

	if (actualMod !== normalizedExpectedModifier) return false;
	if (parsed.codepoint === expectedCodepoint) return true;

	if (parsed.baseLayoutKey !== undefined && parsed.baseLayoutKey === expectedCodepoint) {
		const codepoint = parsed.codepoint;
		const isLatinLetter = codepoint >= 97 && codepoint <= 122;
		const isKnownSymbol = SYMBOL_KEYS.has(String.fromCharCode(codepoint));
		if (!isLatinLetter && !isKnownSymbol) return true;
	}

	return false;
}

export function matchesModifyOtherKeys(data: string, expectedKeycode: number, expectedModifier: number): boolean {
	const match = data.match(/^\x1b\[27;(\d+);(\d+)~$/);
	if (!match) return false;
	const modValue = parseInt(match[1]!, 10);
	const keycode = parseInt(match[2]!, 10);
	return keycode === expectedKeycode && modValue - 1 === expectedModifier;
}

export function parseKittyKey(data: string): string | undefined {
	const kitty = parseKittySequence(data);
	if (!kitty) return undefined;

	const mods: string[] = [];
	const effectiveMod = kitty.modifier & ~LOCK_MASK;
	const supportedModifierMask = MODIFIERS.shift | MODIFIERS.ctrl | MODIFIERS.alt;
	if ((effectiveMod & ~supportedModifierMask) !== 0) return undefined;
	if (effectiveMod & MODIFIERS.shift) mods.push("shift");
	if (effectiveMod & MODIFIERS.ctrl) mods.push("ctrl");
	if (effectiveMod & MODIFIERS.alt) mods.push("alt");

	const keyName = resolveKittyKeyName(kitty);
	if (!keyName) return undefined;
	return mods.length > 0 ? `${mods.join("+")}+${keyName}` : keyName;
}

function parseEventType(eventTypeStr: string | undefined): KeyEventType {
	if (!eventTypeStr) return "press";
	const eventType = parseInt(eventTypeStr, 10);
	if (eventType === 2) return "repeat";
	if (eventType === 3) return "release";
	return "press";
}

function resolveKittyKeyName(parsed: ParsedKittySequence): string | undefined {
	const effectiveCodepoint = getEffectiveCodepoint(parsed);

	if (effectiveCodepoint === CODEPOINTS.escape) return "escape";
	if (effectiveCodepoint === CODEPOINTS.tab) return "tab";
	if (effectiveCodepoint === CODEPOINTS.enter || effectiveCodepoint === CODEPOINTS.kpEnter) return "enter";
	if (effectiveCodepoint === CODEPOINTS.space) return "space";
	if (effectiveCodepoint === CODEPOINTS.backspace) return "backspace";
	if (effectiveCodepoint === FUNCTIONAL_CODEPOINTS.delete) return "delete";
	if (effectiveCodepoint === FUNCTIONAL_CODEPOINTS.insert) return "insert";
	if (effectiveCodepoint === FUNCTIONAL_CODEPOINTS.home) return "home";
	if (effectiveCodepoint === FUNCTIONAL_CODEPOINTS.end) return "end";
	if (effectiveCodepoint === FUNCTIONAL_CODEPOINTS.pageUp) return "pageUp";
	if (effectiveCodepoint === FUNCTIONAL_CODEPOINTS.pageDown) return "pageDown";
	if (effectiveCodepoint === ARROW_CODEPOINTS.up) return "up";
	if (effectiveCodepoint === ARROW_CODEPOINTS.down) return "down";
	if (effectiveCodepoint === ARROW_CODEPOINTS.left) return "left";
	if (effectiveCodepoint === ARROW_CODEPOINTS.right) return "right";
	if (effectiveCodepoint >= 97 && effectiveCodepoint <= 122) return String.fromCharCode(effectiveCodepoint);
	if (SYMBOL_KEYS.has(String.fromCharCode(effectiveCodepoint))) return String.fromCharCode(effectiveCodepoint);

	return undefined;
}

function getEffectiveCodepoint(parsed: ParsedKittySequence): number {
	const isLatinLetter = parsed.codepoint >= 97 && parsed.codepoint <= 122;
	const isKnownSymbol = SYMBOL_KEYS.has(String.fromCharCode(parsed.codepoint));
	return isLatinLetter || isKnownSymbol ? parsed.codepoint : (parsed.baseLayoutKey ?? parsed.codepoint);
}

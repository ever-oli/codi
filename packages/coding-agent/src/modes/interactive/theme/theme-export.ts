type ThemeExportSection = {
	pageBg?: string | number;
	cardBg?: string | number;
	infoBg?: string | number;
};

type ThemeJsonLike = {
	vars?: Record<string, string | number>;
	colors: {
		userMessageBg: string | number;
	};
	export?: ThemeExportSection;
};

export function ansi256ToHex(index: number): string {
	const basicColors = [
		"#000000",
		"#800000",
		"#008000",
		"#808000",
		"#000080",
		"#800080",
		"#008080",
		"#c0c0c0",
		"#808080",
		"#ff0000",
		"#00ff00",
		"#ffff00",
		"#0000ff",
		"#ff00ff",
		"#00ffff",
		"#ffffff",
	];
	if (index < 16) {
		return basicColors[index];
	}
	if (index < 232) {
		const cubeIndex = index - 16;
		const r = Math.floor(cubeIndex / 36);
		const g = Math.floor((cubeIndex % 36) / 6);
		const b = cubeIndex % 6;
		const toHex = (value: number) => (value === 0 ? 0 : 55 + value * 40).toString(16).padStart(2, "0");
		return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
	}
	const gray = 8 + (index - 232) * 10;
	const grayHex = gray.toString(16).padStart(2, "0");
	return `#${grayHex}${grayHex}${grayHex}`;
}

export function resolveThemeValueToHex(
	value: string | number | undefined,
	vars: Record<string, string | number> | undefined,
	fallback: string,
	resolveVarRefs: (value: string | number, vars: Record<string, string | number>) => string | number,
): string {
	if (value === undefined) {
		return fallback;
	}
	const resolved = resolveVarRefs(value, vars ?? {});
	if (typeof resolved === "number") {
		return ansi256ToHex(resolved);
	}
	if (resolved === "") {
		return fallback;
	}
	return resolved;
}

export function isThemeJsonLight(
	themeJson: ThemeJsonLike,
	resolveVarRefs: (value: string | number, vars: Record<string, string | number>) => string | number,
	hexToRgb: (hex: string) => { r: number; g: number; b: number },
): boolean {
	const vars = themeJson.vars ?? {};
	const background = resolveThemeValueToHex(
		themeJson.export?.pageBg,
		vars,
		resolveThemeValueToHex(themeJson.colors.userMessageBg, vars, "#000000", resolveVarRefs),
		resolveVarRefs,
	);
	const { r, g, b } = hexToRgb(background);
	const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
	return luminance >= 0.6;
}

export function resolveCssThemeColors(
	resolvedColors: Record<string, string | number>,
	defaultText: string,
): Record<string, string> {
	const cssColors: Record<string, string> = {};
	for (const [key, value] of Object.entries(resolvedColors)) {
		if (typeof value === "number") {
			cssColors[key] = ansi256ToHex(value);
		} else if (value === "") {
			cssColors[key] = defaultText;
		} else {
			cssColors[key] = value;
		}
	}
	return cssColors;
}

export function getThemeExportColorsFromJson(
	themeJson: ThemeJsonLike,
	resolveVarRefs: (value: string | number, vars: Record<string, string | number>) => string | number,
): { pageBg?: string; cardBg?: string; infoBg?: string } {
	const exportSection = themeJson.export;
	if (!exportSection) return {};

	const vars = themeJson.vars ?? {};
	const resolveExportValue = (value: string | number | undefined): string | undefined => {
		if (value === undefined) return undefined;
		const resolved = resolveVarRefs(value, vars);
		if (typeof resolved === "number") {
			return ansi256ToHex(resolved);
		}
		return resolved || undefined;
	};

	return {
		pageBg: resolveExportValue(exportSection.pageBg),
		cardBg: resolveExportValue(exportSection.cardBg),
		infoBg: resolveExportValue(exportSection.infoBg),
	};
}

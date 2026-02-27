"""
problems.py — Problem discovery, metadata loading, diff rendering, and helpers.
"""
from __future__ import annotations

import difflib
import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Supported problem file extensions
CODE_EXTENSIONS = {".py", ".jl", ".R"}


def _is_code_file(p: Path) -> bool:
    return p.is_file() and p.suffix in CODE_EXTENSIONS


def scan_problems(problems_dir: Path) -> list[Path]:
    if not problems_dir.exists():
        return []
    return sorted(p for p in problems_dir.iterdir() if _is_code_file(p))


def scan_collections(root: Path) -> list[Path]:
    """
    Recursively find all directories in `root` that contain at least one code file.
    Includes `root` itself if it contains code files.
    """
    if not root.exists():
        return []

    collections = set()

    # Check root itself first
    if any(_is_code_file(p) for p in root.iterdir()):
        collections.add(root)

    # Walk directory tree
    for path in root.rglob("*"):
        if path.is_dir():
            has_code = any(_is_code_file(p) for p in path.iterdir())
            if has_code:
                collections.add(path)

    # Sort by path depth then name for nice display order
    return sorted(list(collections), key=lambda p: (len(p.parts), p.name))


def get_problem_id(problem_path: Path, root: Path) -> str:
    """
    Get a stable ID for the problem.
    - If direct child of root: use filename stem (backward compatibility)
    - Else: use relative path string without extension
    """
    try:
        rel = problem_path.relative_to(root)
    except ValueError:
        # Fallback if somehow path is not relative to root
        return problem_path.stem

    if len(rel.parts) == 1:
        return problem_path.stem

    # Use forward slashes for ID regardless of OS
    return str(rel.with_suffix("")).replace("\\", "/")


def load_problem_meta(path: Path) -> dict:
    """Load SOLUTION and DESCRIPTION from a problem file.

    For .py files, tries to import and read SOLUTION/DESCRIPTION variables.
    For other file types (.jl, .R, etc.), reads raw text as the solution.
    """
    # Non-Python files: just read raw text
    if path.suffix != ".py":
        return {"solution": path.read_text(), "description": ""}

    spec = importlib.util.spec_from_file_location("_prob", path)
    if spec is None or spec.loader is None:
        return {"solution": path.read_text(), "description": ""}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        pass
    return {
        "solution":    getattr(mod, "SOLUTION",    path.read_text()),
        "description": getattr(mod, "DESCRIPTION", ""),
    }


def build_side_by_side(ref_code: str, user_code: str) -> Table:
    """Side-by-side diff with per-line colour tinting."""
    ref_lines  = ref_code.splitlines()
    user_lines = user_code.splitlines()

    def hl(line: str, base_style: str) -> Text:
        if not line.strip():
            return Text("")
        return Text.from_markup(f"[{base_style}]{escape(line)}[/]")

    table = Table(
        show_header=True,
        header_style="bold dim",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("reference", no_wrap=False, ratio=1)
    table.add_column("yours",     no_wrap=False, ratio=1)

    matcher = difflib.SequenceMatcher(None, ref_lines, user_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in ref_lines[i1:i2]:
                t = hl(line, "dim white")
                table.add_row(t, t)
        elif tag == "replace":
            ref_chunk  = ref_lines[i1:i2]
            user_chunk = user_lines[j1:j2]
            length = max(len(ref_chunk), len(user_chunk))
            ref_chunk  += [""] * (length - len(ref_chunk))
            user_chunk += [""] * (length - len(user_chunk))
            for rl, ul in zip(ref_chunk, user_chunk):
                table.add_row(hl(rl, "red"), hl(ul, "green"))
        elif tag == "delete":
            for line in ref_lines[i1:i2]:
                table.add_row(hl(line, "red"), Text(""))
        elif tag == "insert":
            for line in user_lines[j1:j2]:
                table.add_row(Text(""), hl(line, "green"))

    return table


def status_label(row: sqlite3.Row | None) -> tuple[str, str]:
    if row is None:
        return "New", "white"
    nxt = datetime.fromisoformat(row["next_review"])
    if datetime.now() >= nxt:
        return "Due", "bold red"
    diff = nxt - datetime.now()
    return (f"In {diff.days}d", "dim") if diff.days >= 1 else ("Due soon", "bold red")


def max_rating_for(attempts: int) -> int:
    if attempts <= 1: return 4
    if attempts == 2: return 3
    if attempts == 3: return 2
    return 1

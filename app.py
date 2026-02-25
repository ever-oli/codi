#!/usr/bin/env python3
"""
Codi — Spaced repetition for ML code.
Drop .py scripts into PROBLEMS_DIR (default: ./problems).
Run: uv run app.py
"""
from __future__ import annotations

import difflib
import importlib.util
import os
import sqlite3
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown as MarkdownWidget,
    RichLog,
    Static,
)

from themes import TERMINAL_SEXY_THEMES

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PROBLEMS_DIR = Path(os.environ.get("PROBLEMS_DIR", "./problems"))
DB_PATH      = Path(os.environ.get("DB_PATH",      "study_data.db"))
EDITOR       = os.environ.get("EDITOR", "hx")
_TMP         = Path(tempfile.gettempdir())

RATING_LABELS = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
RATING_DESC   = {
    1: "forgot it completely",
    2: "got it with real effort",
    3: "got it with some hesitation",
    4: "recalled perfectly",
}

# ── Database ──────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            problem_id  TEXT    PRIMARY KEY,
            interval    REAL    DEFAULT 1,
            easiness    REAL    DEFAULT 2.5,
            reps        INTEGER DEFAULT 0,
            next_review TEXT    DEFAULT (datetime('now')),
            last_output TEXT    DEFAULT ''
        )
    """)
    # streak table — one row per calendar day a review was submitted
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            day TEXT PRIMARY KEY
        )
    """)
    conn.commit()
    return conn


def get_row(conn: sqlite3.Connection, pid: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM reviews WHERE problem_id=?", (pid,)
    ).fetchone()


def sm2_update(conn: sqlite3.Connection, pid: str, rating: int) -> None:
    """SM-2 algorithm. rating: 1=Again 2=Hard 3=Good 4=Easy"""
    quality = {1: 0, 2: 3, 3: 4, 4: 5}[rating]
    row = get_row(conn, pid)
    iv, ef, reps = (row["interval"], row["easiness"], row["reps"]) if row else (1.0, 2.5, 0)

    if quality < 3:
        iv, reps = 1.0, 0
    else:
        iv   = 1 if reps == 0 else (6 if reps == 1 else round(iv * ef, 1))
        reps += 1
        ef   = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

    next_rev = (datetime.now() + timedelta(days=iv)).isoformat()
    conn.execute("""
        INSERT INTO reviews (problem_id, interval, easiness, reps, next_review)
        VALUES (?,?,?,?,?)
        ON CONFLICT(problem_id) DO UPDATE SET
            interval=excluded.interval,
            easiness=excluded.easiness,
            reps=excluded.reps,
            next_review=excluded.next_review
    """, (pid, iv, ef, reps, next_rev))
    # record today as an active day
    conn.execute(
        "INSERT OR IGNORE INTO activity (day) VALUES (?)",
        (datetime.now().date().isoformat(),),
    )
    conn.commit()


def get_streak(conn: sqlite3.Connection) -> int:
    """Return current consecutive-day streak (today counts if active today)."""
    rows = conn.execute(
        "SELECT day FROM activity ORDER BY day DESC"
    ).fetchall()
    if not rows:
        return 0
    days = [datetime.fromisoformat(r["day"]).date() for r in rows]
    today = datetime.now().date()
    streak = 0
    cursor = today
    for day in days:
        if day == cursor:
            streak += 1
            cursor -= timedelta(days=1)
        elif day == cursor + timedelta(days=1):
            # today not yet done — still on yesterday's streak
            streak += 1
            cursor = day - timedelta(days=1)
        else:
            break
    return streak


def reset_progress(conn: sqlite3.Connection, pid: str) -> None:
    conn.execute("DELETE FROM reviews WHERE problem_id=?", (pid,))
    conn.commit()

# ── Helpers ───────────────────────────────────────────────────────────────────
def scan_problems() -> list[Path]:
    if not PROBLEMS_DIR.exists():
        return []
    return sorted(p for p in PROBLEMS_DIR.iterdir() if p.suffix == ".py")


def load_problem_meta(path: Path) -> dict:
    """Load SOLUTION and DESCRIPTION from a problem file without exec."""
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
    """Side-by-side diff with per-line Python syntax highlighting."""
    ref_lines  = ref_code.splitlines()
    user_lines = user_code.splitlines()

    def hl(line: str, base_style: str) -> Text:
        """Highlight a single Python line via Pygments, then tint it."""
        if not line.strip():
            return Text("")
        syn = Syntax(line, "python", theme="ansi_dark", word_wrap=False)
        # Render to a plain Text then apply the diff colour as a dim overlay
        t = Text.from_markup(f"[{base_style}]{escape(line)}[/]")
        return t

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
    return 1  # 4+ → forced Again

# ── AI providers ──────────────────────────────────────────────────────────────
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL",     "gemini-2.0-flash")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
AI_PROVIDER      = os.environ.get("AI_PROVIDER", "openrouter")   # "openrouter" | "gemini"


def _gemini(prompt: str) -> str:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No GEMINI_API_KEY found in .env."
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (result.text or "").strip()
    except Exception as e:
        return f"**Gemini error:** {e}"


def _openrouter(prompt: str) -> str:
    import urllib.request, json as _json
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No OPENROUTER_API_KEY found in .env."
    payload = _json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/ever-oli/codi",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"**OpenRouter error:** {e}"


def ai_call(prompt: str) -> str:
    if AI_PROVIDER == "gemini":
        return _gemini(prompt)
    return _openrouter(prompt)


# ── AI prompts ────────────────────────────────────────────────────────────────
def get_hint(problem_name: str, ref_code: str, user_code: str | None) -> str:
    has_attempt = bool(user_code and user_code.strip())
    if has_attempt:
        diff_text = "\n".join(difflib.unified_diff(
            ref_code.splitlines(), (user_code or "").splitlines(),
            fromfile="reference", tofile="yours", lineterm="",
        ))
        prompt = f"""\
You are a coding tutor helping a student study ML implementations from memory.

Problem: {problem_name}

Reference solution:
```python
{ref_code}
```

The student's current attempt diff vs reference:
```diff
{diff_text}
```

Give a single Socratic hint that nudges them toward what they're missing \
without revealing the answer. Be concise (2-4 sentences max). \
Focus on the most important gap in their attempt."""
    else:
        prompt = f"""\
You are a coding tutor helping a student study ML implementations from memory.

Problem: {problem_name}

Reference solution:
```python
{ref_code}
```

The student hasn't written anything yet. Give a single Socratic hint to \
help them get started — what is the core concept or structure they need to \
think about? Be concise (2-4 sentences max). Do not give away the answer."""
    return ai_call(prompt)


def get_suggest_fix(problem_name: str, ref_code: str, user_code: str) -> str:
    diff_text = "\n".join(difflib.unified_diff(
        ref_code.splitlines(), user_code.splitlines(),
        fromfile="reference", tofile="yours", lineterm="",
    ))
    prompt = f"""\
You are a coding tutor reviewing a student's ML implementation attempt.

Problem: {problem_name}

Reference solution:
```python
{ref_code}
```

Student's attempt diff vs reference:
```diff
{diff_text}
```

Be direct and specific. List exactly what is wrong and what needs to change \
to match the reference. Use short bullet points. Do not be vague. \
Do not explain concepts they already got right."""
    return ai_call(prompt)


def get_explain(problem_name: str, ref_code: str) -> str:
    prompt = f"""\
You are a coding tutor explaining an ML concept to a student who just finished \
(or attempted) an implementation exercise.

Problem: {problem_name}

Reference solution:
```python
{ref_code}
```

Give a clear, concise explanation (5-8 sentences) of:
1. What this component does conceptually in the broader ML context
2. Why each key design decision in the implementation exists
3. One common real-world mistake or misconception to watch out for

Write for someone who can code but is still building intuition. \
Do not just restate the code line by line."""
    return ai_call(prompt)


# ── Shared AI Modal ───────────────────────────────────────────────────────────
class AIModal(ModalScreen):
    """Generic modal that fires an AI fetch in a background thread."""
    BINDINGS = [Binding("escape,q", "dismiss", "Close")]

    def __init__(self, title: str, fetch_fn) -> None:
        super().__init__()
        self._title    = title
        self._fetch_fn = fetch_fn

    def compose(self) -> ComposeResult:
        with Vertical(id="hint-box"):
            yield Label(f"[bold]{escape(self._title)}[/]", id="hint-title")
            yield MarkdownWidget("*asking Codi…*", id="hint-md")

    def on_mount(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        text = self._fetch_fn()
        self.app.call_from_thread(self._display, text)

    def _display(self, text: str) -> None:
        self.query_one("#hint-md", MarkdownWidget).update(text)


# ── Confirm Modal ─────────────────────────────────────────────────────────────
class ConfirmModal(ModalScreen[bool]):
    """Simple yes/no confirmation."""
    BINDINGS = [
        Binding("y", "confirm(True)",  "Yes"),
        Binding("n", "confirm(False)", "No"),
        Binding("escape", "confirm(False)", "No"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(f"[bold]{escape(self._message)}[/]", id="modal-title")
            yield Label("")
            yield Label("  [y]  Yes")
            yield Label("  [n]  No")

    def action_confirm(self, result: bool) -> None:
        self.dismiss(result)


# ── Rating Modal ──────────────────────────────────────────────────────────────
class RatingModal(ModalScreen[int]):
    BINDINGS = [
        Binding("1", "rate(1)", "Again"),
        Binding("2", "rate(2)", "Hard"),
        Binding("3", "rate(3)", "Good"),
        Binding("4", "rate(4)", "Easy"),
        Binding("escape", "dismiss(0)", "Skip"),
    ]

    def __init__(self, max_r: int, attempts: int) -> None:
        super().__init__()
        self.max_r    = max_r
        self.attempts = attempts

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(
                f"[white]attempt {self.attempts}[/]  —  "
                f"max: [dim]{RATING_LABELS[self.max_r]}[/]",
                id="modal-title",
            )
            yield Label("")
            for i in range(1, 5):
                line = f"  [{i}]  {RATING_LABELS[i]:<7} {RATING_DESC[i]}"
                if i <= self.max_r:
                    yield Label(line)
                else:
                    yield Label(f"[dim]{line}  ✕[/]")
            yield Label("")
            yield Label("  [Esc] skip without updating", id="modal-skip")

    def action_rate(self, rating: int) -> None:
        if rating <= self.max_r:
            self.dismiss(rating)


# ── Study Screen ──────────────────────────────────────────────────────────────
class StudyScreen(Screen):
    BINDINGS = [
        Binding("e", "edit",         "Edit"),
        Binding("s", "submit",       "Submit"),
        Binding("h", "hint",         "Hint"),
        Binding("f", "suggest_fix",  "Fix"),
        Binding("x", "explain",      "Explain"),
        Binding("q", "back",         "Menu"),
    ]

    def __init__(self, problem: Path) -> None:
        super().__init__()
        self.problem   = problem
        self.meta      = load_problem_meta(problem)
        self.work_file = _TMP / f"codi_{problem.stem}.py"
        self.conn      = get_db()
        self.attempts  = 0
        self.has_diff  = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="problem-bar")
        yield RichLog(id="diff-pane", classes="full-pane", highlight=False, markup=True, wrap=False, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        desc = self.meta["description"]
        desc_part = f"  [dim italic]{escape(desc)}[/]" if desc else ""
        self.query_one("#problem-bar", Static).update(
            f"[bold white]{escape(self.problem.name)}[/]{desc_part}"
        )
        log = self.query_one("#diff-pane", RichLog)
        row = get_row(self.conn, self.problem.stem)
        if row and row["last_output"] and row["last_output"] != "✓  perfect match":
            log.write("[dim]── last session ──[/]")
            log.write(Syntax(row["last_output"], "diff", theme="ansi_dark", word_wrap=False))
            log.write("[dim]press  e  to start new attempt[/]")
        else:
            log.write("[dim]press  e  to open editor[/]")

    def action_hint(self) -> None:
        ref  = self.meta["solution"]
        user = self.work_file.read_text() if self.work_file.exists() else None
        self.app.push_screen(AIModal(
            f"hint  ·  {self.problem.name}",
            lambda: get_hint(self.problem.name, ref, user),
        ))

    def action_suggest_fix(self) -> None:
        if not self.work_file.exists():
            self.query_one("#diff-pane", RichLog).write("\n[dim]edit first — press  e[/]")
            return
        ref  = self.meta["solution"]
        user = self.work_file.read_text()
        self.app.push_screen(AIModal(
            f"suggest fix  ·  {self.problem.name}",
            lambda: get_suggest_fix(self.problem.name, ref, user),
        ))

    def action_explain(self) -> None:
        ref = self.meta["solution"]
        self.app.push_screen(AIModal(
            f"explain  ·  {self.problem.name}",
            lambda: get_explain(self.problem.name, ref),
        ))

    def action_edit(self) -> None:
        self.attempts += 1
        if not self.work_file.exists():
            self.work_file.write_text(f"# {self.problem.name}\n\n")
        with self.app.suspend():
            subprocess.run([EDITOR, str(self.work_file)])
        self.call_after_refresh(self._show_diff)

    def _show_diff(self) -> None:
        log = self.query_one("#diff-pane", RichLog)
        log.clear()

        user_code = self.work_file.read_text() if self.work_file.exists() else ""
        ref_code  = self.meta["solution"]

        if user_code.splitlines() == ref_code.splitlines():
            log.write("[white]✓  perfect match[/]")
            summary = "✓  perfect match"
        else:
            log.write(build_side_by_side(ref_code, user_code))
            summary = "\n".join(difflib.unified_diff(
                ref_code.splitlines(), user_code.splitlines(),
                fromfile="reference", tofile="yours", lineterm="",
            ))

        max_r = max_rating_for(self.attempts)
        if self.attempts >= 4:
            log.write(f"\n[bold red]attempt {self.attempts} — press  s  to record (forced: Again)[/]")
        else:
            log.write(
                f"\n[dim]attempt {self.attempts}  ·  max: {RATING_LABELS[max_r]}"
                f"  ·  s = submit    e = retry    x = explain[/]"
            )

        self.conn.execute(
            "INSERT INTO reviews (problem_id, last_output) VALUES (?,?) "
            "ON CONFLICT(problem_id) DO UPDATE SET last_output=excluded.last_output",
            (self.problem.stem, summary),
        )
        self.conn.commit()
        self.has_diff = True

    def action_submit(self) -> None:
        if not self.has_diff:
            self.query_one("#diff-pane", RichLog).write("\n[dim]edit first — press  e[/]")
            return

        max_r = max_rating_for(self.attempts)
        if max_r == 1:
            log = self.query_one("#diff-pane", RichLog)
            log.write("\n[bold red]forced: Again  (4+ attempts)[/]")
            sm2_update(self.conn, self.problem.stem, 1)
            self.work_file.unlink(missing_ok=True)
            self.app.pop_screen()
        else:
            self.app.push_screen(RatingModal(max_r, self.attempts), self._rated)

    def _rated(self, rating: int | None) -> None:
        if rating:
            sm2_update(self.conn, self.problem.stem, rating)
        self.work_file.unlink(missing_ok=True)
        self.app.pop_screen()

    def action_back(self) -> None:
        if self.work_file.exists():
            self.app.push_screen(
                ConfirmModal("Discard in-progress work and go back?"),
                self._confirm_back,
            )
        else:
            self.app.pop_screen()

    def _confirm_back(self, confirmed: bool | None) -> None:
        if confirmed:
            self.work_file.unlink(missing_ok=True)
            self.app.pop_screen()


# ── Search / Filter bar ───────────────────────────────────────────────────────
class SearchBar(Static):
    """A slim inline search input rendered above the table."""
    DEFAULT_CSS = """
    SearchBar {
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    SearchBar Input {
        border: none;
        height: 1;
        background: $surface;
        color: $foreground;
        padding: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="search…", id="search-input")


# ── Menu Screen ───────────────────────────────────────────────────────────────
class MenuScreen(Screen):
    BINDINGS = [
        Binding("r",     "refresh",   "Refresh"),
        Binding("/",     "focus_search", "Search"),
        Binding("escape","clear_search", "Clear", show=False),
        Binding("q",     "quit_app",  "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.conn        = get_db()
        self._all_rows: list[tuple] = []
        self._filter     = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="stats-bar")
        yield SearchBar(id="search-bar")
        yield DataTable(id="table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns("Status", "Problem", "Reps", "Interval", "Next review")
        self._refresh()

    # ── data loading ──────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        problems = scan_problems()
        due = new = upcoming = 0
        rows: list[tuple] = []

        for p in problems:
            row          = get_row(self.conn, p.stem)
            label, color = status_label(row)
            reps         = str(row["reps"])      if row else "0"
            interval     = f"{row['interval']}d" if row else "—"
            nxt          = row["next_review"][:10] if row else "—"
            sort_key     = 0 if label in ("Due", "Due soon") else (1 if label == "New" else 2)
            if label in ("Due", "Due soon"): due      += 1
            elif label == "New":             new      += 1
            else:                            upcoming += 1
            rows.append((sort_key, label, color, p, reps, interval, nxt))

        self._all_rows = sorted(rows, key=lambda x: x[0])

        streak = get_streak(self.conn)
        streak_str = f"  [bold yellow]🔥 {streak}d streak[/]" if streak >= 2 else ""
        self.query_one("#stats-bar", Static).update(
            f"  [bold red]{due} due[/]  [bold]{new} new[/]"
            f"  [dim]{upcoming} upcoming  ·  {len(problems)} total[/]"
            + streak_str
        )
        self._render_table()

    def _render_table(self) -> None:
        t = self.query_one(DataTable)
        t.clear()
        q = self._filter.lower()
        for _, label, color, p, reps, interval, nxt in self._all_rows:
            if q and q not in p.name.lower():
                continue
            t.add_row(
                f"[{color}]{label}[/]", p.name, reps, interval, nxt,
                key=str(p),
            )

    # ── search ────────────────────────────────────────────────────────────────
    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_clear_search(self) -> None:
        inp = self.query_one("#search-input", Input)
        inp.value = ""
        self._filter = ""
        self._render_table()
        self.query_one(DataTable).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter = event.value
        self._render_table()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.query_one(DataTable).focus()

    # ── navigation ───────────────────────────────────────────────────────────
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(StudyScreen(Path(str(event.row_key.value))))

    def on_screen_resume(self) -> None:
        self._refresh()

    def action_refresh(self) -> None:
        self._refresh()

    def action_quit_app(self) -> None:
        self.app.exit()

    # ── reset progress ────────────────────────────────────────────────────────
    BINDINGS = [
        Binding("r",      "refresh",      "Refresh"),
        Binding("/",      "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear",  show=False),
        Binding("d",      "reset_row",    "Reset",  show=False),
        Binding("q",      "quit_app",     "Quit"),
    ]

    def action_reset_row(self) -> None:
        t = self.query_one(DataTable)
        if t.cursor_row is None:
            return
        row_key = t.get_row_at(t.cursor_row)
        # row_key[1] is the problem filename
        self.app.push_screen(
            ConfirmModal(f"Reset progress for  {row_key[1]}?"),
            lambda confirmed: self._do_reset(confirmed, row_key[1]),
        )

    def _do_reset(self, confirmed: bool | None, filename: str) -> None:
        if confirmed:
            stem = Path(filename).stem
            reset_progress(self.conn, stem)
            self._refresh()


# ── App ───────────────────────────────────────────────────────────────────────
class MLStudyApp(App):
    TITLE = "CODI"
    CSS = """
    Screen        { background: $background; color: $foreground; }
    Header        { background: $surface; color: $foreground; }
    Footer        { background: $surface; color: $accent; }

    #stats-bar    { height: 1; padding: 0 2; background: $surface; color: $accent; }
    #table        { height: 1fr; border: solid $primary; }

    #problem-bar  { height: 2; padding: 0 2; background: $surface; color: $accent; }
    .full-pane    { height: 1fr; border: solid $primary; padding: 1 2; overflow-y: auto; }

    #modal-box  {
        background: $surface;
        border: double $primary;
        padding: 2 4;
        width: 56;
        height: 15;
        align: center middle;
    }
    #modal-title { text-style: bold; margin-bottom: 1; }
    #modal-skip  { color: $accent; }
    RatingModal  { align: center middle; background: $background 70%; }
    ConfirmModal { align: center middle; background: $background 70%; }

    #hint-box {
        background: $surface;
        border: double $primary;
        padding: 2 4;
        width: 80%;
        height: 60%;
        align: center middle;
    }
    #hint-title  { text-style: bold; margin-bottom: 1; }
    #hint-md     { height: 1fr; overflow-y: auto; background: $surface; }
    AIModal { align: center middle; background: $background 70%; }
    """

    def on_mount(self) -> None:
        for theme in TERMINAL_SEXY_THEMES:
            self.register_theme(theme)
        self.push_screen(MenuScreen())


if __name__ == "__main__":
    MLStudyApp().run()

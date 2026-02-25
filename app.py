#!/usr/bin/env python3
"""
mlstudy-tui — Spaced repetition for ML code.
Drop .py scripts into PROBLEMS_DIR (default: ./problems).
Run: uv run app.py
"""
from __future__ import annotations

import difflib
import os
import sqlite3
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen, ModalScreen
from textual.widgets import DataTable, Footer, Header, Label, Markdown as MarkdownWidget, RichLog, Static

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
    conn.commit()

# ── Helpers ───────────────────────────────────────────────────────────────────
def scan_problems() -> list[Path]:
    if not PROBLEMS_DIR.exists():
        return []
    return sorted(p for p in PROBLEMS_DIR.iterdir() if p.suffix == ".py")


def build_side_by_side(ref_code: str, user_code: str) -> Table:
    """Build a Rich Table with side-by-side diff: reference left, yours right."""
    ref_lines  = ref_code.splitlines()
    user_lines = user_code.splitlines()

    table = Table(
        show_header=True,
        header_style="bold #8b5e3c",
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
                t = Text(line, style="dim white")
                table.add_row(t, t)
        elif tag == "replace":
            ref_chunk  = ref_lines[i1:i2]
            user_chunk = user_lines[j1:j2]
            # pad shorter side with empty strings
            length = max(len(ref_chunk), len(user_chunk))
            ref_chunk  += [""] * (length - len(ref_chunk))
            user_chunk += [""] * (length - len(user_chunk))
            for rl, ul in zip(ref_chunk, user_chunk):
                left  = Text(rl, style="red")   if rl else Text("")
                right = Text(ul, style="green")  if ul else Text("")
                table.add_row(left, right)
        elif tag == "delete":
            for line in ref_lines[i1:i2]:
                table.add_row(Text(line, style="red"), Text(""))
        elif tag == "insert":
            for line in user_lines[j1:j2]:
                table.add_row(Text(""), Text(line, style="green"))

    return table


def status_label(row: sqlite3.Row | None) -> tuple[str, str]:
    if row is None:
        return "New", "white"
    nxt = datetime.fromisoformat(row["next_review"])
    if datetime.now() >= nxt:
        return "Due", "#9e0b0f"
    diff = nxt - datetime.now()
    return (f"In {diff.days}d", "#8b5e3c") if diff.days >= 1 else ("Due soon", "#9e0b0f")


def max_rating_for(attempts: int) -> int:
    if attempts <= 1: return 4
    if attempts == 2: return 3
    if attempts == 3: return 2
    return 1  # 4+ → forced Again

# ── AI providers ──────────────────────────────────────────────────────────────
GEMINI_MODEL     = "gemini-3-flash-preview"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
AI_PROVIDER      = os.environ.get("AI_PROVIDER", "gemini")   # "gemini" | "openrouter"


def _gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No GEMINI_API_KEY found in .env."
    try:
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()
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
            "HTTP-Referer":  "https://github.com/mlstudy-tui",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"**OpenRouter error:** {e}"


def ai_call(prompt: str) -> str:
    if AI_PROVIDER == "openrouter":
        return _openrouter(prompt)
    return _gemini(prompt)


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


# ── Hint Modal ────────────────────────────────────────────────────────────────
class HintModal(ModalScreen):
    BINDINGS = [Binding("escape,h,q", "dismiss", "Close")]

    def __init__(self, problem_name: str, ref_code: str, user_code: str | None) -> None:
        super().__init__()
        self.problem_name = problem_name
        self.ref_code     = ref_code
        self.user_code    = user_code

    def compose(self) -> ComposeResult:
        with Vertical(id="hint-box"):
            yield Label(f"[bold #8b5e3c]hint  ·  {escape(self.problem_name)}[/]", id="hint-title")
            yield MarkdownWidget("*asking Codi…*", id="hint-md")

    def on_mount(self) -> None:
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self) -> None:
        text = get_hint(self.problem_name, self.ref_code, self.user_code)
        self.app.call_from_thread(self._display, text)

    def _display(self, text: str) -> None:
        self.query_one("#hint-md", MarkdownWidget).update(text)


# ── Suggest Fix Modal ─────────────────────────────────────────────────────────
class SuggestFixModal(ModalScreen):
    BINDINGS = [Binding("escape,f,q", "dismiss", "Close")]

    def __init__(self, problem_name: str, ref_code: str, user_code: str) -> None:
        super().__init__()
        self.problem_name = problem_name
        self.ref_code     = ref_code
        self.user_code    = user_code

    def compose(self) -> ComposeResult:
        with Vertical(id="hint-box"):
            yield Label(f"[bold #9e0b0f]suggest fix  ·  {escape(self.problem_name)}[/]", id="hint-title")
            yield MarkdownWidget("*asking Codi…*", id="hint-md")

    def on_mount(self) -> None:
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self) -> None:
        text = get_suggest_fix(self.problem_name, self.ref_code, self.user_code)
        self.app.call_from_thread(self._display, text)

    def _display(self, text: str) -> None:
        self.query_one("#hint-md", MarkdownWidget).update(text)


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
                f"max: [#8b5e3c]{RATING_LABELS[self.max_r]}[/]",
                id="modal-title",
            )
            yield Label("")
            for i in range(1, 5):
                line = f"  [{i}]  {RATING_LABELS[i]:<7} {RATING_DESC[i]}"
                if i <= self.max_r:
                    yield Label(line)
                else:
                    yield Label(f"[#3a3a3a]{line}  ✕[/]")
            yield Label("")
            yield Label("  [Esc] skip without updating", id="modal-skip")

    def action_rate(self, rating: int) -> None:
        if rating <= self.max_r:
            self.dismiss(rating)
        # silently block keys above ceiling

# ── Study Screen ──────────────────────────────────────────────────────────────
class StudyScreen(Screen):
    BINDINGS = [
        Binding("e", "edit",         "Edit"),
        Binding("s", "submit",       "Submit"),
        Binding("h", "hint",         "Hint"),
        Binding("f", "suggest_fix",  "Fix"),
        Binding("q", "back",         "Menu"),
    ]

    def __init__(self, problem: Path) -> None:
        super().__init__()
        self.problem   = problem
        self.work_file = _TMP / f"mlstudy_{problem.stem}.py"
        self.conn      = get_db()
        self.attempts  = 0
        self.has_diff  = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="problem-bar")
        yield RichLog(id="diff-pane", classes="full-pane", highlight=False, markup=True, wrap=False, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#problem-bar", Static).update(
            f"[bold white]{escape(self.problem.name)}[/]  "
            f"[#8b5e3c]─  e edit   s submit   q menu[/]"
        )
        log = self.query_one("#diff-pane", RichLog)
        row = get_row(self.conn, self.problem.stem)
        if row and row["last_output"] and row["last_output"] != "✓  perfect match":
            log.write("[#8b5e3c]── last session ──[/]")
            log.write(Syntax(row["last_output"], "diff", theme="ansi_dark", word_wrap=False))
            log.write("[#8b5e3c]press  e  to start new attempt[/]")
        else:
            log.write("[#8b5e3c]press  e  to open editor[/]")

    def action_hint(self) -> None:
        user_code = self.work_file.read_text() if self.work_file.exists() else None
        self.app.push_screen(HintModal(self.problem.name, self.problem.read_text(), user_code))

    def action_suggest_fix(self) -> None:
        if not self.work_file.exists():
            self.query_one("#diff-pane", RichLog).write("\n[#8b5e3c]edit first — press  e[/]")
            return
        user_code = self.work_file.read_text()
        self.app.push_screen(SuggestFixModal(self.problem.name, self.problem.read_text(), user_code))

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
        ref_code  = self.problem.read_text()

        if user_code.splitlines() == ref_code.splitlines():
            log.write("[white]✓  perfect match[/]")
            summary = "✓  perfect match"
        else:
            log.write(build_side_by_side(ref_code, user_code))
            # keep unified diff as plaintext for next-session preview
            summary = "\n".join(difflib.unified_diff(
                ref_code.splitlines(), user_code.splitlines(),
                fromfile="reference", tofile="yours", lineterm="",
            ))

        max_r = max_rating_for(self.attempts)
        if self.attempts >= 4:
            log.write(f"\n[#9e0b0f]attempt {self.attempts} — press  s  to record (forced: Again)[/]")
        else:
            log.write(
                f"\n[#8b5e3c]attempt {self.attempts}  ·  max: {RATING_LABELS[max_r]}"
                f"  ·  s = submit    e = retry[/]"
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
            log = self.query_one("#diff-pane", RichLog)
            log.write("\n[#8b5e3c]edit first — press  e[/]")
            return

        max_r = max_rating_for(self.attempts)
        if max_r == 1:
            log = self.query_one("#diff-pane", RichLog)
            log.write("\n[#9e0b0f]forced: Again  (4+ attempts)[/]")
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
        self.work_file.unlink(missing_ok=True)
        self.app.pop_screen()

# ── Menu Screen ───────────────────────────────────────────────────────────────
class MenuScreen(Screen):
    BINDINGS = [
        Binding("r", "refresh",  "Refresh"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.conn = get_db()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="stats-bar")
        yield DataTable(id="table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns("Status", "Problem", "Reps", "Interval", "Next review")
        self._refresh()

    def _refresh(self) -> None:
        t        = self.query_one(DataTable)
        t.clear()
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

        for _, label, color, p, reps, interval, nxt in sorted(rows, key=lambda x: x[0]):
            t.add_row(
                f"[{color}]{label}[/]", p.name, reps, interval, nxt,
                key=str(p),
            )

        self.query_one("#stats-bar", Static).update(
            f"  [#9e0b0f bold]{due} due[/]  [white bold]{new} new[/]"
            f"  [#8b5e3c]{upcoming} upcoming  ·  {len(problems)} total[/]"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(StudyScreen(Path(str(event.row_key.value))))

    def on_screen_resume(self) -> None:
        self._refresh()

    def action_refresh(self) -> None:
        self._refresh()

    def action_quit_app(self) -> None:
        self.app.exit()

# ── App ───────────────────────────────────────────────────────────────────────
class MLStudyApp(App):
    TITLE = "mlstudy"
    CSS = """
    Screen        { background: #0c0c0c; color: #ffffff; }
    Header        { background: #191919; color: #ffffff; }
    Footer        { background: #191919; color: #8b5e3c; }

    #stats-bar    { height: 1; padding: 0 2; background: #191919; color: #8b5e3c; }
    #table        { height: 1fr; border: solid #8b5e3c; }

    #problem-bar  { height: 1; padding: 0 2; background: #191919; color: #8b5e3c; }
    .full-pane    { height: 1fr; border: solid #8b5e3c; padding: 1 2; overflow-y: auto; }

    #modal-box  {
        background: #191919;
        border: double #8b5e3c;
        padding: 2 4;
        width: 56;
        height: 15;
        align: center middle;
    }
    #modal-title { text-style: bold; margin-bottom: 1; }
    #modal-skip  { color: #8b5e3c; }
    RatingModal  { align: center middle; background: #0c0c0c 70%; }

    #hint-box {
        background: #191919;
        border: double #8b5e3c;
        padding: 2 4;
        width: 80%;
        height: 60%;
        align: center middle;
    }
    #hint-title  { text-style: bold; margin-bottom: 1; }
    #hint-md     { height: 1fr; overflow-y: auto; background: #191919; }
    HintModal       { align: center middle; background: #0c0c0c 70%; }
    SuggestFixModal { align: center middle; background: #0c0c0c 70%; }
    """

    def on_mount(self) -> None:
        self.push_screen(MenuScreen())


if __name__ == "__main__":
    MLStudyApp().run()

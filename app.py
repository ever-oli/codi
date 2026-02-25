#!/usr/bin/env python3
"""
Codi — Spaced repetition for ML code.
Drop .py scripts into PROBLEMS_DIR (default: ./problems).
Run: uv run app.py
"""
from __future__ import annotations

import difflib
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from rich.markup import escape
from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from ai import get_explain, get_hint, get_suggest_fix
from db import get_db, get_row, get_streak, reset_progress, sm2_update
from modals import AIModal, ConfirmModal, RatingModal
from problems_utils import (
    build_side_by_side,
    load_problem_meta,
    max_rating_for,
    scan_problems,
    status_label,
)
from themes import TERMINAL_SEXY_THEMES

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PROBLEMS_DIR = Path(os.environ.get("PROBLEMS_DIR", "./problems"))
DB_PATH      = Path(os.environ.get("DB_PATH",      "study_data.db"))
EDITOR       = os.environ.get("EDITOR", "hx")
_TMP         = Path(tempfile.gettempdir())

RATING_LABELS = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}


# ── Study Screen ──────────────────────────────────────────────────────────────
class StudyScreen(Screen):
    BINDINGS = [
        Binding("e", "edit",        "Edit"),
        Binding("s", "submit",      "Submit"),
        Binding("h", "hint",        "Hint"),
        Binding("f", "suggest_fix", "Fix"),
        Binding("x", "explain",     "Explain"),
        Binding("q", "back",        "Menu"),
    ]

    def __init__(self, problem: Path) -> None:
        super().__init__()
        self.problem   = problem
        self.meta      = load_problem_meta(problem)
        self.work_file = _TMP / f"codi_{problem.stem}.py"
        self.conn      = get_db(DB_PATH)
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


# ── Search bar ────────────────────────────────────────────────────────────────
class SearchBar(Static):
    DEFAULT_CSS = """
    SearchBar { height: 1; padding: 0 2; background: $surface; }
    SearchBar Input {
        border: none; height: 1;
        background: $surface; color: $foreground; padding: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="search…", id="search-input")


# ── Menu Screen ───────────────────────────────────────────────────────────────
class MenuScreen(Screen):
    BINDINGS = [
        Binding("r",      "refresh",      "Refresh"),
        Binding("/",      "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear",  show=False),
        Binding("d",      "reset_row",    "Reset",  show=False),
        Binding("q",      "quit_app",     "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.conn        = get_db(DB_PATH)
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

    def _refresh(self) -> None:
        problems = scan_problems(PROBLEMS_DIR)
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

    def on_input_submitted(self, _: Input.Submitted) -> None:
        self.query_one(DataTable).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(StudyScreen(Path(str(event.row_key.value))))

    def on_screen_resume(self) -> None:
        self._refresh()

    def action_refresh(self) -> None:
        self._refresh()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_reset_row(self) -> None:
        t = self.query_one(DataTable)
        if t.cursor_row is None:
            return
        row_key = t.get_row_at(t.cursor_row)
        self.app.push_screen(
            ConfirmModal(f"Reset progress for  {row_key[1]}?"),
            lambda confirmed: self._do_reset(confirmed, row_key[1]),
        )

    def _do_reset(self, confirmed: bool | None, filename: str) -> None:
        if confirmed:
            reset_progress(self.conn, Path(filename).stem)
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

    #modal-box {
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
    #hint-title { text-style: bold; margin-bottom: 1; }
    #hint-md    { height: 1fr; overflow-y: auto; background: $surface; }
    AIModal     { align: center middle; background: $background 70%; }
    """

    def on_mount(self) -> None:
        for theme in TERMINAL_SEXY_THEMES:
            self.register_theme(theme)
        self.push_screen(MenuScreen())


if __name__ == "__main__":
    MLStudyApp().run()

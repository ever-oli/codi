"""
modals.py — Reusable Textual modal screens: AIModal, ConfirmModal, RatingModal.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from rich.markup import escape
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Markdown as MarkdownWidget, ListView, ListItem

RATING_LABELS = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
RATING_DESC   = {
    1: "forgot it completely",
    2: "got it with real effort",
    3: "got it with some hesitation",
    4: "recalled perfectly",
}


class AIModal(ModalScreen):
    """Generic modal that fires an AI fetch in a background thread."""
    BINDINGS = [Binding("escape,q", "dismiss", "Close")]

    def __init__(self, title: str, fetch_fn) -> None:
        super().__init__()
        self._title    = title
        self._fetch_fn = fetch_fn

    def compose(self):
        yield Vertical(
            Label(f"[bold]{escape(self._title)}[/]", id="hint-title"),
            MarkdownWidget("*asking Codi…*", id="hint-md"),
            id="hint-box"
        )

    def on_mount(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from ai import render_latex
        text = render_latex(self._fetch_fn())
        self.app.call_from_thread(self._display, text)

    def _display(self, text: str) -> None:
        self.query_one("#hint-md", MarkdownWidget).update(text)


class ConfirmModal(ModalScreen[bool]):
    """Simple yes / no confirmation modal."""
    BINDINGS = [
        Binding("y",      "confirm(True)",  "Yes"),
        Binding("n",      "confirm(False)", "No"),
        Binding("escape", "confirm(False)", "No"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self):
        yield Vertical(
            Label(f"[bold]{escape(self._message)}[/]", id="modal-title"),
            Label(""),
            Label("  [y]  Yes"),
            Label("  [n]  No"),
            id="modal-box"
        )

    def action_confirm(self, result: bool) -> None:
        self.dismiss(result)


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

    def compose(self):
        yield Vertical(
            Label(
                f"[white]attempt {self.attempts}[/]  —  "
                f"max: [dim]{RATING_LABELS[self.max_r]}[/]",
                id="modal-title",
            ),
            Label(""),
            *[
                Label(
                    f"  [{i}]  {RATING_LABELS[i]:<7} {RATING_DESC[i]}"
                    if i <= self.max_r else
                    f"[dim]  [{i}]  {RATING_LABELS[i]:<7} {RATING_DESC[i]}  ✕[/]"
                )
                for i in range(1, 5)
            ],
            Label(""),
            Label("  [Esc] skip without updating", id="modal-skip"),
            id="modal-box"
        )

    def action_rate(self, rating: int) -> None:
        if rating <= self.max_r:
            self.dismiss(rating)


class CollectionSelectModal(ModalScreen[Path]):
    """Modal to select a problem collection (folder)."""
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, collections: list[Path], current: Path) -> None:
        super().__init__()
        self.collections = collections
        self.current     = current

    def compose(self):
        # We'll re-use the modal-box styling but maybe make it taller via CSS in App
        yield Vertical(
            Label("[bold]Select Collection[/]", id="modal-title"),
            ListView(
                *[
                    ListItem(
                        Label(
                            f"[green]●[/] {self._display_name(c)}" if c == self.current
                            else f"  {self._display_name(c)}"
                        ),
                        id=f"col-{i}"
                    )
                    for i, c in enumerate(self.collections)
                ],
                id="collection-list"
            ),
            id="collection-box"  # new ID for styling if needed
        )

    def _display_name(self, path: Path) -> str:
        # Show relative path from problems dir, or just name
        # For now let's just show the name or a shortened path
        # Assuming all are inside a common root, we can show relative path?
        # But for now let's just show the path name or parts
        if path.name == "problems":
            return "Main Collection"
        return str(path.relative_to(path.parent.parent)) if path.parent.name != "problems" else path.name

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = int(event.item.id.split("-")[1])
        self.dismiss(self.collections[idx])

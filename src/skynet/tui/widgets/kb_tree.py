"""KB tree — browsable list of internal_kb/notes/ (plan §13)."""

from __future__ import annotations

from textual.widgets import Tree


class KbTree(Tree):
    """A tree of KB notes grouped by tag."""

    DEFAULT_CSS = """
    KbTree {
        border: round $success;
        background: $surface-darken-1;
        padding: 0 1;
        max-width: 42;
        min-width: 30;
        display: none;   /* toggle with ctrl+k */
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        # Forward widget kwargs (id, classes) to the base Tree.
        super().__init__("internal_kb", *args, **kwargs)

    def populate(self, notes: dict[str, dict]) -> None:
        """notes: {note_id: {"title": str, "tags": [str]}}."""
        self.clear()
        by_tag: dict[str, list[str]] = {}
        for nid, meta in notes.items():
            for tag in meta.get("tags") or ["untagged"]:
                by_tag.setdefault(tag, []).append(nid)
        for tag in sorted(by_tag):
            branch = self.root.add(f"#{tag}", expand=True)
            for nid in sorted(by_tag[tag]):
                title = notes[nid].get("title") or nid
                branch.add_leaf(f"{nid} — {title}")

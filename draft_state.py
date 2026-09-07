"""Draft state persistence and projection.

The draft *log* (an ordered list of pick records) is the single source of
truth, persisted in ``draft_autosave.json`` as ``{"players": [...],
"draft_log": [...]}``. Everything else the apps need — each team's picks and
the flat "everything drafted" list — is *derived* from the log via
``project()`` rather than stored separately. This removes the old
second-source-of-truth file (draft_state.json) that could drift.
"""
import json
import os

DEFAULT_SAVE_FILE = "draft_autosave.json"


def load_draft(path: str = DEFAULT_SAVE_FILE) -> dict:
    """Load the persisted draft ({players, draft_log}); {} if absent/corrupt."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_draft(state: dict, path: str = DEFAULT_SAVE_FILE) -> None:
    """Persist the draft state. Failures are swallowed (best-effort autosave)."""
    try:
        with open(path, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def project(draft_log, players=None):
    """Derive ``(teams, drafted)`` from a draft log.

    ``teams`` maps each player to their picks in draft order; passing
    ``players`` seeds the dict so players with no picks yet still appear.
    ``drafted`` is the flat list of every ingredient taken, in pick order.
    """
    teams = {p: [] for p in (players or [])}
    drafted = []
    for rec in draft_log:
        plyr = rec.get("Player")
        ing = rec.get("Ingredient")
        if ing is None:
            continue
        teams.setdefault(plyr, []).append(ing)
        drafted.append(ing)
    return teams, drafted


def my_picks_for(teams, your_name):
    """Convenience: the picks belonging to ``your_name`` (empty if unknown)."""
    return teams.get(your_name, [])

"""Tests for the log-projection state module.

my_picks/drafted are now *derived* from the draft log rather than stored in a
separate file, so these lock the projection against the real 2025 draft.
"""
import json
import os

import draft_state

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log_2025():
    with open(os.path.join(HERE, "draft_autosave.json")) as f:
        return json.load(f)


def test_project_matches_2025_draft():
    data = _log_2025()
    teams, drafted = draft_state.project(data["draft_log"], data["players"])
    # every player is present, even before deriving picks
    assert set(teams) == set(data["players"])
    # 7 rounds * 9 players = 63 total picks
    assert len(drafted) == 63
    # Will's derived team matches the recorded picks
    will_recorded = [r["Ingredient"] for r in data["draft_log"] if r["Player"] == "Will"]
    assert teams["Will"] == will_recorded
    assert draft_state.my_picks_for(teams, "Will") == will_recorded


def test_project_seeds_empty_players():
    teams, drafted = draft_state.project([], ["A", "B"])
    assert teams == {"A": [], "B": []}
    assert drafted == []


def test_project_skips_records_without_ingredient():
    log = [{"Player": "A", "Ingredient": "Cascade"}, {"Player": "A"}]
    teams, drafted = draft_state.project(log, ["A"])
    assert teams["A"] == ["Cascade"]
    assert drafted == ["Cascade"]


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "draft_autosave.json")
    state = {"players": ["A"], "draft_log": [{"Player": "A", "Ingredient": "Saaz"}]}
    draft_state.save_draft(state, path)
    assert draft_state.load_draft(path) == state


def test_load_missing_returns_empty(tmp_path):
    assert draft_state.load_draft(str(tmp_path / "nope.json")) == {}

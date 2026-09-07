"""Headless smoke tests for the Streamlit app and its two modes.

Uses Streamlit's AppTest harness to execute each script top-to-bottom with
default widget values and assert it runs without raising. This is the safety
net for the Phase-1 refactor: a stale reference, a wrong call-site arity, or a
broken mode split would make a script error and fail here.
"""
import os

import pytest

from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT = os.path.join(HERE, "draft_mode.py")
BREWERY = os.path.join(HERE, "brewery_mode.py")
APP = os.path.join(HERE, "app.py")


@pytest.fixture(autouse=True)
def _run_in_repo_root():
    # draft_mode autosaves to draft_autosave.json (the real 2025 results).
    # Snapshot and restore so the smoke tests can never mutate that data file.
    autosave = os.path.join(HERE, "draft_autosave.json")
    backup = None
    if os.path.exists(autosave):
        with open(autosave) as f:
            backup = f.read()
    cwd = os.getcwd()
    os.chdir(HERE)  # scripts read data files by relative path
    try:
        yield
    finally:
        os.chdir(cwd)
        if backup is not None:
            with open(autosave, "w") as f:
                f.write(backup)


def test_draft_mode_runs_without_exception():
    at = AppTest.from_file(DRAFT, default_timeout=30).run()
    assert not at.exception, at.exception


def test_draft_mode_renders_expected_tabs():
    at = AppTest.from_file(DRAFT, default_timeout=30).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert "Draft Board" in labels
    assert "Recommendations" in labels


def test_draft_mode_renders_roster_strip():
    at = AppTest.from_file(DRAFT, default_timeout=30).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    # The always-visible roster strip (one labeled box per required slot) and the
    # turn header render above the tabs. (The checked-in autosave is a completed
    # draft, so the header reads "Draft complete" rather than "Round N".)
    assert ("Round" in md) or ("Draft complete" in md)
    for label in ("**Malt**", "**Hop**", "**Yeast**", "**Adjunct**"):
        assert label in md


def test_draft_mode_recommendation_panel_and_draft_buttons():
    """An in-progress draft renders the inline recommendation panel with
    one-click Draft buttons (the unified board+recs cockpit)."""
    at = AppTest.from_file(DRAFT, default_timeout=30)
    at.session_state["players"] = ["Alice", "Bob", "Cara", "Dan"]
    at.session_state["draft_log"] = [
        {"Round": 1, "Overall": 1, "Player": "Alice", "Ingredient": "Citra", "Category": "Hop"},
        {"Round": 1, "Overall": 2, "Player": "Bob", "Ingredient": "Cascade", "Category": "Hop"},
    ]
    at.run()
    assert not at.exception, at.exception
    keys = [b.key for b in at.button]
    # One-click draft buttons exist for both the board rows and the inline
    # recommendation panel (the unified cockpit).
    assert any(k and k.startswith("rec-draft-") for k in keys)
    assert any(k and k.startswith("draft-") for k in keys)


def test_draft_mode_mock_simulator_runs():
    """The persona-based mock draft simulator runs a full draft without error
    and produces a pick log."""
    at = AppTest.from_file(DRAFT, default_timeout=60).run()
    assert not at.exception
    at.number_input(key="sim_players").set_value(4).run()
    at.number_input(key="sim_rounds").set_value(3).run()
    at.button(key="run_mock").click().run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Simulation Results" in md


def test_brewery_mode_runs_without_exception():
    # Empty sandbox: should render, inform the user, and stop cleanly.
    at = AppTest.from_file(BREWERY, default_timeout=30).run()
    assert not at.exception, at.exception


def test_brewery_mode_produces_recommendations_from_a_sandbox():
    at = AppTest.from_file(BREWERY, default_timeout=30).run()
    assert not at.exception
    # Select a couple of ingredients into the sandbox, independent of any draft.
    at.multiselect(key="brewery_Hop").select("Cascade").run()
    assert not at.exception
    # With a selection, the "What to Add Next" analysis renders a dataframe.
    assert len(at.dataframe) >= 1


def test_app_entrypoint_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception

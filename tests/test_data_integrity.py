"""Tests for load_data hardening and the data validator."""
import json
import os

import pandas as pd
import pytest

import draft_core as dc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def repo_data():
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        return dc.load_data()
    finally:
        os.chdir(cwd)


# Ingredients the (current/2026) style matrix references that did not exist on
# the 2025 sheet. Allowed here so the 2025 characterization tests still catch
# accidental orphans (typos) without flagging deliberate forward-looking entries.
POST_2025_MATRIX_ADDITIONS = {
    "Columbus/Tomahawk/Zeus",
    "Saflager SH-45 (Thiol Enhancing Dry Lager Yeast)",
    "Abstrax terpenes (any variety of Quantum, Omni, or BrewGas)",
    "Abstrax SkyFarm fruit flavors (any)",
    # 2026 sheet revision (Sep 2026)
    "Chinook", "Cluster", "Mt Hood", "Northern Brewer", "Nugget", "Warrior",
    "Willamette",
    "Belgian Witbier (WLP400, WY3944, Imperial B44, LalBrew Wit)",
    "Irish Ale (WLP004, WY1084, Imperial A44)",
    "Cinnamon", "Cranberries", "Ginger", "Pumpkin",
}


def test_repo_data_has_no_orphan_style_ingredients(repo_data):
    """Safety check: every ingredient a style requires must be draftable from
    the sheet. Evaluated against the 2025 sheet (load_data defaults); the only
    allowed gaps are ingredients added to the matrix for a later season."""
    ingredients, style_matrix = repo_data[0], repo_data[1]
    report = dc.validate_data(ingredients, style_matrix)
    assert set(report["in_matrix_not_sheet"]) <= POST_2025_MATRIX_ADDITIONS


def test_current_season_sheet_has_no_orphan_style_ingredients():
    """Same invariant, but for the sheet the live app actually draws from
    (the 2026 set via league config). Guards against a converter typo
    silently orphaning a style — the required gate the plan calls for."""
    from config import load_league_config

    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        cfg = load_league_config()
        ingredients, style_matrix, *_ = dc.load_data(
            ingredients_path=cfg["ingredients_path"]
        )
    finally:
        os.chdir(cwd)
    report = dc.validate_data(ingredients, style_matrix)
    assert report["in_matrix_not_sheet"] == []


def test_validate_data_detects_missing_ingredient():
    df = pd.DataFrame({"Hop": ["Cascade", "Citra"]})
    style_matrix = {"Fake IPA": {"Hop": ["Cascade", "Nonexistent Hop"]}}
    report = dc.validate_data(df, style_matrix)
    assert "Nonexistent Hop" in report["in_matrix_not_sheet"]
    assert "Citra" in report["in_sheet_not_matrix"]


def test_load_data_missing_required_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dc.load_data(ingredients_path=str(tmp_path / "nope.csv"))


def test_load_required_json_malformed_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json ")
    with pytest.raises(ValueError):
        dc._load_required_json(str(p), "test")


def test_load_optional_json_missing_returns_none(tmp_path):
    assert dc._load_optional_json(str(tmp_path / "nope.json")) is None


def test_load_optional_json_malformed_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json ")
    assert dc._load_optional_json(str(p)) is None

"""Tests for the league config loader.

The DEFAULTS must equal the historical hardcoded values (this is what makes the
config extraction behavior-preserving), and overrides must overlay cleanly.
"""
import json
import os

from config import DEFAULTS, load_league_config


def test_defaults_match_historical_values():
    assert DEFAULTS["rounds"] == 7
    assert DEFAULTS["flex_slots"] == 3
    assert DEFAULTS["required_categories"] == {
        "Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1
    }
    assert DEFAULTS["category_aliases"]["Hop"] == ["Hop", "Hops"]
    # Defaults mirror historical (2025) behavior; the live season is set only
    # in the checked-in league_config.json (see test below).
    assert DEFAULTS["ingredient_year"] == 2025
    assert DEFAULTS["ingredients_path"] == "ingredients_2025.csv"


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = load_league_config(str(tmp_path / "does_not_exist.json"))
    assert cfg["rounds"] == DEFAULTS["rounds"]
    assert cfg["required_categories"] == DEFAULTS["required_categories"]


def test_repo_config_loads_and_matches_defaults():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_league_config(os.path.join(here, "league_config.json"))
    # Rules are unchanged for 2026; only the ingredient set advances.
    assert cfg["rounds"] == 7
    assert cfg["flex_slots"] == 3
    assert cfg["required_categories"] == DEFAULTS["required_categories"]


def test_repo_config_points_at_current_season():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_league_config(os.path.join(here, "league_config.json"))
    # The live app draws from the 2026 sheet via config, not the 2025 default.
    assert cfg["ingredient_year"] == 2026
    assert cfg["ingredients_path"] == "ingredients_2026.csv"
    assert os.path.exists(os.path.join(here, cfg["ingredients_path"]))


def test_partial_override_overlays_on_defaults(tmp_path):
    p = tmp_path / "league_config.json"
    p.write_text(json.dumps({"rounds": 8, "flex_slots": 2}))
    cfg = load_league_config(str(p))
    assert cfg["rounds"] == 8
    assert cfg["flex_slots"] == 2
    # untouched keys still present from defaults
    assert cfg["required_categories"] == DEFAULTS["required_categories"]
    assert "category_aliases" in cfg


def test_malformed_file_falls_back(tmp_path):
    p = tmp_path / "league_config.json"
    p.write_text("{ not valid json ")
    cfg = load_league_config(str(p))
    assert cfg["rounds"] == DEFAULTS["rounds"]

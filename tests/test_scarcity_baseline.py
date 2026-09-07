"""Tests for scripts/build_scarcity_baseline.py and the retirement of the
static scarcity file from the engine path."""
import json
import os
import sys

import pandas as pd

import draft_core as dc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

import build_scarcity_baseline as bsb  # noqa: E402

CFG = {
    "required_categories": {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1},
    "category_aliases": {"Base Malt": ["Base Malt"], "Hop": ["Hop"], "Yeast": ["Yeast"],
                         "Adjunct": ["Adjunct"], "Specialty": ["Specialty"]},
}


def _synthetic():
    sheet = pd.DataFrame({
        "Base Malt": ["M1", "M2", None],
        "Hop": ["H1", "H2", "H3"],
        "Yeast": ["Y1", None, None],
        "Adjunct": ["A1", "A2", None],
        "Specialty": ["S1", "S2", "S3"],
    })
    matrix = {
        "Style A": {"Base Malt": ["M1"], "Hop": ["H1", "H2", "H3"], "Yeast": ["Y1"],
                    "Adjunct": ["A1"], "Specialty": ["S1"]},
        "Style B": {"Base Malt": ["M1", "M2"], "Hop": ["H1"], "Yeast": ["Y1"],
                    "Adjunct": ["A1", "A2"], "Specialty": ["S2"]},
        # S3 on the sheet but in no style; "Ghost" in a style but not on the sheet
        "Style C": {"Base Malt": ["Ghost"], "Hop": [], "Yeast": [], "Adjunct": [], "Specialty": []},
    }
    return sheet, matrix


def test_build_rows_synthetic_shape_and_values():
    sheet, matrix = _synthetic()
    sim = {"H1": [{"ingredient": "H2", "score": 0.8}, {"ingredient": "Ghost", "score": 0.9},
                  {"ingredient": "H3", "score": 0.1}]}
    rows = bsb.build_rows(sheet, matrix, CFG, similarity=sim, num_players=4)
    by = {r["Ingredient"]: r for r in rows}
    # only sheet ingredients that some style models; ordered by category then name
    assert [r["Ingredient"] for r in rows] == ["M1", "M2", "H1", "H2", "H3", "Y1", "A1", "A2", "S1", "S2"]
    assert "Ghost" not in by and "S3" not in by
    assert by["M1"]["Style Coverage"] == 2 and by["M2"]["Style Coverage"] == 1
    # signature: best style list size -> M1 in a 1-list (8/9), H2 only in a 3-list (8/11)
    assert by["M1"]["Signature"] == round(8 / 9, 4)
    assert by["H2"]["Signature"] == round(8 / 11, 4)
    # close substitutes count only modelled neighbours above threshold
    assert by["H1"]["Close Substitutes"] == 1
    # baseline scarcity = 1 - e^-(demand/supply): yeast 4/1, hop 4/3, specialty 1/2
    import math
    assert by["Y1"]["Baseline Scarcity"] == round(1 - math.exp(-4 / 1), 4)
    assert by["H1"]["Baseline Scarcity"] == round(1 - math.exp(-4 / 3), 4)
    assert by["S1"]["Baseline Scarcity"] == round(1 - math.exp(-(4 // 3) / 2), 4)
    for r in rows:
        assert 0.0 <= r["Baseline Scarcity"] < 1.0 and 0.0 < r["Signature"] <= 1.0


def test_build_rows_deterministic():
    sheet, matrix = _synthetic()
    assert bsb.render(bsb.build_rows(sheet, matrix, CFG)) == bsb.render(bsb.build_rows(sheet, matrix, CFG))


def test_committed_snapshot_matches_generator():
    from config import load_league_config
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        cfg = load_league_config()
        ings, sm, *_ = dc.load_data(ingredients_path=cfg["ingredients_path"])
        sim = dc.load_similarity(cfg)
        _, _, squash = dc.scoring_params(cfg)
        text = bsb.render(bsb.build_rows(ings, sm, cfg, sim, num_players=9, squash=squash))
        committed = open("ingredient_scarcity.json", encoding="utf-8").read()
    finally:
        os.chdir(cwd)
    assert committed == text
    rows = json.loads(committed)
    assert len(rows) == 217  # every 2026 sheet ingredient is modelled


def test_load_data_scarcity_file_is_optional(tmp_path):
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        out = dc.load_data(scarcity_path=str(tmp_path / "absent.json"))
    finally:
        os.chdir(cwd)
    assert out[2].empty  # scarcity_df slot preserved, empty when file absent


def test_engine_ignores_scarcity_df(tmp_path):
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        ings, sm, sc, _, sb, i2c = dc.load_data(ingredients_path="ingredients_2026.csv")
    finally:
        os.chdir(cwd)
    req = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}
    a = dc.next_best_picks([], [], ings, sm, sc, req, 3, i2c, sb, top_k=50)
    b = dc.next_best_picks([], [], ings, sm, pd.DataFrame(), req, 3, i2c, sb, top_k=50)
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))

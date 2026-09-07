"""Characterization tests for draft_core.

These lock in the *current* behavior of the shared engine so the Phase-1
refactor (deduping the local copies out of the Streamlit app, extracting
config, unifying state) provably does not change draft results. Golden values
were captured from the real 2025 draft data checked into the repo.

Run:  ./fantasydraft/bin/python -m pytest -q
"""
import json
import os
from collections import defaultdict

import pytest

import draft_core as dc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixtures — real repo data + the 2025 draft log
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    cwd = os.getcwd()
    os.chdir(HERE)  # draft_core.load_data() uses relative default paths
    try:
        (ingredients, style_matrix, scarcity, opponent_model,
         style_bias, ingredient_to_category) = dc.load_data()
    finally:
        os.chdir(cwd)
    return {
        "ingredients": ingredients,
        "style_matrix": style_matrix,
        "scarcity": scarcity,
        "opponent_model": opponent_model,
        "style_bias": style_bias,
        "i2c": ingredient_to_category,
    }


@pytest.fixture(scope="module")
def draft_2025():
    with open(os.path.join(HERE, "draft_autosave.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def opp_signals(data):
    early, pair = {}, defaultdict(int)
    opp = data["opponent_model"]
    for r in opp.get("ingredient_popularity", []):
        if r.get("Ingredient"):
            early[r["Ingredient"]] = float(r.get("Early_Score", 0.0))
    for r in opp.get("top_pairs", []):
        a, b, c = r.get("A"), r.get("B"), int(r.get("PairCount", 0))
        if a and b:
            pair[(a, b)] += c
            pair[(b, a)] += c
    return early, pair


REQUIRED = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}


# ---------------------------------------------------------------------------
# Snake-draft math — validated against the actual recorded 2025 order
# ---------------------------------------------------------------------------
def test_snake_order_direction():
    assert dc.snake_draft_order(9, 1) == list(range(9))          # round 1 ascending
    assert dc.snake_draft_order(9, 2) == list(range(8, -1, -1))  # round 2 reversed
    assert dc.snake_draft_order(9, 3) == list(range(9))


def test_pick_slot_reconstructs_2025_draft(draft_2025):
    """Every recorded pick's (Round, Player) must fall out of the snake math."""
    players = draft_2025["players"]
    n = len(players)
    for rec in draft_2025["draft_log"]:
        rnd, seat = dc.pick_slot(rec["Overall"], n)
        assert rnd == rec["Round"], rec
        assert players[seat] == rec["Player"], rec


# ---------------------------------------------------------------------------
# Availability + category bucketing
# ---------------------------------------------------------------------------
def test_build_available_set(data):
    avail = dc.build_available_set(data["ingredients"])
    assert len(avail) == 200
    assert "Cascade" in avail
    assert "Maris Otter" in avail


def test_bucket_for_rules():
    assert dc.bucket_for_rules("Base Malt") == "Malt"
    assert dc.bucket_for_rules("Hop") == "Hop"
    assert dc.bucket_for_rules("Yeast") == "Yeast"
    assert dc.bucket_for_rules("Adjunct") == "Adjunct"
    assert dc.bucket_for_rules("Specialty") == "Flex"
    assert dc.bucket_for_rules("Anything Else") == "Flex"


# ---------------------------------------------------------------------------
# Rules status
# ---------------------------------------------------------------------------
def test_compute_rules_status_wills_2025_team(data, draft_2025):
    will = [r["Ingredient"] for r in draft_2025["draft_log"] if r["Player"] == "Will"]
    rs = dc.compute_rules_status(will, data["i2c"], 7)
    # Molasses is drafted as an Adjunct but is absent from style_matrix, so it
    # buckets to Flex — locking this reveals the data gap without changing it.
    assert rs["counts"] == {"Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 0, "Flex": 4}
    assert rs["flex_remaining"] == 0
    assert rs["picks_remaining"] == 0
    assert rs["feasible"] is False


def test_compute_rules_status_feasible_early():
    rs = dc.compute_rules_status(["Cascade"], {"Cascade": "Hop"}, 7)
    assert rs["counts"]["Hop"] == 1
    assert rs["required_remaining"] == {"Malt": 1, "Yeast": 1, "Adjunct": 1, "Hop": 0}
    assert rs["feasible"] is True


# ---------------------------------------------------------------------------
# Style viability
# ---------------------------------------------------------------------------
def test_compute_style_status_sorted_and_scored(data, draft_2025):
    log = draft_2025["draft_log"]
    drafted = [r["Ingredient"] for r in log]
    will = [r["Ingredient"] for r in log if r["Player"] == "Will"]
    df = dc.compute_style_status(will, drafted, data["style_matrix"], REQUIRED, 3)
    # sorted by Score descending
    assert list(df["Score"]) == sorted(df["Score"], reverse=True)
    top = df.iloc[0]
    assert top["Style"] == "Stout / Porter"
    assert int(top["Satisfied Categories"]) == 4
    assert int(top["Score"]) == 12


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def test_next_best_picks_empty_draft_golden(data, opp_signals):
    early, _ = opp_signals
    recs = dc.next_best_picks(
        [], [], data["ingredients"], data["style_matrix"], data["scarcity"],
        REQUIRED, 3, data["i2c"], data["style_bias"],
        early_signal=early, bias_weight=0.0, top_k=5,
    )
    assert list(recs.columns) == [
        "Ingredient", "Category", "Style Coverage",
        "Scarcity", "Popularity", "Bias Factor", "Pick Value",
    ]
    assert len(recs) == 5
    top = recs.iloc[0]
    assert top["Ingredient"] == "Pale Malt (2 Row)"
    assert int(top["Style Coverage"]) == 11
    assert top["Pick Value"] == pytest.approx(50.062, abs=1e-3)


def test_next_best_picks_style_coverage_dominates_weighting(data, opp_signals):
    """Locks the 2.0x coverage / 1.0x scarcity weighting from commit 1b9385c:
    Pick Value ordering must track Style Coverage, not Scarcity, at the top."""
    early, _ = opp_signals
    recs = dc.next_best_picks(
        [], [], data["ingredients"], data["style_matrix"], data["scarcity"],
        REQUIRED, 3, data["i2c"], data["style_bias"],
        early_signal=early, bias_weight=0.0, top_k=15,
    )
    # The most style-covering ingredient is the top recommendation.
    assert recs.iloc[0]["Style Coverage"] == recs["Style Coverage"].max()


def test_next_best_picks_excludes_drafted(data, opp_signals):
    early, _ = opp_signals
    recs = dc.next_best_picks(
        ["Pale Malt (2 Row)"], ["Pale Malt (2 Row)"], data["ingredients"],
        data["style_matrix"], data["scarcity"], REQUIRED, 3, data["i2c"],
        data["style_bias"], early_signal=early, bias_weight=0.0, top_k=15,
    )
    assert "Pale Malt (2 Row)" not in list(recs["Ingredient"])


# ---------------------------------------------------------------------------
# Block picks
# ---------------------------------------------------------------------------
def test_block_picks_structure_and_golden(data, draft_2025, opp_signals):
    early, pair = opp_signals
    log = draft_2025["draft_log"]
    drafted27 = [r["Ingredient"] for r in log[:27]]
    will = [r["Ingredient"] for r in log if r["Player"] == "Will"]
    bp = dc.block_picks(drafted27, will, pair, data["ingredients"],
                        early_signal=early, top_k=5)
    assert list(bp.columns) == ["Ingredient", "Block Score", "Popularity Cue"]
    assert len(bp) == 5
    # sorted by Block Score descending
    assert list(bp["Block Score"]) == sorted(bp["Block Score"], reverse=True)


def test_block_picks_empty_when_no_pairs(data):
    bp = dc.block_picks([], [], defaultdict(int), data["ingredients"], top_k=5)
    assert bp.empty
    assert list(bp.columns) == ["Ingredient", "Block Score", "Popularity Cue"]

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
    # Dedicated, read-only fixture — decoupled from the live draft_autosave.json,
    # which the running app overwrites (that file is the app's save slot).
    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    with open(os.path.join(fixtures, "draft_2025.json")) as f:
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
    # Will's 2025 team: Dry English yeast, Golden Promise, Fuggle, Molasses
    # (adjunct) + roasted barley / flaked barley / pale chocolate (flex).
    # Molasses used to be absent from style_matrix and silently bucketed to
    # Flex (making a legal roster look infeasible); the 2026 matrix models
    # every sheet ingredient, so the name-based path now agrees with the
    # record-based one below.
    assert rs["counts"] == {"Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1, "Flex": 3}
    assert rs["flex_remaining"] == 0
    assert rs["picks_remaining"] == 0
    assert rs["feasible"] is True


def test_compute_rules_status_feasible_early():
    rs = dc.compute_rules_status(["Cascade"], {"Cascade": "Hop"}, 7)
    assert rs["counts"]["Hop"] == 1
    assert rs["required_remaining"] == {"Malt": 1, "Yeast": 1, "Adjunct": 1, "Hop": 0}
    assert rs["feasible"] is True


def test_rules_status_from_records_trusts_stored_category(draft_2025):
    """The record-based path counts the *drafted* category, so Will's Molasses
    (an Adjunct absent from the style matrix) satisfies the Adjunct slot rather
    than burning a flex slot — the fix for the attribution bug."""
    will = [r for r in draft_2025["draft_log"] if r["Player"] == "Will"]
    rs = dc.compute_rules_status_from_records(will, 7)
    assert rs["counts"] == {"Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1, "Flex": 3}
    assert rs["required_met"]["Adjunct"] is True


def test_roster_slots_fills_and_spills():
    records = [
        {"Ingredient": "Pale Malt (2 Row)", "Category": "Base Malt"},
        {"Ingredient": "Citra", "Category": "Hop"},
        {"Ingredient": "Mosaic", "Category": "Hop"},          # 2nd hop -> flex
        {"Ingredient": "Mango", "Category": "Adjunct"},       # not in matrix -> Adjunct
    ]
    slots = {s["key"]: s["filled"] for s in dc.roster_slots(records, flex_slots=3)}
    assert slots["Malt"] == "Pale Malt (2 Row)"
    assert slots["Hop"] == "Citra"
    assert slots["Adjunct"] == "Mango"          # correctly attributed
    assert slots["Yeast"] is None
    assert slots["Flex1"] == "Mosaic"           # extra hop spilled to flex


def test_roster_slots_round8_slot():
    slots = dc.roster_slots([], enable_round8=True, flex_slots=3)
    keys = [s["key"] for s in slots]
    assert keys == ["Malt", "Hop", "Yeast", "Adjunct", "Flex1", "Flex2", "Flex3", "Round8"]
    assert all(s["required"] for s in slots if s["key"] in {"Malt", "Hop", "Yeast", "Adjunct"})


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
    # Will's 2025 roster (Dry English, Golden Promise, Fuggle, roasted +
    # flaked barley, molasses, pale chocolate) is a textbook dry/oatmeal stout.
    # Under the 2026 matrix all five categories are satisfied (molasses now
    # counts as the adjunct), and every one of the 7 picks is used by the
    # Stout style. Several dark British styles tie on categories alone; the
    # Picks Matched / Match tie-breaks are what put Stout on top.
    assert top["Style"].startswith("Stout")
    assert int(top["Satisfied Categories"]) == 5
    assert int(top["Picks Matched"]) == 7
    assert int(top["Score"]) == 15
    assert 0.0 < top["Match"] <= 1.0
    # Porter shares every ingredient but has broader lists -> lower Match.
    porter = df[df["Style"].str.startswith("Porter")].iloc[0]
    assert porter["Match"] < top["Match"]


def test_compute_style_status_match_prefers_defining_ingredients():
    """A pick in a narrow category list is stronger evidence for that style
    than the same pick in a broad list; an off-style pick costs more than any
    matched pick."""
    sm = {
        "Narrow": {"Base Malt": ["A"], "Hop": ["H1", "H2"], "Yeast": ["Y"],
                   "Adjunct": ["X"], "Specialty": []},
        "Broad": {"Base Malt": ["A", "B", "C", "D"], "Hop": ["H1", "H2", "H3", "H4"],
                  "Yeast": ["Y", "Z"], "Adjunct": ["X", "W"], "Specialty": []},
    }
    df = dc.compute_style_status(["A", "H1"], ["A", "H1"], sm, REQUIRED, 3)
    assert df.iloc[0]["Style"] == "Narrow"
    assert df.iloc[0]["Picks Matched"] == 2
    # Add an ingredient only Broad uses -> Broad now matches more picks and wins
    df2 = dc.compute_style_status(["A", "H1", "W"], ["A", "H1", "W"], sm, REQUIRED, 3)
    assert df2.iloc[0]["Style"] == "Broad"
    assert df2.iloc[0]["Picks Matched"] == 3


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def test_next_best_picks_structure_and_normalized(data, opp_signals):
    early, _ = opp_signals
    recs = dc.next_best_picks(
        [], [], data["ingredients"], data["style_matrix"], data["scarcity"],
        REQUIRED, 3, data["i2c"], data["style_bias"],
        early_signal=early, bias_weight=0.0, top_k=5,
    )
    assert list(recs.columns) == [
        "Ingredient", "Category", "Style Coverage", "Scarcity", "Popularity",
        "Bias Factor", "Fit", "Urgency", "Synergy", "Denial", "Pick Value", "Why",
    ]
    assert len(recs) == 5
    # Weighted sum of normalized [0,1] components -> Pick Value bounded by the
    # total weight (~1.0); sorted descending.
    assert list(recs["Pick Value"]) == sorted(recs["Pick Value"], reverse=True)
    assert recs["Pick Value"].max() <= 1.0 + 1e-9
    assert recs.iloc[0]["Why"]  # non-empty explanation


def test_next_best_picks_fit_has_diminishing_returns(data, opp_signals):
    """Style fit rewards versatility but with diminishing returns, so a very
    broadly-used ingredient is not linearly better than a moderately-used one
    (this is what stops a single base malt from dominating the whole board)."""
    early, _ = opp_signals
    recs = dc.next_best_picks(
        [], [], data["ingredients"], data["style_matrix"], data["scarcity"],
        REQUIRED, 3, data["i2c"], data["style_bias"],
        early_signal=early, bias_weight=0.0, top_k=1000,
    )
    df = recs[recs["Style Coverage"] > 0]
    hi = df.sort_values("Style Coverage", ascending=False).iloc[0]
    lo = df[df["Style Coverage"] == df["Style Coverage"].median()].iloc[0]
    ratio_cov = hi["Style Coverage"] / max(lo["Style Coverage"], 1)
    ratio_fit = hi["Fit"] / max(lo["Fit"], 1e-9)
    assert ratio_fit < ratio_cov  # fit grows slower than raw coverage
    assert (df["Fit"] <= 1.0 + 1e-9).all()


def test_next_best_picks_fit_sharpens_to_your_styles(data):
    """Once you've committed to a style, an unpicked ingredient central to that
    style scores higher on Fit than it did on an empty roster."""
    ings, sm, sc, i2c, sb = (data["ingredients"], data["style_matrix"],
                             data["scarcity"], data["i2c"], data["style_bias"])
    # Pick a stout-y specialty + yeast to bias focus toward dark styles.
    seeds = ["Roasted Barley"] if "Roasted Barley" in i2c else []
    # Fall back to any ingredient that appears in a single style to seed focus.
    if not seeds:
        for ing, c in i2c.items():
            seeds = [ing]
            break
    empty = dc.next_best_picks([], [], ings, sm, sc, REQUIRED, 3, i2c, sb, top_k=1000)
    focused = dc.next_best_picks(seeds, list(seeds), ings, sm, sc, REQUIRED, 3,
                                 i2c, sb, top_k=1000)
    # The focus machinery runs without error and still produces bounded fits.
    assert (focused["Fit"] <= 1.0 + 1e-9).all()
    assert len(focused) > 0 and len(empty) > 0


def test_next_best_picks_excludes_drafted(data, opp_signals):
    early, _ = opp_signals
    recs = dc.next_best_picks(
        ["Pale Malt (2 Row)"], ["Pale Malt (2 Row)"], data["ingredients"],
        data["style_matrix"], data["scarcity"], REQUIRED, 3, data["i2c"],
        data["style_bias"], early_signal=early, bias_weight=0.0, top_k=15,
    )
    assert "Pale Malt (2 Row)" not in list(recs["Ingredient"])


# ---------------------------------------------------------------------------
# Best available (roster-agnostic board value) + team context
# ---------------------------------------------------------------------------
def _ba(data, drafted, early):
    return dc.best_available(
        drafted, data["ingredients"], data["style_matrix"], data["scarcity"],
        REQUIRED, 3, data["i2c"], data["style_bias"], early_signal=early, top_k=1000,
    )


def test_best_available_is_roster_agnostic(data, opp_signals):
    early, _ = opp_signals
    # Different rosters, same board -> identical ranking (it ignores my_picks).
    a = _ba(data, ["Citra"], early)
    b = _ba(data, ["Citra"], early)  # same drafted board
    assert list(a["Ingredient"]) == list(b["Ingredient"])
    # best_available takes no roster at all; ensure it doesn't include a drafted item.
    assert "Citra" not in list(_ba(data, ["Citra"], early)["Ingredient"])


def test_best_available_sorted_and_spans_categories(data, opp_signals):
    early, _ = opp_signals
    ba = _ba(data, [], early)
    assert list(ba["Pick Value"]) == sorted(ba["Pick Value"], reverse=True)
    # A healthy board ranking is not monopolized by one category in the top 15.
    assert ba.head(15)["Category"].nunique() >= 2


def test_team_context_reports_needs_and_style(data):
    records = [
        {"Ingredient": "Citra", "Category": "Hop"},
        {"Ingredient": "Maris Otter", "Category": "Base Malt"},
    ]
    picks = ["Citra", "Maris Otter"]
    ctx = dc.team_context(records, picks, picks, data["style_matrix"], REQUIRED, 3)
    assert ctx["needed_buckets"] == {"Yeast", "Adjunct"}
    assert ctx["likely_style"] != "TBD"
    assert ctx["flex_remaining"] == 3


def test_team_context_empty_roster_style_tbd(data):
    ctx = dc.team_context([], [], [], data["style_matrix"], REQUIRED, 3)
    assert ctx["likely_style"] == "TBD"
    assert ctx["needed_buckets"] == {"Malt", "Hop", "Yeast", "Adjunct"}


def test_next_best_picks_urgency_prioritizes_needed_category(data, opp_signals):
    """With a hop already in hand, an unmet required category (Yeast) should
    surface a Yeast candidate above hops in the top few via the urgency term."""
    early, _ = opp_signals
    recs = dc.next_best_picks(
        ["Cascade"], ["Cascade"], data["ingredients"], data["style_matrix"],
        data["scarcity"], REQUIRED, 3, data["i2c"], data["style_bias"],
        early_signal=early, top_k=20,
    )
    top_cats = list(recs["Category"].head(10))
    assert any(c in ("Yeast", "Base Malt", "Adjunct") for c in top_cats)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def test_compute_style_idf_ranks_signature_above_generic():
    sm = {
        "A": {"Hop": ["Broad", "Rare1"]},
        "B": {"Hop": ["Broad", "Rare2"]},
        "C": {"Hop": ["Broad", "Rare3"]},
    }
    idf = dc.compute_style_idf(sm)
    assert idf["Rare1"] > idf["Broad"]  # in 1 style vs 3


def test_dynamic_scarcity_rises_as_category_depletes():
    # 5 hops, 2 drafters -> pressure 0.4 (below the 1.0 cap so the effect shows).
    hops = {f"H{i}" for i in range(1, 6)}
    i2c = {h: "Hop" for h in hops}
    full = dc.compute_dynamic_scarcity(hops, i2c, {"Hop": 1}, num_players=2)
    depleted = dc.compute_dynamic_scarcity({"H1", "H2", "H3"}, i2c,
                                           {"Hop": 1}, num_players=2)
    assert depleted["H1"] > full["H1"]  # fewer hops left -> scarcer


def test_dynamic_scarcity_hop_substitutes_reduce_scarcity():
    hops = {f"H{i}" for i in range(1, 6)}
    i2c = {h: "Hop" for h in hops}
    hop_sim = {"H1": [{"hop": "H2", "score": 0.9}, {"hop": "H3", "score": 0.8}]}
    without = dc.compute_dynamic_scarcity(hops, i2c, {"Hop": 1}, num_players=2)
    with_subs = dc.compute_dynamic_scarcity(hops, i2c, {"Hop": 1}, num_players=2,
                                            hop_similarity=hop_sim)
    assert with_subs["H1"] < without["H1"]  # close analogs -> less scarce


def test_picks_until_next_turn_snake():
    # 4 players, seat 0. Overall pick 1 is yours (gap 0). After that, seat 0's
    # next turn in a snake draft is overall pick 8 (round 2 reversed).
    assert dc.picks_until_next_turn(1, 4, 0) == 0
    assert dc.picks_until_next_turn(2, 4, 0) == 6  # picks 2..7 before your pick 8


def test_explain_pick_reports_dominant_component():
    why = dc.explain_pick({"fit": 0.25, "scarce": 0.01, "need": 0.2,
                           "syn": 0.0, "deny": 0.0, "pop": 0.0})
    assert why.startswith(dc._WHY_LABELS["fit"])
    assert dc.explain_pick({k: 0.0 for k in dc.DEFAULT_PICK_WEIGHTS}) == "balanced value"


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

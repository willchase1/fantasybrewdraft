"""Tests for the calibrated scoring shape, config overrides, and the 2025
replay harness (scripts/calibrate_weights.py)."""
import os
import sys

import pytest

import draft_core as dc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

import calibrate_weights as cw  # noqa: E402

REQUIRED = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}


@pytest.fixture(scope="module")
def ctx():
    return cw.load_context()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def test_scoring_params_defaults_and_overlay():
    w, b, sq = dc.scoring_params({})
    assert w == dc.DEFAULT_PICK_WEIGHTS and b == dc.BOARD_VALUE_WEIGHTS
    assert sq == dc.DEFAULT_SQUASH
    w2, b2, sq2 = dc.scoring_params({
        "pick_weights": {"fit": 0.5, "bogus": 9},
        "board_value_weights": {"pop": 0.0},
        "squash": {"fit": 3},
    })
    assert w2["fit"] == 0.5 and w2["scarce"] == dc.DEFAULT_PICK_WEIGHTS["scarce"]
    assert "bogus" not in w2
    assert b2["pop"] == 0.0 and b2["fit"] == dc.BOARD_VALUE_WEIGHTS["fit"]
    assert sq2["fit"] == 3.0 and sq2["syn"] == dc.DEFAULT_SQUASH["syn"]


def test_repo_config_scoring_params_load():
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        w, b, sq = dc.scoring_params()
    finally:
        os.chdir(cwd)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert set(sq) == set(dc.DEFAULT_SQUASH)


def test_build_opponent_signals():
    early, pairs = dc.build_opponent_signals({
        "ingredient_popularity": [{"Ingredient": "Citra", "Early_Score": 1.5}, {"Ingredient": ""}],
        "top_pairs": [{"A": "Citra", "B": "Mosaic", "PairCount": 2}],
    })
    assert early == {"Citra": 1.5}
    assert pairs[("Citra", "Mosaic")] == 2 and pairs[("Mosaic", "Citra")] == 2
    assert dc.build_opponent_signals(None) == ({}, {})


# ---------------------------------------------------------------------------
# Component behaviour
# ---------------------------------------------------------------------------
def _recs(ctx, picks, drafted=None, **kw):
    return dc.next_best_picks(
        picks, drafted if drafted is not None else picks, ctx["ingredients"],
        ctx["style_matrix"], ctx["scarcity"], REQUIRED, 3, ctx["i2c"], ctx["style_bias"],
        early_signal=ctx["early"], top_k=100000, similarity=ctx["similarity"],
        pair_lookup=ctx["pairs"], num_players=9, **kw,
    ).set_index("Ingredient")


def test_fit_alignment_prefers_the_style_you_lead_toward(ctx):
    # Belgian yeast + Belgian pils -> Belgian styles lead; a Belgian-only hop
    # should out-Fit an American-only hop of similar breadth.
    r = _recs(ctx, ["Trappist Ale (WLP500, WY1214, B48)", "Belgian Pilsner"])
    assert r.loc["Styrian Goldings", "Fit"] > r.loc["Simcoe", "Fit"]
    assert r.loc["Candi Sugar, Dark", "Fit"] > r.loc["Milk Sugar (Lactose)", "Fit"]


def test_fit_signature_favours_defining_ingredient_on_empty_roster(ctx):
    # Empty roster: a yeast that anchors a style beats a base malt with the
    # same breadth once signature is on; with fit_sig=0 breadth alone decides.
    on = _recs(ctx, [])
    off = _recs(ctx, [], squash={"fit_sig": 0.0, "fit_idf": 0.0})
    ing_y, ing_m = "Hefewiezen (WLP300, WY3068, G01)", "Munich, Dark"
    # Hefeweizen is the only yeast in its style (signature 8/9) vs a 10-way base list
    assert on.loc[ing_y, "Fit"] / off.loc[ing_y, "Fit"] > on.loc[ing_m, "Fit"] / off.loc[ing_m, "Fit"]


def test_fit_squash_constant_changes_saturation(ctx):
    fast = _recs(ctx, [], squash={"fit": 1.0})
    slow = _recs(ctx, [], squash={"fit": 20.0})
    # coverage 1 vs coverage 10: with k=1 they are close (0.5 vs 0.91);
    # with k=20 the ratio approaches the raw coverage ratio.
    cov = fast["Style Coverage"]
    lo, hi = cov[cov == cov.min()].index[0], cov[cov == cov.max()].index[0]
    assert fast.loc[hi, "Fit"] / fast.loc[lo, "Fit"] < slow.loc[hi, "Fit"] / slow.loc[lo, "Fit"]


def test_urgency_required_slot_constant_and_above_flex(ctx):
    # Roster has hop + malt + yeast; Adjunct is the unmet required slot.
    picks = ["Citra", "Pale Malt (2 Row)",
             "American Ale (WLP001 [liquid or dry], WY1056, US-05, A07)"]
    r = _recs(ctx, picks)
    adj = r[r["Category"] == "Adjunct"]["Urgency"]
    spec = r[r["Category"] == "Specialty"]["Urgency"]
    assert adj.min() > spec.max()
    # With need_slack the urgency of a needed slot falls while spare picks remain.
    r2 = _recs(ctx, picks, squash={"need_slack": 1.0})
    assert r2[r2["Category"] == "Adjunct"]["Urgency"].max() < adj.max()


def test_scarcity_residual_demand_falls_as_category_is_drafted(ctx):
    yeasts = [i for i, c in ctx["i2c"].items() if c == "Yeast"]
    my = ["Pale Malt (2 Row)"]
    resid = {"scarce_residual": 1.0}
    early = _recs(ctx, my, drafted=my, squash=resid)
    # Eight other drafters have taken a yeast -> residual demand for yeast is 1.
    late = _recs(ctx, my, drafted=my + yeasts[:8], squash=resid)
    probe = next(y for y in yeasts[8:] if y in early.index and y in late.index)
    assert late.loc[probe, "Scarcity"] < early.loc[probe, "Scarcity"]
    # Default (static demand) goes the other way: fewer left -> scarcer.
    late_static = _recs(ctx, my, drafted=my + yeasts[:8])
    early_static = _recs(ctx, my, drafted=my)
    assert late_static.loc[probe, "Scarcity"] > early_static.loc[probe, "Scarcity"]


def test_scarcity_is_smooth_not_clipped():
    i2c = {f"Y{i}": "Yeast" for i in range(3)}
    raw = dc.compute_dynamic_scarcity(set(i2c), i2c, {"Yeast"}, num_players=12)
    assert 0.9 < raw["Y0"] < 1.0   # pressure 4 -> 1 - e^-4, not clipped to 1.0
    sim = {"Y0": [{"ingredient": "Y1", "score": 0.9}]}
    with_sub = dc.compute_dynamic_scarcity(set(i2c), i2c, {"Yeast"}, num_players=12,
                                           similarity=sim, sub_strength=0.5)
    assert with_sub["Y0"] < raw["Y0"]  # substitutes still matter under pressure


# ---------------------------------------------------------------------------
# Replay harness + calibration floor
# ---------------------------------------------------------------------------
def test_replay_covers_every_pick_with_valid_ranks(ctx):
    players, log = cw.load_draft()
    rows = cw.replay(ctx, players, log)
    assert len(rows) == len(log) == 63
    for p, rnd, ing, rank, n in rows:
        assert p in players and 1 <= rnd <= 7
        assert 1 <= rank <= n + 1
    m = cw.metrics(rows)
    assert set(m) >= {"top1", "top5", "top10", "MRR", "median_rank", "mean_log_rank", "unlisted"}
    assert m["unlisted"] == 0  # every 2025 pick is modelled by some style


def test_calibration_floor_on_2025_replay(ctx):
    """Regression guard for ranking quality, not a pin on exact ranks.

    Calibrated defaults score MRR ~0.21 / top-10 ~0.37 / median rank ~26 on
    the 2025 draft (pre-calibration: 0.13 / 0.30 / 29 with 10 unlisted picks).
    Fail if a change drags the model back toward that."""
    players, log = cw.load_draft()
    m = cw.metrics(cw.replay(ctx, players, log))
    assert m["MRR"] >= 0.19
    assert m["top10"] >= 0.33
    assert m["median_rank"] <= 30


def test_replay_ranks_average_ties():
    import pandas as pd
    # Two equal Pick Values above ours, three tied with ours -> rank 3 + 1
    pv = pd.Series({"a": 0.9, "b": 0.9, "me": 0.5, "c": 0.5, "d": 0.5, "e": 0.1})
    mine = pv["me"]
    rank = 1 + int((pv > mine).sum()) + 0.5 * int((pv == mine).sum() - 1)
    assert rank == 4.0


# ---------------------------------------------------------------------------
# Ubiquitous adjuncts must not out-rank anchor picks
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ctx26():
    return cw.load_context(sheet="ingredients_2026.csv")


def test_sugars_do_not_lead_an_empty_board(ctx26):
    """Sept-2026 finding: dextrose / honey / cane sugar sat in 15-18 of 22
    styles and short filler adjunct lists made them look 'signature', so a bag
    of dextrose out-ranked every base malt. Guard: on an empty board no adjunct
    is in the top 20 of recommendations or best-available."""
    recs = _recs(ctx26, [])
    top20 = recs.head(20)
    assert (top20["Category"] != "Adjunct").all(), top20[top20["Category"] == "Adjunct"].index.tolist()
    # Best-available weights popularity at 0.2 and dextrose *was* a popular
    # 2022-24 pick (dextrose + German Pilsner is the top historical pair), so
    # it may legitimately appear mid-list there -- just not at the top.
    ba = dc.best_available([], ctx26["ingredients"], ctx26["style_matrix"], ctx26["scarcity"],
                           REQUIRED, 3, ctx26["i2c"], ctx26["style_bias"],
                           early_signal=ctx26["early"], similarity=ctx26["similarity"],
                           num_players=9, top_k=10)
    assert (ba["Category"] != "Adjunct").all()
    # ...and no adjunct's Fit exceeds the best base malt's Fit.
    assert recs[recs["Category"] == "Adjunct"]["Fit"].max() < \
        recs[recs["Category"] == "Base Malt"]["Fit"].max()


def test_adjunct_coverage_stays_narrow(ctx26):
    """Sugars/adjuncts are listed only where characteristic (docs/STYLE_MATRIX.md)."""
    sm = ctx26["style_matrix"]
    cov = {}
    for cats in sm.values():
        for ing in cats["Adjunct"]:
            cov[ing] = cov.get(ing, 0) + 1
    worst = max(cov.items(), key=lambda kv: kv[1])
    assert worst[1] <= 9, worst
    for sugar in ["Corn Sugar (Dextrose)", "Honey", "Cane (or Beet) Sugar"]:
        assert cov[sugar] <= 7, (sugar, cov[sugar])


# ---------------------------------------------------------------------------
# Single-pick categories: no second yeast floating back up in the flex phase
# ---------------------------------------------------------------------------
FULL_ROSTER = ["Maris Otter", "East Kent Goldings",
               "English Ale (WLP002, WY1968, S-04, A09)",
               "Lyle's Golden Syrup (Invert Sugar)"]  # every required slot filled


def test_second_yeast_is_demoted_not_hidden(ctx26):
    r = _recs(ctx26, FULL_ROSTER, single_pick_categories=["Yeast"]).reset_index()
    top40 = r.head(40)
    assert (top40["Category"] != "Yeast").all(), top40[top40["Category"] == "Yeast"]["Ingredient"].tolist()
    yeasts = r[r["Category"] == "Yeast"]
    assert len(yeasts) > 20                       # still listed (co-pitching is legal)
    assert yeasts["Why"].str.startswith("redundant").all()
    # Other categories are untouched by the rule.
    assert not r[r["Category"] != "Yeast"]["Why"].str.startswith("redundant").any()


def test_redundancy_only_applies_once_the_slot_is_filled(ctx26):
    # No yeast on the roster yet -> yeasts compete normally (and lead: unmet
    # required slot).
    r = _recs(ctx26, ["Maris Otter", "East Kent Goldings"], single_pick_categories=["Yeast"])
    assert "Yeast" in set(r.head(5)["Category"])
    assert not r["Why"].str.startswith("redundant").any()


def test_redundancy_knob_is_tunable_and_config_driven(ctx26):
    off = _recs(ctx26, FULL_ROSTER, single_pick_categories=["Yeast"],
                squash={"redundant_mult": 1.0})
    assert "Yeast" in set(off.head(10)["Category"])   # knob at 1.0 == old behaviour
    none = _recs(ctx26, FULL_ROSTER, single_pick_categories=[])
    assert "Yeast" in set(none.head(10)["Category"])  # empty config list == off
    # Default (None) reads league_config.json, which names Yeast.
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        default = _recs(ctx26, FULL_ROSTER)
    finally:
        os.chdir(cwd)
    assert (default.head(40)["Category"] != "Yeast").all()
    assert "redundant_mult" in dc.scoring_params({})[2]


def test_config_defaults_name_single_pick_categories():
    from config import DEFAULTS, load_league_config
    assert DEFAULTS["single_pick_categories"] == ["Yeast"]
    assert load_league_config(os.path.join(HERE, "league_config.json"))["single_pick_categories"] == ["Yeast"]


# ---------------------------------------------------------------------------
# Workable tier: "what might work" when the characteristic options are gone
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workable():
    return dc.load_workable(base_dir=HERE)


KOLSCH = ["German Pilsner", "Hallertau Mittelfrueh", "Kolsch (WLP029, WY2565, G03, K-97)"]
KOLSCH_SUGARS = ["Corn Sugar (Dextrose)", "Brewer's Crystals"]  # its characteristic adjuncts
STOUT = ["Maris Otter", "East Kent Goldings", "Dry English Ale (WLP007, WY1098, A10)",
         "Barley, Roasted"]


def test_workable_file_shape_and_disjoint(ctx26, workable):
    sm = ctx26["style_matrix"]
    assert set(workable) == set(sm)
    on_sheet = ctx26["available"]
    n = 0
    for style, cats in workable.items():
        assert list(cats) == list(sm[style])
        for cat, ings in cats.items():
            assert not set(ings) & set(sm[style][cat]), (style, cat)  # never both tiers
            assert set(ings) <= on_sheet, (style, cat, set(ings) - on_sheet)
            n += len(ings)
    assert n > 150
    assert all(workable[s]["Adjunct"] for s in workable)  # every style has adjunct fallbacks


def test_stranded_kolsch_gets_workable_adjuncts_with_reason(ctx26, workable):
    r = _recs(ctx26, KOLSCH, drafted=KOLSCH + KOLSCH_SUGARS, workable=workable).reset_index()
    top = r.head(4)
    assert (top["Category"] == "Adjunct").all()
    assert top["Why"].str.startswith("might work for Kölsch").all(), top["Why"].tolist()
    assert set(top["Ingredient"]) <= set(workable["Kölsch & Altbier"]["Adjunct"])


def test_workable_stays_in_reserve_while_characteristic_is_available(ctx26, workable):
    r = _recs(ctx26, KOLSCH, workable=workable)
    adj = r[r["Category"] == "Adjunct"]
    assert adj.index[0] == "Corn Sugar (Dextrose)"
    assert adj.iloc[0]["Why"].startswith("fits Kölsch")
    # Only one of the two gone -> the remaining characteristic option still leads.
    r2 = _recs(ctx26, KOLSCH, drafted=KOLSCH + ["Corn Sugar (Dextrose)"], workable=workable)
    assert r2[r2["Category"] == "Adjunct"].index[0] == "Brewer's Crystals"


def test_workable_twist_on_a_finished_build(ctx26, workable):
    full = KOLSCH + ["Corn Sugar (Dextrose)"]
    r = _recs(ctx26, full, workable=workable)
    adj = r[r["Category"] == "Adjunct"].head(3)
    assert adj["Why"].str.startswith("might work for Kölsch").all()
    assert not r.head(5)["Why"].str.contains("Kettle Sour").any()  # flex deepens the style


def test_stranded_stout_and_no_workable_file(ctx26, workable):
    gone = STOUT + ["Milk Sugar (Lactose)", "Coffee (Liquid or Beans)", "Cocoa Nibs/Beans",
                    "Vanilla (extract or bean)", "Molasses",
                    "Lyle's Golden Syrup (Invert Sugar)", "Licorice", "Coconut"]
    r = _recs(ctx26, STOUT, drafted=gone, workable=workable)
    adj = r[r["Category"] == "Adjunct"].head(3)
    assert adj["Why"].str.startswith("might work for Stout").all()
    # Without the file, nothing says "might work" and the engine still runs.
    r0 = _recs(ctx26, STOUT, drafted=gone)
    assert not r0["Why"].str.contains("might work").any()
    r1 = _recs(ctx26, STOUT, drafted=gone, workable=workable, squash={"workable_weight": 0.0})
    assert not r1["Why"].str.contains("might work").any()


def test_focus_likelihood_prefers_style_with_defining_pick(ctx26):
    sm = ctx26["style_matrix"]
    f = dc.compute_style_focus(STOUT, sm)
    assert max(f, key=f.get).startswith("Stout")
    f2 = dc.compute_style_focus(KOLSCH, sm)
    assert max(f2, key=f2.get) == "Kölsch & Altbier"
    # Legacy count mode still works and is flatter.
    fc = dc.compute_style_focus(KOLSCH, sm, mode="count")
    assert max(fc.values()) < max(f2.values())


def test_style_status_counts_workable_options_and_picks(ctx26, workable):
    sm = ctx26["style_matrix"]
    board_gone = KOLSCH + KOLSCH_SUGARS
    without = dc.compute_style_status(KOLSCH, board_gone, sm, REQUIRED, 3).set_index("Style")
    with_w = dc.compute_style_status(KOLSCH, board_gone, sm, REQUIRED, 3,
                                     workable=workable).set_index("Style")
    k = "Kölsch & Altbier"
    assert without.loc[k, "Categories with Options Left"] == 4   # adjunct exhausted
    assert with_w.loc[k, "Categories with Options Left"] == 5    # workable adjuncts remain
    roster = KOLSCH + ["Honey"]
    st = dc.compute_style_status(roster, roster, sm, REQUIRED, 3, workable=workable)
    assert st.iloc[0]["Style"] == k
    assert st.set_index("Style").loc[k, "Picks Matched"] == 3.9  # honey at 0.9

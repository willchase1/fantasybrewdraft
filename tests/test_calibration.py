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

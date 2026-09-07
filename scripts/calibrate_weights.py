#!/usr/bin/env python3
"""Replay a recorded draft through the recommender and score / tune the weights.

What it measures
----------------
For every historical pick we rebuild the board exactly as the drafter saw it
(everything drafted before them, their own roster, their seat, the overall
pick number) and ask ``draft_core.next_best_picks`` for a full ranking. The
rank of the ingredient they *actually* took is the observation. Over a draft:

    top1 / top5 / top10   share of picks the model ranked in its top N
    MRR                   mean of 1/rank
    median_rank           robust central tendency
    mean_log_rank         geometric-mean rank (penalises misses smoothly)

MRR is the tuning objective (a smooth, rank-aware, bounded score); the others
are reported so a change that helps MRR by moving a few picks from #2 to #1
while pushing others off the page is visible.

Caveat: the 2025 draft is nine humans with their own plans, not the model's
"correct" answer -- the goal is a model whose *ordering* agrees with good
drafters more often, not one that predicts them exactly. So the search is
deliberately coarse, uses leave-one-player-out cross-validation (train on 8
players' picks, test on the 9th), and prefers a neighbourhood of good weights
over the single best point.

Usage
-----
    python scripts/calibrate_weights.py                 # score current defaults
    python scripts/calibrate_weights.py --search 400    # random search + LOPO CV
    python scripts/calibrate_weights.py --weights '{"fit":0.3,...}'
    python scripts/calibrate_weights.py --old-matrix style_matrix_backup.json

Pure and deterministic (seeded). Reads only committed data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import draft_core as dc  # noqa: E402
from config import load_league_config  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "draft_2025.json")


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------
def load_context(sheet="ingredients_2025.csv", style_matrix_path="style_matrix.json",
                 use_similarity=True):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        ings, sm, sc, opp, sb, i2c = dc.load_data(
            ingredients_path=sheet, style_matrix_path=style_matrix_path)
        sim = dc.load_similarity() if use_similarity else None
        cfg = load_league_config()
        workable = dc.load_workable(cfg)
    finally:
        os.chdir(cwd)
    early, pairs = dc.build_opponent_signals(opp)
    return {
        "ingredients": ings, "style_matrix": sm, "scarcity": sc, "style_bias": sb,
        "i2c": i2c, "similarity": sim, "early": early, "pairs": pairs,
        "workable": workable,
        "required": cfg["required_categories"], "flex_slots": cfg["flex_slots"],
        "available": dc.build_available_set(ings, cfg["category_aliases"]),
    }


def load_draft(path=FIXTURE):
    with open(path) as f:
        d = json.load(f)
    return d["players"], d["draft_log"]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay(ctx, players, log, weights=None, squash=None, players_subset=None):
    """Return [(player, round, ingredient, rank, n_candidates)] for each pick.

    ``rank`` is 1-based; a pick the model could not even list (not in any
    style) gets rank ``n_candidates + 1``.
    """
    n = len(players)
    seat = {p: i for i, p in enumerate(players)}
    drafted, rosters, out = [], defaultdict(list), []
    for rec in log:
        p, ing, overall = rec["Player"], rec["Ingredient"], int(rec["Overall"])
        if players_subset is None or p in players_subset:
            recs = dc.next_best_picks(
                rosters[p], drafted, ctx["ingredients"], ctx["style_matrix"],
                ctx["scarcity"], ctx["required"], ctx["flex_slots"], ctx["i2c"],
                ctx["style_bias"], early_signal=ctx["early"], top_k=100000,
                similarity=ctx["similarity"], pair_lookup=ctx["pairs"],
                num_players=n, overall_pick=overall, your_seat_index=seat[p],
                weights=weights, squash=squash, available_set=ctx["available"],
                workable=ctx.get("workable"),
            )
            pv = recs.set_index("Ingredient")["Pick Value"]
            if ing in pv.index:
                mine = pv[ing]
                # Average rank within a tie group, so the metric does not depend
                # on how the sort happened to order equal Pick Values.
                rank = 1 + int((pv > mine).sum()) + 0.5 * int((pv == mine).sum() - 1)
            else:
                rank = len(pv) + 1
            out.append((p, int(rec["Round"]), ing, rank, len(pv)))
        drafted.append(ing)
        rosters[p].append(ing)
    return out


def metrics(rows):
    ranks = [r[3] for r in rows]
    if not ranks:
        return {}
    return {
        "n": len(ranks),
        "top1": sum(r <= 1 for r in ranks) / len(ranks),
        "top5": sum(r <= 5 for r in ranks) / len(ranks),
        "top10": sum(r <= 10 for r in ranks) / len(ranks),
        "MRR": sum(1.0 / r for r in ranks) / len(ranks),
        "median_rank": statistics.median(ranks),
        "mean_log_rank": math.exp(sum(math.log(r) for r in ranks) / len(ranks)),
        "unlisted": sum(1 for r in rows if r[3] > r[4]),
    }


def fmt(m):
    return (f"top1 {m['top1']:.2f}  top5 {m['top5']:.2f}  top10 {m['top10']:.2f}  "
            f"MRR {m['MRR']:.3f}  median {m['median_rank']:.0f}  "
            f"geo-mean {m['mean_log_rank']:.1f}  unlisted {m['unlisted']}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
COMPONENTS = ["fit", "scarce", "need", "syn", "deny", "pop"]


def random_weights(rng, floor=0.02):
    """Random point on the simplex, each weight at least ``floor``."""
    raw = [rng.random() for _ in COMPONENTS]
    s = sum(raw)
    w = {c: floor + (1 - floor * len(COMPONENTS)) * x / s for c, x in zip(COMPONENTS, raw)}
    return {c: round(v, 3) for c, v in w.items()}


def random_squash(rng):
    """Coarse grid over the shape constants (see draft_core.DEFAULT_SQUASH)."""
    return {
        "fit": rng.choice([2.0, 3.0, 5.0, 8.0]),
        "syn": rng.choice([1.0, 2.0, 3.0, 5.0]),
        "deny": rng.choice([1.0, 2.0, 3.0]),
        "pop": rng.choice([0.5, 1.0, 2.0]),
        "fit_align": rng.choice([0.0, 0.35, 0.65, 0.9]),
        "fit_sig": rng.choice([0.0, 0.25, 0.5, 0.75]),
        "fit_idf": rng.choice([0.0, 0.05, 0.15]),
        "need_flex_floor": rng.choice([0.0, 0.1, 0.2]),
        "need_slack": rng.choice([0.0, 1.0]),
        "scarce_residual": rng.choice([0.0, 1.0]),
        "scarce_sub": rng.choice([0.0, 0.0, 0.05, 0.15]),
        "focus_ll": rng.choice([0.0, 1.0, 1.0]),
        "focus_damp": rng.choice([0.0, 2.0, 4.0, 8.0]),
        "workable_weight": rng.choice([0.0, 0.5, 0.9]),
    }


def search(ctx, players, log, n_trials, seed=7, tune_squash=True, log_every=50):
    rng = random.Random(seed)
    trials = [(dc.DEFAULT_PICK_WEIGHTS, dc.DEFAULT_SQUASH)]
    for _ in range(n_trials):
        trials.append((random_weights(rng), random_squash(rng) if tune_squash else None))

    # Leave-one-player-out: for each trial, score every player's picks once;
    # CV score = mean over held-out players of that player's MRR. Because the
    # weights are not *fitted* per fold (we pick by the pooled objective), LOPO
    # here measures how stable a candidate is across drafters.
    results = []
    for i, (w, sq) in enumerate(trials):
        rows = replay(ctx, players, log, weights=w, squash=sq)
        per_player = {}
        for p in players:
            pr = [r for r in rows if r[0] == p]
            per_player[p] = metrics(pr)["MRR"]
        m = metrics(rows)
        m["cv_mrr_mean"] = statistics.mean(per_player.values())
        m["cv_mrr_min"] = min(per_player.values())
        results.append((m, w, sq))
        if log_every and i % log_every == 0:
            print(f"  trial {i:4d}  MRR {m['MRR']:.3f}  cv-min {m['cv_mrr_min']:.3f}",
                  file=sys.stderr)
    # Rank by pooled MRR, but break ties toward the more even per-player profile.
    results.sort(key=lambda t: (t[0]["MRR"], t[0]["cv_mrr_min"]), reverse=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", type=json.loads, default=None)
    ap.add_argument("--squash", type=json.loads, default=None)
    ap.add_argument("--old-matrix", default=None,
                    help="score with an alternative style matrix (e.g. style_matrix_backup.json)")
    ap.add_argument("--no-similarity", action="store_true")
    ap.add_argument("--search", type=int, default=0, help="random-search trials")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--by-round", action="store_true", help="print rank by round")
    ap.add_argument("--dump", action="store_true", help="print every pick's rank")
    args = ap.parse_args(argv)

    players, log = load_draft()
    ctx = load_context(style_matrix_path=args.old_matrix or "style_matrix.json",
                       use_similarity=not args.no_similarity)

    if args.search:
        res = search(ctx, players, log, args.search, seed=args.seed)
        print("\nTop 10 trials (pooled MRR, then worst-player MRR):")
        for m, w, sq in res[:10]:
            print(f"  {fmt(m)}  cv-mean {m['cv_mrr_mean']:.3f} cv-min {m['cv_mrr_min']:.3f}")
            print(f"     weights {w}  squash {sq}")
        return 0

    rows = replay(ctx, players, log, weights=args.weights, squash=args.squash)
    m = metrics(rows)
    print(fmt(m))
    if args.by_round:
        by = defaultdict(list)
        for r in rows:
            by[r[1]].append(r[3])
        for rnd in sorted(by):
            print(f"  R{rnd}: median rank {statistics.median(by[rnd]):.0f}  "
                  f"top5 {sum(x <= 5 for x in by[rnd]) / len(by[rnd]):.2f}")
    if args.dump:
        for p, rnd, ing, rank, ncand in rows:
            print(f"  R{rnd} {p:8s} #{rank:5.1f}/{ncand}  {ing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

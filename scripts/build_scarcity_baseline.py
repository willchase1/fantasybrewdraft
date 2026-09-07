#!/usr/bin/env python3
"""Rebuild ``ingredient_scarcity.json`` -- the *informational* season snapshot.

Status of this file
-------------------
The recommendation engine no longer reads it. ``draft_core.next_best_picks``
computes scarcity live from the board (``compute_dynamic_scarcity``) and the
style idf at call time (``compute_style_idf`` -- 22 styles x ~30 names, a few
microseconds, so no idf file is needed). ``draft_core.load_data`` treats this
file as optional.

The historical file was ``Scarcity Score = 1 / Style Coverage`` -- circular and
not a scarcity. This generator replaces it with a reproducible pre-draft
snapshot from the sheet, style matrix, league config and similarity files:

    Ingredient          sheet name
    Category            style-matrix category
    Style Coverage      styles that use it
    Signature           max over its styles of 8/(8+|category list|) -- how
                        defining it is for its best style (see draft_core Fit)
    Close Substitutes   analogs with similarity >= SIMILARITY_CLOSE_THRESHOLD
    Baseline Scarcity   compute_dynamic_scarcity at a full board (nothing
                        drafted), i.e. category demand / supply, 1 - e^-p;
                        substitute discount at the engine's default strength

Handy for pre-draft prep ("what is contested from pick 1") and for the
print sheet; nothing in the live engine depends on it.

Usage
-----
    python scripts/build_scarcity_baseline.py           # write ingredient_scarcity.json
    python scripts/build_scarcity_baseline.py --check   # exit 1 if committed file differs

Deterministic: rows sorted by (Category, Ingredient); values rounded to 4 dp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CATEGORY_ORDER = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty"]


def build_rows(ingredients_df, style_matrix, config, similarity=None, num_players=None,
               squash=None):
    """Pure: rows for the snapshot from in-memory inputs."""
    import draft_core as dc

    squash = {**dc.DEFAULT_SQUASH, **(squash or {})}
    available = dc.build_available_set(ingredients_df, config["category_aliases"])
    n_players = num_players or config.get("num_players", 9)

    cat_of, coverage, best_sig = {}, {}, {}
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing not in available:
                    continue
                cat_of[ing] = cat
                coverage[ing] = coverage.get(ing, 0) + 1
                sig = 8.0 / (8.0 + len(ings))
                best_sig[ing] = max(best_sig.get(ing, 0.0), sig)

    modeled = set(cat_of)
    scarcity = dc.compute_dynamic_scarcity(
        modeled, cat_of, set(config["required_categories"]), n_players,
        similarity=similarity, sub_strength=squash["scarce_sub"],
    )
    rows = []
    for ing in sorted(modeled, key=lambda i: (CATEGORY_ORDER.index(cat_of[i])
                                              if cat_of[i] in CATEGORY_ORDER else 99, i)):
        close = 0
        if similarity and ing in similarity:
            close = sum(1 for r in similarity[ing]
                        if dc._sim_name(r) in modeled
                        and r.get("score", 0) >= dc.SIMILARITY_CLOSE_THRESHOLD)
        rows.append({
            "Ingredient": ing,
            "Category": cat_of[ing],
            "Style Coverage": coverage[ing],
            "Signature": round(best_sig[ing], 4),
            "Close Substitutes": close,
            "Baseline Scarcity": round(scarcity[ing], 4),
        })
    return rows


def render(rows) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "ingredient_scarcity.json"))
    ap.add_argument("--players", type=int, default=9,
                    help="table size for the demand side (default 9, the 2025 draft)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    import draft_core as dc
    from config import load_league_config

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        cfg = load_league_config()
        ings, sm, *_ = dc.load_data(ingredients_path=cfg["ingredients_path"])
        sim = dc.load_similarity(cfg)
        _, _, squash = dc.scoring_params(cfg)
    finally:
        os.chdir(cwd)

    text = render(build_rows(ings, sm, cfg, sim, num_players=args.players, squash=squash))
    if args.check:
        if not os.path.exists(args.out):
            print(f"ERROR: {args.out} missing")
            return 1
        with open(args.out, encoding="utf-8") as f:
            if f.read() != text:
                print(f"ERROR: {args.out} differs from generator output")
                return 1
        print("OK: ingredient_scarcity.json is up to date")
        return 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

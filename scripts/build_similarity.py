#!/usr/bin/env python3
"""Build ingredient similarity files from curated descriptor tables.

Method (documented so it can be argued with)
--------------------------------------------
Each ingredient is a row in ``data/<kind>_descriptors.csv``: a handful of
0-3 intensity descriptors (aroma / flavor axes), a few categorical facts
(origin, purpose, family, ...), and a few continuous facts (alpha acid, color,
attenuation, ...). The generator turns each row into a non-negative vector:

* descriptor columns   -> used as-is (0..3)
* categorical columns  -> one-hot, scaled by a per-column weight (a small
                          nudge toward "same origin / same purpose", never
                          enough to make two hops similar on its own)
* continuous columns   -> rescaled to 0..3 by a per-column transform
                          (log for malt color, since 10L vs 20L matters and
                          400L vs 500L does not)

Raw similarity = cosine of the two vectors, in [0,1] because every component
is non-negative. Raw cosines of non-negative vectors have a high floor (two
unrelated yeasts still share "clean", "attenuation" and "temperature" mass),
and the floor differs per table, so the raw value is **re-based on the median
pairwise cosine of that table**:

    score = max(0, (cos - median) / (1 - median))

so 0 means "no more alike than a typical pair on this sheet", 1.0 means
indistinguishable on these axes (e.g. two sheet rows that are literally the
same strain), and the engine's "close analog" threshold (0.3) means the same
thing for hops, yeasts and malts. The median used is printed on each run.

For each ingredient we emit the top ``--k`` neighbours with score >= ``--min``
(shape ``{name: [{"ingredient": other, "score": 0.xx}, ...]}``, sorted by score
then name) plus the dense pairwise matrix as CSV for tooling.

Deterministic: same CSV in -> byte-identical JSON/CSV out.

Usage
-----
    python scripts/build_similarity.py            # all kinds -> repo root
    python scripts/build_similarity.py hop yeast  # a subset
    python scripts/build_similarity.py --check    # exit 1 if committed files differ

Adding an ingredient: add a row to the descriptor CSV (name must match the
ingredient sheet exactly; ``--check`` also verifies full-sheet coverage), re-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)


def _log_color(v: float) -> float:
    # 1.5L -> 0.2, 10L -> 1.1, 40L -> 1.7, 120L -> 2.2, 500L -> 2.9
    return 3.0 * math.log10(float(v) + 1.0) / math.log10(600.0)


def _lin(lo: float, hi: float):
    def f(v: float) -> float:
        return 3.0 * min(1.0, max(0.0, (float(v) - lo) / (hi - lo)))
    return f


# Per-kind spec: which sheet columns it covers, which CSV columns are what.
SPECS = {
    "hop": {
        "csv": "hop_descriptors.csv",
        "sheet_columns": ["Hop"],
        "categorical": {"origin": 0.6, "purpose": 1.0},
        "continuous": {"alpha_mid": (_lin(2.0, 17.0), 1.0)},
        "ignore": ["alpha_lo", "alpha_hi"],  # display-only (ingredient_profile)
        "out": "hop_similarity",
    },
    "yeast": {
        "csv": "yeast_descriptors.csv",
        "sheet_columns": ["Yeast"],
        "categorical": {"family": 1.5, "ferment_type": 1.0},
        "continuous": {
            "attenuation_mid": (_lin(65.0, 82.0), 1.0),
            "temp_mid_f": (_lin(48.0, 98.0), 1.5),
            "flocculation": (_lin(1.0, 3.0), 0.7),
        },
        "ignore": ["temp_lo_f", "temp_hi_f"],  # display-only (ingredient_profile)
        "out": "yeast_similarity",
    },
    "malt": {
        "csv": "malt_descriptors.csv",
        "sheet_columns": ["Base Malt", "Specialty Malt"],
        "categorical": {"kind": 1.5, "grain": 1.2, "origin": 0.4},
        "continuous": {
            "color_L": (_log_color, 2.0),
            "diastatic": (_lin(0.0, 1.0), 1.0),
        },
        "ignore": ["sheet_category"],
        "out": "malt_similarity",
    },
}


def load_table(path: str) -> list[OrderedDict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [OrderedDict(r) for r in csv.DictReader(f)]
    names = [r["name"] for r in rows]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit(f"{path}: duplicate names {sorted(dupes)}")
    return rows


def vectorize(rows, spec):
    """Return (names, vectors) with a shared, deterministic dimension order."""
    cat_cols = spec["categorical"]
    cont_cols = spec["continuous"]
    skip = set(cat_cols) | set(cont_cols) | set(spec["ignore"]) | {"name"}
    desc_cols = [c for c in rows[0].keys() if c not in skip]
    cat_levels = {c: sorted({r[c] for r in rows}) for c in cat_cols}

    names, vecs = [], []
    for r in rows:
        v = [float(r[c]) for c in desc_cols]
        for c, w in cat_cols.items():
            v.extend(w if r[c] == lvl else 0.0 for lvl in cat_levels[c])
        for c, (fn, w) in cont_cols.items():
            v.append(w * fn(r[c]))
        names.append(r["name"])
        vecs.append(v)
    return names, vecs


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def similarity_matrix(names, vecs):
    n = len(names)
    m = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine(vecs[i], vecs[j])
            m[i][j] = m[j][i] = s
    return m


def rebase(m):
    """Affine re-base so the median off-diagonal cosine maps to 0 (see module doc)."""
    vals = sorted(m[i][j] for i in range(len(m)) for j in range(i + 1, len(m)))
    if not vals:
        return m, 0.0
    med = vals[len(vals) // 2]
    if med >= 1.0:
        return m, med
    out = [[max(0.0, min(1.0, (x - med) / (1.0 - med))) for x in row] for row in m]
    for i in range(len(out)):
        out[i][i] = 1.0
    return out, med


def neighbours(names, m, k: int, min_score: float) -> dict:
    out = {}
    for i, a in enumerate(names):
        cands = [(names[j], round(m[i][j], 3)) for j in range(len(names)) if j != i]
        cands = [(b, s) for b, s in cands if s >= min_score]
        cands.sort(key=lambda t: (-t[1], t[0]))
        out[a] = [{"ingredient": b, "score": s} for b, s in cands[:k]]
    return out


def render_json(d: dict) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False) + "\n"


def render_matrix_csv(names, m) -> str:
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([""] + names)
    for a, row in zip(names, m):
        w.writerow([a] + [f"{x:.3f}" for x in row])
    return buf.getvalue()


def sheet_names(spec, sheet_path: str) -> set:
    with open(sheet_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = set()
    for col in spec["sheet_columns"]:
        out |= {r[col].strip() for r in rows if r.get(col) and r[col].strip()}
    return out


def build(kind: str, k: int, min_score: float, sheet_path: str | None):
    spec = SPECS[kind]
    rows = load_table(os.path.join(DATA, spec["csv"]))
    problems = []
    if sheet_path:
        on_sheet = sheet_names(spec, sheet_path)
        have = {r["name"] for r in rows}
        for n in sorted(on_sheet - have):
            problems.append(f"{kind}: on sheet but no descriptor row: {n}")
        for n in sorted(have - on_sheet):
            problems.append(f"{kind}: descriptor row not on sheet: {n}")
    names, vecs = vectorize(rows, spec)
    m, median = rebase(similarity_matrix(names, vecs))
    return {
        "json": render_json(neighbours(names, m, k, min_score)),
        "csv": render_matrix_csv(names, m),
        "problems": problems,
        "n": len(names),
        "median": median,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kinds", nargs="*", default=[],
                    help=f"which tables to build: {', '.join(SPECS)} (default all)")
    ap.add_argument("--k", type=int, default=8, help="neighbours per ingredient")
    ap.add_argument("--min", type=float, default=0.15, help="minimum neighbour score")
    ap.add_argument("--sheet", default=None,
                    help="ingredient sheet to validate coverage against "
                         "(default: league_config ingredients_path)")
    ap.add_argument("--out-dir", default=ROOT)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if outputs differ or coverage fails")
    args = ap.parse_args(argv)
    kinds = args.kinds or list(SPECS)
    bad = [k for k in kinds if k not in SPECS]
    if bad:
        ap.error(f"unknown kind(s) {bad}; choose from {list(SPECS)}")

    if args.sheet is None:
        from config import load_league_config
        args.sheet = os.path.join(ROOT, load_league_config()["ingredients_path"])

    rc = 0
    for kind in kinds:
        res = build(kind, args.k, args.min, args.sheet)
        for p in res["problems"]:
            print("WARN:", p)
            rc = 1 if args.check else rc
        base = os.path.join(args.out_dir, SPECS[kind]["out"])
        targets = {base + ".json": res["json"], base + "_matrix.csv": res["csv"]}
        for path, text in targets.items():
            if args.check:
                if not os.path.exists(path):
                    print(f"ERROR: {path} missing")
                    rc = 1
                    continue
                with open(path, encoding="utf-8") as f:
                    if f.read() != text:
                        print(f"ERROR: {path} differs from generator output")
                        rc = 1
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
        print(f"{kind}: {res['n']} ingredients, median raw cosine {res['median']:.3f} "
              f"-> {os.path.basename(base)}.json"
              + (" (checked)" if args.check else " (written)"))
    if args.check and rc == 0:
        print("OK: similarity files are up to date")
    return rc


if __name__ == "__main__":
    sys.exit(main())

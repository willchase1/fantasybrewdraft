# Scoring calibration against the 2025 draft

`scripts/calibrate_weights.py` replays the recorded 2025 draft
(`tests/fixtures/draft_2025.json`, 9 players × 7 rounds = 63 picks) through
`draft_core.next_best_picks`. For each pick the board is rebuilt exactly as the
drafter saw it — everything drafted before them, their roster, their seat and
overall pick number — and the model produces a full ranking of the ~200
candidates. The rank of the ingredient they actually took is the observation.

    python scripts/calibrate_weights.py               # score current defaults
    python scripts/calibrate_weights.py --by-round    # + median rank per round
    python scripts/calibrate_weights.py --dump        # every pick's rank
    python scripts/calibrate_weights.py --search 300  # random search + LOPO stability

Metrics: **top-1 / top-5 / top-10** hit rate, **MRR** (mean 1/rank — the tuning
objective), **median rank**, geometric-mean rank. Ties in Pick Value are scored
as the average rank of the tie group so the metric does not depend on sort
order. The opponent model (`opponent_model.json`) covers 2022–2024 only, so
the 2025 draft is a clean holdout.

## Results

| configuration | top-1 | top-5 | top-10 | MRR | median rank | unlisted |
|---|---|---|---|---|---|---|
| **Before** — 2025 matrix, old hop file, old engine, old weights | 0.05 | 0.21 | 0.30 | 0.133 | 29 | 10 |
| New matrix + similarity, old engine, old weights | 0.10 | 0.24 | 0.37 | 0.191 | 27 | 0 |
| New matrix + similarity, new engine, calibrated defaults | 0.13 | 0.30 | 0.37 | 0.213 | 26 | 0 |
| **After** — + Sep-2026 sheet additions, narrow adjunct lists, adjunct-neutral signature | 0.13 | 0.30 | 0.38 | 0.220 | 26 | 0 |
| **After** — + single-pick redundancy demotion (second yeast) | 0.13 | 0.30 | 0.38 | 0.223 | 22 | 0 |
| (rejected) coordinate-ascent optimum, in-sample | 0.17 | 0.32 | 0.40 | 0.248 | 23 | 0 |

"Unlisted" = picks the model could not even rank because no style contained
them (the 64-ingredient gap fixed in `docs/STYLE_MATRIX.md`). By round, the
calibrated model has median rank **8 in R1 and 6 in R2** (the anchor picks),
then 25–46 in R3–R7 where drafters choose among ~50 specialty malts and ~45
adjuncts for a style — that part is idiosyncratic and no weight setting helps
much. 13 of 63 picks rank in the top 3 (German Pilsner, English Ale, Golden
Promise, Fuggle, EKG, Saaz, Tettnang, Munich Dark, US-05, Molasses, Coconut…).

### Why the "optimum" was rejected

Coordinate ascent over weights and shape constants reaches MRR ~0.24–0.25, but on
three train/test splits by player it beat the untuned defaults on only two and
lost on one. With 63 observations, differences under ~0.02 MRR are noise
(one pick moving from rank 2 to rank 1 is +0.008). We therefore kept only
changes that recur across every split or are structurally principled and
neutral, and left the six weights at their original values — the search never
moved them by more than 0.05 anyway:

    DEFAULT_PICK_WEIGHTS = {fit 0.30, scarce 0.25, need 0.20, syn 0.15, deny 0.07, pop 0.03}

The random search also kept surfacing high `deny` weights (0.2–0.3). That is an
artifact: Denial = historical co-occurrence with opponents' recent picks ×
popularity, i.e. a mainstream-ness prior in disguise. Recommending "block your
opponent" a quarter of the time is not what the club wants from the tool, so
`deny` stays at 0.07.

## What changed in the engine (`draft_core.py`)

Shape constants now live in `DEFAULT_SQUASH` and can be overridden per season
from `league_config.json` (`pick_weights`, `board_value_weights`, `squash`);
`scoring_params()` resolves them and `draft_mode.py` passes them through.

| constant | old | new | evidence |
|---|---|---|---|
| `fit` squash k | 5 | **2** | robust across all splits: 2 styles of coverage already count as versatile; beyond that alignment and signature should decide |
| `syn` squash k | 3 | **5** | robust across all splits: synergy stays nearly linear over the observed pair counts |
| Fit alignment | 0.5 + 0.5·(focus *share*) | `(1-a) + a·(focus / max focus)`, **a = 0.65** | share-based alignment barely separated on-style from off-style (a leading style with 20 % share gave ×0.6 vs ×0.5); normalised alignment gives the leading style ×1.0 |
| Fit signature | — | `(1-g) + g·8/(8+|list|)`, **g = 0.5** | a Belgian yeast in a Dubbel (4 options) is more defining than a base malt (10) — lifts R1 yeasts; neutral-to-positive on MRR |
| Fit idf tie-break | 0.05 | 0.05 | kept; removing it collapsed top-1 from 0.13 to 0.02 through ties |
| Urgency | `need_base·(0.6+0.4·risk)`, flex base 0.2 | `min(1, 0.7·need + 0.3·snipe_risk)`, flex floor **0** | snipe risk now applies to flex picks too |
| Urgency slack | — | option `need_slack` (**off**) | scaling required-slot urgency by spare picks is rational but wrong empirically: drafters fill required, contested slots first (R2 median rank 6 vs 29) |
| Scarcity transform | `min(1, pressure)` | `1 − e^(−pressure)` | smooth; substitutes keep mattering under heavy pressure instead of being clipped away |
| Scarcity demand | static (`num_players`) | option `scarce_residual` (**off**) | residual demand ("who still needs a yeast") is principled but predicts worse (MRR 0.213 → 0.162): the table kept drafting yeast/malt in R2 as if demand had not moved |
| Substitute discount | hops only, 0.15 per close analog | all categories, `scarce_sub` (**0 = off**) | at any strength the discount lowered every metric (0.204 → 0.158 at 0.15): drafters take the mainstream ingredient that *has* many analogs. Hop-blend **synergy** from the similarity files is kept (+0.003 MRR) |
| Fit signature for adjuncts | as other categories | **neutral (0.5)** | Sep 2026: short filler adjunct lists (a Pils lists two sugars) made dextrose look "defining" (Fit 0.74 vs 0.67 for German Pilsner) and it ranked #2 on an empty board. Adjunct lists describe tolerance, not identity; with narrow sugar membership in the matrix as well, the first adjunct now appears at #52 and MRR rose 0.213 → 0.220. Guarded by `test_sugars_do_not_lead_an_empty_board` |

| Redundancy (single-pick categories) | — | Pick Value × `redundant_mult` (**0.25**) for a candidate whose bucket is in `league_config.single_pick_categories` (`["Yeast"]`) and already filled | Flex phase: with every required slot met, `need` is flat and mainstream yeasts floated back to #2/#6 on Fit + popularity. A second yeast is co-pitching — legal but rare — so it is demoted, not hidden (`Why` = "redundant · already have a yeast"; first yeast now #181 of 213). No 2025 roster took a second yeast, so the replay only improves: MRR 0.220 → 0.223, median rank 26 → 22. Guarded by `test_second_yeast_is_demoted_not_hidden` |

`compute_style_status` also gained *Picks Matched* and *Match* tie-breaks (see
`docs/STYLE_MATRIX.md`).

### Single-pick categories (Sep 2026)

`league_config.json` → `"single_pick_categories": ["Yeast"]` (also in
`config.DEFAULTS`). `next_best_picks(..., single_pick_categories=...)` reads it
from config when not passed; `draft_mode.py` passes it explicitly. The strength
lives with the other shape constants (`squash.redundant_mult`, 1.0 = off) so it
is overridable per season and shows up in `scoring_params()`. Generalising to a
per-category "max useful count" (e.g. soft-capping a third base malt) is a
small extension of the same hook if the club ever wants it.

### Sugar over-counting (Sep 2026)

Symptom found in testing: Corn Sugar (Dextrose) #2 and Honey #4 among
recommendations on an empty 2026 board, Dextrose #1 in best-available. Two
causes, two fixes:

1. **Data.** Sugars were listed wherever they *could* be used (cane sugar and
   honey in 18 of 22 styles). The matrix now lists sugars only where they are
   characteristic (see `docs/STYLE_MATRIX.md`, rule 2): dextrose 15 → 7 styles,
   honey 18 → 6, cane 18 → 5, no adjunct above 8. Honey and cane fell to
   ranks 61 and 68 from this alone.
2. **Engine.** Dextrose stayed at #7 because the *signature* term rewards short
   category lists, and the lager styles' two-item filler adjunct lists are the
   shortest lists in the matrix. Signature is now neutral for the Adjunct
   category. Dextrose → #52; it still appears mid-list in best-available (#17)
   through the popularity prior, which is correct — dextrose + German Pilsner is
   the most common 2022–24 pair.

Re-running the coordinate ascent after these changes reproduces the earlier
picture (in-sample 0.248 via `fit_idf 0`, `pop 0.5` — noise-level, not adopted).

## How to re-tune next season

1. Add the new draft as `tests/fixtures/draft_<year>.json` (same shape) and
   point `calibrate_weights.load_draft` at it (or pass both and pool).
2. `python scripts/calibrate_weights.py --search 300` for the landscape;
   then hand-check candidate settings with `--weights`/`--squash` and
   `--by-round`. Prefer changes that hold on player-wise splits.
3. Put the chosen values in `league_config.json` under `pick_weights` /
   `squash` rather than editing `draft_core.py`, and update the floor in
   `tests/test_calibration.py::test_calibration_floor_on_2025_replay` if the
   baseline moves.

The mock-draft personas in `draft_mode.py` (`OPPONENT_PERSONAS`) are weight
overlays on `PICK_WEIGHTS`; the same `--weights` JSON can be A/B'd there.

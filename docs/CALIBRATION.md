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
| **After** — new matrix + similarity, new engine, calibrated defaults | 0.13 | 0.30 | 0.37 | 0.213 | 26 | 0 |
| (rejected) coordinate-ascent optimum, in-sample | 0.16 | 0.32 | 0.40 | 0.241 | 23 | 0 |

"Unlisted" = picks the model could not even rank because no style contained
them (the 64-ingredient gap fixed in `docs/STYLE_MATRIX.md`). By round, the
calibrated model has median rank **10 in R1 and 6 in R2** (the anchor picks),
then 25–46 in R3–R7 where drafters choose among ~50 specialty malts and ~45
adjuncts for a style — that part is idiosyncratic and no weight setting helps
much. 13 of 63 picks rank in the top 3 (German Pilsner, English Ale, Golden
Promise, Fuggle, EKG, Saaz, Tettnang, Munich Dark, US-05, Molasses, Coconut…).

### Why the "optimum" was rejected

Coordinate ascent over weights and shape constants reaches MRR 0.241, but on
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

`compute_style_status` also gained *Picks Matched* and *Match* tie-breaks (see
`docs/STYLE_MATRIX.md`).

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

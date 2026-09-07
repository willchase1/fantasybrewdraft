# Cowork — next data/scoring pass

Two items for the next Cowork (data & scoring) session. Both live in the layer
Cowork owns (`draft_core.py` scoring + the `scripts/` generators + calibration).
The app/UI side is handled separately and needs no changes for either.

Working agreement unchanged: keep `draft_core` pure/unit-tested, config-driven
(`league_config.json`), and `./fantasydraft/bin/python -m pytest -q` green. Any
data edit must keep `validate_data(...)["in_matrix_not_sheet"] == []`.

---

## 1. Single-pick categories (the "second yeast" fix)

**Symptom.** Once a team has filled all required slots (the flex phase), the
recommender floats a **second yeast** back up. Reproduction against the current
engine — a roster of malt + hop + yeast + adjunct (all required met) returns
yeasts at ranks **5 and 10**. A second yeast (co-pitching) is rarely wanted,
unlike a 2nd hop / malt / adjunct which are common flex picks.

**Why it happens.** With all required categories satisfied, the `need`/urgency
term is uniform, so `fit` + `popularity` carry mainstream yeasts back into the
top of the list. Nothing penalizes a *redundant* pick in a single-use category.

**Proposed fix (config-driven).**
- Add to `league_config.json` / `config.DEFAULTS`:
  `"single_pick_categories": ["Yeast"]` — required categories where a second
  pick is almost never useful.
- In `next_best_picks`, apply a strong redundancy penalty to a candidate whose
  rule bucket is in `single_pick_categories` **and** already filled by the
  roster (e.g. multiply its final `Pick Value` by a small factor, or add a large
  negative `redundancy` component). Keep it a named, tunable knob like the other
  scoring params so it shows up in `scoring_params()`.
- Do **not** hard-filter (co-pitching is legal, just rare) — demote so it can
  still appear far down / when explicitly filtered to Yeast.

**Validate.** Re-run `python scripts/calibrate_weights.py` and confirm the fit
metrics hold — second-yeast picks are near-absent in the 2025 fixture, so top-1 /
MRR should be unchanged. Add a unit test: a roster with a yeast ranks no yeast in
the top N (mirrors the reproduction above). Note the change in `docs/CALIBRATION.md`.

Consider generalizing later (per-category "max useful count") if the club ever
wants, e.g., to also soft-cap base malts — but Yeast is the only clear single-use
case today, so ship just that.

---

## 2. New ingredients + reweight

When the updated ingredient sheet lands, re-run the full pipeline and recalibrate:

1. `python scripts/convert_print_sheet.py "<new 20xx print sheet>.csv" ingredients_<year>.csv`
   and point `league_config.json` `ingredients_path` at it (mirror how 2026 was wired).
2. `python scripts/build_style_matrix.py` (then `--check`) — place any genuinely
   new ingredients into the styles where they're *characteristic*; keep ubiquitous
   adjuncts (sugars, honey, lactose) constrained so they don't inflate `fit`
   coverage (this was the fix that dropped Corn Sugar 15→7, Honey 18→6 styles).
3. `python scripts/build_similarity.py` — extend hop/yeast/malt similarity to
   cover the new ingredients (fill any missing rows).
4. `python scripts/build_scarcity_baseline.py` — refresh the informational
   baseline.
5. `python scripts/calibrate_weights.py --search 300` — re-tune only if a change
   holds across the player-wise splits; otherwise leave weights and just refresh
   the data. Update `docs/CALIBRATION.md` and `docs/STYLE_MATRIX.md`.

Gate before hand-back: `validate_data(2026 sheet)["in_matrix_not_sheet"] == []`
and the full test suite green.

# Fantasy Brew Draft — Data & Scoring Modeling Scope (hand-off brief)

**Audience:** a standalone assistant ("Fable") tasked with improving the *data
and scoring* behind the draft tool — **not** the app UI.
**Repo:** `fantasybrewdraft` (Python 3.12 + Streamlit + pandas). Work branch:
`2026-polish`.
**Golden rule:** the recommendation engine lives entirely in
[`draft_core.py`](../draft_core.py) as **pure, unit-tested functions**. Keep it
that way — no Streamlit imports, no heavy ML, all inputs/outputs plain
dicts/DataFrames. Every change must keep `./fantasydraft/bin/python -m pytest -q`
green.

---

## 0. Orientation — how a pick gets scored

`draft_core.next_best_picks(...)` ranks every still-available ingredient as a
**weighted sum of six components, each squashed to [0,1]** (so a weight reads as
"share of the decision"):

```
PickValue = w_fit·Fit + w_scarce·Scarcity + w_need·Urgency
          + w_syn·Synergy + w_deny·Denial + w_pop·Popularity
```

- **Fit** — versatility with diminishing returns (`style_cov/(style_cov+5)`),
  sharpened toward the styles the roster already leans into (`compute_style_focus`),
  with a tiny idf tie-break (`compute_style_idf`).
- **Scarcity** — `compute_dynamic_scarcity`: live board supply vs. demand per
  category, capped at 1.0; hops get a discount when close analogs remain.
- **Urgency** — unmet required slot × snake-draft survival risk
  (`picks_until_next_turn`, `_survival_risk`).
- **Synergy** — co-occurrence with the roster (`opponent_model` typed pair
  arrays) + hop-blend affinity (`hop_similarity`).
- **Denial** — value to an opponent picking before your next turn.
- **Popularity** — historical prior (`opponent_model` early signal).

Current weights (in `draft_core.py`):
```
DEFAULT_PICK_WEIGHTS = {fit:0.30, scarce:0.25, need:0.20, syn:0.15, deny:0.07, pop:0.03}
BOARD_VALUE_WEIGHTS  = {fit:0.50, scarce:0.30, need:0,    syn:0,    deny:0,    pop:0.20}  # "best available"
```
`best_available()` = `next_best_picks` with an empty roster + `BOARD_VALUE_WEIGHTS`
(roster-agnostic board value). `explain_pick()` turns the weighted contributions
into the human "Why" string.

**Config:** rules, categories, and the active ingredient sheet come from
[`league_config.json`](../league_config.json) via [`config.py`](../config.py)
(2026 season → `ingredients_2026.csv`). Don't hardcode paths.

**Data invariant (must always hold):**
`draft_core.validate_data(ingredients, style_matrix)["in_matrix_not_sheet"] == []`
— every ingredient a style references must be draftable from the sheet. Enforced
by `tests/test_data_integrity.py`.

---

## Workstream 1 — Style-matrix coverage

**File:** [`style_matrix.json`](../style_matrix.json) — 17 beer styles, each a
map `{ "Base Malt": [...], "Hop": [...], "Yeast": [...], "Adjunct": [...],
"Specialty": [...] }` of ingredient **names** that belong to that style.

**How it feeds the engine:** it is the candidate universe *and* the basis for
Fit/idf/focus and for `compute_style_status` (viable-styles). **An ingredient
that appears in no style is invisible to recommendations** (it can still be
drafted from the board, but is never *suggested*).

**Gap:** `validate_data(...)["in_sheet_not_matrix"]` = **64 draftable ingredients
modeled by no style** — 21 Adjuncts, 43 Specialty malts/grains (e.g. `Barley,
Flaked`; `Brown Malt`; `Biscuit Malt`; `Abstrax terpenes …`). These never surface
as recommendations or best-available.

**Task:**
1. Assign each unmodeled adjunct/specialty ingredient to the styles where it
   genuinely belongs (e.g. `Brown Malt`, `Special B` → British/Belgian dark
   styles; `Oats, Flaked` → Stout/NEIPA-ish; flavor adjuncts → the fruit/spec
   styles). Some (pure process aids, exotic terpene sprays) may legitimately
   stay unmodeled — document which and why.
2. Optionally review existing membership for accuracy (are the hop lists per
   style sensible? is any style over-broad?).
3. Consider whether 17 styles is the right granularity for this club's ingredient
   pool, or whether a few should be split/merged.

**Acceptance:** `validate_data` still returns `in_matrix_not_sheet == []`;
`in_sheet_not_matrix` shrinks to a documented, intentional remainder; the
2025-draft characterization tests in `tests/test_draft_core.py` still pass (or
are updated with a clear rationale if style membership deliberately changes
rankings).

---

## Workstream 2 — Similarity data

**Files:** [`hop_similarity.json`](../hop_similarity.json) (43 hops; each maps to
a ranked neighbor list `[{"hop": <name>, "score": 0..1}, ...]`) and
[`hop_similarity_matrix.csv`](../hop_similarity_matrix.csv) (dense pairwise
matrix).

**How it feeds the engine:** drives the hop-blend term in **Synergy** and the
substitute discount in **Scarcity**; also powers the "Hop similarity finder" tool.

**Gaps:**
- **Provenance unknown / not reproducible** — no committed script generates these
  files; the scoring methodology is undocumented.
- **Hops only** — there is no yeast-to-yeast or malt-to-malt similarity, so
  Synergy/Scarcity substitution is blind outside hops.
- **Missing entries** — e.g. `Columbus/Tomahawk/Zeus` (net-new 2026 hop) has no
  row.

**Task:**
1. Define and **document** a similarity method (e.g. cosine over curated
   flavor-descriptor / aroma-oil vectors, or published hop-substitution charts)
   and write a reproducible generator script under `scripts/` (mirror the style
   of [`scripts/convert_print_sheet.py`](../scripts/convert_print_sheet.py) —
   pure, CLI-runnable, deterministic).
2. Regenerate `hop_similarity.json` covering **all** hops on the current sheet.
3. Extend to **yeast** and **malt** similarity (same JSON shape, new files, e.g.
   `yeast_similarity.json` / `malt_similarity.json`) so Synergy/Scarcity can use
   them. Coordinate the new file names so `draft_core` can load them (a small
   loader change is expected — keep it config-driven).

**Acceptance:** a documented, re-runnable generator; full-sheet coverage; scores
in [0,1]; tests that assert shape + symmetry + that a known-similar pair scores
higher than a known-dissimilar one.

---

## Workstream 3 — Weights & scoring calibration

**Where:** the six-component model and weight presets in
[`draft_core.py`](../draft_core.py) (`next_best_picks`, `DEFAULT_PICK_WEIGHTS`,
`BOARD_VALUE_WEIGHTS`, the `_squash` transforms, `_survival_risk`).

**Task:**
1. **Calibrate the weights** against the recorded 2025 draft in
   [`tests/fixtures/draft_2025.json`](../tests/fixtures/draft_2025.json) (9
   players, 63 picks). Useful objective: at each historical pick, how highly did
   the model rank the ingredient that was actually taken? Tune weights /
   squash constants to raise that hit-rate without overfitting.
2. Sanity-check component behavior: does Scarcity actually rise as a category
   depletes? Does Urgency escalate as your snake turn approaches? Are the
   `_squash` denominators (fit `+5`, synergy `+3`, denial `+2`, pop `+1`)
   reasonable, or should they be data-derived?
3. Consider exposing a couple of high-impact knobs via `league_config.json`
   rather than hardcoding, if that helps season-to-season tuning.
4. The mock simulator (`draft_mode.py`, `OPPONENT_PERSONAS`) is a ready-made
   harness — persona weight profiles can be A/B'd there.

**Acceptance:** documented calibration method + before/after hit-rate on the 2025
fixture; changes covered by unit tests; no regression in existing
`tests/test_draft_core.py` behavior tests (update them intentionally if the
ranking philosophy changes, with rationale).

---

## Workstream 4 — Scarcity / idf generation scripts

**Files:** [`ingredient_scarcity.json`](../ingredient_scarcity.json) (116 rows:
`Ingredient`, `Style Coverage`, `Scarcity Score`).

**Status:** the live engine now computes scarcity dynamically
(`compute_dynamic_scarcity`) and only falls back to this static file when a row
is missing; its provenance is also undocumented (the historical values are just
`1/StyleCoverage`, which is circular).

**Task:** author reproducible `scripts/` generators (pure, CLI, deterministic)
that rebuild season data from the sheet + `style_matrix.json`:
- a **style-idf** file (or confirm computing it at load is sufficient),
- a **baseline scarcity** file computed from real substitute counts at a full
  board (not `1/coverage`), *or* a documented decision to retire the static file
  entirely in favor of the dynamic function.

**Acceptance:** running the script(s) reproduces the committed data
deterministically; a short README note explains inputs/outputs; tests cover the
generator on a small synthetic input.

---

## Working agreement

- Branch from `2026-polish`; keep `draft_core.py` pure and fully unit-tested.
- Run `./fantasydraft/bin/python -m pytest -q` before every hand-back; it must be
  green (currently 56 tests).
- Any data edit: re-run `validate_data` for the 2026 sheet and keep
  `in_matrix_not_sheet == []`.
- Prefer config (`league_config.json`) over hardcoded constants; prefer a
  committed generator script over hand-edited derived data.
- Leave the 2025 golden fixture (`tests/fixtures/draft_2025.json`) untouched — it
  is the calibration/regression baseline.

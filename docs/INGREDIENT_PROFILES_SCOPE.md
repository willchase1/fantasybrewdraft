# Scoping — richer ingredient profiles in the detail modal

**Goal.** The ingredient look-up modal should show brewer-useful specs, not just
styles + substitutes: for a **hop** its usage (bittering / aroma / dual), alpha
acid, and flavor/aroma notes; for a **malt** its color (SRM/°L) and flavor
notes; for a **yeast** its attenuation, temperature range, ale/lager type, and
flavor character.

**Good news: the data already exists.** Cowork's similarity descriptor tables in
`data/` already carry all of this as structured columns — this is mostly a
*surfacing* job, not a data-collection one.

Division of labor (unchanged): the **data + a pure `draft_core` helper** are
Cowork's (this doc); the **modal rendering** is Claude Code's follow-up (§5),
gated on the helper landing.

---

## 1. What's already in `data/`

- **`hop_descriptors.csv`** (51 rows): `name, origin, purpose` (bittering /
  aroma / dual), `alpha_mid`, and 0–3 flavor-axis scores: `citrus, tropical,
  stone_fruit, berry, melon_candy, floral, herbal, spicy, pine_resin, earthy,
  woody, grassy, dank, coconut_cream, wine_gooseberry, clean_bitter`.
- **`malt_descriptors.csv`** (83 rows): `name, sheet_category, kind`
  (extract / grain), `grain, origin, color_L` (°L ≈ SRM), `diastatic`, and axes:
  `bready, biscuit_toast, caramel_toffee, dark_fruit, roast_coffee_choc,
  burnt_acrid, nutty, smoky, grainy_neutral, body_head, sweet, melanoidin_rich`.
- **`yeast_descriptors.csv`** (32 rows): `name, family, ferment_type`
  (ale / lager), `attenuation_mid, flocculation, temp_mid_f`, and axes: `esters,
  phenols, clean_neutral, malt_forward, dry_crisp, haze_bio, sour, lager_sulfur`.

So the numeric/spec fields are done. The two missing pieces are (a) a reliable
name→row lookup and (b) turning the 0–3 axis scores into short human phrases.

---

## 2. Name matching (the one wrinkle)

Hop/malt descriptor `name`s largely match the sheet. **Yeast names carry strain
codes** — e.g. `"Abbey Ale (WLP540, EY1762, BE-256)"` — while the board may use
a shorter label. Reuse the existing normalization already used for similarity
(`_sim_name` / `load_similarity` / `category_aliases`) so the profile lookup
resolves the same way the substitution finder does. Any ingredient with no
descriptor row must degrade gracefully (helper returns `None` for the profile;
the modal just omits the section).

**Gate:** every sheet hop/malt/yeast should resolve to a descriptor row, or be
listed as a known gap. Add a `--check`-style coverage assertion (mirrors
`validate_data`) so a new-ingredient year can't silently lose profiles.

---

## 3. Proposed pure helper — `draft_core.ingredient_profile(...)`

A read-only helper (like `ingredient_detail`), config-driven and unit-tested:

```
ingredient_profile(ing, descriptors, board_category=None) -> dict | None
```

Returns a normalized, **already-humanized** profile so the UI stays dumb:

- **Hop** → `{"kind": "Hop", "usage": "Dual-purpose", "alpha": "9% AA",
  "notes": ["citrus", "tropical", "pine"]}`
- **Malt** → `{"kind": "Malt", "color": "40 °L", "form": "grain",
  "notes": ["caramel/toffee", "dark fruit"]}`
- **Yeast** → `{"kind": "Yeast", "type": "Ale", "attenuation": "76%",
  "temp": "66–70 °F", "notes": ["estery", "phenolic"]}`

Design notes:
- **notes** = the top-N axes (score ≥ threshold, e.g. ≥2), mapped to friendly
  labels via a small module-level `AXIS_LABELS` dict (`caramel_toffee →
  "caramel/toffee"`, `pine_resin → "pine/resin"`, …). N≈3 keeps it projector-
  legible. This axis→words mapping and the threshold are brewer's-judgement —
  own them here and note choices in a short `docs/INGREDIENT_PROFILES.md`.
- Keep it a **loader + pure function**: `load_descriptors(config)` reading paths
  named in `league_config.json` (`descriptor_files: {Hop, "Base Malt"/Specialty,
  Yeast}`, parallel to `similarity_files`), plus the pure `ingredient_profile`.
- Return strings pre-formatted (units included) so there's zero formatting logic
  in the app.

Optionally fold the profile into `ingredient_detail`'s return (add a
`"profile"` key) instead of a separate call — either is fine; a separate helper
keeps `ingredient_detail`'s existing tests untouched.

---

## 4. Tests (Cowork)

- `ingredient_profile` for one hop, one malt, one yeast → correct usage/alpha,
  color, attenuation/temp/type, and the expected top notes given known axis
  scores (use small synthetic descriptor dicts, like the existing
  `style_build_plan`/`ingredient_detail` tests).
- Unknown ingredient → `None`.
- Coverage check: all sheet hops/malts/yeasts resolve (or an explicit gap list).
- `./fantasydraft/bin/python -m pytest -q` stays green.

---

## 5. UI follow-up (Claude Code — after the helper lands)

Once `ingredient_profile` exists, I'll render a category-aware **specs block** at
the top of `show_ingredient_dialog` (above "Fits styles"): a compact line like
`Dual-purpose · 9% AA · citrus, tropical, pine` for hops, `40 °L · caramel/toffee,
dark fruit` for malts, `Ale · 76% attenuation · 66–70 °F · estery, phenolic` for
yeasts. No app-side formatting — just display the helper's strings. I may also
show the one-line note inline in the board/build-planner rows later, but the
modal is the primary surface. No engine/scoring change; profiles are
informational only (they do **not** feed `Pick Value`).

---

## Out of scope / notes
- Adjuncts have no descriptor table yet (`data/adjunct_descriptors.csv` is the
  natural extension if the club wants adjunct notes + substitution later — not
  required here).
- This is display-only. If we ever want flavor axes to influence synergy/fit,
  that's a separate scoring proposal.

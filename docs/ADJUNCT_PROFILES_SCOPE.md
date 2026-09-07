# Adjunct profiles — build brief (2026-09-07)

**Context:** every adjunct on the board renders a blank detail modal. On the 2026 sheet
**51 of 51 adjuncts** return `ingredient_profile(...) is None` — it's the last display-only
ingredient class without a spec sheet (Hop / Yeast / Base Malt / Specialty Malt are all at 100%
coverage). This brief adds `data/adjunct_descriptors.csv` + the code to render it, mirroring the
round-5 pattern exactly. **Display-only: nothing here feeds scoring or similarity in V1.**

The 2025 adjunct column (45 items) is a strict subset of 2026 (51), so covering the 51 below
covers both sheets. `usage` is a column no other descriptor table has — use it as the branch marker.

## The pattern to mirror (all already generic — see `docs/INGREDIENT_PROFILES.md`)

- **`config.py` `DEFAULTS["descriptor_files"]`** — add `"Adjunct": "data/adjunct_descriptors.csv"`.
  (`category_aliases["Adjunct"]` already exists: `["Adjunct","Adjuncts","Adjuncts/Spices/Fruits"]`.)
  If `league_config.json` overrides `descriptor_files`, add the key there too.
- **`draft_core.load_descriptors()`** — already loads whatever `descriptor_files` names; no change.
- **`draft_core.validate_descriptors()`** — already generic; once the file is configured, adjuncts
  are gated automatically (missing rows → `gaps["Adjunct"]`).
- **`draft_core.ingredient_profile()`** — add one Adjunct branch (below). Loose-key name matching
  (`_lookup_row` / `_profile_key`) is reused as-is.
- **`_profile_notes(row, _ADJUNCT_AXES)`** — reuse for the notes list; add `_ADJUNCT_AXES` + any new
  `AXIS_LABELS`.

## `data/adjunct_descriptors.csv` — schema

```
name,type,form,usage,fermentable,origin,<axis columns 0-3…>
```

- `name` — the board string. Loose match strips ` (...)`, so `Corn Sugar (Dextrose)` matches a row
  named either `Corn Sugar` or the full string; full is clearer. (See gotchas.)
- `type` — controlled: `fruit / citrus_peel / sugar / syrup / spice / herb / roast / wood / nut /
  dairy_sugar / fungus / extract / other`. Drives the summary's lead noun.
- `form` — free display string: `puree / zest / syrup / powder / bean / nib / chip / whole / dried`…
- `usage` — `flavoring / fermentable / both`. Drives the summary verb. **This column is the branch
  marker** — no other table has it.
- `fermentable` — `0/1`. Sugars/syrups = 1; **lactose = 0** (unfermentable, adds body/sweetness);
  fruit is partially fermentable → your call, default 1 for fresh fruit.
- axis columns `0–3` — reuse existing axis keys wherever they fit so `AXIS_LABELS` already covers
  them: `citrus, berry, stone_fruit, tropical, floral, herbal, earthy, woody, sweet, dark_fruit,
  roast_coffee_choc, coconut_cream, sour`. Add a small adjunct-specific set (and their labels):
  `melon → "melon"`, `spice_warm → "warm spice"`, `pepper_heat → "chili heat"`,
  `vanilla → "vanilla"`, `boozy → "boozy/rich"`. **Axis set is a starter — finalize by ear;** keep
  `AXIS_LABELS` in sync. Notes = axes scoring ≥ `PROFILE_NOTE_THRESHOLD` (2), max
  `PROFILE_NOTE_MAX` (3), strongest first — same rule as the others.

## `ingredient_profile()` — the Adjunct branch

Add **before** the malt fallback (the unconditional `else` at the bottom), after the yeast branch:

```python
if cat == "Adjunct" or "usage" in row:
    typ   = _ADJUNCT_TYPE.get(str(row.get("type", "")).lower(), "Adjunct")
    usage = _ADJUNCT_USAGE.get(str(row.get("usage", "")).lower(), None)
    ferment = bool(int(_num(row.get("fermentable"), 0) or 0))
    form  = str(row.get("form", "")).replace("_", " ") or None
    origin = _ORIGIN.get(row.get("origin"), row.get("origin"))
    notes = _profile_notes(row, _ADJUNCT_AXES)
    return {"kind": "Adjunct", "type": typ, "form": form, "usage": usage,
            "fermentable": ferment, "origin": origin, "notes": notes,
            "summary": join([typ, usage, ", ".join(notes)])}
```

New maps beside `_HOP_PURPOSE` / `_YEAST_TYPE`:

```python
_ADJUNCT_TYPE  = {"fruit": "Fruit", "citrus_peel": "Citrus", "sugar": "Sugar", "syrup": "Syrup",
                  "spice": "Spice", "herb": "Herb", "roast": "Roast", "wood": "Wood",
                  "nut": "Nut", "dairy_sugar": "Lactose", "fungus": "Fungus",
                  "extract": "Flavoring", "other": "Adjunct"}
_ADJUNCT_USAGE = {"flavoring": "flavoring", "fermentable": "fermentable", "both": "fermentable + flavor"}
_ADJUNCT_AXES  = ["citrus", "berry", "stone_fruit", "tropical", "melon", "floral", "herbal",
                  "spice_warm", "pepper_heat", "vanilla", "roast_coffee_choc", "coconut_cream",
                  "woody", "earthy", "sour", "sweet", "dark_fruit", "boozy"]
```

Summary reads like the others, e.g. `Fruit · flavoring · mango, tropical fruit` ·
`Sugar · fermentable · boozy/rich` · `Lactose · flavoring · sweet` ·
`Spice · flavoring · warm spice`.

## Tests — `tests/test_profiles.py`

1. **Flip the sentinel (this test will FAIL until you do):** the last line of
   `test_real_profiles_read_sensibly` currently asserts
   `ingredient_profile("Honey", descriptors, "Adjunct") is None  # no adjunct table`.
   Change it to assert a real profile — e.g. `honey = ...; assert honey and honey["summary"]`.
2. **Extend the coverage gate** `test_every_sheet_hop_malt_yeast_has_a_profile`: add
   `("Adjunct", "Adjunct")` to the `(col, cat)` loop. The `gaps == {}` assert now also covers
   adjuncts on **both** the 2026 and 2025 sheets → every one of the 51 needs a row with a non-empty
   summary. (Rename the test if you like; cosmetic.)
3. **Add a unit test** `test_adjunct_profile` on a synthetic row in `SYN` (mirror
   `test_hop_profile`): assert `kind == "Adjunct"`, the `type`/`usage` labels, and the notes order.

## Similarity — explicitly OUT for V1

Do **not** add an `adjunct` entry to `SPECS` in `scripts/build_similarity.py`. That script reads its
own `SPECS`, not `descriptor_files`, so the new CSV won't touch the hop/yeast/malt similarity files —
`build_similarity.py --check` must still print **OK** (byte-identical). Adjunct substitution /
similarity is a future round (it's where a `substitutes` column would eventually go).

## Name-matching gotchas

`_profile_key` strips ` (...)` / ` [...]` and collapses whitespace/case. So these board strings all
resolve on their base name — the CSV `name` may be the base or the full string:

- `Corn Sugar (Dextrose)` → `corn sugar` · `Milk Sugar (Lactose)` → `milk sugar`
- `Lyle's Golden Syrup (Invert Sugar)` → `lyle's golden syrup`
- `Abstrax terpenes (any variety of Quantum, Omni, or BrewGas)` → `abstrax terpenes`
- `Abstrax SkyFarm fruit flavors (any)` → `abstrax skyfarm fruit flavors`
- `Rice Syrup (added "syrup" to differentiate…)` → `rice syrup`

Comma-suffixed variants are **distinct rows**, not parentheticals — keep them separate:
`Candi Sugar, Amber / Clear / Dark` and `Orange Peel, Bitter / Sweet`.

## Worklist — the 51 adjuncts (2026), grouped by proposed `type`

- **sugar / syrup (15):** Cane (or Beet) Sugar · Corn Sugar (Dextrose) · Candi Sugar, Amber ·
  Candi Sugar, Clear · Candi Sugar, Dark · Brown Sugar · Turbinado Sugar · Demerara Sugar ·
  Molasses · Maple Syrup · Honey · Rice Syrup · Lyle's Golden Syrup (Invert Sugar) ·
  Brewer's Crystals · Milk Sugar (Lactose) *(→ `dairy_sugar`, fermentable 0)*
- **fruit (15):** Apricots · Mango · Blackberries · Blueberries · Cherries · Nectarines · Peaches ·
  Pineapple · Raspberries · Passion Fruit · Dragon Fruit · Guava · Watermelon · Cranberries ·
  Pumpkin *(gourd — often reads as `spice_warm` too)*
- **citrus_peel (5):** Lime Peel/Juice · Lemon Peel/Juice · Grapefruit Peel · Orange Peel, Bitter ·
  Orange Peel, Sweet
- **spice / herb (8):** Coriander · Cinnamon · Ginger · Grains of Paradise · Juniper · Licorice ·
  Chili Peppers *(→ `pepper_heat`)* · Vanilla (extract or bean)
- **roast (3):** Cocoa Nibs/Beans · Coffee (Liquid or Beans) · Coconut *(`coconut_cream`)*
- **wood (1):** Oak
- **fungus (2):** Mushrooms · Truffles
- **extract (2):** Abstrax terpenes (…) · Abstrax SkyFarm fruit flavors (any)

## Verify

```
./fantasydraft/bin/python -m pytest -q tests/test_profiles.py   # + full suite
./fantasydraft/bin/python scripts/build_similarity.py --check   # still OK (unchanged)
```

## Acceptance

- `data/adjunct_descriptors.csv` covers every adjunct on the 2026 and 2025 sheets.
- `ingredient_profile(<any adjunct>, descriptors, "Adjunct")` → dict with a non-empty `summary`.
- `validate_descriptors(...) == {}` including the Adjunct category.
- `build_similarity.py --check` prints OK (similarity files byte-identical).
- The `Honey is None` sentinel is flipped; new adjunct unit test passes.

## Notes for whoever authors the cells

- Values are brewer's judgement — the axis 0–3 scores are "how loudly does this note read in the
  finished beer," same scale as the hop/malt/yeast sheets. Spot-check the summaries out loud.
- Keep it projector-legible: the notes cap at 3, so score conservatively — only a 2 or 3 shows.
- Don't git-commit from an automated VM (strands `.git/index.lock`); hand the diff back to Will.

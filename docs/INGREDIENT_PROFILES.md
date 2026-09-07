# Ingredient profiles (detail-modal spec sheets)

`draft_core.ingredient_profile(ing, descriptors, board_category=None)` turns a
row of the curated descriptor tables in `data/` into a brewer-facing spec
sheet with every value **pre-formatted for display** — the UI shows strings,
it does not compute anything. Profiles are informational only; nothing here
feeds `Pick Value`.

    descriptors = draft_core.load_descriptors()          # paths from league_config.json
    draft_core.ingredient_profile("Citra", descriptors, "Hop")
    -> {"kind": "Hop", "usage": "Aroma", "alpha": "11–13% AA", "origin": "USA",
        "notes": ["citrus", "tropical fruit"],
        "summary": "Aroma · 11–13% AA · citrus, tropical fruit"}

| kind  | fields | example summary |
|---|---|---|
| Hop   | `usage` (Bittering / Aroma / Dual-purpose), `alpha` (range, or midpoint if no range), `origin`, `notes` | `Dual-purpose · 4.5–7% AA · citrus, floral` |
| Malt  | `color` (°L ≈ SRM), `form` (base malt (pale) / crystal/caramel malt / roasted malt / flaked grain / extract …), `grain`, `origin`, `diastatic` (bool), `notes` | `140 °L · crystal/caramel malt · dark fruit, caramel/toffee, sweet` |
| Yeast | `type` (Ale / Lager / Hybrid / Kveik), `attenuation`, `temp` (range °F), `flocculation`, `family`, `notes` | `Ale · 78% attenuation · 68–85 °F · phenolic (clove/pepper), dry/crisp, estery/fruity` |

`summary` is the one-liner for the modal / board rows; the individual fields
are there if the UI wants a table. Returns `None` for anything without a
descriptor row (all adjuncts today — `data/adjunct_descriptors.csv` is the
natural extension).

## Data

`league_config.json` → `descriptor_files` names the tables per category
(malt covers Base Malt + Specialty); `config.DEFAULTS` mirrors it. The same
tables drive `scripts/build_similarity.py`. Two **display-only** column pairs
were added for the profiles — `alpha_lo`/`alpha_hi` (hops) and
`temp_lo_f`/`temp_hi_f` (yeasts), typical published ranges — and are listed
in the similarity generator's `ignore` list, so the similarity files are
byte-identical with or without them.

## Name matching

Exact sheet name first; otherwise a loose key — case- and whitespace-
insensitive with anything in `(...)`/`[...]` dropped — so a board label like
`Abbey Ale` or `voss kveik` resolves to `Abbey Ale (WLP540, EY1762, BE-256)`.
`board_category` narrows the lookup to one table when the UI knows it; an
unknown hint (e.g. `Adjunct`) falls back to searching every table.

## Coverage gate

`draft_core.validate_descriptors(ingredients_df, descriptors)` returns
`{category: [sheet ingredients with no row]}`, mirroring `validate_data`.
`tests/test_profiles.py::test_every_sheet_hop_malt_yeast_has_a_profile` runs it
against the 2026 and 2025 sheets and additionally asserts every hop / malt /
yeast yields a non-empty summary, so a new-ingredient year cannot silently
lose profiles. (`python scripts/build_similarity.py --check` reports the same
gaps as WARN lines from the generator's side.)

## Notes: axis → words

`AXIS_LABELS` in `draft_core.py` maps each 0–3 descriptor axis to a phrase;
an axis is mentioned when its score is ≥ `PROFILE_NOTE_THRESHOLD` (2) and at
most `PROFILE_NOTE_MAX` (3) are shown, strongest first, ties in table order.
Brewer's-judgement choices worth knowing:

* `pine_resin` → "pine/resin", `wine_gooseberry` → "white wine/gooseberry",
  `coconut_cream` → "coconut/cream", `clean_bitter` → "clean bittering" (so a
  Magnum/Warrior profile says what it is *for* rather than showing nothing).
* `roast_coffee_choc` → "roast/coffee/chocolate", `melanoidin_rich` →
  "rich/melanoidin", `body_head` → "body & head" (flaked grains).
* `phenols` → "phenolic (clove/pepper)", `haze_bio` → "hazy/juicy",
  `lager_sulfur` → "lager sulfur", `clean_neutral` → "clean". A yeast can
  legitimately show both "estery/fruity" and "clean" (Voss: orange esters,
  otherwise neutral).
* Threshold 2 rather than 1 keeps notes to what a brewer would actually
  taste; a 1 is "a hint of", which is noise on a projector.

Editing a note is a one-line change in `AXIS_LABELS`; editing a spec is a cell
in `data/*_descriptors.csv` (re-run `build_similarity.py` if you touched a
scored axis, `--check` otherwise).

# Style matrix — design notes (2026)

`style_matrix.json` is **generated**: edit `scripts/build_style_matrix.py`, run it,
commit both. `python scripts/build_style_matrix.py --check` fails if the JSON has
drifted from the generator (wire it into CI / pre-commit if you like).

## Why it was rebuilt

The 2025 matrix (kept as `style_matrix_backup.json`) had 17 styles and left 64 of
the 204 draftable ingredients — 21 adjuncts and 43 specialty malts/grains — in
no style at all, so they could never be *recommended*. More importantly, its
structure biased the engine:

* **Catch-all styles.** *Stout / Porter* listed all 44 hops and 25 base malts;
  *Fruit Wheat / Sour* listed 29 of 30 yeasts. Membership in those styles carried
  no information, yet each added +1 to every ingredient's Style Coverage (the
  input to the Fit term).
* **Duplicate styles.** *Best Bitter*, *British Brown / Mild*, *British Strong*
  and *Robust Porter* were byte-identical apart from their specialty lists, so
  every British base malt / hop / yeast got 4× the coverage credit of, say, a
  Kölsch ingredient. *Dubbel*, *Dark Strong*, *Saison* and *Bière de Garde* were
  nearly so.
* **Missing styles.** No home for Hefeweizen, Kölsch/Alt, NEIPA, Märzen/Dunkel/
  Bock, American lager, kveik — although the 2025 draft contained clear NEIPA,
  Helles and kettle-sour builds, and the yeast sheet is half lager/kveik strains.
* **No adjuncts** on most styles even though the rules require one adjunct pick,
  so a Pils or IPA roster could never fully "satisfy" its own style.

## Design rules

1. **Fingerprint, not catch-all.** A style lists only ingredients actually used
   in it. Workhorses (2-row, Pale Malt Extract, US-05, Magnum) appear in many
   styles by nature; the engine's idf term handles that. Nothing is in every
   style: the widest ingredients are cane sugar / honey (18 of 22).
2. **Every style has an Adjunct list** so it can be fully satisfied under the
   rules. Where the style traditionally has none (Pils, Kölsch) the list is the
   handful a brewer would plausibly draft without ruining it (dextrose, rice
   syrup, brewer's crystals, cane sugar).
3. **Distinct styles only.** Two styles that would share >90 % of their lists are
   merged into one name (Dubbel / Dark Strong; Saison / Bière de Garde; Best
   Bitter / ESB / English Pale). The name lists what's folded in.
4. **Names are verbatim sheet strings**, including sheet quirks
   (`Hefewiezen`, `Melanoiden Malt`, `Rice Syrup (added "syrup" …)`). The
   generator validates against the active sheet and refuses to write orphans.
5. **Category keys are fixed** — `Base Malt, Hop, Yeast, Adjunct, Specialty`, in
   that order — because `draft_core` derives each ingredient's category from the
   matrix.

## The 22 styles

| Family | Styles |
|---|---|
| Lager | Pale Lager (Pils / Helles / Czech) · Amber & Dark Lager (Märzen / Dunkel / Bock) · American & Mexican Lager / Cream Ale |
| German / Belgian ale | Kölsch & Altbier · Hefeweizen / Dunkelweizen · Witbier / Belgian Wheat · Saison / Bière de Garde · Belgian Blonde / Tripel / Golden Strong · Belgian Dubbel / Dark Strong · Nordic Farmhouse / Kveik (Sahti) |
| American ale | American Pale Ale / West Coast & Cold IPA · Hazy / New England IPA · American Amber / Red / Brown |
| British ale | Best Bitter / ESB / English Pale · Mild / Brown Ale · English Strong / Barleywine / Wee Heavy · Porter (Brown / Robust / Baltic) · Stout (Dry / Oatmeal / Sweet / Foreign) · Imperial / Pastry Stout |
| Sour / fruit / specialty | Kettle Sour / Fruit Sour (Berliner / Gose) · American Wheat / Fruit & Spice Beer · Smoked & Wood-Aged (Rauchbier / Barrel) |

Notable placements for the previously unmodeled ingredients:

* **Flaked / torrefied / raw barley, oats (flaked, malted, golden naked), chit
  malt** → Stout, Porter, NEIPA, Witbier, Saison, Kettle Sour, American Wheat
  (body / haze / head builders).
* **Brown, Biscuit, Victory, Special Roast, Toasted, Cookie, Honey Malt** → the
  British and American amber/brown families (the "biscuity" family), Porter,
  Mild.
* **Crystal 10–120L** → light ones to APA / NEIPA / American Wheat; mid to
  Bitter / Amber / Porter / Stout; dark (plus Special B) to Strong Ale,
  Imperial Stout, Dubbel, Smoked & Wood-Aged.
* **Carafa I, Carafoam, Carared, Caramunich II, Melanoiden** → the German
  cara family: Pale / Amber & Dark Lager, Kölsch & Alt, Hefeweizen, Belgian
  darks, Smoked.
* **Rye (malt, flaked, extract), Spelt, Buckwheat, Millet, Quinoa** →
  Saison, Nordic Farmhouse, American Amber (red rye), American Wheat.
  `Extract, Sorghum` → American Wheat / Fruit (gluten-free fruit beer).
* **Corn (flaked, grits), Rice flaked** → American & Mexican Lager, Cold IPA,
  Kettle Sour, American Wheat.
* **Smoked & Peat Smoked Malt** → Smoked & Wood-Aged, Porter, Imperial Stout,
  Wee Heavy, Amber & Dark Lager (rauch-märzen), Nordic Farmhouse.
* **Fruit** (cherries, mango, passion fruit, guava, pineapple, dragon fruit,
  watermelon, peaches, nectarines) → Kettle Sour, American Wheat / Fruit,
  Hazy IPA; cherries also Dubbel / Dark Strong, Imperial Stout, Wood-Aged.
* **Dessert adjuncts** (cocoa, coffee, vanilla, coconut, lactose, maple,
  molasses) → Imperial / Pastry Stout, Stout, Porter, Mild / Brown, American
  Brown; lactose + vanilla also Hazy IPA (milkshake) and Kettle Sour.
* **Spice / herb** — coriander & grains of paradise → Witbier, Saison, American
  Wheat; juniper → Nordic Farmhouse, Saison, American Wheat; chili → Pastry
  Stout, Mexican Lager, West Coast IPA, American Wheat; licorice → Porter,
  Stout, Mild, Strong Ale, Wood-Aged.
* **Oak** → Smoked & Wood-Aged, Imperial Stout, Barleywine, Porter, Kettle Sour.
* **Mushrooms, Truffles** → Imperial / Pastry Stout, Smoked & Wood-Aged
  (umami stouts are a real thing; if the club never brews one, drop them from
  those two lists and they become an intentional remainder).
* **Abstrax terpenes** → West Coast IPA, Hazy IPA, American Wheat.
  **Abstrax SkyFarm fruit flavors** → Hazy IPA, Kettle Sour, American Wheat.

**Intentional remainder: none.** Every sheet ingredient is modeled by at least one
style (`validate_data(...)["in_sheet_not_matrix"] == []`). Sorghum extract,
mushrooms and truffles are the marginal calls — see above.

## Effects on the engine

* Style Coverage now ranges 1–18 across 22 styles with a healthy middle
  (most ingredients in 3–8 styles) instead of the old bimodal shape; Fit's
  `+5` squash constant is revisited in the calibration workstream.
* `compute_style_status` gained **Picks Matched** and **Match** columns and
  uses them as tie-breaks (see its docstring) — with distinct-but-related
  styles (Stout vs Porter vs Imperial Stout) the old `2×satisfied + options`
  score ties constantly. Likely-style for the nine 2025 rosters now reads:
  Chris B → Hazy IPA, Will & Ed → Stout, Matt → Kettle Sour, Jordan & Jim →
  Dubbel / Dark Strong, Chris E → Saison, Geoff → Pale Lager, Mike → Imperial
  / Pastry Stout (ties with American Brown on picks).
* `style_bias.json` families were re-pointed at the new names.
* `compute_rules_status` (name-based) now agrees with the record-based path for
  the 2025 rosters — previously Will's Molasses fell to Flex because no style
  listed it.

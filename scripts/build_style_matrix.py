#!/usr/bin/env python3
"""Build ``style_matrix.json`` from curated ingredient families.

Why a generator instead of a hand-edited JSON
---------------------------------------------
The style matrix is the recommendation engine's *candidate universe* and the
basis for Fit / idf / focus / style viability. It is inherently curated
brewing knowledge, but hand-editing 20+ styles x 5 categories x ~800 names
drifts out of sync with the ingredient sheet and accumulates copy-paste
duplicates. Composing each style from named families keeps the intent
legible ("British hops", "German cara malts"), makes membership diffs easy to
review, and lets us validate every name against the sheet before writing.

Design rules (see docs/STYLE_MATRIX.md):
  * Every style lists only the ingredients actually used in that style, so a
    style's ingredient set is a distinctive *fingerprint* (no catch-alls).
  * Every style has an Adjunct list -- league rules require one adjunct pick,
    so a style with no adjuncts can never be fully satisfied.
  * Workhorse ingredients (2-row, US-05) legitimately appear in many styles;
    the engine's idf term is what rewards signature ingredients over them.
  * Names must match the ingredient sheet exactly (validated below).

Usage
-----
    python scripts/build_style_matrix.py                # writes style_matrix.json
    python scripts/build_style_matrix.py --check        # validate only, exit 1 on drift
    python scripts/build_style_matrix.py --sheet ingredients_2026.csv --out style_matrix.json

Deterministic: same inputs -> byte-identical output (lists are de-duplicated
and sorted, JSON keys keep insertion order, ``ensure_ascii=False``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CATEGORIES = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty"]

# ---------------------------------------------------------------------------
# Ingredient families -- names are verbatim from the 2026 sheet
# ---------------------------------------------------------------------------

# Base malts
PILS_GER = ["German Pilsner", "Floor Malted Pilsner",
            "American (or North American) Pilsner", "Belgian Pilsner",
            "Extra Light/Pils Malt Extract"]
PILS_BEL = ["Belgian Pilsner", "Floor Malted Pilsner", "Belgian Pale",
            "German Pilsner", "American (or North American) Pilsner",
            "Extra Light/Pils Malt Extract"]
VIENNA_MUNICH = ["German Vienna", "American (or North American) Vienna",
                 "Munich, Light", "Munich, Dark", "Munich Malt Extract"]
PALE_US = ["Pale Malt (2 Row)", "Pale Malt (6 Row)", "Pale Malt Extract",
           "Light Malt Extract", "Golden Malt Extract"]
PALE_UK = ["Maris Otter", "Golden Promise", "English Pale Ale",
           "Pale Malt Extract", "Amber Malt Extract"]
WHEAT_BASE = ["White Wheat", "Wheat Malt Extract"]
DARK_EXTRACT = ["Dark Malt Extract", "Amber Malt Extract"]

# Hops
NOBLE = ["Saaz", "Hallertau Mittelfrueh", "Hallertau Tradition",
         "Tettnang / Tettnanger", "Perle", "Saphir", "Strisselspalt", "Liberty",
         "Mt Hood"]
GER_BITTER = ["Magnum", "Perle"]
UK_HOPS = ["East Kent Goldings", "Fuggle", "Challenger", "Progress",
           "Styrian Goldings"]
BELGIAN_HOPS = ["Saaz", "Styrian Goldings", "Strisselspalt",
                "Hallertau Mittelfrueh", "Hallertau Tradition",
                "East Kent Goldings", "Perle", "Magnum"]
US_C = ["Cascade", "Centennial", "Citra", "Simcoe", "Amarillo",
        "Columbus/Tomahawk/Zeus", "Mosaic", "Magnum", "Idaho 7", "Strata",
        "El Dorado", "Loral", "Lotus", "Krush (HBC 586)", "Lemondrop", "Sabro",
        "BRU-1", "Triumph", "Sorachi Ace", "Azacca", "Chinook", "Nugget",
        "Warrior"]
NZ_AUS = ["Galaxy", "Nelson Sauvin", "Motueka", "Nectaron", "Riwaka", "Rakau",
          "Kohatu", "Ella", "Topaz"]
GER_NEW_WORLD = ["Ariana", "Calista"]
HAZY = ["Citra", "Mosaic", "Galaxy", "Nelson Sauvin", "Strata", "Sabro",
        "El Dorado", "Nectaron", "Riwaka", "Motueka", "Idaho 7",
        "Krush (HBC 586)", "Azacca", "Lotus", "BRU-1", "Rakau", "Kohatu", "Ella",
        "Topaz", "Amarillo", "Simcoe", "Columbus/Tomahawk/Zeus", "Ariana",
        "Calista", "Lemondrop", "Loral", "Sorachi Ace", "Triumph", "Warrior",
        "Chinook"]
DARK_ALE_BITTER = ["Magnum", "Columbus/Tomahawk/Zeus", "Perle", "Cascade",
                   "Centennial", "Nugget", "Chinook", "Northern Brewer",
                   "Willamette"]
# Classic / high-alpha American workhorses (2026 additions)
US_CLASSIC = ["Chinook", "Nugget", "Warrior", "Northern Brewer", "Cluster",
              "Willamette", "Mt Hood"]

# Yeasts (exact sheet strings)
Y = {
    "abbey": "Abbey Ale (WLP540, EY1762, BE-256)",
    "alt": "Alt (WLP023, WY1007, G02)",
    "us": "American Ale (WLP001 [liquid or dry], WY1056, US-05, A07)",
    "belgian": "Belgian Ale (WLP570, WY1388)",
    "ardennes": "Belgian Ardennes (WY3522, B45)",
    "conan": "Conan (WLP095, OYL-052, A04, LalBrew New England [Dry])",
    "czech": "Czech Lager (WLP802, WY2001, L28)",
    "dryeng": "Dry English Ale (WLP007, WY1098, A10)",
    "eastcoast": "East Coast Ale (WLP008, A15)",
    "english": "English Ale (WLP002, WY1968, S-04, A09)",
    "farmhouse": "Farmhouse (WY3726, OYL-217, Dry)",
    "gerale": "German Ale (WY1007, G02)",
    "gerlager": "German Lager (WLP830, WY2124, L13, 34/70)",
    "hefe": "Hefewiezen (WLP300, WY3068, G01)",
    "wit": "Belgian Witbier (WLP400, WY3944, Imperial B44, LalBrew Wit)",
    "irish": "Irish Ale (WLP004, WY1084, Imperial A44)",
    "hornindal": "Hornindal Kveik (WLP521, OYL-091)",
    "kolsch": "Kolsch (WLP029, WY2565, G03, K-97)",
    "london3": "London III (WLP013, A38, WY1318)",
    "lutra": "Lutra Kveik (OYL-071, OYL-071DRY)",
    "mexlager": "Mexican Lager (WLP940, OYL-113, L09)",
    "munichlager": "Munich Lager (WLP838, WY2308, Dry)",
    "nova": "NovaLager (Lalbrew Y055) (Dry)",
    "philly": "Philly Sour or Sourvisiae (Dry)",
    "pomona": "Pomona (Lalbrew Y067) (Dry)",
    "sh45": "Saflager SH-45 (Thiol Enhancing Dry Lager Yeast)",
    "saison": "Saison (WLP565, WY3724)",
    "sflager": "San Fran Lager (WLP810, L05, WY2112)",
    "scottish": "Scottish Ale (WLP028, WY1728, A31)",
    "trappist": "Trappist Ale (WLP500, WY1214, B48)",
    "verdant": "Verdant IPA (Dry)",
    "voss": "Voss Kveik (A43, OYL-061, Dry)",
}
LAGER_CLEAN = [Y["gerlager"], Y["munichlager"], Y["czech"], Y["nova"],
               Y["sflager"], Y["lutra"], Y["mexlager"]]
ENGLISH = [Y["english"], Y["dryeng"], Y["london3"], Y["scottish"], Y["eastcoast"]]
DARK_ALE_YEASTS = ENGLISH + [Y["irish"]]
TRAPPIST = [Y["trappist"], Y["abbey"], Y["ardennes"], Y["belgian"]]
KVEIK = [Y["voss"], Y["hornindal"], Y["lutra"]]

# Adjuncts
RICE_SYRUP = ('Rice Syrup (added "syrup" to differentiate from flaked rice '
              'in specialty malts)')
TERPENES = "Abstrax terpenes (any variety of Quantum, Omni, or BrewGas)"
SKYFARM = "Abstrax SkyFarm fruit flavors (any)"
LACTOSE = "Milk Sugar (Lactose)"
# Sugars are deliberately *narrow*: a style lists only the sugars that are
# characteristic of it (candi in Belgians, invert in British ales, dextrose in
# lagers / dry IPAs), never "you could add it" -- otherwise a bag of dextrose
# out-versatiles a base malt in the Fit term. See docs/STYLE_MATRIX.md.
SUGARS_LAGER = ["Corn Sugar (Dextrose)", RICE_SYRUP]
SUGARS_BEL_PALE = ["Candi Sugar, Clear", "Candi Sugar, Amber",
                   "Cane (or Beet) Sugar", "Corn Sugar (Dextrose)", "Honey"]
SUGARS_BEL_DARK = ["Candi Sugar, Dark", "Candi Sugar, Amber", "Candi Sugar, Clear",
                   "Brown Sugar", "Demerara Sugar", "Turbinado Sugar",
                   "Cane (or Beet) Sugar"]
SUGARS_UK_PALE = ["Lyle's Golden Syrup (Invert Sugar)", "Demerara Sugar",
                  "Turbinado Sugar", "Cane (or Beet) Sugar"]
SUGARS_UK_DARK = ["Lyle's Golden Syrup (Invert Sugar)", "Brown Sugar",
                  "Demerara Sugar", "Molasses"]
SPICE_BAKING = ["Cinnamon", "Ginger", "Pumpkin"]
FRUIT_STONE_BERRY = ["Apricots", "Blackberries", "Blueberries", "Cherries",
                     "Raspberries", "Peaches", "Nectarines", "Cranberries"]
FRUIT_TROPICAL = ["Mango", "Passion Fruit", "Pineapple", "Guava",
                  "Dragon Fruit", "Watermelon", "Coconut"]
CITRUS = ["Orange Peel, Sweet", "Orange Peel, Bitter", "Lemon Peel/Juice",
          "Lime Peel/Juice", "Grapefruit Peel"]
DESSERT = ["Cocoa Nibs/Beans", "Coffee (Liquid or Beans)",
           "Vanilla (extract or bean)", "Coconut", LACTOSE, "Maple Syrup",
           "Molasses", "Cinnamon"]
UMAMI = ["Mushrooms", "Truffles"]

# Specialty malts / grains
CARA_GER = ["Carahell", "Caravienne", "Caramunich I", "Caramunich II",
            "Carared", "Carafoam", "Carapils/Dextrine/Carafoam",
            "Melanoiden Malt", "Aromatic"]
CARAFA = ["Carafa I", "Carafa II", "Carafa III"]
DEXTRINE = ["Carafoam", "Carapils/Dextrine/Carafoam"]
CRYSTAL_LIGHT = ["Caramel/Crystal Malt - 10L", "Caramel/Crystal Malt - 20L",
                 "Caramel/Crystal Malt - 30L", "Caramel/Crystal Malt - 40L"]
CRYSTAL_MID = ["Caramel/Crystal Malt - 40L", "Caramel/Crystal Malt - 60L",
               "Caramel/Crystal Malt - 80L"]
CRYSTAL_DARK = ["Caramel/Crystal Malt - 60L", "Caramel/Crystal Malt - 80L",
                "Caramel/Crystal Malt -120L", "Special B"]
BISCUITY = ["Biscuit Malt", "Victory", "Special Roast", "Toasted Malt",
            "Cookie Malt (Viking Malt)", "Honey Malt"]
ROAST = ["Chocolate Malt", "Pale Chocolate Malt", "Black Malt",
         "Barley, Roasted", "Blackprinze Malt", "Wheat, Roasted", "Coffee Malt"]
OATS = ["Oats, Flaked", "Oats, Malted", "Oats, Golden Naked"]
WHEATS = ["Wheat, Flaked", "Wheat, Torrified", "Wheat, Red"]
BARLEY_ADJ = ["Barley, Flaked", "Barley, Torrefied", "Barley, Raw"]
RYE = ["Rye Malt", "Rye, Flaked", "Extract, Rye"]
CORN_RICE = ["Corn, Flaked", "Corn, Grits", "Rice, Flaked"]
ANCIENT = ["Millet", "Buckwheat", "Quinoa", "Spelt Malt"]
SMOKED = ["Smoked Malt", "Peat Smoked Malt"]
EXTRA_DARK = ["Extract, Extra Dark"]


def _u(*lists):
    """Union of lists / single names, de-duplicated, sorted for determinism."""
    out = set()
    for x in lists:
        if isinstance(x, str):
            out.add(x)
        else:
            out.update(x)
    return sorted(out)


# ---------------------------------------------------------------------------
# Styles -- the order here is the order in the JSON / UI
# ---------------------------------------------------------------------------
def build_styles() -> dict:
    S = {}

    # -- Lagers ---------------------------------------------------------------
    S["Pale Lager (Pils / Helles / Czech)"] = {
        "Base Malt": _u(PILS_GER, "German Pale Ale", "German Vienna",
                        "Light Malt Extract"),
        "Hop": _u(NOBLE, "Magnum"),
        "Yeast": _u(LAGER_CLEAN, Y["sh45"]),
        "Adjunct": _u(SUGARS_LAGER),
        "Specialty": _u(DEXTRINE, "Carahell", "Caravienne", "Melanoiden Malt",
                        "Chit Malt"),
    }
    S["Amber & Dark Lager (Märzen / Dunkel / Bock)"] = {
        "Base Malt": _u(VIENNA_MUNICH, "German Pilsner", "Floor Malted Pilsner",
                        "Red X", DARK_EXTRACT),
        "Hop": _u(NOBLE, "Magnum", "Northern Brewer"),
        "Yeast": _u(LAGER_CLEAN),
        "Adjunct": _u("Corn Sugar (Dextrose)", "Brewer's Crystals"),
        "Specialty": _u(CARA_GER, CARAFA, "Chocolate Malt", EXTRA_DARK,
                        "Smoked Malt"),
    }
    S["American & Mexican Lager / Cream Ale"] = {
        "Base Malt": _u("Pale Malt (6 Row)", "Pale Malt (2 Row)",
                        "American (or North American) Pilsner",
                        "American (or North American) Vienna", "German Pilsner",
                        "Pale Malt Extract", "Light Malt Extract",
                        "Extra Light/Pils Malt Extract"),
        "Hop": _u("Liberty", "Saaz", "Hallertau Mittelfrueh",
                  "Hallertau Tradition", "Tettnang / Tettnanger", "Perle",
                  "Magnum", "Cascade", "Sorachi Ace", "Cluster", "Mt Hood",
                  "Willamette"),
        "Yeast": _u(Y["mexlager"], Y["sflager"], Y["nova"], Y["gerlager"],
                    Y["lutra"], Y["kolsch"], Y["us"], Y["sh45"], Y["czech"]),
        "Adjunct": _u(SUGARS_LAGER, "Brewer's Crystals", "Lime Peel/Juice",
                      "Chili Peppers"),
        "Specialty": _u(CORN_RICE, DEXTRINE, "Millet", "Chit Malt"),
    }

    # -- German / Belgian ales -------------------------------------------------
    S["Kölsch & Altbier"] = {
        "Base Malt": _u("German Pale Ale", "German Pilsner",
                        "Floor Malted Pilsner",
                        "American (or North American) Pilsner", "Munich, Light",
                        "German Vienna", "Red X", "Pale Malt Extract"),
        "Hop": _u(NOBLE, "Magnum", "Northern Brewer"),
        "Yeast": _u(Y["kolsch"], Y["alt"], Y["gerale"], Y["sflager"], Y["lutra"]),
        "Adjunct": _u("Corn Sugar (Dextrose)", "Brewer's Crystals"),
        "Specialty": _u("Caravienne", "Carahell", "Caramunich I", "Caramunich II",
                        "Carafa I", "Carafa II", DEXTRINE, "Melanoiden Malt",
                        "Wheat, Flaked"),
    }
    S["Hefeweizen / Dunkelweizen"] = {
        "Base Malt": _u(WHEAT_BASE, "German Pilsner", "Floor Malted Pilsner",
                        "American (or North American) Pilsner", "Munich, Light",
                        "Munich, Dark", "German Vienna"),
        "Hop": _u("Hallertau Mittelfrueh", "Hallertau Tradition",
                  "Tettnang / Tettnanger", "Perle", "Saaz", "Saphir"),
        "Yeast": _u(Y["hefe"]),
        "Adjunct": _u("Apricots", "Peaches", "Blueberries"),
        "Specialty": _u(WHEATS, "Caramunich I", "Carafa II", "Carafa III",
                        "Chocolate Malt", "Melanoiden Malt", "Aromatic",
                        "Carahell", "Carafoam", "Wheat, Roasted"),
    }
    S["Witbier / Belgian Wheat"] = {
        "Base Malt": _u(PILS_BEL, WHEAT_BASE),
        "Hop": _u("Saaz", "Hallertau Mittelfrueh", "Strisselspalt",
                  "Styrian Goldings", "Perle", "East Kent Goldings"),
        "Yeast": _u(Y["wit"], Y["belgian"], Y["ardennes"], Y["farmhouse"],
                    Y["saison"], Y["hefe"], Y["abbey"]),
        "Adjunct": _u("Coriander", CITRUS, "Grains of Paradise", "Honey",
                      "Candi Sugar, Clear", "Apricots", "Raspberries", "Peaches",
                      "Cranberries", "Ginger"),
        "Specialty": _u(WHEATS, "Barley, Raw", "Oats, Flaked", "Oats, Malted",
                        DEXTRINE),
    }
    S["Saison / Bière de Garde"] = {
        "Base Malt": _u(PILS_BEL, "Munich, Light", "Munich, Dark",
                        "German Vienna", "White Wheat"),
        "Hop": _u(BELGIAN_HOPS, "Sorachi Ace", "Nelson Sauvin", "Motueka"),
        "Yeast": _u(Y["saison"], Y["farmhouse"], Y["ardennes"], Y["belgian"],
                    Y["hornindal"], Y["voss"]),
        "Adjunct": _u(SUGARS_BEL_PALE, "Turbinado Sugar", "Grains of Paradise",
                      "Coriander", "Orange Peel, Bitter", "Lemon Peel/Juice",
                      "Juniper", "Ginger", "Apricots", "Peaches", "Nectarines",
                      "Blackberries", "Raspberries", "Cranberries"),
        "Specialty": _u(WHEATS, RYE, ANCIENT, OATS, "Barley, Raw", "Aromatic",
                        "Caravienne", "Carahell", "Caramunich I", DEXTRINE,
                        "Melanoiden Malt", "Biscuit Malt", "Special B",
                        "Honey Malt"),
    }
    S["Belgian Blonde / Tripel / Golden Strong"] = {
        "Base Malt": _u(PILS_BEL, "Pale Malt (2 Row)", "Light Malt Extract"),
        "Hop": _u(BELGIAN_HOPS),
        "Yeast": _u(TRAPPIST),
        "Adjunct": _u(SUGARS_BEL_PALE, "Coriander", "Orange Peel, Bitter"),
        "Specialty": _u(DEXTRINE, "Carahell", "Caravienne", "Aromatic",
                        "Melanoiden Malt", "Honey Malt", "Wheat, Flaked",
                        "Wheat, Torrified"),
    }
    S["Belgian Dubbel / Dark Strong"] = {
        "Base Malt": _u("Belgian Pilsner", "Belgian Pale", "Floor Malted Pilsner",
                        "Munich, Light", "Munich, Dark", "German Pilsner",
                        "American (or North American) Pilsner",
                        "Munich Malt Extract", DARK_EXTRACT),
        "Hop": _u(BELGIAN_HOPS),
        "Yeast": _u(TRAPPIST),
        "Adjunct": _u(SUGARS_BEL_DARK, "Cherries", "Cinnamon"),
        "Specialty": _u("Special B", "Aromatic", "Caramunich I", "Caramunich II",
                        "Caravienne", "Carahell", "Carared", "Melanoiden Malt",
                        CARAFA, "Chocolate Malt", "Pale Chocolate Malt",
                        "Biscuit Malt", "Caramel/Crystal Malt - 80L",
                        "Caramel/Crystal Malt -120L", "Honey Malt", EXTRA_DARK),
    }
    S["Nordic Farmhouse / Kveik (Sahti)"] = {
        "Base Malt": _u("German Pilsner", "Floor Malted Pilsner",
                        "American (or North American) Pilsner",
                        "Pale Malt (2 Row)", "Munich, Light", "German Vienna",
                        "American (or North American) Vienna", "Maris Otter",
                        "White Wheat", "Pale Malt Extract"),
        "Hop": _u("Saaz", "Hallertau Mittelfrueh", "Hallertau Tradition",
                  "Tettnang / Tettnanger", "Perle", "East Kent Goldings",
                  "Fuggle", "Styrian Goldings", "Magnum"),
        "Yeast": _u(KVEIK, Y["farmhouse"], Y["saison"]),
        "Adjunct": _u("Juniper", "Honey", "Blueberries", "Raspberries",
                      "Blackberries", "Cranberries"),
        "Specialty": _u(RYE, OATS, ANCIENT, "Wheat, Flaked", "Wheat, Red",
                        "Barley, Raw", "Caramunich I", "Caravienne", "Carahell",
                        "Smoked Malt", "Aromatic", "Melanoiden Malt"),
    }

    # -- American ales ---------------------------------------------------------
    S["American Pale Ale / West Coast & Cold IPA"] = {
        "Base Malt": _u(PALE_US, "Maris Otter", "Golden Promise",
                        "English Pale Ale", "German Pale Ale",
                        "American (or North American) Pilsner", "German Pilsner",
                        "American (or North American) Vienna",
                        "Extra Light/Pils Malt Extract"),
        "Hop": _u(US_C, NZ_AUS, GER_NEW_WORLD),
        "Yeast": _u(Y["us"], Y["eastcoast"], Y["dryeng"], Y["gerlager"],
                    Y["nova"], Y["sh45"], Y["lutra"], Y["mexlager"], Y["voss"],
                    Y["kolsch"]),
        "Adjunct": _u(SUGARS_LAGER, "Grapefruit Peel", "Orange Peel, Sweet",
                      TERPENES, "Chili Peppers"),
        "Specialty": _u(CRYSTAL_LIGHT, DEXTRINE, "Carahell", "Caravienne",
                        "Victory", "Rice, Flaked", "Corn, Flaked", "Chit Malt",
                        "Wheat, Torrified"),
    }
    S["Hazy / New England IPA"] = {
        "Base Malt": _u("Pale Malt (2 Row)", "Maris Otter", "Golden Promise",
                        "English Pale Ale", "German Pale Ale",
                        "American (or North American) Pilsner", WHEAT_BASE,
                        "Pale Malt Extract", "Light Malt Extract",
                        "Extra Light/Pils Malt Extract"),
        "Hop": _u(HAZY),
        "Yeast": _u(Y["conan"], Y["verdant"], Y["london3"], Y["pomona"],
                    Y["hornindal"], Y["voss"], Y["english"], Y["us"],
                    Y["eastcoast"], Y["lutra"]),
        "Adjunct": _u(LACTOSE, "Vanilla (extract or bean)", TERPENES, SKYFARM,
                      "Mango", "Passion Fruit", "Pineapple", "Guava",
                      "Dragon Fruit", "Peaches", "Nectarines", "Coconut",
                      "Watermelon", "Orange Peel, Sweet", "Grapefruit Peel"),
        "Specialty": _u(OATS, WHEATS, "Chit Malt", DEXTRINE, "Honey Malt",
                        "Carahell", "Barley, Flaked", "Spelt Malt", "Rye Malt",
                        "Caramel/Crystal Malt - 10L",
                        "Caramel/Crystal Malt - 20L"),
    }
    S["American Amber / Red / Brown"] = {
        "Base Malt": _u("Pale Malt (2 Row)", "Pale Malt (6 Row)", "Maris Otter",
                        "American (or North American) Vienna", "German Vienna",
                        "Munich, Light", "Red X", "Pale Malt Extract",
                        "Amber Malt Extract", "Munich Malt Extract",
                        "Golden Malt Extract"),
        "Hop": _u("Cascade", "Centennial", "Columbus/Tomahawk/Zeus", "Simcoe",
                  "Amarillo", "Magnum", "Citra", "Mosaic", "Perle", "Liberty",
                  "Triumph", "Loral", US_CLASSIC),
        "Yeast": _u(Y["us"], Y["eastcoast"], Y["dryeng"], Y["kolsch"],
                    Y["sflager"], Y["voss"], Y["lutra"], Y["english"], Y["irish"]),
        "Adjunct": _u("Brown Sugar", "Maple Syrup", "Honey", "Cocoa Nibs/Beans",
                      "Coffee (Liquid or Beans)", "Coconut", "Pumpkin",
                      "Cinnamon"),
        "Specialty": _u("Caramel/Crystal Malt - 20L", "Caramel/Crystal Malt - 30L",
                        CRYSTAL_MID, "Caramel/Crystal Malt -120L", "Carared",
                        "Caramunich I", "Caramunich II", BISCUITY, "Brown Malt",
                        "Chocolate Malt", "Pale Chocolate Malt", "Carafa I",
                        "Carafa II", "Melanoiden Malt", "Aromatic", "Special B",
                        "Coffee Malt", RYE),
    }

    # -- British ales ----------------------------------------------------------
    S["Best Bitter / ESB / English Pale"] = {
        "Base Malt": _u(PALE_UK, "Pale Malt (2 Row)", "Golden Malt Extract",
                        "Light Malt Extract"),
        "Hop": _u(UK_HOPS, "Willamette"),
        "Yeast": _u(ENGLISH),
        "Adjunct": _u(SUGARS_UK_PALE),
        "Specialty": _u("Caramel/Crystal Malt - 20L", "Caramel/Crystal Malt - 30L",
                        CRYSTAL_MID, BISCUITY, "Wheat, Torrified",
                        "Barley, Torrefied", DEXTRINE, "Pale Chocolate Malt"),
    }
    S["Mild / Brown Ale"] = {
        "Base Malt": _u(PALE_UK, "Pale Malt (2 Row)", "Munich, Light",
                        "Dark Malt Extract", "Munich Malt Extract"),
        "Hop": _u(UK_HOPS, "Willamette"),
        "Yeast": _u(DARK_ALE_YEASTS),
        "Adjunct": _u(SUGARS_UK_DARK, "Cocoa Nibs/Beans",
                      "Coffee (Liquid or Beans)", "Licorice"),
        "Specialty": _u(CRYSTAL_MID, "Caramel/Crystal Malt -120L", "Special B",
                        "Brown Malt", "Chocolate Malt", "Pale Chocolate Malt",
                        "Black Malt", "Coffee Malt", BISCUITY, "Wheat, Torrified",
                        "Barley, Torrefied", "Carafa II", EXTRA_DARK, "Aromatic",
                        "Melanoiden Malt"),
    }
    S["English Strong / Barleywine / Wee Heavy"] = {
        "Base Malt": _u(PALE_UK, "Pale Malt (2 Row)", "Munich, Light",
                        "Light Malt Extract", "Golden Malt Extract",
                        "Dark Malt Extract"),
        "Hop": _u(UK_HOPS, "Magnum", "Columbus/Tomahawk/Zeus", "Cascade",
                  "Centennial", "Nugget", "Warrior", "Chinook", "Northern Brewer"),
        "Yeast": _u(DARK_ALE_YEASTS, Y["us"]),
        "Adjunct": _u(SUGARS_UK_DARK, "Turbinado Sugar", "Cane (or Beet) Sugar",
                      "Maple Syrup", "Licorice", "Oak"),
        "Specialty": _u(CRYSTAL_MID, "Caramel/Crystal Malt -120L", "Special B",
                        BISCUITY, "Brown Malt", SMOKED, "Barley, Roasted",
                        "Pale Chocolate Malt", "Chocolate Malt", "Aromatic",
                        "Melanoiden Malt", "Wheat, Torrified", EXTRA_DARK),
    }
    S["Porter (Brown / Robust / Baltic)"] = {
        "Base Malt": _u(PALE_UK, "Pale Malt (2 Row)", "Munich, Light",
                        "Munich, Dark", "German Pilsner", "Dark Malt Extract",
                        "Munich Malt Extract"),
        "Hop": _u(UK_HOPS, DARK_ALE_BITTER, "Hallertau Tradition", "Saaz"),
        "Yeast": _u(DARK_ALE_YEASTS, Y["us"], Y["gerlager"], Y["munichlager"],
                    Y["nova"]),
        "Adjunct": _u("Molasses", "Brown Sugar",
                      "Lyle's Golden Syrup (Invert Sugar)", "Licorice",
                      "Vanilla (extract or bean)", "Coffee (Liquid or Beans)",
                      "Cocoa Nibs/Beans", "Coconut", "Maple Syrup", "Oak",
                      "Pumpkin", "Cinnamon"),
        "Specialty": _u("Brown Malt", ROAST, CARAFA, CRYSTAL_MID,
                        "Caramel/Crystal Malt -120L", "Special B", BISCUITY,
                        SMOKED, "Oats, Flaked", "Barley, Flaked",
                        "Wheat, Torrified", EXTRA_DARK, "Caramunich II",
                        "Aromatic", "Melanoiden Malt"),
    }
    S["Stout (Dry / Oatmeal / Sweet / Foreign)"] = {
        "Base Malt": _u(PALE_UK, "Pale Malt (2 Row)", "Pale Malt (6 Row)",
                        "Munich, Light", "Dark Malt Extract",
                        "Light Malt Extract"),
        "Hop": _u(UK_HOPS, DARK_ALE_BITTER),
        "Yeast": _u(DARK_ALE_YEASTS, Y["us"]),
        "Adjunct": _u(LACTOSE, "Coffee (Liquid or Beans)", "Cocoa Nibs/Beans",
                      "Vanilla (extract or bean)", "Molasses",
                      "Lyle's Golden Syrup (Invert Sugar)", "Licorice", "Coconut"),
        "Specialty": _u(ROAST, BARLEY_ADJ, OATS, "Carafa II", "Carafa III",
                        "Wheat, Flaked", CRYSTAL_MID, "Caramel/Crystal Malt -120L",
                        "Special B", "Brown Malt", "Chit Malt", "Carafoam",
                        "Special Roast", "Victory", EXTRA_DARK),
    }
    S["Imperial / Pastry Stout"] = {
        "Base Malt": _u(PALE_UK, "Pale Malt (2 Row)", "Munich, Light",
                        "Munich, Dark", "Dark Malt Extract", "Light Malt Extract",
                        "Golden Malt Extract"),
        "Hop": _u(DARK_ALE_BITTER, "Simcoe", "Warrior", UK_HOPS),
        "Yeast": _u(DARK_ALE_YEASTS, Y["us"], Y["voss"]),
        "Adjunct": _u(DESSERT, "Chili Peppers", "Cherries", "Raspberries",
                      "Blackberries", "Blueberries", "Oak", "Licorice",
                      "Brown Sugar", "Turbinado Sugar", UMAMI,
                      "Candi Sugar, Dark", "Pumpkin"),
        "Specialty": _u(ROAST, CARAFA, CRYSTAL_DARK, "Brown Malt", OATS,
                        "Barley, Flaked", "Wheat, Flaked", "Special Roast",
                        SMOKED, EXTRA_DARK, "Caramunich II", "Aromatic",
                        "Melanoiden Malt", "Honey Malt",
                        "Cookie Malt (Viking Malt)"),
    }

    # -- Sours, fruit, specialty ------------------------------------------------
    S["Kettle Sour / Fruit Sour (Berliner / Gose)"] = {
        "Base Malt": _u("German Pilsner", "Floor Malted Pilsner",
                        "American (or North American) Pilsner", "Belgian Pilsner",
                        WHEAT_BASE, "Extra Light/Pils Malt Extract",
                        "Pale Malt (2 Row)"),
        "Hop": _u("Saaz", "Hallertau Mittelfrueh", "Hallertau Tradition",
                  "Tettnang / Tettnanger", "Perle", "Strisselspalt", "Liberty",
                  "Motueka", "Citra", "Galaxy", "Nelson Sauvin", "Mosaic"),
        "Yeast": _u(Y["philly"], Y["us"], Y["lutra"], Y["kolsch"], Y["hefe"],
                    Y["wit"], Y["voss"], Y["hornindal"], Y["nova"], Y["gerale"]),
        "Adjunct": _u(FRUIT_STONE_BERRY, FRUIT_TROPICAL, CITRUS, "Coriander",
                      LACTOSE, "Vanilla (extract or bean)", "Ginger", SKYFARM,
                      "Oak"),
        "Specialty": _u(WHEATS, "Barley, Raw", "Oats, Flaked", "Oats, Malted",
                        DEXTRINE, "Chit Malt", "Rice, Flaked", "Spelt Malt"),
    }
    S["American Wheat / Fruit & Spice Beer"] = {
        "Base Malt": _u(WHEAT_BASE, PALE_US, "American (or North American) Pilsner",
                        "Extra Light/Pils Malt Extract", "German Pale Ale"),
        "Hop": _u("Cascade", "Liberty", "Saaz", "Hallertau Mittelfrueh",
                  "Hallertau Tradition", "Tettnang / Tettnanger", "Perle",
                  "Lemondrop", "Citra", "Amarillo", "Motueka", "Mosaic",
                  "El Dorado", "Sorachi Ace", "Kohatu", "Rakau", "Ella", "Topaz",
                  "Mt Hood", "Willamette", "Cluster"),
        "Yeast": _u(Y["us"], Y["kolsch"], Y["lutra"], Y["voss"], Y["hornindal"],
                    Y["gerale"], Y["hefe"], Y["wit"], Y["eastcoast"], Y["nova"]),
        "Adjunct": _u(FRUIT_STONE_BERRY, FRUIT_TROPICAL, CITRUS, "Honey",
                      "Vanilla (extract or bean)", "Coriander",
                      "Grains of Paradise", "Juniper", "Chili Peppers",
                      SPICE_BAKING, SKYFARM, TERPENES),
        "Specialty": _u(WHEATS, OATS, DEXTRINE, "Carahell", "Honey Malt",
                        "Caramel/Crystal Malt - 10L", "Caramel/Crystal Malt - 20L",
                        CORN_RICE, ANCIENT, "Extract, Sorghum", "Chit Malt"),
    }
    S["Smoked & Wood-Aged (Rauchbier / Barrel)"] = {
        "Base Malt": _u("Munich, Light", "Munich, Dark", "German Vienna",
                        "American (or North American) Vienna", "German Pilsner",
                        "Maris Otter", "English Pale Ale", "Golden Promise",
                        "Pale Malt (2 Row)", "Munich Malt Extract", DARK_EXTRACT),
        "Hop": _u("Hallertau Mittelfrueh", "Hallertau Tradition",
                  "Tettnang / Tettnanger", "Perle", "Saaz", "Magnum",
                  "Northern Brewer", UK_HOPS),
        "Yeast": _u(Y["gerlager"], Y["munichlager"], Y["nova"], Y["sflager"],
                    Y["scottish"], Y["english"], Y["dryeng"], Y["irish"], Y["us"],
                    Y["trappist"], Y["abbey"], Y["alt"]),
        "Adjunct": _u("Oak", "Vanilla (extract or bean)",
                      "Coffee (Liquid or Beans)", "Cocoa Nibs/Beans",
                      "Maple Syrup", "Molasses", "Cherries", "Coconut",
                      "Candi Sugar, Dark", "Licorice", "Chili Peppers", UMAMI,
                      "Cinnamon"),
        "Specialty": _u(SMOKED, "Caramunich I", "Caramunich II", "Carahell",
                        "Caravienne", "Carared", CARAFA, "Melanoiden Malt",
                        "Aromatic", "Chocolate Malt", "Pale Chocolate Malt",
                        CRYSTAL_DARK, "Brown Malt", "Biscuit Malt",
                        "Special Roast", "Barley, Roasted", EXTRA_DARK),
    }

    for name, cats in S.items():
        assert list(cats) == CATEGORIES, f"{name}: category keys out of order"
        assert cats["Adjunct"], f"{name}: every style needs an Adjunct list"
    return S


# ---------------------------------------------------------------------------
# Validation against the ingredient sheet
# ---------------------------------------------------------------------------
def validate(styles: dict, sheet_path: str) -> tuple[list, list]:
    """Return (in_matrix_not_sheet, in_sheet_not_matrix) using draft_core."""
    import pandas as pd
    import draft_core

    df = pd.read_csv(sheet_path)
    rep = draft_core.validate_data(df, styles)
    return rep["in_matrix_not_sheet"], rep["in_sheet_not_matrix"]


def render(styles: dict) -> str:
    return json.dumps(styles, indent=2, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", default=None,
                    help="ingredient sheet CSV (default: league_config ingredients_path)")
    ap.add_argument("--out", default=os.path.join(ROOT, "style_matrix.json"))
    ap.add_argument("--check", action="store_true",
                    help="validate and compare with --out; exit 1 on drift")
    args = ap.parse_args(argv)

    if args.sheet is None:
        from config import load_league_config
        args.sheet = os.path.join(ROOT, load_league_config()["ingredients_path"])

    styles = build_styles()
    orphans, unmodeled = validate(styles, args.sheet)
    if orphans:
        print("ERROR: style matrix references ingredients not on the sheet:")
        for o in orphans:
            print("  ", o)
        return 1
    text = render(styles)
    n_ing = len({i for c in styles.values() for l in c.values() for i in l})
    print(f"{len(styles)} styles, {n_ing} distinct ingredients modeled, "
          f"{len(unmodeled)} sheet ingredients unmodeled")
    for u in unmodeled:
        print("   unmodeled:", u)

    if args.check:
        if not os.path.exists(args.out):
            print(f"ERROR: {args.out} missing")
            return 1
        with open(args.out, encoding="utf-8") as f:
            if f.read() != text:
                print(f"ERROR: {args.out} differs from generator output; re-run without --check")
                return 1
        print("OK: style_matrix.json is up to date")
        return 0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

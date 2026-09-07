"""Tests for ingredient profiles: load_descriptors / ingredient_profile /
validate_descriptors (display-only spec sheets from data/*_descriptors.csv)."""
import os

import pandas as pd
import pytest

import draft_core as dc
from config import DEFAULTS, load_league_config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Pure helper on small synthetic tables
# ---------------------------------------------------------------------------
SYN = {
    "Hop": {
        "Testra": {"name": "Testra", "origin": "US", "purpose": "dual", "alpha_mid": "9.0",
                   "alpha_lo": "8", "alpha_hi": "11", "citrus": "3", "tropical": "2",
                   "pine_resin": "2", "floral": "1", "earthy": "0"},
        "Bitterbomb": {"name": "Bitterbomb", "origin": "DE", "purpose": "bittering",
                       "alpha_mid": "14.0", "clean_bitter": "3", "citrus": "1"},
    },
    "Yeast": {
        "Belgian Test (WLP999, WY9999)": {
            "name": "Belgian Test (WLP999, WY9999)", "family": "belgian", "ferment_type": "ale",
            "attenuation_mid": "76", "flocculation": "2", "temp_mid_f": "68",
            "temp_lo_f": "66", "temp_hi_f": "70", "esters": "2", "phenols": "3",
            "clean_neutral": "0", "malt_forward": "1"},
    },
    "Base Malt": {
        "Crystal Test": {"name": "Crystal Test", "kind": "crystal", "grain": "barley",
                         "origin": "none", "color_L": "40", "diastatic": "0",
                         "caramel_toffee": "3", "dark_fruit": "2", "sweet": "2",
                         "bready": "0"},
    },
}
SYN["Specialty"] = SYN["Base Malt"]


def test_hop_profile():
    p = dc.ingredient_profile("Testra", SYN, "Hop")
    assert p["kind"] == "Hop" and p["usage"] == "Dual-purpose"
    assert p["alpha"] == "8–11% AA" and p["origin"] == "USA"
    assert p["notes"] == ["citrus", "tropical fruit", "pine/resin"]  # score>=2, best first, max 3
    assert p["summary"] == "Dual-purpose · 8–11% AA · citrus, tropical fruit, pine/resin"
    # No range columns -> falls back to the midpoint.
    q = dc.ingredient_profile("Bitterbomb", SYN)
    assert q["alpha"] == "14% AA" and q["usage"] == "Bittering" and q["notes"] == ["clean bittering"]


def test_yeast_profile_and_loose_name_match():
    p = dc.ingredient_profile("Belgian Test (WLP999, WY9999)", SYN, "Yeast")
    assert p["kind"] == "Yeast" and p["type"] == "Ale"
    assert p["attenuation"] == "76% attenuation" and p["temp"] == "66–70 °F"
    assert p["flocculation"] == "medium flocculation"
    assert p["notes"] == ["phenolic (clove/pepper)", "estery/fruity"]  # 3 before 2
    assert p["summary"].startswith("Ale · 76% attenuation · 66–70 °F · ")
    # Board label without the strain codes, different case, no category hint.
    assert dc.ingredient_profile("belgian test", SYN) == p


def test_malt_profile():
    p = dc.ingredient_profile("Crystal Test", SYN, "Specialty")
    assert p["kind"] == "Malt" and p["color"] == "40 °L"
    assert p["form"] == "crystal/caramel malt" and p["grain"] == "barley"
    assert p["diastatic"] is False and p["origin"] is None
    assert p["notes"] == ["caramel/toffee", "dark fruit", "sweet"]
    assert p["summary"] == "40 °L · crystal/caramel malt · caramel/toffee, dark fruit, sweet"


def test_unknown_ingredient_and_empty_tables():
    assert dc.ingredient_profile("Nope", SYN) is None
    assert dc.ingredient_profile("Testra", {}) is None
    assert dc.ingredient_profile("Testra", SYN, "Adjunct") is not None  # unknown hint -> search all


def test_notes_threshold_and_cap():
    row = {"name": "X", "purpose": "aroma", "alpha_mid": "5",
           "citrus": "3", "floral": "3", "herbal": "3", "spicy": "3", "earthy": "1"}
    p = dc.ingredient_profile("X", {"Hop": {"X": row}})
    assert len(p["notes"]) == dc.PROFILE_NOTE_MAX
    assert "earthy" not in p["notes"]           # below threshold
    assert p["notes"] == ["citrus", "floral", "herbal"]  # ties keep axis order


# ---------------------------------------------------------------------------
# Repo data: loader + coverage gate
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def descriptors():
    return dc.load_descriptors(load_league_config(os.path.join(HERE, "league_config.json")),
                               base_dir=HERE)


def test_load_descriptors_shape(descriptors):
    assert set(descriptors) == {"Hop", "Yeast", "Base Malt", "Specialty"}
    assert descriptors["Base Malt"] is descriptors["Specialty"]  # shared file, loaded once
    assert "Citra" in descriptors["Hop"] and "Maris Otter" in descriptors["Base Malt"]
    assert DEFAULTS["descriptor_files"]["Hop"] == "data/hop_descriptors.csv"


def test_load_descriptors_missing_file_is_empty(tmp_path):
    d = dc.load_descriptors({"descriptor_files": {"Hop": "nope.csv"}}, base_dir=str(tmp_path))
    assert d == {"Hop": {}}
    assert dc.ingredient_profile("Citra", d) is None


@pytest.mark.parametrize("sheet", ["ingredients_2026.csv", "ingredients_2025.csv"])
def test_every_sheet_hop_malt_yeast_has_a_profile(descriptors, sheet):
    """The coverage gate: a new-ingredient year cannot silently lose profiles."""
    df = pd.read_csv(os.path.join(HERE, sheet))
    cfg = load_league_config(os.path.join(HERE, "league_config.json"))
    gaps = dc.validate_descriptors(df, descriptors, cfg["category_aliases"])
    assert gaps == {}, gaps
    for col, cat in [("Hop", "Hop"), ("Yeast", "Yeast"), ("Base Malt", "Base Malt"),
                     ("Specialty Malt", "Specialty")]:
        for name in df[col].dropna():
            p = dc.ingredient_profile(name, descriptors, cat)
            assert p and p["summary"], name


def test_validate_descriptors_reports_gaps(descriptors):
    df = pd.DataFrame({"Hop": ["Citra", "Imaginary Hop"], "Yeast": ["Nope Yeast", None]})
    gaps = dc.validate_descriptors(df, descriptors,
                                   {"Hop": ["Hop"], "Yeast": ["Yeast"]})
    assert gaps == {"Hop": ["Imaginary Hop"], "Yeast": ["Nope Yeast"]}


def test_real_profiles_read_sensibly(descriptors):
    citra = dc.ingredient_profile("Citra", descriptors, "Hop")
    assert citra["usage"] == "Aroma" and citra["alpha"] == "11–13% AA"
    assert citra["notes"][:2] == ["citrus", "tropical fruit"]
    hefe = dc.ingredient_profile("Hefewiezen (WLP300, WY3068, G01)", descriptors, "Yeast")
    assert hefe["type"] == "Ale" and hefe["temp"] == "64–72 °F"
    assert "phenolic (clove/pepper)" in hefe["notes"]
    lager = dc.ingredient_profile("German Lager (WLP830, WY2124, L13, 34/70)", descriptors, "Yeast")
    assert lager["type"] == "Lager" and lager["temp"] == "48–56 °F"
    choc = dc.ingredient_profile("Chocolate Malt", descriptors, "Specialty")
    assert choc["color"] == "350 °L" and choc["form"] == "roasted malt"
    assert choc["notes"][0] == "roast/coffee/chocolate"
    assert dc.ingredient_profile("Honey", descriptors, "Adjunct") is None  # no adjunct table

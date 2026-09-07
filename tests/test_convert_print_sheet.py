"""Tests for the 2026 print-sheet -> clean-schema converter."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

import convert_print_sheet as cps  # noqa: E402

PRINT_SHEET = os.path.join(
    HERE,
    "RHBC Fantasy Brewing Draft Ingredients - Print Version 2026 - 2026 Ingredients.csv",
)


def test_parse_counts_match_expected():
    cats = cps.apply_corrections(cps.parse_print_sheet(PRINT_SHEET))
    assert {k: len(v) for k, v in cats.items()} == {
        "Base Malt": 25,
        "Hop": 44,
        "Yeast": 30,
        "Adjunct": 47,
        "Specialty Malt": 58,
    }


def test_corrections_applied():
    cats = cps.apply_corrections(cps.parse_print_sheet(PRINT_SHEET))
    base, adjunct, specialty = cats["Base Malt"], cats["Adjunct"], cats["Specialty Malt"]
    # Typos fixed to their canonical spellings.
    assert "Maris Otter" in base and "Marris Otter" not in base
    assert "Demerara Sugar" in adjunct and "Dememera Sugar" not in adjunct
    # Rice Syrup normalized to the long canonical form used by style_matrix.
    assert any(a.startswith("Rice Syrup (") for a in adjunct)
    assert "Rice Syrup" not in adjunct
    # Duplicate Licorice dropped from Specialty (kept only as an Adjunct).
    assert "Licorice" not in specialty
    assert "Licorice" in adjunct


def test_net_new_2026_ingredients_present():
    cats = cps.apply_corrections(cps.parse_print_sheet(PRINT_SHEET))
    assert "Columbus/Tomahawk/Zeus" in cats["Hop"]
    assert any("Saflager SH-45" in y for y in cats["Yeast"])
    assert any("Abstrax terpenes" in a for a in cats["Adjunct"])


def test_write_clean_csv_roundtrips(tmp_path):
    import csv

    out = tmp_path / "ingredients_out.csv"
    cats = cps.convert(PRINT_SHEET, str(out))
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == cps.CLEAN_HEADER
    # Each category column holds exactly its ingredients (order preserved).
    for ci, cat in enumerate(cps.CLEAN_HEADER):
        col = [r[ci] for r in rows[1:] if ci < len(r) and r[ci]]
        assert col == cats[cat]

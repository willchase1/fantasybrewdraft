"""Convert the wide "print version" ingredient sheet into the clean 5-column
schema the app reads (matching ``ingredients_2025.csv``).

The print sheet is a presentation export, not a data file: a category-banner
header row, a "Choose at least 1 / Drafted By:" label row, interleaved empty
"Drafted By:" columns, category spillover across two physical columns (hops,
adjuncts, specialty), and trailing prose notes (allergy/manufacturer). Rather
than teach ``draft_core.load_data`` to parse this one year's spreadsheet
formatting, we convert it once, offline, and commit the clean output so the
loader keeps reading a stable schema and the characterization tests stay valid.

Usage:
    python scripts/convert_print_sheet.py \
        "RHBC Fantasy Brewing Draft Ingredients - Print Version 2026 - 2026 Ingredients.csv" \
        ingredients_2026.csv
"""
import csv
import sys

# Physical column indices in the print sheet that hold ingredient values,
# grouped by the clean-schema category they map to. Verified against the 2026
# sheet: row 0 holds category banners, row 1 holds "Choose at least 1 /
# Drafted By:" labels, and the odd "Drafted By:" columns (1, 4, 6, 9, 11, 13)
# and trailing note columns (>=14) carry no draftable ingredients.
PRINT_COLUMN_MAP = {
    "Base Malt": [0],
    "Hop": [2, 3],
    "Yeast": [5],
    "Adjunct": [7, 8],
    "Specialty Malt": [10, 12],
}

CLEAN_HEADER = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty Malt"]

DATA_START_ROW = 2  # rows 0-1 are the two header rows

# Known data-entry issues in the 2026 print sheet, kept explicit and reviewable
# so a future year is a data edit rather than a code change. Renames map the
# printed string -> the canonical spelling used by the derived data files
# (style_matrix.json, ingredient_scarcity.json) and the 2025 sheet; an
# uncorrected typo would orphan the style-matrix entry that references it.
CORRECTIONS = {
    "Marris Otter": "Maris Otter",
    "Dememera Sugar": "Demerara Sugar",
    # The 2026 sheet shortened this to "Rice Syrup"; the canonical name used by
    # the 2025 sheet and style_matrix.json is the long form. Normalize to it so
    # the style that references it stays satisfiable in both years.
    "Rice Syrup": (
        'Rice Syrup (added "syrup" to differentiate from flaked rice '
        "in specialty malts)"
    ),
}

# Ingredients to drop from a specific category (data-entry duplicates). Licorice
# is an Adjunct (and appears in the Adjunct column); it was also mistakenly
# listed under Specialty. Keep only the Adjunct copy.
DROP = {
    "Specialty Malt": {"Licorice"},
}


def parse_print_sheet(path):
    """Return ``{category: [ingredient, ...]}`` from the wide print sheet.

    De-duplicates within a category while preserving sheet order.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    width = max((len(r) for r in rows), default=0)
    cats = {cat: [] for cat in CLEAN_HEADER}
    seen = {cat: set() for cat in CLEAN_HEADER}
    for row in rows[DATA_START_ROW:]:
        padded = row + [""] * (width - len(row))  # right-pad ragged rows
        for cat, cols in PRINT_COLUMN_MAP.items():
            for ci in cols:
                val = padded[ci].strip()
                if val and val not in seen[cat]:
                    seen[cat].add(val)
                    cats[cat].append(val)
    return cats


def apply_corrections(cats, corrections=CORRECTIONS, drop=DROP):
    """Apply the typo/rename map and per-category drops, de-duplicating."""
    out = {}
    for cat, items in cats.items():
        dropset = (drop or {}).get(cat, set())
        fixed, seen = [], set()
        for it in items:
            it = (corrections or {}).get(it, it)
            if it in dropset or it in seen:
                continue
            seen.add(it)
            fixed.append(it)
        out[cat] = fixed
    return out


def write_clean_csv(cats, out_path):
    """Emit the padded 5-column clean schema (same shape as 2025)."""
    n = max((len(v) for v in cats.values()), default=0)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        for i in range(n):
            w.writerow([cats[cat][i] if i < len(cats[cat]) else ""
                        for cat in CLEAN_HEADER])


def convert(print_csv_path, out_path="ingredients_2026.csv"):
    """Full pipeline: parse -> correct -> write. Returns the cleaned dict."""
    cats = apply_corrections(parse_print_sheet(print_csv_path))
    write_clean_csv(cats, out_path)
    return cats


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    print_csv_path = argv[0]
    out_path = argv[1] if len(argv) > 1 else "ingredients_2026.csv"
    cats = convert(print_csv_path, out_path)
    print(f"Wrote {out_path}:")
    for cat, items in cats.items():
        print(f"  {cat}: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

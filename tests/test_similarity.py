"""Tests for the similarity generator (scripts/build_similarity.py), the
committed similarity files, and the engine's use of them."""
import csv
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

import build_similarity as bs  # noqa: E402
import draft_core as dc  # noqa: E402

KINDS = ["hop", "yeast", "malt"]


def _matrix(path):
    rows = list(csv.reader(open(path, newline="", encoding="utf-8")))
    names = rows[0][1:]
    m = {a: {b: float(x) for b, x in zip(names, r[1:])} for a, r in zip(names, rows[1:])}
    return names, m


# ---------------------------------------------------------------------------
# Generator behaviour on a tiny synthetic table
# ---------------------------------------------------------------------------
def test_cosine_bounds_and_identity():
    assert bs.cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert bs.cosine([1, 0], [0, 1]) == 0.0
    assert bs.cosine([0, 0], [1, 1]) == 0.0


def test_vectorize_onehot_and_continuous():
    rows = [
        {"name": "A", "origin": "US", "x": "3", "alpha_mid": "10"},
        {"name": "B", "origin": "DE", "x": "3", "alpha_mid": "10"},
    ]
    spec = {"categorical": {"origin": 0.5}, "continuous": {"alpha_mid": (bs._lin(0, 20), 1.0)},
            "ignore": []}
    names, vecs = bs.vectorize(rows, spec)
    assert names == ["A", "B"]
    # descriptor x, then one-hot [DE, US] * 0.5, then alpha scaled (10/20*3 = 1.5)
    assert vecs[0] == [3.0, 0.0, 0.5, 1.5]
    assert vecs[1] == [3.0, 0.5, 0.0, 1.5]


def test_rebase_maps_median_to_zero_and_keeps_diagonal():
    m = [[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]]
    out, med = bs.rebase(m)
    assert med == 0.5
    assert out[0][0] == out[1][1] == out[2][2] == 1.0
    assert out[1][2] == 0.0                     # the median pair
    assert out[0][1] == pytest.approx(0.6)      # (0.8-0.5)/(0.5)
    assert out[0][2] == 0.0                     # below median clamps to 0
    assert out[0][1] == out[1][0]               # symmetric


def test_neighbours_sorted_filtered_and_capped():
    names = ["A", "B", "C", "D"]
    m = [[1, .9, .1, .5], [.9, 1, .3, .2], [.1, .3, 1, .0], [.5, .2, .0, 1]]
    nb = bs.neighbours(names, m, k=2, min_score=0.15)
    assert [r["ingredient"] for r in nb["A"]] == ["B", "D"]
    assert nb["A"][0]["score"] == 0.9
    assert [r["ingredient"] for r in nb["C"]] == ["B"]   # 0.1 and 0.0 filtered
    assert nb["D"][0]["ingredient"] == "A"


def test_generator_is_deterministic(tmp_path):
    a = bs.build("hop", 8, 0.15, None)
    b = bs.build("hop", 8, 0.15, None)
    assert a["json"] == b["json"] and a["csv"] == b["csv"]


def test_descriptor_tables_have_no_duplicate_names():
    for kind in KINDS:
        rows = bs.load_table(os.path.join(bs.DATA, bs.SPECS[kind]["csv"]))
        assert len({r["name"] for r in rows}) == len(rows)


# ---------------------------------------------------------------------------
# Committed data: coverage, shape, symmetry, sanity
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sheet_path():
    from config import load_league_config
    cfg = load_league_config(os.path.join(HERE, "league_config.json"))
    return os.path.join(HERE, cfg["ingredients_path"])


@pytest.mark.parametrize("kind", KINDS)
def test_committed_files_match_generator_and_cover_sheet(kind, sheet_path):
    res = bs.build(kind, 8, 0.15, sheet_path)
    assert res["problems"] == [], res["problems"]
    base = os.path.join(HERE, bs.SPECS[kind]["out"])
    assert open(base + ".json", encoding="utf-8").read() == res["json"]
    assert open(base + "_matrix.csv", encoding="utf-8").read() == res["csv"]


@pytest.mark.parametrize("kind", KINDS)
def test_similarity_json_shape_and_range(kind):
    data = json.load(open(os.path.join(HERE, f"{kind}_similarity.json")))
    assert isinstance(data, dict) and data
    for name, recs in data.items():
        assert isinstance(recs, list) and len(recs) <= 8
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)
        for r in recs:
            assert set(r) == {"ingredient", "score"}
            assert 0.0 <= r["score"] <= 1.0
            assert r["ingredient"] != name
            assert r["ingredient"] in data  # neighbours are themselves on the sheet


@pytest.mark.parametrize("kind", KINDS)
def test_similarity_matrix_symmetric_unit_diagonal(kind):
    names, m = _matrix(os.path.join(HERE, f"{kind}_similarity_matrix.csv"))
    for a in names:
        assert m[a][a] == 1.0
        for b in names:
            assert m[a][b] == m[b][a]
            assert 0.0 <= m[a][b] <= 1.0


@pytest.mark.parametrize("a,b,c", [
    # (similar pair) should beat (dissimilar pair): a~b > a~c
    ("Cascade", "Centennial", "Saaz"),
    ("Citra", "Galaxy", "East Kent Goldings"),
    ("Saaz", "Hallertau Mittelfrueh", "Columbus/Tomahawk/Zeus"),
    ("East Kent Goldings", "Fuggle", "Mosaic"),
])
def test_hop_known_pairs(a, b, c):
    _, m = _matrix(os.path.join(HERE, "hop_similarity_matrix.csv"))
    assert m[a][b] > m[a][c]
    assert m[a][b] >= dc.SIMILARITY_CLOSE_THRESHOLD


@pytest.mark.parametrize("a,b,c", [
    ("German Lager (WLP830, WY2124, L13, 34/70)", "Czech Lager (WLP802, WY2001, L28)",
     "Hefewiezen (WLP300, WY3068, G01)"),
    ("Saison (WLP565, WY3724)", "Farmhouse (WY3726, OYL-217, Dry)",
     "American Ale (WLP001 [liquid or dry], WY1056, US-05, A07)"),
    ("Voss Kveik (A43, OYL-061, Dry)", "Hornindal Kveik (WLP521, OYL-091)",
     "Munich Lager (WLP838, WY2308, Dry)"),
    ("Conan (WLP095, OYL-052, A04, LalBrew New England [Dry])", "Verdant IPA (Dry)",
     "Czech Lager (WLP802, WY2001, L28)"),
])
def test_yeast_known_pairs(a, b, c):
    _, m = _matrix(os.path.join(HERE, "yeast_similarity_matrix.csv"))
    assert m[a][b] > m[a][c]
    assert m[a][b] >= dc.SIMILARITY_CLOSE_THRESHOLD


def test_sour_yeast_has_no_close_substitute():
    data = json.load(open(os.path.join(HERE, "yeast_similarity.json")))
    recs = data["Philly Sour or Sourvisiae (Dry)"]
    assert all(r["score"] < dc.SIMILARITY_CLOSE_THRESHOLD for r in recs)


@pytest.mark.parametrize("a,b,c", [
    ("Maris Otter", "Golden Promise", "Black Malt"),
    ("Caramel/Crystal Malt - 40L", "Caramel/Crystal Malt - 60L", "Oats, Flaked"),
    ("Chocolate Malt", "Pale Chocolate Malt", "Carahell"),
    ("Oats, Flaked", "Wheat, Flaked", "Special B"),
    ("Munich, Light", "German Vienna", "Rice, Flaked"),
    ("Smoked Malt", "Peat Smoked Malt", "Caramel/Crystal Malt - 10L"),
])
def test_malt_known_pairs(a, b, c):
    _, m = _matrix(os.path.join(HERE, "malt_similarity_matrix.csv"))
    assert m[a][b] > m[a][c]
    assert m[a][b] >= dc.SIMILARITY_CLOSE_THRESHOLD


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------
def test_load_similarity_merges_configured_files():
    sim = dc.load_similarity(base_dir=HERE)
    assert "Citra" in sim                    # hop
    assert "Maris Otter" in sim              # base malt
    assert "Special B" in sim                # specialty
    assert "Saison (WLP565, WY3724)" in sim  # yeast
    assert dc._sim_name(sim["Citra"][0]) in sim


def test_load_similarity_tolerates_missing_files(tmp_path):
    cfg = {"similarity_files": {"Hop": "nope.json"}}
    assert dc.load_similarity(cfg, base_dir=str(tmp_path)) == {}


def test_sim_name_accepts_legacy_hop_key():
    assert dc._sim_name({"hop": "Citra", "score": 0.9}) == "Citra"
    assert dc._sim_name({"ingredient": "Citra", "score": 0.9}) == "Citra"


def test_dynamic_scarcity_substitutes_discount_any_category():
    i2c = {"Y1": "Yeast", "Y2": "Yeast", "Y3": "Yeast", "M1": "Base Malt",
           "M2": "Base Malt", "M3": "Base Malt"}
    sim = {"Y1": [{"ingredient": "Y2", "score": 0.8}, {"ingredient": "Y3", "score": 0.7}],
           "M1": [{"ingredient": "M2", "score": 0.1}]}  # M2 too dissimilar to count
    avail = set(i2c)
    raw = dc.compute_dynamic_scarcity(avail, i2c, {"Yeast", "Base Malt"}, 2, similarity=sim)
    assert raw["Y1"] < raw["Y2"]           # Y1 has two close analogs still on the board
    assert raw["M1"] == raw["M2"] == raw["M3"]
    # once Y2/Y3 are gone, Y1's discount disappears
    raw2 = dc.compute_dynamic_scarcity({"Y1", "M1"}, i2c, {"Yeast", "Base Malt"}, 2,
                                       similarity=sim)
    assert raw2["Y1"] > raw["Y1"]


def test_next_best_picks_accepts_similarity_and_legacy_hop_similarity(tmp_path):
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        ings, sm, sc, _, sb, i2c = dc.load_data(ingredients_path="ingredients_2026.csv")
        sim = dc.load_similarity()
    finally:
        os.chdir(cwd)
    req = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}
    a = dc.next_best_picks(["Citra"], ["Citra"], ings, sm, sc, req, 3, i2c, sb,
                           top_k=1000, similarity=sim)
    legacy = {k: v for k, v in sim.items() if i2c.get(k) == "Hop"}
    b = dc.next_best_picks(["Citra"], ["Citra"], ings, sm, sc, req, 3, i2c, sb,
                           top_k=1000, hop_similarity=legacy)
    # Hop synergy identical either way; a Citra-like hop gets synergy, Saaz does not.
    sa = a.set_index("Ingredient")["Synergy"]
    sb_ = b.set_index("Ingredient")["Synergy"]
    assert sa["Galaxy"] == sb_["Galaxy"] > 0
    assert sa["Saaz"] == 0.0
    # Yeast scarcity is discounted only when the merged lookup is supplied.
    assert a.set_index("Ingredient")["Scarcity"]["Czech Lager (WLP802, WY2001, L28)"] \
        <= b.set_index("Ingredient")["Scarcity"]["Czech Lager (WLP802, WY2001, L28)"]

"""Tests for load_data hardening and the data validator."""
import json
import os

import pandas as pd
import pytest

import draft_core as dc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def repo_data():
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        return dc.load_data()
    finally:
        os.chdir(cwd)


def test_repo_data_has_no_orphan_style_ingredients(repo_data):
    """The load-time invariant: every ingredient a style requires must be
    draftable from the sheet. This is the safety check the plan calls for."""
    ingredients, style_matrix = repo_data[0], repo_data[1]
    report = dc.validate_data(ingredients, style_matrix)
    assert report["in_matrix_not_sheet"] == []


def test_validate_data_detects_missing_ingredient():
    df = pd.DataFrame({"Hop": ["Cascade", "Citra"]})
    style_matrix = {"Fake IPA": {"Hop": ["Cascade", "Nonexistent Hop"]}}
    report = dc.validate_data(df, style_matrix)
    assert "Nonexistent Hop" in report["in_matrix_not_sheet"]
    assert "Citra" in report["in_sheet_not_matrix"]


def test_load_data_missing_required_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dc.load_data(ingredients_path=str(tmp_path / "nope.csv"))


def test_load_required_json_malformed_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json ")
    with pytest.raises(ValueError):
        dc._load_required_json(str(p), "test")


def test_load_optional_json_missing_returns_none(tmp_path):
    assert dc._load_optional_json(str(tmp_path / "nope.json")) is None


def test_load_optional_json_malformed_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json ")
    assert dc._load_optional_json(str(p)) is None

"""Single source of truth for league configuration.

Draft rules (rounds, flex slots, required categories) and the category alias
lists used to read the ingredient sheet all live in ``league_config.json``.
Every app and tool loads them from here instead of hardcoding — so a new
season or a differently-shaped league is a data edit, not a code change.

The DEFAULTS below intentionally mirror the historical hardcoded values, so
if ``league_config.json`` is missing or partial, behavior is unchanged.
"""
import json
import os

DEFAULTS = {
    "rounds": 7,
    "flex_slots": 3,
    "required_categories": {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1},
    "category_aliases": {
        "Base Malt": ["Base Malt", "Base Malts", "Base Malts and Extracts"],
        "Hop": ["Hop", "Hops"],
        "Yeast": ["Yeast", "Yeasts"],
        "Adjunct": ["Adjunct", "Adjuncts", "Adjuncts/Spices/Fruits"],
        "Specialty": [
            "Specialty", "Specialty Malt", "Specialty Malts",
            "Specialty Malt and Flaked Grains", "Specialty Malts and Flaked Grains",
            "Flaked Grains", "Flaked/Other Grains",
        ],
        "Extra": ["Extra", "Extras"],
    },
}

DEFAULT_CONFIG_PATH = "league_config.json"


def load_league_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Return league config, overlaying values from ``path`` onto DEFAULTS.

    A missing or malformed file falls back to DEFAULTS (per key), so partial
    configs are fine and a corrupt file never crashes the app.
    """
    cfg = {k: (v.copy() if isinstance(v, (dict, list)) else v)
           for k, v in DEFAULTS.items()}
    if os.path.exists(path):
        try:
            with open(path) as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                cfg.update(overrides)
        except (json.JSONDecodeError, OSError):
            pass  # keep DEFAULTS
    return cfg

import pandas as pd
import json
import os
from collections import defaultdict, Counter

from config import load_league_config


def _load_required_json(path, label):
    """Load a required JSON file, raising a clear error if missing/malformed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required {label} file not found: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse {label} file {path}: {e}") from e


def _load_optional_json(path):
    """Load an optional JSON file; return None if absent or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_data(
    ingredients_path: str = "ingredients_2025.csv",
    style_matrix_path: str = "style_matrix.json",
    scarcity_path: str = "ingredient_scarcity.json",
    opponent_model_path: str = "opponent_model.json",
    style_bias_path: str = "style_bias.json",
):
    """Load core data files used by both the tracker and mobile tools.

    Returns a tuple of:
        ingredients_df, style_matrix, scarcity_df, opponent_model,
        style_bias, ingredient_to_category
    """
    # Required files: fail loud and clear rather than with a raw traceback.
    if not os.path.exists(ingredients_path):
        raise FileNotFoundError(f"Required ingredients file not found: {ingredients_path}")
    ingredients = pd.read_csv(ingredients_path)
    style_matrix = _load_required_json(style_matrix_path, "style matrix")
    scarcity = pd.DataFrame(_load_required_json(scarcity_path, "ingredient scarcity"))

    # Optional files: absent or malformed degrades to None (feature off).
    opponent_model = _load_optional_json(opponent_model_path)
    style_bias = _load_optional_json(style_bias_path)
    ingredient_to_category = {}
    for style, cats in style_matrix.items():
        for cat, ing_list in cats.items():
            for ing in ing_list:
                ingredient_to_category[ing] = cat
    return (
        ingredients,
        style_matrix,
        scarcity,
        opponent_model,
        style_bias,
        ingredient_to_category,
    )


def build_available_set(ingredients_df, category_aliases=None):
    """Return the set of ingredient strings that are available to draft.

    ``category_aliases`` maps a category to the sheet column names that may
    hold it; defaults to the league config so callers don't hardcode it.
    """
    if category_aliases is None:
        category_aliases = load_league_config()["category_aliases"]

    def extract(alias_list):
        vals = []
        for colname in alias_list:
            if colname in ingredients_df.columns:
                vals.extend(ingredients_df[colname].dropna().unique().tolist())
        return set(vals)

    avail = set()
    for aliases in category_aliases.values():
        avail |= extract(aliases)
    return avail


def validate_data(ingredients_df, style_matrix, category_aliases=None):
    """Cross-check the style matrix against the ingredient sheet.

    Returns a report dict:
      - ``in_matrix_not_sheet``: ingredients a style requires but that are not
        draftable from the sheet. These are the dangerous ones — a style that
        references them can never be fully satisfied, silently skewing style
        viability and recommendations.
      - ``in_sheet_not_matrix``: draftable ingredients that no style uses
        (informational; e.g. an adjunct not modeled in any style).
    Empty lists mean the two sources are consistent.
    """
    available = build_available_set(ingredients_df, category_aliases)
    matrix_ings = set()
    for cats in style_matrix.values():
        for ings in cats.values():
            matrix_ings.update(ings)
    return {
        "in_matrix_not_sheet": sorted(matrix_ings - available),
        "in_sheet_not_matrix": sorted(available - matrix_ings),
    }


def ingredient_style_bias(ingredient, style_matrix, style_bias):
    if not style_bias:
        return 1.0
    bias_factor = 1.0
    for family, data in style_bias.items():
        styles = data.get("styles", [])
        weight = data.get("weight", 1.0)
        for style in styles:
            if style in style_matrix:
                for cat, ings in style_matrix[style].items():
                    if ingredient in ings:
                        bias_factor = max(bias_factor, weight)
    return bias_factor


def bucket_for_rules(category_label: str) -> str:
    if category_label == "Base Malt":
        return "Malt"
    if category_label == "Hop":
        return "Hop"
    if category_label == "Yeast":
        return "Yeast"
    if category_label == "Adjunct":
        return "Adjunct"
    return "Flex"


def snake_draft_order(num_players, current_round):
    """Seat order (0-indexed) for a round in a snake draft.

    Round 1 (and every odd round) runs ascending 0..N-1; even rounds reverse.
    """
    if current_round % 2 == 1:
        return list(range(num_players))
    return list(range(num_players - 1, -1, -1))


def pick_slot(overall_pick, num_players):
    """Map a 1-indexed overall pick number to (round, seat_index_0based).

    This is the single source of truth for the snake-draft math that the
    tracker UI uses to decide whose turn it is.
    """
    current_round = ((overall_pick - 1) // num_players) + 1
    order = snake_draft_order(num_players, current_round)
    idx_in_order = (overall_pick - 1) % num_players
    return current_round, order[idx_in_order]


def compute_rules_status(my_picks, ingredient_to_category, total_picks, config=None):
    if config is None:
        config = load_league_config()
    counts = {"Malt": 0, "Hop": 0, "Yeast": 0, "Adjunct": 0, "Flex": 0}
    for ing in my_picks:
        ui_cat = ingredient_to_category.get(ing, "Specialty")
        bucket = bucket_for_rules(ui_cat)
        counts[bucket] += 1

    # Required categories come from config (as category names); map them onto
    # rule buckets so e.g. "Base Malt" -> "Malt".
    required_min = {}
    for cat, n in config["required_categories"].items():
        required_min[bucket_for_rules(cat)] = n
    required_met = {k: counts[k] >= v for k, v in required_min.items()}
    required_remaining = {k: max(0, v - counts[k]) for k, v in required_min.items()}

    satisfied_core = sum(min(counts[k], 1) for k in required_min.keys())
    flex_used = max(0, len(my_picks) - satisfied_core)
    flex_remaining = max(0, config.get("flex_slots", 3) - flex_used)

    picks_remaining = max(0, total_picks - len(my_picks))

    required_slots_left = sum(required_remaining.values())
    feasible = required_slots_left <= picks_remaining

    status = {
        "counts": counts,
        "required_met": required_met,
        "required_remaining": required_remaining,
        "flex_used": flex_used,
        "flex_remaining": flex_remaining,
        "picks_remaining": picks_remaining,
        "required_slots_left": required_slots_left,
        "feasible": feasible,
    }
    return status


def compute_style_status(my_picks, drafted, style_matrix, required, flex_slots):
    status = []
    drafted_set = set(drafted)
    my_set = set(my_picks)

    for style, cats in style_matrix.items():
        cat_choices_remaining = {}
        for cat, ing_list in cats.items():
            remaining_ing = [ing for ing in ing_list if ing not in drafted_set or ing in my_set]
            cat_choices_remaining[cat] = len(remaining_ing)

        satisfied = sum(any(ing in my_set for ing in ings) for cat, ings in cats.items())
        options = sum(1 for k, v in cat_choices_remaining.items() if v > 0)
        score = satisfied * 2 + options
        status.append(
            {
                "Style": style,
                "Satisfied Categories": satisfied,
                "Categories with Options Left": options,
                "Score": score,
            }
        )
    df = pd.DataFrame(status).sort_values(by=["Score", "Satisfied Categories"], ascending=False)
    return df


def next_best_picks(
    my_picks,
    drafted,
    ingredients,
    style_matrix,
    scarcity_df,
    required,
    flex_slots,
    ingredient_to_category,
    style_bias,
    early_signal=None,
    bias_weight: float = 0.0,
    top_k: int = 15,
):
    drafted_set = set(drafted)
    my_set = set(my_picks)
    available_set = build_available_set(ingredients)

    cand = []
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing in available_set and ing not in drafted_set:
                    cand.append((ing, cat, style))
    if not cand:
        return pd.DataFrame(
            columns=[
                "Ingredient",
                "Category",
                "Style Coverage",
                "Scarcity",
                "Popularity",
                "Bias Factor",
                "Pick Value",
            ]
        )

    coverage = defaultdict(set)
    for ing, cat, style in cand:
        coverage[ing].add(style)

    sc = scarcity_df.set_index("Ingredient") if not scarcity_df.empty else pd.DataFrame()
    rows = []
    have_counts = Counter([ingredient_to_category.get(i, "Unknown") for i in my_set])
    cat_need_factor = {}
    for cat in ["Yeast", "Hop", "Adjunct", "Base Malt", "Specialty"]:
        required_min = required.get(cat, 0)
        have = have_counts.get(cat, 0)
        cat_need_factor[cat] = 1.5 if have < required_min else 1.0

    for ing, styles in coverage.items():
        cat = ingredient_to_category.get(ing, "Unknown")
        style_cov = len(styles)
        if not sc.empty and ing in sc.index and "Scarcity Score" in sc.columns:
            scarcity = float(sc.loc[ing]["Scarcity Score"])
        else:
            scarcity = 1.0 / max(style_cov, 1)
        need_bonus = cat_need_factor.get(cat, 1.0)
        popularity = float(early_signal.get(ing, 0.0)) if early_signal else 0.0
        bias_factor = ingredient_style_bias(ing, style_matrix, style_bias)
        # Favor style coverage more heavily than ingredient scarcity
        pick_value = (
            (style_cov * 2.0 + scarcity * 1.0 + popularity * bias_weight)
            * need_bonus
            * bias_factor
        )
        rows.append(
            {
                "Ingredient": ing,
                "Category": cat,
                "Style Coverage": style_cov,
                "Scarcity": round(scarcity, 3),
                "Popularity": round(popularity, 3),
                "Bias Factor": round(bias_factor, 2),
                "Pick Value": round(pick_value, 3),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["Pick Value", "Bias Factor", "Scarcity", "Style Coverage"],
        ascending=[False, False, False, False],
    )
    return df.head(top_k)


def block_picks(
    drafted,
    my_picks,
    pair_lookup,
    ingredients,
    early_signal=None,
    top_k: int = 15,
):
    available_set = build_available_set(ingredients)
    opp_picks = [d for d in drafted if d not in set(my_picks)]
    recent = list(reversed(opp_picks))[:3]
    suggestions = defaultdict(int)

    for ing in recent:
        for (a, b), cnt in pair_lookup.items():
            if a == ing and b in available_set and b not in set(drafted):
                suggestions[b] += cnt

    remaining = []
    for ing, score in suggestions.items():
        popularity = float(early_signal.get(ing, 0.0)) if early_signal else 0.0
        remaining.append((ing, score, popularity))
    if not remaining:
        return pd.DataFrame(columns=["Ingredient", "Block Score", "Popularity Cue"])

    df = pd.DataFrame(remaining, columns=["Ingredient", "Block Score", "Popularity Cue"])
    df = df.sort_values(["Block Score", "Popularity Cue"], ascending=[False, False]).head(top_k)
    return df

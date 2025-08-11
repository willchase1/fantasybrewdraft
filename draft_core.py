import pandas as pd
import json
import os
from collections import defaultdict, Counter


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
    ingredients = pd.read_csv(ingredients_path)
    with open(style_matrix_path) as f:
        style_matrix = json.load(f)
    with open(scarcity_path) as f:
        scarcity = pd.DataFrame(json.load(f))
    opponent_model = None
    if os.path.exists(opponent_model_path):
        try:
            with open(opponent_model_path) as f:
                opponent_model = json.load(f)
        except Exception:
            opponent_model = None
    style_bias = None
    if os.path.exists(style_bias_path):
        try:
            with open(style_bias_path) as f:
                style_bias = json.load(f)
        except Exception:
            style_bias = None
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


def build_available_set(ingredients_df):
    """Return the set of ingredient strings that are available to draft."""
    base_aliases = ["Base Malt", "Base Malts", "Base Malts and Extracts"]
    hop_aliases = ["Hop", "Hops"]
    yeast_aliases = ["Yeast", "Yeasts"]
    adjunct_aliases = ["Adjunct", "Adjuncts", "Adjuncts/Spices/Fruits"]
    specialty_aliases = [
        "Specialty", "Specialty Malt", "Specialty Malts",
        "Specialty Malt and Flaked Grains", "Specialty Malts and Flaked Grains",
        "Flaked Grains", "Flaked/Other Grains",
    ]
    extra_aliases = ["Extra", "Extras"]

    def extract(alias_list):
        vals = []
        for colname in alias_list:
            if colname in ingredients_df.columns:
                vals.extend(ingredients_df[colname].dropna().unique().tolist())
        return set(vals)

    avail = set()
    for aliases in [
        base_aliases,
        hop_aliases,
        yeast_aliases,
        adjunct_aliases,
        specialty_aliases,
        extra_aliases,
    ]:
        avail |= extract(aliases)
    return avail


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


def compute_rules_status(my_picks, ingredient_to_category, total_picks):
    counts = {"Malt": 0, "Hop": 0, "Yeast": 0, "Adjunct": 0, "Flex": 0}
    for ing in my_picks:
        ui_cat = ingredient_to_category.get(ing, "Specialty")
        bucket = bucket_for_rules(ui_cat)
        counts[bucket] += 1

    required_min = {"Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}
    required_met = {k: counts[k] >= v for k, v in required_min.items()}
    required_remaining = {k: max(0, v - counts[k]) for k, v in required_min.items()}

    satisfied_core = sum(min(counts[k], 1) for k in required_min.keys())
    flex_used = max(0, len(my_picks) - satisfied_core)
    flex_remaining = max(0, 3 - flex_used)

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
        pick_value = (
            (style_cov * 1.0 + scarcity * 2.0 + popularity * bias_weight)
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

import argparse
from collections import defaultdict

import pandas as pd

from draft_core import (
    load_data,
    compute_style_status,
    next_best_picks,
    block_picks,
)


def main():
    parser = argparse.ArgumentParser(description="Offline draft advisor for mobile use")
    parser.add_argument(
        "--my-picks", default="", help="Comma-separated list of your picks so far"
    )
    parser.add_argument(
        "--drafted", default="", help="Comma-separated list of all drafted ingredients"
    )
    parser.add_argument(
        "--bias", type=float, default=0.0, help="Opponent bias weight (0=off)"
    )
    args = parser.parse_args()

    (
        ingredients,
        style_matrix,
        scarcity_df,
        opponent_model,
        style_bias,
        ingredient_to_category,
    ) = load_data()

    # Build opponent cues if model is available
    early_signal = {}
    pair_lookup = defaultdict(int)
    if opponent_model:
        for rec in opponent_model.get("ingredient_popularity", []):
            ing = rec.get("Ingredient")
            if ing:
                early_signal[ing] = float(rec.get("Early_Score", 0.0))
        for rec in opponent_model.get("top_pairs", []):
            a = rec.get("A")
            b = rec.get("B")
            c = int(rec.get("PairCount", 0))
            if a and b:
                pair_lookup[(a, b)] += c
                pair_lookup[(b, a)] += c

    my_picks = [x.strip() for x in args.my_picks.split(",") if x.strip()]
    drafted = [x.strip() for x in args.drafted.split(",") if x.strip()] or my_picks.copy()

    required = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}
    flex_slots = 3

    style_df = compute_style_status(
        my_picks, drafted, style_matrix, required, flex_slots
    )
    rec_df = next_best_picks(
        my_picks,
        drafted,
        ingredients,
        style_matrix,
        scarcity_df,
        required,
        flex_slots,
        ingredient_to_category,
        style_bias,
        early_signal,
        bias_weight=args.bias,
    )
    block_df = block_picks(
        drafted, my_picks, pair_lookup, ingredients, early_signal
    )

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    print("\nViable Styles:\n", style_df.head(10).to_string(index=False))
    print("\nTop Recommendations:\n", rec_df.to_string(index=False))
    print("\nBlock Suggestions:\n", block_df.to_string(index=False))


if __name__ == "__main__":
    main()

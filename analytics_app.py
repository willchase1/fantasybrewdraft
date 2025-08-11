import streamlit as st
from collections import defaultdict
from draft_core import (
    load_data,
    compute_style_status,
    next_best_picks,
    block_picks,
)
from draft_state import load_state

st.set_page_config(page_title="Draft Analytics", layout="wide")

# Load data once
(
    ingredients,
    style_matrix,
    scarcity_df,
    opponent_model,
    style_bias,
    ingredient_to_category,
) = load_data()

# Precompute opponent signals
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

# Load shared draft state
state = load_state()
my_picks = state["my_picks"]
drafted = state["drafted"]

st.sidebar.metric("My picks", len(my_picks))
st.sidebar.metric("Total drafted", len(drafted))
if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

required = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}
flex_slots = 3

st.header("Viable Styles")
viab = compute_style_status(my_picks, drafted, style_matrix, required, flex_slots)
st.dataframe(viab, use_container_width=True)

st.header("Top Recommendations")
recs = next_best_picks(
    my_picks,
    drafted,
    style_matrix,
    scarcity_df,
    required,
    flex_slots,
    top_k=15,
    bias_weight=1.0,
)
st.dataframe(recs, use_container_width=True)

st.header("Block Suggestions")
blocks = block_picks(
    drafted,
    my_picks,
    pair_lookup,
    ingredients,
    early_signal=early_signal,
    top_k=15,
)
st.dataframe(blocks, use_container_width=True)

"""My Brewery — personal recipe/prep sandbox.

Decoupled from any live draft: you freely pick ingredients into a *sandbox* and
the shared engine (draft_core) tells you which styles become viable, what to add
next, and hop substitutions. This is the seed of the Phase-2 personal tool
(recipe builder + strategy lab); it deliberately reuses the same scoring the
draft board uses so both modes stay consistent.
"""
import streamlit as st

import draft_core
from config import load_league_config
from draft_core import build_available_set, compute_style_status, next_best_picks

st.header("🧪 My Brewery")
st.caption(
    "A personal sandbox — pick ingredients freely to see what you could brew. "
    "Nothing here touches a live draft."
)

LEAGUE = load_league_config()


@st.cache_data
def _load():
    return draft_core.load_data()


@st.cache_data
def _load_hop_similarity():
    return draft_core._load_optional_json("hop_similarity.json") or {}


(
    ingredients,
    style_matrix,
    scarcity_df,
    opponent_model,
    style_bias,
    ingredient_to_category,
) = _load()
hop_similarity = _load_hop_similarity()

# Opponent popularity is a draft-strategy signal; in the brewery it stays off so
# recommendations reflect brewing merit (style coverage + scarcity) only.
early_signal = {}

# --- Build the selectable ingredient pool by category (alias-aware) ---
pool = {}  # category label -> sorted list of ingredient names
for category_label, alias_list in LEAGUE["category_aliases"].items():
    names = set()
    for colname in alias_list:
        if colname in ingredients.columns:
            names.update(str(v) for v in ingredients[colname].dropna().unique().tolist())
    if names:
        pool[category_label] = sorted(names)

# --- Sandbox selection ---
st.subheader("Your ingredients")
sandbox = []
cols = st.columns(min(3, len(pool)) or 1)
for i, (category_label, options) in enumerate(pool.items()):
    with cols[i % len(cols)]:
        chosen = st.multiselect(category_label, options, key=f"brewery_{category_label}")
        sandbox.extend(chosen)

if st.button("Clear selection"):
    for category_label in pool:
        st.session_state.pop(f"brewery_{category_label}", None)
    st.rerun()

required = LEAGUE["required_categories"]
flex_slots = LEAGUE["flex_slots"]

if not sandbox:
    st.info("Pick a few ingredients above to see viable styles and suggestions.")
    st.stop()

# In a personal sandbox, "drafted" (unavailable) == what you've already added,
# so recommendations exclude your current picks but nothing else is off-limits.
drafted = sandbox

tab_styles, tab_recs, tab_hops = st.tabs(
    ["Viable Styles", "What to Add Next", "Hop Substitutes"]
)

with tab_styles:
    st.caption("Styles you could still build with your current ingredients.")
    viab = compute_style_status(sandbox, drafted, style_matrix, required, flex_slots)
    st.dataframe(viab, use_container_width=True)

with tab_recs:
    st.caption("Ingredients that open up the most styles from here.")
    recs = next_best_picks(
        sandbox, drafted, ingredients, style_matrix, scarcity_df, required,
        flex_slots, ingredient_to_category, style_bias,
        early_signal=early_signal, bias_weight=0.0, top_k=15,
    )
    st.dataframe(recs, use_container_width=True)

with tab_hops:
    if not hop_similarity:
        st.info("Add hop_similarity.json to enable substitution suggestions.")
    else:
        your_hops = [ing for ing in sandbox if ing in hop_similarity]
        if not your_hops:
            st.caption("Add a hop to your selection to see substitutes.")
        for hop in your_hops:
            alts = [rec.get("hop") for rec in hop_similarity.get(hop, []) if rec.get("hop")]
            st.markdown(f"**{hop}** → " + (", ".join(alts) if alts else "_no data_"))

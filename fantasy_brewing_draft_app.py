import streamlit as st
import pandas as pd
from draft_core import load_data
from draft_state import load_state, save_state

st.set_page_config(page_title="Fantasy Brewing Draft Board", layout="wide")

# Load data once
ingredients, *_ = load_data()

# Load shared draft state
state = load_state()
my_picks = state["my_picks"]
drafted = state["drafted"]

st.sidebar.header("Session")
if st.sidebar.button("Reset draft"):
    my_picks.clear()
    drafted.clear()
    save_state(my_picks, drafted)
    st.experimental_rerun()

st.sidebar.metric("My picks", len(my_picks))
st.sidebar.metric("Total drafted", len(drafted))

# Build long list of available ingredients by category
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

def add_from_aliases(alias_list, category_label, rows):
    for colname in alias_list:
        if colname in ingredients.columns:
            for val in ingredients[colname].dropna().unique().tolist():
                rows.append({"Ingredient": str(val), "Category": category_label})

long_rows = []
add_from_aliases(base_aliases, "Base Malt", long_rows)
add_from_aliases(hop_aliases, "Hop", long_rows)
add_from_aliases(yeast_aliases, "Yeast", long_rows)
add_from_aliases(adjunct_aliases, "Adjunct", long_rows)
add_from_aliases(specialty_aliases, "Specialty", long_rows)
add_from_aliases(extra_aliases, "Extra", long_rows)

df_long = pd.DataFrame(long_rows).drop_duplicates()

# Remove drafted ingredients
available_df = df_long[~df_long["Ingredient"].isin(drafted)]

# Category summary
all_categories = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty", "Extra"]
avail_summary = available_df.groupby("Category")["Ingredient"].nunique().reindex(all_categories).fillna(0).astype(int)
st.caption("Available now → " + " | ".join([f"{cat}: {avail_summary.loc[cat]}" for cat in all_categories]))

# Display by category
for cat in all_categories:
    sub = available_df[available_df["Category"] == cat]
    if sub.empty:
        continue
    with st.expander(f"{cat} ({len(sub)})", expanded=(cat in ["Base Malt", "Yeast", "Hop"])):
        for ing in sorted(sub["Ingredient"].tolist()):
            cols = st.columns([6,1.2,1.6])
            cols[0].markdown(f"**{ing}**")
            if cols[1].button("I drafted", key=f"mine-{ing}"):
                my_picks.append(ing)
                drafted.append(ing)
                save_state(my_picks, drafted)
                st.experimental_rerun()
            if cols[2].button("Someone else", key=f"taken-{ing}"):
                drafted.append(ing)
                save_state(my_picks, drafted)
                st.experimental_rerun()

st.divider()
st.subheader("My Picks")
st.write(my_picks if my_picks else "No picks yet.")

st.subheader("All Drafted")
st.write(drafted if drafted else "Nothing drafted yet.")

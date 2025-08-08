
import streamlit as st
import pandas as pd
import json
from collections import defaultdict, Counter

st.set_page_config(page_title="Fantasy Brewing Draft Advisor", layout="wide")

# --- Global CSS for better row UX ---
st.markdown("""
<style>
.hover-row {
    padding: 6px 8px;
    border-radius: 6px;
    transition: background-color 0.12s ease-in-out;
}
.hover-row:hover {
    background-color: rgba(200, 200, 200, 0.18);
}
.block-container .stExpander {
    margin-bottom: 0.35rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    ingredients = pd.read_csv("ingredients_2025.csv")
    with open("style_matrix.json") as f:
        style_matrix = json.load(f)
    with open("ingredient_scarcity.json") as f:
        scarcity = pd.DataFrame(json.load(f))
    # Opponent model is optional
    opponent_model = None
    try:
        with open("opponent_model.json") as f:
            opponent_model = json.load(f)
    except Exception:
        pass
    return ingredients, style_matrix, scarcity, opponent_model

ingredients, style_matrix, scarcity_df, opponent_model = load_data()

def build_available_set(ingredients_df):
    base_aliases = ["Base Malt", "Base Malts", "Base Malts and Extracts"]
    hop_aliases = ["Hop", "Hops"]
    yeast_aliases = ["Yeast", "Yeasts"]
    adjunct_aliases = ["Adjunct", "Adjuncts", "Adjuncts/Spices/Fruits"]
    specialty_aliases = [
        "Specialty", "Specialty Malt", "Specialty Malts", 
        "Specialty Malt and Flaked Grains", "Specialty Malts and Flaked Grains",
        "Flaked Grains", "Flaked/Other Grains"
    ]
    extra_aliases = ["Extra", "Extras"]

    def extract(alias_list):
        vals = []
        for colname in alias_list:
            if colname in ingredients_df.columns:
                vals.extend(ingredients_df[colname].dropna().unique().tolist())
        return set(vals)

    avail = set()
    for aliases in [base_aliases, hop_aliases, yeast_aliases, adjunct_aliases, specialty_aliases, extra_aliases]:
        avail |= extract(aliases)
    return avail


# Load style bias if present
style_bias = None
try:
    with open("style_bias.json") as f:
        style_bias = json.load(f)
except Exception:
    style_bias = None

# Helper to get style bias weight for an ingredient
def ingredient_style_bias(ingredient, style_matrix, style_bias):
    if not style_bias:
        return 1.0
    bias_factor = 1.0
    # Find styles containing this ingredient
    for family, data in style_bias.items():
        styles = data.get("styles", [])
        weight = data.get("weight", 1.0)
        for style in styles:
            if style in style_matrix:
                for cat, ings in style_matrix[style].items():
                    if ingredient in ings:
                        bias_factor = max(bias_factor, weight)
    return bias_factor


# --- Helper maps ---
ingredient_to_category = {}
for style, cats in style_matrix.items():
    for cat, ing_list in cats.items():
        for ing in ing_list:
            ingredient_to_category[ing] = cat

all_categories = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty"]

# --- Rulebook-aware requirements status ---
TOTAL_PICKS = 7  # 1 malt, 1 hop, 1 yeast, 1 adjunct, plus 3 flex

def bucket_for_rules(category_label: str) -> str:
    # Map UI categories to rulebook buckets
    if category_label == "Base Malt":
        return "Malt"
    if category_label == "Hop":
        return "Hop"
    if category_label == "Yeast":
        return "Yeast"
    if category_label == "Adjunct":
        return "Adjunct"
    # Specialty/Extra -> only count toward Flex
    return "Flex"

def compute_rules_status(my_picks, ingredient_to_category):
    # Count picks by rule bucket
    counts = {"Malt":0, "Hop":0, "Yeast":0, "Adjunct":0, "Flex":0}
    # Reverse map ingredient->UI category, then to rule bucket
    for ing in my_picks:
        ui_cat = ingredient_to_category.get(ing, "Specialty")
        bucket = bucket_for_rules(ui_cat)
        counts[bucket] += 1

    # Required minimums
    required_min = {"Malt":1, "Hop":1, "Yeast":1, "Adjunct":1}
    required_met = {k: counts[k] >= v for k,v in required_min.items()}
    required_remaining = {k: max(0, v - counts[k]) for k,v in required_min.items()}

    # Flex usage: extra picks beyond first required four count as Flex
    satisfied_core = sum(min(counts[k], 1) for k in required_min.keys())
    flex_used = max(0, len(my_picks) - satisfied_core)
    flex_remaining = max(0, 3 - flex_used)

    picks_remaining = max(0, TOTAL_PICKS - len(my_picks))

    # Can we still satisfy the remaining required within remaining picks?
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
        "feasible": feasible
    }
    return status


# --- Sidebar controls ---
st.sidebar.header("Draft Setup")
num_players = st.sidebar.number_input("Number of players", min_value=4, max_value=20, value=10, step=1)
draft_position = st.sidebar.number_input("Your draft position (Round 1)", min_value=1, max_value=num_players, value=min(num_players, 10), step=1)
st.sidebar.caption("Snake draft: end of round 1 means first pick in round 2.")

st.sidebar.header("Room Bias (opponent behavior)")
bias_choice = st.sidebar.selectbox("How aggressively should we anticipate snipes?", ["Off","Conservative","Aggressive"], index=1)
bias_weight = {"Off":0.0, "Conservative":0.75, "Aggressive":1.5}[bias_choice]
if opponent_model is None and bias_choice != "Off":
    st.sidebar.warning("No opponent model file found. Bias effects will be limited.")

st.sidebar.header("Category Requirements")
required = {"Base Malt": 1, "Hop": 1, "Yeast": 1, "Adjunct": 1}
flex_slots = st.sidebar.number_input("Flex slots", min_value=0, max_value=5, value=3, step=1)


st.sidebar.header("Session")

reload_data = st.sidebar.button("Reload data files", key="reload_data_btn")
if reload_data:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.rerun()

reset = st.sidebar.button("Reset session", type="primary")

if "my_picks" not in st.session_state or reset:
    st.session_state["my_picks"] = []
if "drafted" not in st.session_state or reset:
    st.session_state["drafted"] = []

my_picks = st.session_state["my_picks"]
drafted = st.session_state["drafted"]


# --- Live rule status panel ---
rules = compute_rules_status(my_picks, ingredient_to_category)

st.sidebar.header("Your Draft Status")
colA, colB = st.sidebar.columns(2)
with colA:
    st.metric("Picks Used", f"{len(my_picks)}/{TOTAL_PICKS}")
with colB:
    st.metric("Flex Left", f"{rules['flex_remaining']}/3")

st.sidebar.caption("Required categories (need 1 each):")
req_cols = st.sidebar.columns(4)
req_map = {"Malt":"Malt", "Hop":"Hop", "Yeast":"Yeast", "Adjunct":"Adjunct"}
i = 0
for key,label in req_map.items():
    met = rules["required_met"][key]
    rem = rules["required_remaining"][key]
    emoji = "✅" if met else "⚠️"
    req_cols[i].markdown(f"{emoji} **{label}**")
    if not met:
        req_cols[i].caption(f"Need {rem}")
    i += 1

if not rules["feasible"]:
    st.sidebar.error("Warning: Not enough picks remaining to satisfy all required categories.")
else:
    st.sidebar.success(f"Picks remaining: {rules['picks_remaining']}")


# --- Simple accessors for opponent data ---
ingredient_popularity = {}
early_signal = {}
pair_lookup = defaultdict(int)

if opponent_model:
    # ingredient_popularity records contain Category, Ingredient, Picks, Avg_Slot, Early_Score, etc.
    for rec in opponent_model.get("ingredient_popularity", []):
        ing = rec.get("Ingredient")
        if ing:
            ingredient_popularity[ing] = rec
            early_signal[ing] = float(rec.get("Early_Score", 0.0))
    for rec in opponent_model.get("top_pairs", []):
        a = rec.get("A"); b = rec.get("B"); c = int(rec.get("PairCount", 0))
        if a and b:
            pair_lookup[(a,b)] += c
            pair_lookup[(b,a)] += c

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Draft Board", "Style Viability", "Recommendations", "Blocks (deny-their-build)"])

# --- Draft Board ---
with tab1:
    st.subheader("Available Ingredients")

    
    # Build long list of available ingredients by category, based on sheet columns (alias-aware)
    long_rows = []
    base_aliases = ["Base Malt", "Base Malts", "Base Malts and Extracts"]
    hop_aliases = ["Hop", "Hops"]
    yeast_aliases = ["Yeast", "Yeasts"]
    adjunct_aliases = ["Adjunct", "Adjuncts", "Adjuncts/Spices/Fruits"]
    specialty_aliases = [
        "Specialty", "Specialty Malt", "Specialty Malts",
        "Specialty Malt and Flaked Grains", "Specialty Malts and Flaked Grains",
        "Flaked Grains", "Flaked/Other Grains"
    ]
    extra_aliases = ["Extra", "Extras"]

    def add_from_aliases(alias_list, category_label):
        for colname in alias_list:
            if colname in ingredients.columns:
                for val in ingredients[colname].dropna().unique().tolist():
                    long_rows.append({"Ingredient": str(val), "Category": category_label})

    add_from_aliases(base_aliases, "Base Malt")
    add_from_aliases(hop_aliases, "Hop")
    add_from_aliases(yeast_aliases, "Yeast")
    add_from_aliases(adjunct_aliases, "Adjunct")
    add_from_aliases(specialty_aliases, "Specialty")
    add_from_aliases(extra_aliases, "Extra")

    df_long = pd.DataFrame(long_rows).drop_duplicates()

    # Remove those already drafted
    df_long = df_long[~df_long["Ingredient"].isin(drafted)]
    
    # Quick availability summary
    avail_summary = df_long.groupby("Category")["Ingredient"].nunique().reindex(all_categories).fillna(0).astype(int)
    st.caption("Available now → " + " | ".join([f"{cat}: {avail_summary.loc[cat]}" for cat in all_categories]))


    # Show by category — keep Base Malt, Yeast, Hop open by default
    for cat in all_categories:
        sub = df_long[df_long["Category"]==cat]
        with st.expander(f"{cat} ({len(sub)})", expanded=(cat in ["Base Malt","Yeast","Hop"])):
            for ing in sorted(sub["Ingredient"].tolist()):
                with st.container():
                    st.markdown("<div class='hover-row'>", unsafe_allow_html=True)
                    cols = st.columns([6,1.2,1.6])
                    label = f"**{ing}**"
                    # Show popularity cue if we have it
                    if ing in ingredient_popularity:
                        rec = ingredient_popularity[ing]
                        label += f"  \n<small>pop: {rec.get('Picks',0)} | avg slot: {round(float(rec.get('Avg_Slot',0)),1)}</small>"
                    cols[0].markdown(label, unsafe_allow_html=True)
                    if cols[1].button("I drafted", key=f"mine-{ing}"):
                        my_picks.append(ing)
                        drafted.append(ing)
                        st.rerun()

                    if cols[2].button("Someone else", key=f"taken-{ing}"):
                        drafted.append(ing)
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("My Picks")
    st.write(my_picks if my_picks else "No picks yet.")
    st.subheader("All Drafted (any team)")
    st.write(drafted if drafted else "Nothing drafted yet.")

# --- Style Viability ---
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
        options = sum(1 for k,v in cat_choices_remaining.items() if v > 0)
        score = satisfied*2 + options
        status.append({
            "Style": style,
            "Satisfied Categories": satisfied,
            "Categories with Options Left": options,
            "Score": score
        })
    df = pd.DataFrame(status).sort_values(by=["Score","Satisfied Categories"], ascending=False)
    return df

with tab2:
    st.subheader("Viable Styles (live)")
    viab = compute_style_status(my_picks, drafted, style_matrix, required, flex_slots)
    st.dataframe(viab, use_container_width=True)

# --- Recommendations with opponent bias ---


def next_best_picks(my_picks, drafted, style_matrix, scarcity_df, required, flex_slots, top_k=15, bias_weight=0.0):
    drafted_set = set(drafted)
    my_set = set(my_picks)
    available_set = build_available_set(ingredients)  # only recommend from current pool

    # Candidate pool (intersect style_matrix with available ingredients)
    cand = []
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing in available_set and ing not in drafted_set:
                    cand.append((ing, cat, style))
    if not cand:
        return pd.DataFrame(columns=["Ingredient","Category","Style Coverage","Scarcity","Popularity","Bias Factor","Pick Value"])

    # Style coverage across viable styles
    coverage = defaultdict(set)
    for ing, cat, style in cand:
        coverage[ing].add(style)

    sc = scarcity_df.set_index("Ingredient") if not scarcity_df.empty else pd.DataFrame()
    rows = []
    have_counts = Counter([ingredient_to_category.get(i, "Unknown") for i in my_set])
    cat_need_factor = {}
    for cat in ["Yeast","Hop","Adjunct","Base Malt","Specialty"]:
        required_min = required.get(cat, 0)
        have = have_counts.get(cat, 0)
        cat_need_factor[cat] = 1.5 if have < required_min else 1.0

    for ing, styles in coverage.items():
        cat = ingredient_to_category.get(ing, "Unknown")
        style_cov = len(styles)
        # Dynamic scarcity fallback: rarer if supports fewer styles
        if not sc.empty and ing in sc.index and "Scarcity Score" in sc.columns:
            scarcity = float(sc.loc[ing]["Scarcity Score"])
        else:
            scarcity = 1.0 / max(style_cov, 1)
        need_bonus = cat_need_factor.get(cat, 1.0)
        popularity = float(early_signal.get(ing, 0.0))  # historical early-draft signal
        bias_factor = ingredient_style_bias(ing, style_matrix, style_bias)
        pick_value = (style_cov * 1.0 + scarcity * 2.0 + popularity * bias_weight) * need_bonus * bias_factor
        rows.append({
            "Ingredient": ing,
            "Category": cat,
            "Style Coverage": style_cov,
            "Scarcity": round(scarcity, 3),
            "Popularity": round(popularity, 3),
            "Bias Factor": round(bias_factor, 2),
            "Pick Value": round(pick_value, 3)
        })

    df = pd.DataFrame(rows).sort_values(by=["Pick Value","Bias Factor","Scarcity","Style Coverage"], ascending=[False,False,False,False])
    return df.head(top_k)


with tab3:
    st.subheader("Best Next Picks (live, opponent-aware)")
    recs = next_best_picks(my_picks, drafted, style_matrix, scarcity_df, required, flex_slots, bias_weight=bias_weight)
    st.dataframe(recs, use_container_width=True)

# --- Block Picks (deny their build) ---

def block_picks(drafted, my_picks, top_k=15):
    available_set = build_available_set(ingredients)
    opp_picks = [d for d in drafted if d not in set(my_picks)]
    recent = list(reversed(opp_picks))[:3]  # last 3 non-you picks
    suggestions = defaultdict(int)

    for ing in recent:
        for (a,b), cnt in pair_lookup.items():
            if a == ing and b in available_set and b not in set(drafted):
                suggestions[b] += cnt

    remaining = []
    for ing, score in suggestions.items():
        remaining.append((ing, score, early_signal.get(ing, 0.0)))
    if not remaining:
        return pd.DataFrame(columns=["Ingredient","Block Score","Popularity Cue"])

    df = pd.DataFrame(remaining, columns=["Ingredient","Block Score","Popularity Cue"])
    df = df.sort_values(["Block Score","Popularity Cue"], ascending=[False,False]).head(top_k)
    return df

with tab4:
    st.subheader("Blocks (based on recent opponent picks)")
    if opponent_model is None:
        st.info("Add opponent_model.json to enable block suggestions.")
    blocks = block_picks(drafted, my_picks, top_k=15)
    st.dataframe(blocks, use_container_width=True)

st.caption("Tip: Toggle Room Bias in the sidebar to lean into opponent tendencies. Blocks tab suggests denial picks based on the last few opponent selections.")
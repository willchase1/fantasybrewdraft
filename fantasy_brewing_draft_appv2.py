import streamlit as st
import pandas as pd
import json
import random
import os
import io
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

# --- Persist draft state locally ---
SAVE_FILE = "draft_autosave.json"

def load_state():
    """Load draft state from disk if it exists."""
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state):
    """Persist draft state to disk."""
    try:
        with open(SAVE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

# ---- Availability helper (alias-aware) ----
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

# ---- Style bias (optional) ----
style_bias = None
try:
    with open("style_bias.json") as f:
        style_bias = json.load(f)
except Exception:
    style_bias = None

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

# --- Helper maps ---
ingredient_to_category = {}
for style, cats in style_matrix.items():
    for cat, ing_list in cats.items():
        for ing in ing_list:
            ingredient_to_category[ing] = cat

all_categories = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty"]

# --- Rulebook-aware requirements status ---
DEFAULT_ROUNDS = 7  # base number of rounds before optional round 8

def bucket_for_rules(category_label: str) -> str:
    if category_label == "Base Malt":
        return "Malt"
    if category_label == "Hop":
        return "Hop"
    if category_label == "Yeast":
        return "Yeast"
    if category_label == "Adjunct":
        return "Adjunct"
    return "Flex"  # Specialty/Extra -> Flex only

def compute_rules_status(my_picks, ingredient_to_category, total_picks):
    counts = {"Malt":0, "Hop":0, "Yeast":0, "Adjunct":0, "Flex":0}
    for ing in my_picks:
        ui_cat = ingredient_to_category.get(ing, "Specialty")
        bucket = bucket_for_rules(ui_cat)
        counts[bucket] += 1

    required_min = {"Malt":1, "Hop":1, "Yeast":1, "Adjunct":1}
    required_met = {k: counts[k] >= v for k,v in required_min.items()}
    required_remaining = {k: max(0, v - counts[k]) for k,v in required_min.items()}

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
        "feasible": feasible
    }
    return status

# --- Sidebar controls ---
st.sidebar.header("Draft Setup")
saved_state = load_state()
if "draft_log" not in st.session_state:
    st.session_state["draft_log"] = saved_state.get("draft_log", [])
if "players" not in st.session_state:
    st.session_state["players"] = saved_state.get("players", [])

existing_players = st.session_state.get("players", [])
num_players = st.sidebar.number_input(
    "Number of players", min_value=4, max_value=20,
    value=len(existing_players) if existing_players else 10, step=1
)

players = []
for i in range(int(num_players)):
    default_name = existing_players[i] if i < len(existing_players) else ""
    nm = st.sidebar.text_input(f"Seat {i+1}", value=default_name, key=f"player_{i}")
    players.append(nm.strip() or f"Player {i+1}")
st.session_state["players"] = players
save_state({"players": players, "draft_log": st.session_state.get("draft_log", [])})

enable_round8 = st.sidebar.checkbox("Enable optional 8th round", value=False)
TOTAL_PICKS = DEFAULT_ROUNDS + (1 if enable_round8 else 0)

prev_draft_pos = int(st.session_state.get("draft_pos", 1))
draft_position = st.sidebar.number_input(
    "Your draft position (Round 1)", min_value=1, max_value=num_players,
    value=min(num_players, prev_draft_pos), step=1
)
st.session_state["draft_pos"] = int(draft_position)
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
if reset:
    st.session_state["draft_log"] = []
    save_state({"players": players, "draft_log": []})

draft_log = st.session_state.get("draft_log", [])

# derive team picks and drafted list
teams = {p: [] for p in players}
drafted = []
for rec in draft_log:
    plyr = rec.get("Player")
    ing = rec.get("Ingredient")
    if plyr in teams:
        teams[plyr].append(ing)
    else:
        teams[plyr] = [ing]
    drafted.append(ing)

your_name = players[int(draft_position)-1] if players else ""
my_picks = teams.get(your_name, [])

# --- Live rule status panel ---
rules = compute_rules_status(my_picks, ingredient_to_category, TOTAL_PICKS)

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

# --- Current draft state ---
total_picks_overall = TOTAL_PICKS * int(num_players)
overall_pick = len(draft_log) + 1
current_round = ((overall_pick - 1) // int(num_players)) + 1
order = list(range(int(num_players))) if current_round % 2 == 1 else list(range(int(num_players)-1, -1, -1))
idx_in_order = (overall_pick - 1) % int(num_players)
current_player = players[order[idx_in_order]] if overall_pick <= total_picks_overall else None

# --- Opponent data accessors ---
ingredient_popularity = {}
early_signal = {}
pair_lookup = defaultdict(int)

if opponent_model:
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Draft Board", "Style Viability", "Recommendations",
    "Blocks (deny-their-build)", "Mock Draft Simulator",
    "Results / Export"
])

# --- Draft Board ---
with tab1:
    st.subheader("Available Ingredients")

    if current_player:
        st.info(f"Round {current_round} • Pick {overall_pick} → {current_player}")
    else:
        st.success("Draft complete.")

    def add_pick(player, ing, cat):
        overall = len(st.session_state["draft_log"]) + 1
        round_no = ((overall - 1) // int(num_players)) + 1
        st.session_state["draft_log"].append({
            "Round": round_no,
            "Overall": overall,
            "Player": player,
            "Ingredient": ing,
            "Category": cat
        })
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})
        st.rerun()

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
                    cols = st.columns([6,2])
                    label = f"**{ing}**"
                    if ing in ingredient_popularity:
                        rec = ingredient_popularity[ing]
                        label += f"  \n<small>pop: {rec.get('Picks',0)} | avg slot: {round(float(rec.get('Avg_Slot',0)),1)}</small>"
                    cols[0].markdown(label, unsafe_allow_html=True)
                    if current_player and cols[1].button("Draft", key=f"draft-{cat}-{ing}"):
                        add_pick(current_player, ing, cat)
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
    available_set = build_available_set(ingredients)

    cand = []
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing in available_set and ing not in drafted_set:
                    cand.append((ing, cat, style))
    if not cand:
        return pd.DataFrame(columns=["Ingredient","Category","Style Coverage","Scarcity","Popularity","Bias Factor","Pick Value"])

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
        if not sc.empty and ing in sc.index and "Scarcity Score" in sc.columns:
            scarcity = float(sc.loc[ing]["Scarcity Score"])
        else:
            scarcity = 1.0 / max(style_cov, 1)
        need_bonus = cat_need_factor.get(cat, 1.0)
        popularity = float(early_signal.get(ing, 0.0))
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
    recent = list(reversed(opp_picks))[:3]
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
    st.subheader("Blocks and Opponent Predictions")

    # --- Per-player draft summary and predictions ---
    pred_rows = []
    for p in players:
        picks_p = teams.get(p, [])
        style_guess = "N/A"
        if picks_p:
            viab_p = compute_style_status(picks_p, drafted, style_matrix, required, flex_slots)
            if not viab_p.empty:
                style_guess = viab_p.iloc[0]["Style"]
        recs_p = next_best_picks(picks_p, drafted, style_matrix, scarcity_df, required, flex_slots, top_k=3, bias_weight=bias_weight)
        next_guess = ", ".join(recs_p["Ingredient"].tolist()) if not recs_p.empty else ""
        pred_rows.append({
            "Player": p,
            "Picks": ", ".join(picks_p),
            "Likely Style": style_guess,
            "Likely Next Picks": next_guess
        })
    pred_df = pd.DataFrame(pred_rows)
    st.markdown("### Player Tendencies")
    st.dataframe(pred_df, use_container_width=True)

    st.markdown("### Block Suggestions")
    if opponent_model is None:
        st.info("Add opponent_model.json to enable block suggestions.")
    blocks = block_picks(drafted, my_picks, top_k=15)
    st.dataframe(blocks, use_container_width=True)

# --- Mock draft simulation helpers ---

def sim_available_candidates(style_matrix, ingredients, drafted):
    """Return dict of ingredient->category that are still available and present in 2025 pool."""
    avail_set = build_available_set(ingredients)
    drafted_set = set(drafted)
    cand = {}
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing in avail_set and ing not in drafted_set:
                    cand.setdefault(ing, ingredient_to_category.get(ing, cat))
    return cand

def sim_opponent_pick(round_idx, cand_map, early_signal, base_malt_run=False, yeast_run=False):
    """Choose an opponent pick based on historical early signal and simple scenario toggles."""
    if not cand_map:
        return None, None

    items = list(cand_map.items())
    weights = []
    for ing, cat in items:
        w = 1.0 + float(early_signal.get(ing, 0.0))
        if round_idx == 1 and base_malt_run and cat == "Base Malt":
            w *= 2.0
        if round_idx == 2 and yeast_run and cat == "Yeast":
            w *= 1.8
        if round_idx <= 2 and cat == "Specialty":
            w *= 0.6
        weights.append(max(w, 0.01))

    total = sum(weights)
    if total <= 0:
        weights = [1.0 for _ in weights]
        total = sum(weights)
    probs = [w/total for w in weights]
    idx = random.choices(range(len(items)), weights=probs, k=1)[0]
    ing, cat = items[idx]
    return ing, cat

def simulate_draft(sim_players, sim_rounds, your_pos, base_malt_run, yeast_run, bias_weight):
    """
    Run a full snake draft simulation:
    - Opponents pick based on early_signal + scenario
    - Your picks use the next_best_picks ranking (opponent-aware, bias-weighted)
    Returns: log (list of dict), my_picks_end, drafted_end
    """
    drafted_local = list(drafted)
    my_local = list(my_picks)

    log = []
    overall = 0

    for rnd in range(1, sim_rounds+1):
        order = list(range(1, sim_players+1)) if (rnd % 2 == 1) else list(range(sim_players, 0, -1))
        for seat in order:
            overall += 1
            cand_map = sim_available_candidates(style_matrix, ingredients, drafted_local)

            if seat == your_pos:
                recs = next_best_picks(my_local, drafted_local, style_matrix, scarcity_df, required, flex_slots, top_k=10, bias_weight=bias_weight)
                recs = recs[~recs["Ingredient"].isin(drafted_local)]
                if not recs.empty:
                    row = recs.iloc[0]
                    ing = row["Ingredient"]
                    cat = row["Category"]
                    reason = "your_top_pick"
                else:
                    if not cand_map:
                        break
                    ing, cat = random.choice(list(cand_map.items()))
                    reason = "fallback_random"
                my_local.append(ing)
                drafted_local.append(ing)
                log.append({"Round": rnd, "Overall": overall, "Seat": seat, "Team": "You", "Ingredient": ing, "Category": cat, "Reason": reason})
            else:
                ing, cat = sim_opponent_pick(rnd, cand_map, early_signal, base_malt_run, yeast_run)
                if ing is None:
                    continue
                drafted_local.append(ing)
                log.append({"Round": rnd, "Overall": overall, "Seat": seat, "Team": f"Opp {seat}", "Ingredient": ing, "Category": cat, "Reason": "opp_weighted"})

    return log, my_local, drafted_local

with tab5:
    st.subheader("Mock Draft Simulator")
    st.caption("Simulates a full snake draft using your current settings. Opponents pick via history/scenarios; your picks use the 'Best Next Picks' logic.")

    c1, c2, c3 = st.columns(3)
    sim_players = c1.number_input("Players", min_value=4, max_value=20, value=int(num_players), step=1, key="sim_players")
    sim_rounds = c2.number_input("Rounds", min_value=1, max_value=10, value=7, step=1, key="sim_rounds")
    your_pos_sim = c3.number_input("Your position", min_value=1, max_value=int(sim_players), value=int(draft_position), step=1, key="sim_your_pos")

    c4, c5, c6 = st.columns(3)
    scen_base_malt = c4.checkbox("Round 1 base malt run", value=True, key="scen_base_malt")
    scen_yeast = c5.checkbox("Round 2 yeast run", value=True, key="scen_yeast")
    sim_seed = c6.number_input("Random seed", min_value=0, max_value=10**9, value=42, step=1, key="sim_seed")

    run = st.button("Run mock draft", key="run_mock")
    if run:
        random.seed(int(sim_seed))
        log, my_local, drafted_local = simulate_draft(int(sim_players), int(sim_rounds), int(your_pos_sim), scen_base_malt, scen_yeast, bias_weight)

        st.markdown("#### Simulation Results")
        st.write(f"**Your picks ({len(my_local)}):** " + ", ".join(my_local))
        st.markdown("**Pick Log** (last 40 shown)")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df.tail(40), use_container_width=True)

        st.markdown("#### Final Style Viability (top 15)")
        viab_sim = compute_style_status(my_local, drafted_local, style_matrix, required, flex_slots).head(15)
        st.dataframe(viab_sim, use_container_width=True)

with tab6:
    st.subheader("Draft Results")
    df_log = pd.DataFrame(st.session_state["draft_log"])
    edited = st.data_editor(df_log, num_rows="dynamic", use_container_width=True, key="draft_editor")
    if not edited.equals(df_log):
        st.session_state["draft_log"] = edited.to_dict("records")
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})
        st.rerun()
    if not edited.empty:
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, file_name="draft_results.csv", mime="text/csv")
        excel_buf = io.BytesIO()
        edited.to_excel(excel_buf, index=False)
        st.download_button(
            "Download Excel", excel_buf.getvalue(), file_name="draft_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.caption("Tip: Toggle Room Bias in the sidebar to lean into opponent tendencies. Blocks tab suggests denial picks based on the last few opponent selections.")

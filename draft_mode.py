import streamlit as st
import pandas as pd
import json
import random
import os
import io
import time
from collections import defaultdict
import draft_state
import draft_core
from config import load_league_config
from draft_core import (
    build_available_set,
    validate_data,
    ingredient_style_bias,
    bucket_for_rules,
    compute_rules_status,
    compute_style_status,
    next_best_picks,
    block_picks,
    pick_slot,
)

# Page config is set by the app.py entry point (st.set_page_config must run
# once, before any other Streamlit command). This module is a navigation page.
st.header("🍺 Run a Draft")

# --- Global CSS for better UX ---
st.markdown("""
<style>
/* Timer section styling */
.timer-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    display:none;
}

.timer-display {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    font-family: 'Courier New', monospace;
    font-size: 28px;
    font-weight: bold;
    color: #2c3e50;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
}

.timer-display.warning {
    background: rgba(255, 193, 7, 0.95);
    color: #856404;
}

.timer-display.danger {
    background: rgba(220, 53, 69, 0.95);
    color: white;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* Draft management section */
.draft-mgmt-container {
    background: rgba(52, 152, 219, 0.1);
    border-left: 4px solid #3498db;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}

/* Ingredient row styling - fix hover issues */
.ingredient-row {
    display: flex !important;
    align-items: center !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease-in-out !important;
    border: 1px solid transparent !important;
    background-color: transparent !important;
}

.ingredient-row:hover {
    background-color: rgba(52, 152, 219, 0.1) !important;
    border-color: rgba(52, 152, 219, 0.3) !important;
    transform: translateX(2px) !important;
    box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2) !important;
}

.ingredient-info {
    flex: 1 !important;
    padding-right: 10px !important;
}

.ingredient-button {
    flex-shrink: 0 !important;
}

/* Expander styling */
.block-container .stExpander {
    margin-bottom: 0.5rem;
}

.block-container .stExpander > div > div > div > div {
    padding-top: 1rem;
}

/* Status indicators */
.status-good { color: #27ae60; font-weight: bold; }
.status-warning { color: #f39c12; font-weight: bold; }
.status-danger { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # draft_core is the single source of truth for loading + shaping data.
    return draft_core.load_data()

(
    ingredients,
    style_matrix,
    scarcity_df,
    opponent_model,
    style_bias,
    ingredient_to_category,
) = load_data()


@st.cache_data
def load_hop_similarity_data():
    """Load hop similarity resources."""
    hop_sim = {}
    sim_matrix = pd.DataFrame()
    try:
        with open("hop_similarity.json") as f:
            hop_sim = json.load(f)
    except Exception:
        pass
    try:
        sim_matrix = pd.read_csv("hop_similarity_matrix.csv", index_col=0)
    except Exception:
        pass
    return hop_sim, sim_matrix


hop_similarity_data, hop_similarity_matrix = load_hop_similarity_data()

# --- Persist draft state locally (the log is the single source of truth) ---
def load_state():
    """Load persisted draft ({players, draft_log}) via draft_state."""
    return draft_state.load_draft()

def save_state(state):
    """Persist the draft ({players, draft_log}) via draft_state."""
    draft_state.save_draft(state)

# build_available_set, ingredient_style_bias, ingredient_to_category and
# style_bias now come from draft_core (imported / returned by load_data above).

all_categories = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty"]

# --- Rulebook-aware requirements status ---
LEAGUE = load_league_config()
DEFAULT_ROUNDS = LEAGUE["rounds"]  # base number of rounds before optional round 8
# bucket_for_rules and compute_rules_status now come from draft_core.

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
required = LEAGUE["required_categories"]
flex_slots = st.sidebar.number_input(
    "Flex slots", min_value=0, max_value=5, value=int(LEAGUE["flex_slots"]), step=1
)

st.sidebar.header("Session")
reload_data = st.sidebar.button("Reload data files", key="reload_data_btn")
if reload_data:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.rerun()

# Data health: warn if any style requires an ingredient that isn't draftable.
_health = validate_data(ingredients, style_matrix, LEAGUE["category_aliases"])
if _health["in_matrix_not_sheet"]:
    st.sidebar.warning(
        f"{len(_health['in_matrix_not_sheet'])} style ingredient(s) are missing "
        "from the sheet — some styles can't be satisfied."
    )
    with st.sidebar.expander("Show data issues"):
        st.write("Referenced by a style but not draftable:")
        st.write(_health["in_matrix_not_sheet"])

reset = st.sidebar.button("Reset session", type="primary")
if reset:
    st.session_state["draft_log"] = []
    save_state({"players": players, "draft_log": []})

draft_log = st.session_state.get("draft_log", [])

# derive team picks and drafted list from the log (single source of truth)
teams, drafted = draft_state.project(draft_log, players)

your_name = players[int(draft_position)-1] if players else ""
my_picks = teams.get(your_name, [])

# --- Draft Timer ---
if "timer_duration" not in st.session_state:
    st.session_state.timer_duration = 120  # seconds total
if "timer_started_at" not in st.session_state:
    st.session_state.timer_started_at = time.time()
if "timer_elapsed" not in st.session_state:
    st.session_state.timer_elapsed = 0.0  # accumulated while running/paused
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False

def start_draft_timer():
    if not st.session_state.timer_running:
        st.session_state.timer_running = True
        st.session_state.timer_started_at = time.time()

def pause_draft_timer():
    if st.session_state.timer_running:
        # accumulate time since last start
        st.session_state.timer_elapsed += time.time() - st.session_state.timer_started_at
        st.session_state.timer_running = False

def reset_draft_timer():
    st.session_state.timer_elapsed = 0.0
    st.session_state.timer_running = False
    st.session_state.timer_started_at = time.time()

# Compute remaining time based on total duration minus elapsed (plus live run time)
elapsed = st.session_state.timer_elapsed + (
    time.time() - st.session_state.timer_started_at if st.session_state.timer_running else 0.0
)
remaining = max(0, int(st.session_state.timer_duration - elapsed))
if remaining == 0 and st.session_state.timer_running:
    # auto-pause on zero
    st.session_state.timer_running = False

# Enhanced timer section with better layout
# st.markdown('<div class="timer-container">', unsafe_allow_html=True)

# Timer display with dynamic styling
timer_class = ""
if remaining <= 10:
    timer_class = "danger"
elif remaining <= 30:
    timer_class = "warning"

timer_status = "RUNNING" if st.session_state.timer_running else "PAUSED"
timer_emoji = "▶️" if st.session_state.timer_running else "⏸️"

st.markdown(
    f"""
    <div class="timer-display {timer_class}">
        {timer_emoji} {remaining//60:02d}:{remaining%60:02d} ({timer_status})
    </div>
    """,
    unsafe_allow_html=True,
)

# Timer controls in a cleaner layout
timer_control_cols = st.columns([2, 2, 2, 2])

with timer_control_cols[0]:
    minutes = st.number_input(
        "⏱️ Duration (min)",
        min_value=1,
        max_value=60,
        value=max(1, int(round(st.session_state.timer_duration / 60))),
        key="timer_minutes",
        help="Set the draft timer duration"
    )
    if minutes * 60 != st.session_state.timer_duration:
        st.session_state.timer_duration = minutes * 60

with timer_control_cols[1]:
    if st.session_state.timer_running:
        st.button("⏸️ Pause", on_click=pause_draft_timer, key="timer_pause_btn", use_container_width=True)
    else:
        st.button("▶️ Start", on_click=start_draft_timer, key="timer_start_btn", use_container_width=True)

with timer_control_cols[2]:
    st.button("🔄 Reset", on_click=reset_draft_timer, key="timer_reset_btn", use_container_width=True)

with timer_control_cols[3]:
    # Timer status info
    total_elapsed = int(st.session_state.timer_elapsed + (time.time() - st.session_state.timer_started_at if st.session_state.timer_running else 0.0))
    st.metric("Elapsed", f"{total_elapsed//60:02d}:{total_elapsed%60:02d}")

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- Draft Management Functions ---
def undo_last_pick():
    """Remove the last pick from the draft log."""
    if st.session_state.get("draft_log"):
        st.session_state["draft_log"].pop()
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})

def swap_pick(pick_index, new_ingredient, new_category):
    """Swap an existing pick with a new ingredient."""
    if 0 <= pick_index < len(st.session_state["draft_log"]):
        st.session_state["draft_log"][pick_index]["Ingredient"] = new_ingredient
        st.session_state["draft_log"][pick_index]["Category"] = new_category
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})

# --- Draft Management Controls ---
# st.markdown('<div class="draft-mgmt-container">', unsafe_allow_html=True)
st.markdown("#### 🎯 Draft Management")

mgmt_cols = st.columns([2, 2, 4])

with mgmt_cols[0]:
    if st.button("↶ Undo Last Pick", disabled=len(draft_log)==0, key="undo_btn", use_container_width=True):
        undo_last_pick()
        st.rerun()

with mgmt_cols[1]:
    if st.button("🔄 Manage Swaps/Trades", key="toggle_swap_mode", use_container_width=True):
        st.session_state["show_swap_mode"] = not st.session_state.get("show_swap_mode", False)

st.markdown('</div>', unsafe_allow_html=True)

# --- Swap/Trade Interface ---
if st.session_state.get("show_swap_mode", False):
    st.markdown("### 🔄 Swap/Trade Manager")
    st.caption("Use this to handle trades or optional Round 8 swaps. Select a pick to change and choose a new ingredient.")
    
    if draft_log:
        # Create a dropdown of all picks for swapping
        pick_options = []
        for i, rec in enumerate(draft_log):
            pick_options.append(f"R{rec.get('Round', '?')} Pick {rec.get('Overall', '?')}: {rec.get('Player', '?')} → {rec.get('Ingredient', '?')}")
        
        swap_cols = st.columns([3, 3, 2])
        
        with swap_cols[0]:
            selected_pick_idx = st.selectbox(
                "Select pick to change:",
                range(len(pick_options)),
                format_func=lambda x: pick_options[x],
                key="swap_pick_select"
            )
        
        with swap_cols[1]:
            # Get available ingredients (not currently drafted, or the one being swapped)
            current_ingredient = draft_log[selected_pick_idx].get("Ingredient", "")
            available_ingredients = []
            
            # Build available ingredients from all categories
            for style, cats in style_matrix.items():
                for cat, ings in cats.items():
                    for ing in ings:
                        if ing in build_available_set(ingredients):
                            if ing not in drafted or ing == current_ingredient:
                                available_ingredients.append((ing, ingredient_to_category.get(ing, cat)))
            
            # Remove duplicates and sort
            available_ingredients = list(set(available_ingredients))
            available_ingredients.sort(key=lambda x: (x[1], x[0]))  # Sort by category, then ingredient
            
            if available_ingredients:
                new_ingredient = st.selectbox(
                    "New ingredient:",
                    [ing for ing, cat in available_ingredients],
                    key="swap_ingredient_select"
                )
                new_category = next((cat for ing, cat in available_ingredients if ing == new_ingredient), "Unknown")
            else:
                st.warning("No ingredients available for swap")
                new_ingredient = None
                new_category = None
        
        with swap_cols[2]:
            if new_ingredient and st.button("Execute Swap", key="execute_swap_btn"):
                swap_pick(selected_pick_idx, new_ingredient, new_category)
                st.success(f"Swapped to {new_ingredient}")
                st.rerun()
        
        # Show the pick being modified
        if selected_pick_idx is not None:
            old_rec = draft_log[selected_pick_idx]
            st.info(f"**Changing:** {old_rec.get('Player', '?')} • Round {old_rec.get('Round', '?')} • {old_rec.get('Ingredient', '?')} ({old_rec.get('Category', '?')})")
    else:
        st.info("No picks to swap yet.")

st.divider()

# --- Live rule status panel ---
rules = compute_rules_status(my_picks, ingredient_to_category, TOTAL_PICKS)

st.sidebar.header("Your Draft Status")
colA, colB = st.sidebar.columns(2)
with colA:
    st.metric("Picks Used", f"{len(my_picks)}/{TOTAL_PICKS}")
with colB:
    st.metric("Flex Left", f"{rules['flex_remaining']}/{LEAGUE['flex_slots']}")

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
current_round, current_seat = pick_slot(overall_pick, int(num_players))
current_player = players[current_seat] if overall_pick <= total_picks_overall else None

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Draft Board", "Style Viability", "Recommendations",
    "Blocks (deny-their-build)", "Mock Draft Simulator",
    "Results / Export", "Hop Finder"
])

# --- Draft Board ---
with tab1:
    st.subheader("Available Ingredients")

    if current_player:
        st.info(f"Round {current_round} • Pick {overall_pick} → {current_player}")
    else:
        st.success("Draft complete.")

    # Allow quick text filtering of ingredients
    filter_text = st.text_input("Filter ingredients", key="draft-filter")

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

    # Build long list of available ingredients by category, based on sheet
    # columns (alias-aware). Alias lists come from the league config.
    long_rows = []

    def add_from_aliases(alias_list, category_label):
        for colname in alias_list:
            if colname in ingredients.columns:
                for val in ingredients[colname].dropna().unique().tolist():
                    long_rows.append({"Ingredient": str(val), "Category": category_label})

    for category_label, alias_list in LEAGUE["category_aliases"].items():
        add_from_aliases(alias_list, category_label)

    df_long = pd.DataFrame(long_rows).drop_duplicates()

    # Remove those already drafted
    df_long = df_long[~df_long["Ingredient"].isin(drafted)]

    # Apply text filter if provided
    if filter_text:
        df_long = df_long[df_long["Ingredient"].str.contains(filter_text, case=False)]

    # Quick availability summary
    avail_summary = df_long.groupby("Category")["Ingredient"].nunique().reindex(all_categories).fillna(0).astype(int)
    st.caption("Available now → " + " | ".join([f"{cat}: {avail_summary.loc[cat]}" for cat in all_categories]))

    # Show by category — keep Base Malt, Yeast, Hop open by default
    for cat in all_categories:
        sub = df_long[df_long["Category"]==cat]
        with st.expander(f"{cat} ({len(sub)})", expanded=(cat in ["Base Malt","Yeast","Hop"])):
            for ing in sorted(sub["Ingredient"].tolist()):
                # Create ingredient row with proper layout and hover
                with st.container():
                    cols = st.columns([6, 2])
                    
                    with cols[0]:
                        label = f"**{ing}**"
                        popularity_info = ""
                        if ing in ingredient_popularity:
                            rec = ingredient_popularity[ing]
                            popularity_info = f"<br><small style='opacity: 0.7;'>pop: {rec.get('Picks',0)} | avg slot: {round(float(rec.get('Avg_Slot',0)),1)}</small>"
                        
                        # Create the hoverable ingredient row
                        st.markdown(f"""
                        <div class="ingredient-row">
                            <div class="ingredient-info">
                                {label}{popularity_info}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        if current_player and st.button("Draft", key=f"draft-{cat}-{ing}", use_container_width=True):
                            add_pick(current_player, ing, cat)

    st.divider()
    st.subheader("My Picks")
    st.write(my_picks if my_picks else "No picks yet.")
    st.subheader("All Drafted (any team)")
    st.write(drafted if drafted else "Nothing drafted yet.")

# --- Style Viability (compute_style_status imported from draft_core) ---
with tab2:
    st.subheader("Viable Styles (live)")
    viab = compute_style_status(my_picks, drafted, style_matrix, required, flex_slots)
    st.dataframe(viab, use_container_width=True)

# --- Recommendations with opponent bias (next_best_picks from draft_core) ---

def recommend(my_picks_arg, drafted_arg, top_k=15):
    """Thin adapter binding the shared next_best_picks to this app's data."""
    return next_best_picks(
        my_picks_arg, drafted_arg, ingredients, style_matrix, scarcity_df,
        required, flex_slots, ingredient_to_category, style_bias,
        early_signal=early_signal, bias_weight=bias_weight, top_k=top_k,
    )

with tab3:
    st.subheader("Best Next Picks (live, opponent-aware)")
    recs = recommend(my_picks, drafted)
    st.dataframe(recs, use_container_width=True)

# --- Block Picks (deny their build) — block_picks imported from draft_core ---

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
        recs_p = recommend(picks_p, drafted, top_k=3)
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
    blocks = block_picks(drafted, my_picks, pair_lookup, ingredients, early_signal=early_signal, top_k=15)
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
                recs = recommend(my_local, drafted_local, top_k=10)
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
    st.subheader("Team Summary")
    df_log = pd.DataFrame(
        st.session_state["draft_log"],
        columns=["Round", "Overall", "Player", "Ingredient", "Category"],
    )
    summary_rows = []
    for p in players:
        slots = {"Malt": "", "Hop": "", "Yeast": "", "Adjunct": "",
                 "Flex1": "", "Flex2": "", "Flex3": ""}
        if enable_round8:
            slots["Round8"] = ""
        flex_keys = ["Flex1", "Flex2", "Flex3"]
        flex_idx = 0
        player_rows = df_log[df_log["Player"] == p]
        for _, r in player_rows.iterrows():
            cat = r.get("Category", "")
            ing = r.get("Ingredient", "")
            bucket = bucket_for_rules(str(cat))
            if bucket in ["Malt", "Hop", "Yeast", "Adjunct"] and slots[bucket] == "":
                slots[bucket] = ing
            else:
                if flex_idx < len(flex_keys):
                    slots[flex_keys[flex_idx]] = ing
                    flex_idx += 1
                elif enable_round8:
                    slots["Round8"] = ing
        summary_rows.append({"Player": p, **slots})
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("Draft Results")
    edited = st.data_editor(df_log, num_rows="dynamic", use_container_width=True, key="draft_editor")
    if not edited.equals(df_log):
        st.session_state["draft_log"] = edited.to_dict("records")
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})
        st.rerun()
    if not edited.empty:
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, file_name="draft_results.csv", mime="text/csv")
        # Excel export needs the optional openpyxl engine; degrade gracefully
        # so a missing dependency can't crash the whole Results tab.
        try:
            excel_buf = io.BytesIO()
            edited.to_excel(excel_buf, index=False)
            st.download_button(
                "Download Excel", excel_buf.getvalue(), file_name="draft_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.caption("Install `openpyxl` to enable Excel export.")
with tab7:
    st.subheader('Hop Similarity Finder')
    search = st.text_input('Search hops', key='hop-search')
    hop_list = sorted(hop_similarity_data.keys())
    if search:
        hop_list = [h for h in hop_list if search.lower() in h.lower()]

    for hop in hop_list:
        label = f'~~{hop}~~' if hop in drafted else hop
        st.markdown(f'**{label}**')
        similars = []
        for rec in hop_similarity_data.get(hop, []):
            alt = rec.get('hop')
            alt_label = f'~~{alt}~~' if alt in drafted else alt
            similars.append(alt_label)
        if similars:
            st.caption('Similar: ' + ', '.join(similars))
        else:
            st.caption('No similarity data available.')


st.caption("Tip: Toggle Room Bias in the sidebar to lean into opponent tendencies. Blocks tab suggests denial picks based on the last few opponent selections.")

# Keep timer updating only while running and draft not complete
if st.session_state.get("timer_running") and current_player:
    time.sleep(1)
    st.rerun()

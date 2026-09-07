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
    compute_rules_status_from_records,
    roster_slots,
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
    # The ingredient set (year) comes from league config, not a hardcoded path.
    cfg = load_league_config()
    return draft_core.load_data(ingredients_path=cfg["ingredients_path"])

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
# Full pick records for your seat (keep the stored Category, which the rules
# logic trusts) — ordered as drafted.
my_records = [r for r in draft_log if r.get("Player") == your_name]

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
st.markdown("#### 🎯 Draft Management")

mgmt_cols = st.columns([2, 4])
with mgmt_cols[0]:
    if st.button("↶ Undo Last Pick", disabled=len(draft_log) == 0, key="undo_btn", use_container_width=True):
        undo_last_pick()
        st.rerun()
with mgmt_cols[1]:
    st.caption("Undo removes the most recent pick (press repeatedly to step back).")

# --- Swap/Trade Interface (always discoverable, collapsed by default) ---
with st.expander("🔄 Swaps / Trades / Round 8 swap", expanded=False):
    st.caption("Handle trades or the optional Round 8 swap. Select a pick to change and choose a new ingredient.")
    
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
# Trust the drafted category (record-based) so adjuncts not modeled in the
# style matrix still satisfy the Adjunct requirement rather than a flex slot.
rules = compute_rules_status_from_records(my_records, TOTAL_PICKS)

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

# --- Whose turn + always-visible roster slot strip ---------------------------
# Visible regardless of active tab so, during a fast live draft, you can always
# see whose pick it is and which of your slots still need filling.
if current_player:
    on_the_clock = "🟢 **YOUR PICK**" if current_player == your_name else f"On the clock: **{current_player}**"
    st.markdown(f"### Round {current_round} · Pick {overall_pick} — {on_the_clock}")
else:
    st.markdown("### Draft complete")


def _short(name, n=22):
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "…"


my_slots = roster_slots(my_records, enable_round8=enable_round8,
                        flex_slots=LEAGUE["flex_slots"])
# Track which required buckets are still open — reused by the board/rec badges.
needed_buckets = {s["key"] for s in my_slots if s["required"] and not s["filled"]}

st.caption(f"Your roster — {your_name}" if your_name else "Your roster")
slot_cols = st.columns(len(my_slots))
for col, s in zip(slot_cols, my_slots):
    with col:
        box = st.container(border=True)
        if s["filled"]:
            box.markdown(f"✅ **{s['label']}**")
            box.caption(_short(s["filled"]), help=s["filled"])
        elif s["required"]:
            box.markdown(f"🔴 **{s['label']}**")
            box.caption("needed")
        else:
            box.markdown(f"⬜ {s['label']}")
            box.caption("open")

if not rules["feasible"]:
    st.error("⚠️ Not enough picks remaining to fill all required categories.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Draft Board", "Style Viability", "Recommendations",
    "Blocks (deny-their-build)", "Mock Draft Simulator",
    "Results / Export", "Hop Finder"
])

# --- Draft Board ---
def add_pick(player, ing, cat):
    overall = len(st.session_state["draft_log"]) + 1
    round_no = ((overall - 1) // int(num_players)) + 1
    st.session_state["draft_log"].append({
        "Round": round_no,
        "Overall": overall,
        "Player": player,
        "Ingredient": ing,
        "Category": cat,
    })
    save_state({"players": players, "draft_log": st.session_state["draft_log"]})
    st.rerun()


# Long list of available ingredients by sheet category (alias-aware). Built once
# so both the board and the recommendation panel resolve the *true* board
# category (not the style-matrix one) when a pick is made.
long_rows = []
for category_label, alias_list in LEAGUE["category_aliases"].items():
    for colname in alias_list:
        if colname in ingredients.columns:
            for val in ingredients[colname].dropna().unique().tolist():
                long_rows.append({"Ingredient": str(val), "Category": category_label})
df_long_all = pd.DataFrame(long_rows).drop_duplicates()
board_category_of = dict(zip(df_long_all["Ingredient"], df_long_all["Category"]))


def recommend(my_picks_arg, drafted_arg, top_k=15, seat_index=None, overall=None):
    """Thin adapter binding the shared next_best_picks to this app's data.

    Passes the live opponent/snake context so the scarcity, urgency, synergy
    and denial components are active (they degrade gracefully if data is off).
    """
    return next_best_picks(
        my_picks_arg, drafted_arg, ingredients, style_matrix, scarcity_df,
        required, flex_slots, ingredient_to_category, style_bias,
        early_signal=early_signal, bias_weight=bias_weight, top_k=top_k,
        hop_similarity=hop_similarity_data, pair_lookup=pair_lookup,
        num_players=int(num_players),
        overall_pick=overall if overall is not None else overall_pick,
        your_seat_index=seat_index if seat_index is not None else (int(draft_position) - 1),
    )


def _needed_badge(cat):
    return "⭐ " if bucket_for_rules(cat) in needed_buckets else ""


def render_board(df_long):
    avail = (df_long.groupby("Category")["Ingredient"].nunique()
             .reindex(all_categories).fillna(0).astype(int))
    st.caption("Available now → " + " | ".join(
        f"{cat}: {avail.loc[cat]}" for cat in all_categories))
    for cat in all_categories:
        sub = df_long[df_long["Category"] == cat]
        star = "⭐ " if bucket_for_rules(cat) in needed_buckets else ""
        with st.expander(f"{star}{cat} ({len(sub)})",
                         expanded=(cat in ["Base Malt", "Yeast", "Hop"])):
            for ing in sorted(sub["Ingredient"].tolist()):
                cols = st.columns([6, 2])
                pop = ""
                if ing in ingredient_popularity:
                    rec = ingredient_popularity[ing]
                    pop = f"  ·  pop {rec.get('Picks', 0)}, avg slot {round(float(rec.get('Avg_Slot', 0)), 1)}"
                cols[0].markdown(f"{_needed_badge(cat)}**{ing}**{pop}")
                if current_player and cols[1].button(
                        "Draft", key=f"draft-{cat}-{ing}", use_container_width=True):
                    add_pick(current_player, ing, cat)


def render_recs():
    st.caption("Ranked for your roster & the live board. ⭐ = fills a needed slot.")
    recs = recommend(my_picks, drafted, top_k=12)
    if recs.empty:
        st.info("No candidates to recommend.")
        return
    for _, r in recs.iterrows():
        ing, cat = r["Ingredient"], r["Category"]
        cols = st.columns([6, 2])
        cols[0].markdown(f"{_needed_badge(cat)}**{ing}**  ·  _{cat}_")
        cols[0].caption(f"{r['Why']}  ·  value {r['Pick Value']:.2f}")
        if current_player and cols[1].button(
                "Draft", key=f"rec-draft-{ing}", use_container_width=True):
            add_pick(current_player, ing, board_category_of.get(ing, cat))


with tab1:
    if current_player:
        st.info(f"Round {current_round} • Pick {overall_pick} → {current_player}")
    else:
        st.success("Draft complete.")

    top = st.columns([2, 2, 1])
    filter_text = top[0].text_input("Filter ingredients", key="draft-filter",
                                    label_visibility="collapsed", placeholder="Filter ingredients…")
    compact = top[2].toggle("Phone", key="compact_mode",
                            help="Stacked single-column layout for phones/tablets")

    df_long = df_long_all[~df_long_all["Ingredient"].isin(drafted)]
    if filter_text:
        df_long = df_long[df_long["Ingredient"].str.contains(filter_text, case=False)]

    if compact:
        # Single-column: switch between board and recs (no side-by-side scroll).
        view = st.radio("View", ["🎯 Recommended", "📋 Board"], horizontal=True,
                        label_visibility="collapsed")
        if view.endswith("Recommended"):
            render_recs()
        else:
            render_board(df_long)
    else:
        board_col, rec_col = st.columns([3, 2], gap="large")
        with board_col:
            st.subheader("Available Ingredients")
            render_board(df_long)
        with rec_col:
            st.subheader("Recommended for you")
            render_recs()

    st.divider()
    with st.expander("My picks & everything drafted", expanded=False):
        st.markdown("**My Picks:** " + (", ".join(my_picks) if my_picks else "_none yet_"))
        st.markdown("**All Drafted:** " + (", ".join(drafted) if drafted else "_nothing yet_"))

# --- Style Viability (compute_style_status imported from draft_core) ---
with tab2:
    st.subheader("Viable Styles (live)")
    viab = compute_style_status(my_picks, drafted, style_matrix, required, flex_slots)
    st.dataframe(viab, use_container_width=True)

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

# Opponent archetypes as weight overrides on the shared scoring model. Each
# simulated opponent drafts with the SAME engine you do, just tuned differently,
# and tracks its own roster — so runs stress-test your strategy realistically
# instead of replaying a scripted "base malt run".
OPPONENT_PERSONAS = {
    "Value":      {},                                    # balanced default
    "Hop-head":   {"syn": 0.30, "fit": 0.22, "scarce": 0.20},
    "Chalk":      {"pop": 0.20, "fit": 0.25, "need": 0.18},
    "Needs-first": {"need": 0.40, "fit": 0.22, "scarce": 0.18},
}
_PERSONA_NAMES = list(OPPONENT_PERSONAS)


def _softmax_choice(values, temperature=8.0):
    """Index sampled from a softmax over values (higher = more likely)."""
    import math
    if not values:
        return 0
    m = max(values)
    exps = [math.exp((v - m) * temperature) for v in values]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    return random.choices(range(len(values)), weights=probs, k=1)[0]


def sim_agent_pick(roster, drafted_local, weights, sim_players, overall, seat_index,
                   top_k=8, greedy=False):
    """One pick from the shared recommender for a given roster + persona weights.

    Opponents softmax-sample from the top candidates (realistic variation); you
    can pass greedy=True to always take the top pick.
    """
    recs = next_best_picks(
        roster, drafted_local, ingredients, style_matrix, scarcity_df,
        required, flex_slots, ingredient_to_category, style_bias,
        early_signal=early_signal, hop_similarity=hop_similarity_data,
        pair_lookup=pair_lookup, num_players=sim_players, overall_pick=overall,
        your_seat_index=seat_index, weights=weights, top_k=top_k,
    )
    recs = recs[~recs["Ingredient"].isin(drafted_local)]
    if recs.empty:
        cand_map = sim_available_candidates(style_matrix, ingredients, drafted_local)
        if not cand_map:
            return None, None
        return random.choice(list(cand_map.items()))
    idx = 0 if greedy else _softmax_choice(recs["Pick Value"].tolist())
    row = recs.iloc[idx]
    return row["Ingredient"], row["Category"]


def simulate_draft(sim_players, sim_rounds, your_pos, bias_weight, my_weights=None):
    """Run a full snake draft where every seat uses the shared engine.

    You draft greedily from your recommendations; each opponent seat is assigned
    a persona (weight profile) and softmax-samples its pick from its own
    roster-aware recommendations. Returns (log, your_picks, drafted).
    """
    drafted_local = list(drafted)
    rosters = {seat: [] for seat in range(1, sim_players + 1)}
    my_local = list(my_picks)
    rosters[your_pos] = my_local
    # Deterministic persona per opponent seat (seed set by caller).
    seat_persona = {seat: _PERSONA_NAMES[(seat - 1) % len(_PERSONA_NAMES)]
                    for seat in range(1, sim_players + 1)}

    log = []
    overall = 0
    for rnd in range(1, sim_rounds + 1):
        order = (list(range(1, sim_players + 1)) if rnd % 2 == 1
                 else list(range(sim_players, 0, -1)))
        for seat in order:
            overall += 1
            if seat == your_pos:
                ing, cat = sim_agent_pick(my_local, drafted_local, my_weights,
                                          sim_players, overall, seat - 1, greedy=True)
                reason = "your_top_pick"
                team = "You"
            else:
                persona = seat_persona[seat]
                ing, cat = sim_agent_pick(rosters[seat], drafted_local,
                                          OPPONENT_PERSONAS[persona], sim_players,
                                          overall, seat - 1)
                reason = f"persona:{persona}"
                team = f"Opp {seat}"
            if ing is None:
                continue
            rosters[seat].append(ing)
            drafted_local.append(ing)
            log.append({"Round": rnd, "Overall": overall, "Seat": seat,
                        "Team": team, "Ingredient": ing, "Category": cat,
                        "Reason": reason})

    return log, my_local, drafted_local

with tab5:
    st.subheader("Mock Draft Simulator")
    st.caption("Simulates a full snake draft where every seat uses the shared engine. "
               "Opponents are assigned personas (Value / Hop-head / Chalk / Needs-first) "
               "and roster-aware; you draft greedily from your recommendations.")

    c1, c2, c3 = st.columns(3)
    sim_players = c1.number_input("Players", min_value=4, max_value=20, value=int(num_players), step=1, key="sim_players")
    sim_rounds = c2.number_input("Rounds", min_value=1, max_value=10, value=7, step=1, key="sim_rounds")
    your_pos_sim = c3.number_input("Your position", min_value=1, max_value=int(sim_players), value=int(draft_position), step=1, key="sim_your_pos")

    sim_seed = st.number_input("Random seed", min_value=0, max_value=10**9, value=42, step=1, key="sim_seed")

    run = st.button("Run mock draft", key="run_mock")
    if run:
        random.seed(int(sim_seed))
        log, my_local, drafted_local = simulate_draft(
            int(sim_players), int(sim_rounds), int(your_pos_sim), bias_weight)

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
        player_records = [r for r in st.session_state["draft_log"]
                          if r.get("Player") == p]
        # Single source of truth for slot attribution (shared with the roster
        # strip); trusts the stored Category so adjuncts land in the right slot.
        slots = {s["label"]: (s["filled"] or "")
                 for s in roster_slots(player_records, enable_round8=enable_round8,
                                       flex_slots=LEAGUE["flex_slots"])}
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

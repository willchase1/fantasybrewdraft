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
    best_available,
    team_context,
    style_build_plan,
    ingredient_detail,
    block_picks,
    pick_slot,
)

# Page config is set by the app.py entry point (st.set_page_config must run
# once, before any other Streamlit command). This module is a navigation page.

# --- Global CSS for better UX ---
st.markdown("""
<style>
/* Projector layout: reclaim the big default top gap so the board sits high. */
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }
[data-testid="stHeader"] { background: transparent; }
/* Tighten vertical rhythm between stacked elements in the top bar. */
div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
/* Compact top-bar clock */
.wr-clock {
    font-family: 'Courier New', monospace;
    font-size: 30px;
    font-weight: bold;
    line-height: 1.1;
}
.wr-title { font-size: 26px; font-weight: 800; line-height: 1.0; margin: 0; }
/* Focus-team roster chip line */
.wr-roster { font-size: 15px; line-height: 1.9; }
.wr-chip {
    display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.15);
}
.wr-chip.filled { background: rgba(46, 204, 113, 0.18); }
.wr-chip.need   { background: rgba(231, 76, 60, 0.20); }
.wr-chip.open   { background: rgba(255,255,255,0.06); opacity: 0.75; }
</style>
""", unsafe_allow_html=True)

# Club branding — shown prominently in the top bar (see below).
LOGO_PATH = "rhbc_logo_round.png"

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
    """Load ingredient similarity resources (hops, yeasts, malts) via league config.

    Returns the merged neighbour lookup plus the dense hop matrix (kept for
    tooling). The name is historical; the lookup covers every category that
    ``league_config.json`` names a similarity file for.
    """
    sim = draft_core.load_similarity()
    sim_matrix = pd.DataFrame()
    try:
        sim_matrix = pd.read_csv("hop_similarity_matrix.csv", index_col=0)
    except Exception:
        pass
    return sim, sim_matrix


hop_similarity_data, hop_similarity_matrix = load_hop_similarity_data()

# --- Persist draft state locally (the log is the single source of truth) ---
def load_state():
    """Load persisted draft ({players, draft_log}) via draft_state."""
    return draft_state.load_draft()

def save_state(state):
    """Persist the draft ({players, draft_log}) via draft_state."""
    draft_state.save_draft(state)

all_categories = ["Base Malt", "Hop", "Yeast", "Adjunct", "Specialty"]

# --- Rulebook-aware requirements status ---
LEAGUE = load_league_config()
DEFAULT_ROUNDS = LEAGUE["rounds"]  # base number of rounds before optional round 8

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
    "My seat (for 🟢 highlight only)", min_value=1, max_value=num_players,
    value=min(num_players, prev_draft_pos), step=1
)
st.session_state["draft_pos"] = int(draft_position)
st.sidebar.caption("The board follows whoever is on the clock; this only marks your own turn.")

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

# your_name/draft_position are used ONLY to highlight your own turn — the live
# view otherwise follows the team on the clock.
your_name = players[int(draft_position) - 1] if players else ""

# ---------------------------------------------------------------------------
# Draft timer — state + controls
# ---------------------------------------------------------------------------
if "timer_duration" not in st.session_state:
    st.session_state.timer_duration = 120
if "timer_started_at" not in st.session_state:
    st.session_state.timer_started_at = time.time()
if "timer_elapsed" not in st.session_state:
    st.session_state.timer_elapsed = 0.0
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False


def start_draft_timer():
    if not st.session_state.timer_running:
        st.session_state.timer_running = True
        st.session_state.timer_started_at = time.time()


def pause_draft_timer():
    if st.session_state.timer_running:
        st.session_state.timer_elapsed += time.time() - st.session_state.timer_started_at
        st.session_state.timer_running = False


def reset_draft_timer():
    st.session_state.timer_elapsed = 0.0
    st.session_state.timer_running = False
    st.session_state.timer_started_at = time.time()


elapsed = st.session_state.timer_elapsed + (
    time.time() - st.session_state.timer_started_at if st.session_state.timer_running else 0.0
)
remaining = max(0, int(st.session_state.timer_duration - elapsed))
if remaining == 0 and st.session_state.timer_running:
    st.session_state.timer_running = False

# ---------------------------------------------------------------------------
# Draft management functions
# ---------------------------------------------------------------------------
def undo_last_pick():
    if st.session_state.get("draft_log"):
        st.session_state["draft_log"].pop()
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})


def swap_pick(pick_index, new_ingredient, new_category):
    if 0 <= pick_index < len(st.session_state["draft_log"]):
        st.session_state["draft_log"][pick_index]["Ingredient"] = new_ingredient
        st.session_state["draft_log"][pick_index]["Category"] = new_category
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})


# ---------------------------------------------------------------------------
# Current draft state (whose turn) + opponent accessors
# ---------------------------------------------------------------------------
total_picks_overall = TOTAL_PICKS * int(num_players)
overall_pick = len(draft_log) + 1
current_round, current_seat = pick_slot(overall_pick, int(num_players))
current_player = players[current_seat] if overall_pick <= total_picks_overall else None

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
            pair_lookup[(a, b)] += c
            pair_lookup[(b, a)] += c

# ---------------------------------------------------------------------------
# Board data + core helpers
# ---------------------------------------------------------------------------
def add_pick(player, ing, cat):
    overall = len(st.session_state["draft_log"]) + 1
    round_no = ((overall - 1) // int(num_players)) + 1
    st.session_state["draft_log"].append({
        "Round": round_no, "Overall": overall, "Player": player,
        "Ingredient": ing, "Category": cat,
    })
    save_state({"players": players, "draft_log": st.session_state["draft_log"]})
    # The clock is per-selection: reset to a full duration and start it running
    # for the next picker the moment a pick is made.
    st.session_state.timer_elapsed = 0.0
    st.session_state.timer_started_at = time.time()
    st.session_state.timer_running = True
    st.rerun()


# Long list of available ingredients by sheet category (alias-aware). Built once
# so the board and best-available resolve the *true* board category on a pick.
long_rows = []
for category_label, alias_list in LEAGUE["category_aliases"].items():
    for colname in alias_list:
        if colname in ingredients.columns:
            for val in ingredients[colname].dropna().unique().tolist():
                long_rows.append({"Ingredient": str(val), "Category": category_label})
df_long_all = pd.DataFrame(long_rows).drop_duplicates()
board_category_of = dict(zip(df_long_all["Ingredient"], df_long_all["Category"]))


# Weights / shape constants: code defaults overlaid with any league_config.json
# overrides (``pick_weights``, ``board_value_weights``, ``squash``).
PICK_WEIGHTS, BOARD_WEIGHTS, SQUASH = draft_core.scoring_params()


def recommend(picks_arg, drafted_arg, top_k=15, seat_index=None, overall=None):
    """Adapter binding the shared next_best_picks to this app's live context."""
    return next_best_picks(
        picks_arg, drafted_arg, ingredients, style_matrix, scarcity_df,
        required, flex_slots, ingredient_to_category, style_bias,
        early_signal=early_signal, bias_weight=bias_weight, top_k=top_k,
        hop_similarity=hop_similarity_data, pair_lookup=pair_lookup,
        num_players=int(num_players),
        overall_pick=overall if overall is not None else overall_pick,
        your_seat_index=seat_index if seat_index is not None else (int(draft_position) - 1),
        weights=PICK_WEIGHTS, squash=SQUASH,
    )


def _short(name, n=22):
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Mock draft simulation helpers (used by the Tools drawer)
# ---------------------------------------------------------------------------
def sim_available_candidates(style_matrix, ingredients, drafted):
    avail_set = build_available_set(ingredients)
    drafted_set = set(drafted)
    cand = {}
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing in avail_set and ing not in drafted_set:
                    cand.setdefault(ing, ingredient_to_category.get(ing, cat))
    return cand


# Opponent archetypes as weight overrides on the shared scoring model.
OPPONENT_PERSONAS = {
    "Value":       {},
    "Hop-head":    {"syn": 0.30, "fit": 0.22, "scarce": 0.20},
    "Chalk":       {"pop": 0.20, "fit": 0.25, "need": 0.18},
    "Needs-first": {"need": 0.40, "fit": 0.22, "scarce": 0.18},
}
_PERSONA_NAMES = list(OPPONENT_PERSONAS)


def _softmax_choice(values, temperature=8.0):
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
    recs = next_best_picks(
        roster, drafted_local, ingredients, style_matrix, scarcity_df,
        required, flex_slots, ingredient_to_category, style_bias,
        early_signal=early_signal, hop_similarity=hop_similarity_data,
        pair_lookup=pair_lookup, num_players=sim_players, overall_pick=overall,
        your_seat_index=seat_index, weights={**PICK_WEIGHTS, **(weights or {})},
        squash=SQUASH, top_k=top_k,
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
    drafted_local = list(drafted)
    rosters = {seat: [] for seat in range(1, sim_players + 1)}
    my_local = list(teams.get(players[your_pos - 1], [])) if players else []
    rosters[your_pos] = my_local
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
                reason, team = "your_top_pick", "You"
            else:
                persona = seat_persona[seat]
                ing, cat = sim_agent_pick(rosters[seat], drafted_local,
                                          OPPONENT_PERSONAS[persona], sim_players,
                                          overall, seat - 1)
                reason, team = f"persona:{persona}", f"Opp {seat}"
            if ing is None:
                continue
            rosters[seat].append(ing)
            drafted_local.append(ing)
            log.append({"Round": rnd, "Overall": overall, "Seat": seat,
                        "Team": team, "Ingredient": ing, "Category": cat,
                        "Reason": reason})
    return log, my_local, drafted_local


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def render_compact_timer():
    color = "#e74c3c" if remaining <= 10 else ("#f39c12" if remaining <= 30 else "#2ecc71")
    run = st.session_state.timer_running
    st.markdown(
        f"<div class='wr-clock' style='color:{color};'>⏱ {remaining//60:02d}:{remaining%60:02d}"
        f" {'▶' if run else '⏸'}</div>", unsafe_allow_html=True)
    b = st.columns(3)
    if run:
        b[0].button("⏸", on_click=pause_draft_timer, key="t_pause", use_container_width=True, help="Pause")
    else:
        b[0].button("▶", on_click=start_draft_timer, key="t_start", use_container_width=True, help="Start")
    b[1].button("↺", on_click=reset_draft_timer, key="t_reset", use_container_width=True, help="Reset")
    with b[2].popover("⚙", use_container_width=True):
        mins = st.number_input("Duration (min)", 1, 60,
                               value=max(1, int(round(st.session_state.timer_duration / 60))),
                               key="timer_minutes")
        if mins * 60 != st.session_state.timer_duration:
            st.session_state.timer_duration = mins * 60


def render_roster_line(slots):
    chips = []
    for s in slots:
        if s["filled"]:
            chips.append(f"<span class='wr-chip filled'>✅ {s['label']}: "
                         f"<b>{_short(s['filled'], 16)}</b></span>")
        elif s["required"]:
            chips.append(f"<span class='wr-chip need'>🔴 {s['label']}</span>")
        else:
            chips.append(f"<span class='wr-chip open'>⬜ {s['label']}</span>")
    st.markdown("<div class='wr-roster'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


def render_board(df_long, needed):
    avail = (df_long.groupby("Category")["Ingredient"].nunique()
             .reindex(all_categories).fillna(0).astype(int))
    st.caption("Available → " + " | ".join(f"{c}: {avail.loc[c]}" for c in all_categories))
    for cat in all_categories:
        sub = df_long[df_long["Category"] == cat]
        star = "⭐ " if bucket_for_rules(cat) in needed else ""
        with st.expander(f"{star}{cat} ({len(sub)})",
                         expanded=(cat in ["Base Malt", "Yeast", "Hop"])):
            for ing in sorted(sub["Ingredient"].tolist()):
                cols = st.columns([6, 2])
                pop = ""
                if ing in ingredient_popularity:
                    rec = ingredient_popularity[ing]
                    pop = f"  ·  pop {rec.get('Picks', 0)}, avg {round(float(rec.get('Avg_Slot', 0)), 1)}"
                badge = "⭐ " if bucket_for_rules(cat) in needed else ""
                cols[0].markdown(f"{badge}**{ing}**{pop}")
                if current_player and cols[1].button(
                        "Draft", key=f"draft-{cat}-{ing}", use_container_width=True):
                    add_pick(current_player, ing, cat)


def render_best_available(focus_player, focus_picks, focus_needed):
    # Roster-aware: rank toward the focus team's emerging build (need, synergy,
    # and style focus all active) rather than a generic global board list.
    if current_player and focus_player == current_player:
        st.caption(f"Ranked for **{focus_player}**'s build")
    elif current_player:
        st.caption(f"Ranked for **{focus_player}** · drafting acts for **{current_player}**")
    else:
        st.caption(f"Ranked for **{focus_player}**")
    # Single full-width filter (avoids a cramped toggle wrapping to "N e e d s").
    sel = st.selectbox("Filter", ["All", "⭐ Fills a need"] + all_categories,
                       key="ba_filter", label_visibility="collapsed")
    focus_seat = players.index(focus_player) if focus_player in players else 0
    recs = recommend(focus_picks, drafted, top_k=40, seat_index=focus_seat)
    if sel == "⭐ Fills a need":
        recs = recs[recs["Category"].map(lambda c: bucket_for_rules(c) in focus_needed)]
    elif sel != "All":
        recs = recs[recs["Category"] == sel]
    recs = recs.head(15)
    if recs.empty:
        st.info("No matching ingredients on the board.")
        return
    for _, r in recs.iterrows():
        ing, cat = r["Ingredient"], r["Category"]
        star = "⭐ " if bucket_for_rules(cat) in focus_needed else ""
        cols = st.columns([6, 2])
        cols[0].markdown(f"{star}**{ing}**  ·  _{cat}_")
        cols[0].caption(f"value {r['Pick Value']:.2f}  ·  {r['Why']}")
        if current_player and cols[1].button(
                "Draft", key=f"ba-draft-{ing}", use_container_width=True):
            add_pick(current_player, ing, board_category_of.get(ing, cat))


def render_styles(focus_picks):
    st.caption("Click a style to see how to build it from the board.")
    viab = compute_style_status(focus_picks, drafted, style_matrix, required, flex_slots)
    for _, r in viab.head(15).iterrows():
        style = r["Style"]
        matched = int(r.get("Picks Matched", 0))
        label = f"{style}  ·  matched {matched}" if matched else style
        if st.button(label, key=f"style-{style}", use_container_width=True):
            show_style_dialog(style)


@st.dialog("Ingredient")
def show_ingredient_dialog(ing):
    detail = ingredient_detail(
        ing, style_matrix, drafted, teams=teams, similarity=hop_similarity_data,
        popularity=ingredient_popularity,
        available_set=set(df_long_all["Ingredient"]),
        board_category=board_category_of.get(ing))
    st.markdown(f"### {ing}")
    cat = detail["category"] or "—"
    if detail["drafted_by"]:
        st.caption(f"{cat} · drafted by **{detail['drafted_by']}**")
    else:
        st.caption(f"{cat} · available")
    p = detail["popularity"]
    if p:
        st.caption(f"Historically picked {p.get('Picks', 0)}× · "
                   f"avg slot {round(float(p.get('Avg_Slot', 0)), 1)}")
    st.markdown("**Fits styles:** " + (", ".join(detail["styles"])
                if detail["styles"] else "_not modeled in any style_"))
    subs = [s for s in detail["similar"] if s["available"]]
    if subs:
        st.markdown("**Similar & still available:**")
        st.markdown("\n".join(f"- {s['ingredient']}  ·  {s['score']:.2f}"
                              for s in subs[:8]))
    elif detail["similar"]:
        st.caption("All close substitutes are already drafted.")
    else:
        st.caption("No similarity data for this ingredient.")
    if current_player and ing not in set(drafted):
        if st.button(f"➕ Draft {ing} for {current_player}", use_container_width=True):
            add_pick(current_player, ing, board_category_of.get(ing, cat))


@st.dialog("Build a style", width="large")
def show_style_dialog(style):
    st.markdown(f"### 🧪 Build: {style}")
    st.caption(f"For **{focus_player}**  ·  ✅ you have · 🟢 open · ❌ taken")
    seat = players.index(focus_player) if focus_player in players else 0
    recs = recommend(focus_picks, drafted, top_k=500, seat_index=seat)
    value_of = dict(zip(recs["Ingredient"], recs["Pick Value"]))
    plan = style_build_plan(style, style_matrix, focus_picks, drafted,
                            teams=teams, value_of=value_of)
    for row in plan:
        cat = row["category"]
        st.markdown(f"**{cat}**")
        if row["have"]:
            st.markdown("✅ " + ", ".join(row["have"]))
        open_opts = row["available"]  # excludes your own picks (already drafted)
        for ing in open_opts[:6]:
            c = st.columns([6, 2])
            c[0].markdown(f"🟢 {ing}")
            if current_player and c[1].button("Draft", key=f"sbp-draft-{cat}-{ing}",
                                              use_container_width=True):
                add_pick(current_player, ing, board_category_of.get(ing, cat))
        if len(open_opts) > 6:
            st.caption(f"+{len(open_opts) - 6} more available")
        if not row["have"] and not open_opts:
            st.caption("❌ none left on the board")
        if row["taken"]:
            st.caption("❌ taken: " + ", ".join(f"{i} ({p or '?'})"
                                                for i, p in row["taken"][:6]))
        st.divider()


def render_ingredient_lookup():
    """Searchable 'look up any ingredient' control that opens the detail dialog."""
    lu = st.columns([3, 1])
    names = ["—"] + sorted(df_long_all["Ingredient"].tolist())
    look = lu[0].selectbox("ℹ️ Look up ingredient", names, key="ing_lookup",
                           label_visibility="collapsed")
    if lu[1].button("View", key="ing_lookup_btn", use_container_width=True) and look != "—":
        show_ingredient_dialog(look)


# ===========================================================================
# TOP BAR — always visible: clock · whose turn · scout selector · roster
# ===========================================================================
_has_logo = os.path.exists(LOGO_PATH)
tb = st.columns(([0.7, 2.6, 2.3, 2] if _has_logo else [3, 2.3, 2]),
                gap="small", vertical_alignment="center")
i = 0
if _has_logo:
    tb[0].image(LOGO_PATH, width=76)
    i = 1
with tb[i]:
    if current_player:
        turn = ("🟢 YOUR PICK" if current_player == your_name
                else f"On the clock: {current_player}")
        st.caption(f"Round {current_round} · Pick {overall_pick}")
        st.markdown(f"<div class='wr-title'>{turn}</div>", unsafe_allow_html=True)
        # Who's up next (on deck).
        next_overall = overall_pick + 1
        if next_overall <= total_picks_overall:
            _, next_seat = pick_slot(next_overall, int(num_players))
            nxt = players[next_seat]
            st.caption(f"On deck: **{nxt}**" + (" 🟢" if nxt == your_name else ""))
    else:
        st.markdown("<div class='wr-title'>✅ Draft complete</div>", unsafe_allow_html=True)
with tb[i + 1]:
    render_compact_timer()
with tb[i + 2]:
    default_idx = players.index(current_player) if current_player in players else 0
    focus_player = st.selectbox("View team", players, index=default_idx, key="focus_player")

# Focus team drives the roster line, needs, styles, and best-available ⭐.
focus_records = [r for r in draft_log if r.get("Player") == focus_player]
focus_picks = teams.get(focus_player, [])
focus = team_context(focus_records, focus_picks, drafted, style_matrix,
                     required, flex_slots, enable_round8=enable_round8)
needed_buckets = focus["needed_buckets"]

roster_label = (f"**{focus_player}**" + (" 🟢" if focus_player == your_name else "")
                + f"  —  leaning: {focus['likely_style']}  ·  flex left: {focus['flex_remaining']}")
st.caption(roster_label)
render_roster_line(focus["slots"])
if not focus["feasible"]:
    st.error(f"⚠️ {focus_player} can't fill all required categories with the picks left.")

# ===========================================================================
# MAIN — 3-column war room (fixed-height scroll panels, no page scroll)
# ===========================================================================
PANEL_H = 600
compact = st.session_state.get("compact_mode", False)

df_live = df_long_all[~df_long_all["Ingredient"].isin(drafted)]

if compact:
    st.toggle("📱 Phone layout", key="compact_mode")
    view = st.segmented_control("View", ["📋 Board", "⭐ Recommended", "📊 Styles"],
                                default="⭐ Recommended", key="wr_view")
    render_ingredient_lookup()
    filt = st.text_input("Filter", key="board-filter", placeholder="Filter ingredients…",
                         label_visibility="collapsed")
    df_live_f = df_live[df_live["Ingredient"].str.contains(filt, case=False)] if filt else df_live
    if view == "📋 Board":
        render_board(df_live_f, needed_buckets)
    elif view == "📊 Styles":
        render_styles(focus_picks)
    else:
        render_best_available(focus_player, focus_picks, needed_buckets)
else:
    left, mid, right = st.columns([2, 1.4, 1.2], gap="medium")
    with left:
        head = st.columns([3, 1])
        head[0].subheader("📋 Board")
        head[1].toggle("📱", key="compact_mode", help="Phone / single-column layout")
        with left:
            render_ingredient_lookup()
        filt = left.text_input("Filter", key="board-filter", placeholder="Filter ingredients…",
                               label_visibility="collapsed")
        df_live_f = df_live[df_live["Ingredient"].str.contains(filt, case=False)] if filt else df_live
        with st.container(height=PANEL_H):
            render_board(df_live_f, needed_buckets)
    with mid:
        st.subheader("⭐ Recommended")
        with st.container(height=PANEL_H):
            render_best_available(focus_player, focus_picks, needed_buckets)
    with right:
        st.subheader(f"📊 Viable Styles")
        with st.container(height=PANEL_H):
            render_styles(focus_picks)

# ===========================================================================
# TOOLS DRAWER — admin off-screen until opened
# ===========================================================================
st.divider()
st.caption("🛠️ Tools")

with st.expander("↶ Draft management (undo / swaps / trades)", expanded=False):
    if st.button("↶ Undo last pick", disabled=len(draft_log) == 0, key="undo_btn"):
        undo_last_pick()
        st.rerun()
    st.caption("Undo removes the most recent pick (press repeatedly to step back).")
    st.markdown("**Swap / trade / Round 8 swap**")
    if draft_log:
        pick_options = [
            f"R{r.get('Round','?')} P{r.get('Overall','?')}: {r.get('Player','?')} → {r.get('Ingredient','?')}"
            for r in draft_log
        ]
        sc = st.columns([3, 3, 2])
        with sc[0]:
            selected_pick_idx = st.selectbox("Pick to change", range(len(pick_options)),
                                             format_func=lambda x: pick_options[x],
                                             key="swap_pick_select")
        with sc[1]:
            current_ingredient = draft_log[selected_pick_idx].get("Ingredient", "")
            avail_set = build_available_set(ingredients)
            available_ingredients = sorted({
                (ing, ingredient_to_category.get(ing, cat))
                for style, cats in style_matrix.items()
                for cat, ings in cats.items()
                for ing in ings
                if ing in avail_set and (ing not in drafted or ing == current_ingredient)
            }, key=lambda x: (x[1], x[0]))
            if available_ingredients:
                new_ingredient = st.selectbox("New ingredient",
                                              [ing for ing, _ in available_ingredients],
                                              key="swap_ingredient_select")
                new_category = next((c for ing, c in available_ingredients if ing == new_ingredient), "Unknown")
            else:
                st.warning("No ingredients available for swap")
                new_ingredient = new_category = None
        with sc[2]:
            if new_ingredient and st.button("Execute swap", key="execute_swap_btn"):
                swap_pick(selected_pick_idx, new_ingredient, new_category)
                st.success(f"Swapped to {new_ingredient}")
                st.rerun()
    else:
        st.info("No picks to swap yet.")

with st.expander(f"🎯 Detailed recommendations — {focus_player}", expanded=False):
    focus_seat = players.index(focus_player) if focus_player in players else 0
    recs = recommend(focus_picks, drafted, seat_index=focus_seat)
    st.dataframe(recs, use_container_width=True, hide_index=True)

with st.expander("🛡️ Blocks & opponent predictions", expanded=False):
    pred_rows = []
    for p in players:
        picks_p = teams.get(p, [])
        style_guess = "N/A"
        if picks_p:
            viab_p = compute_style_status(picks_p, drafted, style_matrix, required, flex_slots)
            if not viab_p.empty:
                style_guess = viab_p.iloc[0]["Style"]
        recs_p = recommend(picks_p, drafted, top_k=3, seat_index=players.index(p))
        next_guess = ", ".join(recs_p["Ingredient"].tolist()) if not recs_p.empty else ""
        pred_rows.append({"Player": p, "Picks": ", ".join(picks_p),
                          "Likely Style": style_guess, "Likely Next Picks": next_guess})
    st.markdown("**Player tendencies**")
    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
    st.markdown("**Block suggestions** (deny the focus team's rivals)")
    if opponent_model is None:
        st.info("Add opponent_model.json to enable block suggestions.")
    blocks = block_picks(drafted, focus_picks, pair_lookup, ingredients,
                         early_signal=early_signal, top_k=15)
    st.dataframe(blocks, use_container_width=True, hide_index=True)

with st.expander("🎲 Mock draft simulator", expanded=False):
    st.caption("Every seat drafts with the shared engine; opponents use personas "
               "(Value / Hop-head / Chalk / Needs-first) and are roster-aware.")
    c1, c2, c3 = st.columns(3)
    sim_players = c1.number_input("Players", 4, 20, value=int(num_players), step=1, key="sim_players")
    sim_rounds = c2.number_input("Rounds", 1, 10, value=7, step=1, key="sim_rounds")
    your_pos_sim = c3.number_input("Your position", 1, int(sim_players),
                                   value=int(draft_position), step=1, key="sim_your_pos")
    sim_seed = st.number_input("Random seed", 0, 10**9, value=42, step=1, key="sim_seed")
    if st.button("Run mock draft", key="run_mock"):
        random.seed(int(sim_seed))
        log, my_local, drafted_local = simulate_draft(
            int(sim_players), int(sim_rounds), int(your_pos_sim), bias_weight)
        st.markdown("#### Simulation Results")
        st.write(f"**Your picks ({len(my_local)}):** " + ", ".join(my_local))
        st.markdown("**Pick log** (last 40)")
        st.dataframe(pd.DataFrame(log).tail(40), use_container_width=True, hide_index=True)
        st.markdown("#### Final style viability (top 15)")
        viab_sim = compute_style_status(my_local, drafted_local, style_matrix, required, flex_slots).head(15)
        st.dataframe(viab_sim, use_container_width=True, hide_index=True)

with st.expander("📤 Results / export", expanded=False):
    df_log = pd.DataFrame(st.session_state["draft_log"],
                          columns=["Round", "Overall", "Player", "Ingredient", "Category"])
    summary_rows = []
    for p in players:
        precs = [r for r in st.session_state["draft_log"] if r.get("Player") == p]
        slots = {s["label"]: (s["filled"] or "")
                 for s in roster_slots(precs, enable_round8=enable_round8,
                                       flex_slots=LEAGUE["flex_slots"])}
        summary_rows.append({"Player": p, **slots})
    if summary_rows:
        st.markdown("**Team summary**")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.markdown("**Draft results (editable)**")
    edited = st.data_editor(df_log, num_rows="dynamic", use_container_width=True, key="draft_editor")
    if not edited.equals(df_log):
        st.session_state["draft_log"] = edited.to_dict("records")
        save_state({"players": players, "draft_log": st.session_state["draft_log"]})
        st.rerun()
    if not edited.empty:
        st.download_button("Download CSV", edited.to_csv(index=False).encode("utf-8"),
                           file_name="draft_results.csv", mime="text/csv")
        try:
            excel_buf = io.BytesIO()
            edited.to_excel(excel_buf, index=False)
            st.download_button(
                "Download Excel", excel_buf.getvalue(), file_name="draft_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except ImportError:
            st.caption("Install `openpyxl` to enable Excel export.")

with st.expander("🌿 Ingredient similarity finder (hops · yeasts · malts)", expanded=False):
    search = st.text_input("Search ingredients", key="hop-search")
    hop_list = sorted(hop_similarity_data.keys())
    if search:
        hop_list = [h for h in hop_list if search.lower() in h.lower()]
    for hop in hop_list:
        label = f"~~{hop}~~" if hop in drafted else hop
        st.markdown(f"**{label}**")
        similars = [f"~~{draft_core._sim_name(rec)}~~" if draft_core._sim_name(rec) in drafted
                    else draft_core._sim_name(rec)
                    for rec in hop_similarity_data.get(hop, [])]
        st.caption("Similar: " + ", ".join(similars) if similars else "No similarity data available.")

# Keep the timer ticking while running and the draft is live.
if st.session_state.get("timer_running") and current_player:
    time.sleep(1)
    st.rerun()

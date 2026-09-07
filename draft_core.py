import pandas as pd
import json
import os
from collections import defaultdict, Counter

from config import load_league_config


def _load_required_json(path, label):
    """Load a required JSON file, raising a clear error if missing/malformed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required {label} file not found: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse {label} file {path}: {e}") from e


def _load_optional_json(path):
    """Load an optional JSON file; return None if absent or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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
    # Required files: fail loud and clear rather than with a raw traceback.
    if not os.path.exists(ingredients_path):
        raise FileNotFoundError(f"Required ingredients file not found: {ingredients_path}")
    ingredients = pd.read_csv(ingredients_path)
    style_matrix = _load_required_json(style_matrix_path, "style matrix")
    scarcity = pd.DataFrame(_load_required_json(scarcity_path, "ingredient scarcity"))

    # Optional files: absent or malformed degrades to None (feature off).
    opponent_model = _load_optional_json(opponent_model_path)
    style_bias = _load_optional_json(style_bias_path)
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


def build_available_set(ingredients_df, category_aliases=None):
    """Return the set of ingredient strings that are available to draft.

    ``category_aliases`` maps a category to the sheet column names that may
    hold it; defaults to the league config so callers don't hardcode it.
    """
    if category_aliases is None:
        category_aliases = load_league_config()["category_aliases"]

    def extract(alias_list):
        vals = []
        for colname in alias_list:
            if colname in ingredients_df.columns:
                vals.extend(ingredients_df[colname].dropna().unique().tolist())
        return set(vals)

    avail = set()
    for aliases in category_aliases.values():
        avail |= extract(aliases)
    return avail


def validate_data(ingredients_df, style_matrix, category_aliases=None):
    """Cross-check the style matrix against the ingredient sheet.

    Returns a report dict:
      - ``in_matrix_not_sheet``: ingredients a style requires but that are not
        draftable from the sheet. These are the dangerous ones — a style that
        references them can never be fully satisfied, silently skewing style
        viability and recommendations.
      - ``in_sheet_not_matrix``: draftable ingredients that no style uses
        (informational; e.g. an adjunct not modeled in any style).
    Empty lists mean the two sources are consistent.
    """
    available = build_available_set(ingredients_df, category_aliases)
    matrix_ings = set()
    for cats in style_matrix.values():
        for ings in cats.values():
            matrix_ings.update(ings)
    return {
        "in_matrix_not_sheet": sorted(matrix_ings - available),
        "in_sheet_not_matrix": sorted(available - matrix_ings),
    }


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


def snake_draft_order(num_players, current_round):
    """Seat order (0-indexed) for a round in a snake draft.

    Round 1 (and every odd round) runs ascending 0..N-1; even rounds reverse.
    """
    if current_round % 2 == 1:
        return list(range(num_players))
    return list(range(num_players - 1, -1, -1))


def pick_slot(overall_pick, num_players):
    """Map a 1-indexed overall pick number to (round, seat_index_0based).

    This is the single source of truth for the snake-draft math that the
    tracker UI uses to decide whose turn it is.
    """
    current_round = ((overall_pick - 1) // num_players) + 1
    order = snake_draft_order(num_players, current_round)
    idx_in_order = (overall_pick - 1) % num_players
    return current_round, order[idx_in_order]


def _rules_status_from_counts(counts, n_picks, total_picks, config):
    """Shared tail: derive the rule-status dict from per-bucket counts."""
    # Required categories come from config (as category names); map them onto
    # rule buckets so e.g. "Base Malt" -> "Malt".
    required_min = {}
    for cat, n in config["required_categories"].items():
        required_min[bucket_for_rules(cat)] = n
    required_met = {k: counts[k] >= v for k, v in required_min.items()}
    required_remaining = {k: max(0, v - counts[k]) for k, v in required_min.items()}

    satisfied_core = sum(min(counts[k], 1) for k in required_min.keys())
    flex_used = max(0, n_picks - satisfied_core)
    flex_remaining = max(0, config.get("flex_slots", 3) - flex_used)

    picks_remaining = max(0, total_picks - n_picks)

    required_slots_left = sum(required_remaining.values())
    feasible = required_slots_left <= picks_remaining

    return {
        "counts": counts,
        "required_met": required_met,
        "required_remaining": required_remaining,
        "flex_used": flex_used,
        "flex_remaining": flex_remaining,
        "picks_remaining": picks_remaining,
        "required_slots_left": required_slots_left,
        "feasible": feasible,
    }


def compute_rules_status(my_picks, ingredient_to_category, total_picks, config=None):
    """Rule status from a list of ingredient *names*.

    Categories are re-derived from the style matrix, so ingredients not modeled
    there fall to Flex. Prefer ``compute_rules_status_from_records`` on the live
    path, where the true drafted category is known.
    """
    if config is None:
        config = load_league_config()
    counts = {"Malt": 0, "Hop": 0, "Yeast": 0, "Adjunct": 0, "Flex": 0}
    for ing in my_picks:
        ui_cat = ingredient_to_category.get(ing, "Specialty")
        counts[bucket_for_rules(ui_cat)] += 1
    return _rules_status_from_counts(counts, len(my_picks), total_picks, config)


def compute_rules_status_from_records(my_records, total_picks, config=None):
    """Rule status that trusts each pick's stored ``Category``.

    This is the correct live-path function: the draft board records the true
    category (from the sheet column an ingredient was drafted out of), so
    adjuncts not modeled in the style matrix (Mango, Vanilla, Oak, ...) still
    satisfy the Adjunct requirement instead of silently burning a flex slot.
    """
    if config is None:
        config = load_league_config()
    counts = {"Malt": 0, "Hop": 0, "Yeast": 0, "Adjunct": 0, "Flex": 0}
    for rec in my_records:
        counts[bucket_for_rules(str(rec.get("Category", "")))] += 1
    return _rules_status_from_counts(counts, len(my_records), total_picks, config)


def roster_slots(my_records, enable_round8=False, flex_slots=3,
                 required_categories=None):
    """Ordered roster-slot model derived from a player's pick records.

    Trusts each record's stored ``Category`` (via ``bucket_for_rules``) so slot
    attribution matches what was actually drafted. Required buckets fill first;
    extras spill into flex, then the optional 8th-round slot.

    Returns a list of ``{key, label, filled, required}`` dicts, suitable for
    both the always-visible roster strip and the Results summary table.
    """
    if required_categories is None:
        required_categories = load_league_config()["required_categories"]
    # Preserve config order, mapped to rule buckets (e.g. "Base Malt" -> "Malt").
    required_buckets = []
    for cat in required_categories:
        b = bucket_for_rules(cat)
        if b not in required_buckets:
            required_buckets.append(b)

    filled = {b: None for b in required_buckets}
    flex, round8 = [], None
    for rec in my_records:
        ing = rec.get("Ingredient", "")
        bucket = bucket_for_rules(str(rec.get("Category", "")))
        if bucket in filled and filled[bucket] is None:
            filled[bucket] = ing
        elif enable_round8 and round8 is None and len(flex) >= flex_slots:
            round8 = ing
        else:
            flex.append(ing)

    slots = [{"key": b, "label": b, "filled": filled[b], "required": True}
             for b in required_buckets]
    n_flex = max(flex_slots, len(flex))  # never hide overflow
    for i in range(n_flex):
        slots.append({
            "key": f"Flex{i + 1}", "label": f"Flex {i + 1}",
            "filled": flex[i] if i < len(flex) else None, "required": False,
        })
    if enable_round8:
        slots.append({"key": "Round8", "label": "Oh Sh*t",
                      "filled": round8, "required": False})
    return slots


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


# --- Scoring model -----------------------------------------------------------
# The recommender scores each candidate as a weighted sum of six components,
# each normalized to [0,1] across the candidate set so the weights read as
# "share of the decision" and the per-component contributions are directly
# explainable (see explain_pick).
DEFAULT_PICK_WEIGHTS = {
    "fit": 0.30,     # graded style fit (signature ingredients, aligned to your styles)
    "scarce": 0.25,  # dynamic board scarcity (supply vs. demand, right now)
    "need": 0.20,    # unmet required slot + snake-draft survival risk
    "syn": 0.15,     # synergy with your existing roster
    "deny": 0.07,    # value denied to an opponent picking before your next turn
    "pop": 0.03,     # historical popularity prior (tie-breaker)
}


def compute_style_idf(style_matrix):
    """{ingredient: idf} where idf = ln(N_styles / n_styles_containing_ing).

    Rewards *signature* ingredients (central to few styles) over generic ones
    used everywhere — inverting the old style-coverage weighting that let broad
    base malts dominate.
    """
    import math
    n_styles = len(style_matrix) or 1
    doc_count = defaultdict(int)
    for cats in style_matrix.values():
        seen = set()
        for ings in cats.values():
            for ing in ings:
                if ing not in seen:
                    seen.add(ing)
                    doc_count[ing] += 1
    return {ing: math.log(n_styles / c) for ing, c in doc_count.items() if c > 0}


def compute_style_focus(my_picks, style_matrix):
    """Normalized weight per style reflecting how invested your roster is in it.

    Empty picks -> empty dict (flat: no style preference yet).
    """
    my_set = set(my_picks)
    focus = {}
    for style, cats in style_matrix.items():
        hits = sum(1 for ings in cats.values() if any(i in my_set for i in ings))
        if hits:
            focus[style] = hits
    total = sum(focus.values())
    return {s: v / total for s, v in focus.items()} if total else {}


def compute_dynamic_scarcity(available_now, ingredient_to_category,
                             required_categories, num_players=8,
                             hop_similarity=None):
    """Raw scarcity per available ingredient from live board supply vs. demand.

    Scarcity rises as a category's remaining supply shrinks against the number
    of drafters who still need it. For hops, many close still-available analogs
    reduce scarcity (you can wait). Caller normalizes. Not circular, not static.
    """
    supply = Counter(ingredient_to_category.get(i, "Unknown") for i in available_now)
    req = set(required_categories)
    raw = {}
    for ing in available_now:
        c = ingredient_to_category.get(ing, "Unknown")
        demand = num_players if c in req else max(1, num_players // 3)
        pressure = demand / max(supply.get(c, 1), 1)
        sub_factor = 1.0
        if hop_similarity and c == "Hop":
            close = sum(1 for r in hop_similarity.get(ing, [])
                        if r.get("hop") in available_now and r.get("score", 0) >= 0.3)
            sub_factor = 1.0 / (1.0 + 0.15 * close)
        # Absolute [0,1]: low for everyone when supply is plentiful (nothing
        # scarce early), rising as a category depletes. NOT min-maxed downstream,
        # so the model doesn't invent a "scarcest" pick when none is scarce.
        raw[ing] = min(1.0, pressure * sub_factor)
    return raw


def picks_until_next_turn(overall_pick, num_players, your_seat_index):
    """How many picks occur from ``overall_pick`` until your next turn.

    0 if the current pick is yours; otherwise the count of intervening picks.
    Used to gauge how likely a candidate is to survive until you pick again.
    """
    p = overall_pick
    for count in range(2 * num_players + 2):
        _, seat = pick_slot(p, num_players)
        if seat == your_seat_index:
            return count
        p += 1
    return 2 * num_players


def _survival_risk(popularity, gap):
    """Probability a candidate is taken before your next turn, in [0,1].

    Popularity (~[0,2]) becomes a per-pick hazard; compounded over ``gap`` picks.
    """
    hazard = min(0.5, max(0.0, popularity) * 0.25)
    return 1.0 - (1.0 - hazard) ** max(gap, 0)


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
    *,
    hop_similarity=None,
    pair_lookup=None,
    num_players: int = 8,
    overall_pick: int = 1,
    your_seat_index: int = 0,
    weights=None,
    available_set=None,
):
    """Rank available ingredients by a normalized, weighted, explainable score.

    Backward compatible: legacy positional args are unchanged and the returned
    frame keeps the old columns (Style Coverage, Scarcity, Popularity, Pick
    Value) plus the new per-component columns and a human ``Why`` string. New
    signals (snake urgency, roster synergy, opponent denial, dynamic scarcity)
    are driven by the keyword-only args and degrade gracefully when absent.
    """
    weights = {**DEFAULT_PICK_WEIGHTS, **(weights or {})}
    drafted_set = set(drafted)
    my_set = set(my_picks)
    if available_set is None:
        available_set = build_available_set(ingredients)

    cand = []
    for style, cats in style_matrix.items():
        for cat, ings in cats.items():
            for ing in ings:
                if ing in available_set and ing not in drafted_set:
                    cand.append((ing, cat, style))

    out_cols = [
        "Ingredient", "Category", "Style Coverage", "Scarcity", "Popularity",
        "Bias Factor", "Fit", "Urgency", "Synergy", "Denial", "Pick Value", "Why",
    ]
    if not cand:
        return pd.DataFrame(columns=out_cols)

    coverage = defaultdict(set)
    cat_of_cand = {}
    for ing, cat, style in cand:
        coverage[ing].add(style)
        cat_of_cand[ing] = cat  # category as it appears on the board/matrix

    available_cands = set(coverage)
    idf = compute_style_idf(style_matrix)
    focus = compute_style_focus(my_picks, style_matrix)
    scarcity_raw = compute_dynamic_scarcity(
        available_cands, cat_of_cand, required, num_players, hop_similarity
    )
    gap = picks_until_next_turn(overall_pick, num_players, your_seat_index)

    needed_buckets = set()
    have_counts = Counter(bucket_for_rules(cat_of_cand.get(i) or
                          ingredient_to_category.get(i, "Specialty")) for i in my_set)
    for cat, n in required.items():
        b = bucket_for_rules(cat)
        if have_counts.get(b, 0) < n:
            needed_buckets.add(b)

    my_hops = [p for p in my_set if ingredient_to_category.get(p) == "Hop"]
    opp_recent = [x for x in reversed(drafted) if x not in my_set][:3]

    # Components are computed as absolute [0,1] scores (diminishing-returns
    # squashes rather than min-max) so that when nothing is scarce/synergistic
    # yet, the model doesn't manufacture a "best" one — it just scores low.
    def _squash(x, k):
        return x / (x + k) if x > 0 else 0.0

    rows = []
    for ing, styles in coverage.items():
        cat = cat_of_cand[ing]
        bucket = bucket_for_rules(cat)
        style_cov = len(styles)
        popularity = float(early_signal.get(ing, 0.0)) if early_signal else 0.0

        # Style fit: versatility (optionality) with diminishing returns, sharpened
        # toward the styles your roster is already committing to. idf breaks ties
        # so a signature ingredient edges out an equally-versatile generic one.
        focus_align = max((focus.get(s, 0.0) for s in styles), default=0.0)
        breadth = style_cov / (style_cov + 5.0)
        align_mult = (0.5 + 0.5 * focus_align) if focus else 1.0
        idf_tiebreak = 1.0 + 0.05 * idf.get(ing, 0.0)
        fit = min(1.0, breadth * align_mult * idf_tiebreak)

        # Scarcity: absolute board pressure (already [0,1]).
        scarce = scarcity_raw.get(ing, 0.0)

        # Urgency: unmet required slot, amplified by survival risk before your
        # next snake turn.
        need_base = 1.0 if bucket in needed_buckets else 0.2
        urgency = need_base * (0.6 + 0.4 * _survival_risk(popularity, gap))

        # Synergy: co-occurrence with your picks + hop-blend affinity.
        s_raw = 0.0
        if pair_lookup:
            for mine in my_set:
                s_raw += pair_lookup.get((mine, ing), 0)
        if hop_similarity and bucket == "Hop" and my_hops:
            for h in my_hops:
                for rec in hop_similarity.get(h, []):
                    if rec.get("hop") == ing:
                        s_raw += float(rec.get("score", 0.0))
        synergy = _squash(s_raw, 3.0)

        # Denial: value to an opponent picking before your next turn.
        d_raw = 0.0
        if gap > 0 and pair_lookup:
            for r in opp_recent:
                d_raw += pair_lookup.get((r, ing), 0)
        denial = _squash(d_raw * popularity, 2.0)

        pop = _squash(popularity, 1.0)

        contrib = {
            "fit": weights["fit"] * fit,
            "scarce": weights["scarce"] * scarce,
            "need": weights["need"] * urgency,
            "syn": weights["syn"] * synergy,
            "deny": weights["deny"] * denial,
            "pop": weights["pop"] * pop,
        }
        pick_value = sum(contrib.values())
        rows.append({
            "Ingredient": ing,
            "Category": cat,
            "Style Coverage": style_cov,
            "Scarcity": round(scarce, 3),
            "Popularity": round(pop, 3),
            "Bias Factor": round(ingredient_style_bias(ing, style_matrix, style_bias), 2),
            "Fit": round(fit, 3),
            "Urgency": round(urgency, 3),
            "Synergy": round(synergy, 3),
            "Denial": round(denial, 3),
            "Pick Value": round(pick_value, 4),
            "Why": explain_pick(contrib),
        })

    df = pd.DataFrame(rows).sort_values(
        by=["Pick Value", "Fit", "Scarcity", "Style Coverage"],
        ascending=[False, False, False, False],
    )
    return df.head(top_k)


_WHY_LABELS = {
    "fit": "fits your styles",
    "scarce": "scarce now",
    "need": "fills a needed slot",
    "syn": "pairs with your roster",
    "deny": "denies an opponent",
    "pop": "popular pick",
}


def explain_pick(contrib, threshold=0.02, top_n=2):
    """Human 'why' string: the 1-2 dominant weighted contributions."""
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    parts = [_WHY_LABELS.get(k, k) for k, v in ranked if v > threshold][:top_n]
    return " · ".join(parts) if parts else "balanced value"


# Roster-agnostic "best available" board value: intrinsic worth of an ingredient
# still on the board — versatility (fit), how fast it's drying up (scarce), and
# historical popularity — with the personalized terms (need/synergy/denial)
# switched off. This is the "best player available" ranking, not "best for you".
BOARD_VALUE_WEIGHTS = {
    "fit": 0.5, "scarce": 0.3, "need": 0.0, "syn": 0.0, "deny": 0.0, "pop": 0.2,
}


def best_available(drafted, ingredients, style_matrix, scarcity_df, required,
                   flex_slots, ingredient_to_category, style_bias,
                   early_signal=None, hop_similarity=None, num_players=8,
                   top_k=15):
    """Rank the best ingredients left on the board, independent of any roster.

    A thin wrapper over next_best_picks with an empty roster and the board-value
    weight preset, so the result reflects intrinsic ingredient worth rather than
    a particular team's needs. Same DataFrame shape as next_best_picks.
    """
    return next_best_picks(
        [], drafted, ingredients, style_matrix, scarcity_df, required, flex_slots,
        ingredient_to_category, style_bias, early_signal=early_signal,
        hop_similarity=hop_similarity, num_players=num_players,
        weights=BOARD_VALUE_WEIGHTS, top_k=top_k,
    )


def team_context(records, my_picks, drafted, style_matrix, required, flex_slots,
                 enable_round8=False, config=None):
    """Broadcast-style context for one team: roster, needs, and leaning style.

    Composes the existing roster/rules/style helpers so the UI stays thin and the
    needs logic is unit-testable. Returns:
      slots            - ordered roster_slots() model (filled/empty per slot)
      needed_buckets   - set of unmet required rule buckets (Malt/Hop/Yeast/Adjunct)
      likely_style     - top style the roster is trending toward ("TBD" if no picks)
      flex_remaining, picks_remaining - from the rules status
    """
    if config is None:
        config = load_league_config()
    total_picks = config.get("rounds", 7) + (1 if enable_round8 else 0)
    slots = roster_slots(records, enable_round8=enable_round8, flex_slots=flex_slots)
    status = compute_rules_status_from_records(records, total_picks, config)
    needed_buckets = {b for b, rem in status["required_remaining"].items() if rem > 0}

    likely_style = "TBD"
    if my_picks:
        viab = compute_style_status(my_picks, drafted, style_matrix, required, flex_slots)
        if not viab.empty:
            likely_style = viab.iloc[0]["Style"]

    return {
        "slots": slots,
        "needed_buckets": needed_buckets,
        "likely_style": likely_style,
        "flex_remaining": status["flex_remaining"],
        "picks_remaining": status["picks_remaining"],
    }


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

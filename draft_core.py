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

    ``scarcity_df`` is the *informational* pre-draft snapshot from
    scripts/build_scarcity_baseline.py. The engine computes scarcity live
    (``compute_dynamic_scarcity``) and never reads it; it is optional and an
    empty frame is returned when absent. The positional slot is kept so
    existing callers keep working.
    """
    # Required files: fail loud and clear rather than with a raw traceback.
    if not os.path.exists(ingredients_path):
        raise FileNotFoundError(f"Required ingredients file not found: {ingredients_path}")
    ingredients = pd.read_csv(ingredients_path)
    style_matrix = _load_required_json(style_matrix_path, "style matrix")
    scarcity = pd.DataFrame(_load_optional_json(scarcity_path) or [])

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


def compute_style_status(my_picks, drafted, style_matrix, required, flex_slots,
                         workable=None, workable_weight=None):
    """Style viability for a roster: which styles can still be built, and which
    the roster most resembles.

    Columns:
      Satisfied Categories        - style categories your roster already covers
      Picks Matched               - how many of your picks the style actually uses
      Categories with Options Left- categories with at least one undrafted option
      Match                       - geometric-mean likelihood of your picks under
                                    the style (naive-Bayes style classifier: a pick
                                    is 1/|category list| if the style uses it, a
                                    small floor otherwise), so a roster of
                                    *defining* ingredients ranks the narrow style
                                    above a broad one that merely also lists them
      Score                       - 2*satisfied + options (unchanged legacy scale)

    Sorted by Score, then Picks Matched, then Match, so ``iloc[0]`` is the
    "likely style" even when several styles tie on categories alone.

    ``workable`` (second-tier lists, see ``load_workable``): a category whose
    characteristic options are gone still counts as having options if a
    workable one is on the board; a workable roster pick counts as matched at
    ``workable_weight`` of a characteristic one (both categories satisfied and
    the likelihood term).
    """
    import math
    if workable_weight is None:
        workable_weight = DEFAULT_SQUASH["workable_weight"]
    status = []
    drafted_set = set(drafted)
    my_set = set(my_picks)
    n_my = len(my_set)
    # Likelihood floor for a pick the style does not use: roughly "one of
    # everything on the sheet", so an off-style pick costs far more than a
    # matched pick in even the broadest category list.
    all_ings = {i for cats in style_matrix.values() for l in cats.values() for i in l}
    floor_ll = math.log(1.0 / max(len(all_ings), 1))

    for style, cats in style_matrix.items():
        wcats = (workable or {}).get(style, {})
        cat_choices_remaining = {}
        satisfied = 0.0
        for cat, ing_list in cats.items():
            wlist = wcats.get(cat, []) if workable_weight > 0 else []
            remaining_ing = [ing for ing in ing_list if ing not in drafted_set or ing in my_set]
            remaining_w = [ing for ing in wlist if ing not in drafted_set or ing in my_set]
            cat_choices_remaining[cat] = len(remaining_ing) + len(remaining_w)
            if any(ing in my_set for ing in ing_list):
                satisfied += 1.0
            elif any(ing in my_set for ing in wlist):
                satisfied += workable_weight

        options = sum(1 for k, v in cat_choices_remaining.items() if v > 0)
        score = satisfied * 2 + options

        matched, ll = 0.0, 0.0
        for ing in my_set:
            hit = next((len(l) for l in cats.values() if ing in l), None)
            whit = next((len(l) for l in wcats.values() if ing in l), None) if workable_weight > 0 else None
            if hit:
                matched += 1.0
                ll += math.log(1.0 / hit)
            elif whit:
                matched += workable_weight
                ll += math.log(workable_weight / whit)
            else:
                ll += floor_ll
        match = math.exp(ll / n_my) if n_my else 0.0

        status.append(
            {
                "Style": style,
                "Satisfied Categories": satisfied,
                "Picks Matched": matched,
                "Categories with Options Left": options,
                "Match": round(match, 4),
                "Score": score,
            }
        )
    df = pd.DataFrame(status).sort_values(
        by=["Score", "Picks Matched", "Match"], ascending=False, kind="mergesort"
    )
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

# Shape constants for the six components (calibrated on the 2025 draft replay,
# see docs/CALIBRATION.md and scripts/calibrate_weights.py). Squash-type
# entries are the k in component = x / (x + k): smaller k saturates faster.
DEFAULT_SQUASH = {
    "fit": 2.0,        # style coverage count: 2 styles already reads "versatile"
    "syn": 5.0,        # summed pair counts + hop-blend scores: nearly linear
    "deny": 2.0,       # opponent pair value x popularity
    "pop": 1.0,        # early-signal popularity (~0-2)
    # Fit shape: how much of Fit is "belongs to the style my roster leads
    # toward" (vs. raw versatility); how much a *signature* ingredient (few
    # alternatives in its category within that style) is favoured; and the
    # idf tie-break toward ingredients central to few styles.
    "fit_align": 0.65,
    "fit_sig": 0.5,
    "fit_idf": 0.05,
    # Urgency shape: baseline urgency for a flex (non-required) pick, and
    # whether an unmet required slot's urgency scales with spare picks (1.0)
    # or is constant (0.0). Constant wins on the replay: drafters fill the
    # required, contested slots first regardless of how many picks remain.
    "need_flex_floor": 0.0,
    "need_slack": 0.0,
    # Scarcity shape: 1.0 -> demand for a required category is the number of
    # drafters who have not drafted from it yet; 0.0 -> every drafter, always.
    # Static wins on the replay: after round 1 the table kept taking yeast and
    # base malt in round 2 as if demand had not moved (median rank 6 vs 29).
    "scarce_residual": 0.0,
    # Per close still-available analog, scarcity pressure is divided by
    # (1 + scarce_sub * n_close). Off by default: on the 2025 replay the
    # discount lowered every hit-rate metric at any strength (drafters take
    # the mainstream ingredient that *has* many analogs -- Citra, Crystal 40L,
    # Maris Otter -- rather than waiting on it). See docs/CALIBRATION.md.
    "scarce_sub": 0.0,
    # Redundancy: once the roster already holds a pick from a category in
    # league_config "single_pick_categories" (Yeast), a second one has its Pick
    # Value multiplied by this. Demotes co-pitching to the tail of the list
    # without hiding it (it is legal, just rare). 1.0 disables.
    "redundant_mult": 0.25,
    # Credit for second-tier ("workable") style membership relative to
    # characteristic membership in Fit alignment and style focus. Full credit
    # applies only when the style's characteristic options in that category
    # are off the board, or the roster already holds one (a twist on a
    # finished build); otherwise it is divided by (1 + options still
    # available), so "might work" never outranks "is the style" while the
    # real thing can still be drafted. 0 ignores the workable file entirely.
    "workable_weight": 0.9,
    # Style focus: 1.0 -> likelihood-based (a style missing your defining pick
    # drops sharply); 0.0 -> legacy category-hit count. focus_floor is the
    # "not listed" likelihood denominator; 0 = the number of ingredients in the
    # matrix (one of everything on the sheet), which must exceed the broadest
    # category list or a narrow style that *lacks* your defining pick can
    # out-score the broad style that has it.
    "focus_ll": 1.0,
    "focus_floor": 0.0,
    # Damping added to every list length in the likelihood (1/(|list|+k)) so
    # that hit-vs-miss dominates and list narrowness is a secondary signal.
    "focus_damp": 2.0,
    # Weight of a *workable* roster pick when deciding which style the roster
    # leads toward (deliberately below workable_weight: a workable hit in a
    # short list must not out-vote a characteristic hit in a long one).
    "focus_workable": 0.5,
}


def scoring_params(config=None):
    """(pick_weights, board_value_weights, squash) with league-config overrides.

    ``league_config.json`` may carry ``pick_weights``, ``board_value_weights``
    and ``squash`` dicts; any keys given overlay the code defaults so a season
    can be re-tuned without a code change. Unknown keys are ignored.
    """
    if config is None:
        config = load_league_config()

    def overlay(base, key):
        over = config.get(key) or {}
        return {k: float(over.get(k, v)) for k, v in base.items()}

    return (overlay(DEFAULT_PICK_WEIGHTS, "pick_weights"),
            overlay(BOARD_VALUE_WEIGHTS, "board_value_weights"),
            overlay(DEFAULT_SQUASH, "squash"))


def build_opponent_signals(opponent_model):
    """(early_signal, pair_lookup) from an opponent_model dict (or None).

    early_signal: {ingredient: Early_Score}; pair_lookup: {(a, b): count},
    symmetric. This is the single place the model file's shape is interpreted.
    """
    early, pairs = {}, defaultdict(int)
    if not opponent_model:
        return early, pairs
    for rec in opponent_model.get("ingredient_popularity", []):
        ing = rec.get("Ingredient")
        if ing:
            early[ing] = float(rec.get("Early_Score", 0.0))
    for rec in opponent_model.get("top_pairs", []):
        a, b, c = rec.get("A"), rec.get("B"), int(rec.get("PairCount", 0))
        if a and b:
            pairs[(a, b)] += c
            pairs[(b, a)] += c
    return early, pairs


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


def compute_style_focus(my_picks, style_matrix, workable=None, workable_weight=0.5,
                        mode="likelihood", miss_floor=None, damp=8.0):
    """Normalized weight per style reflecting how invested your roster is in it.

    ``mode="likelihood"`` (default): each style's weight is the naive-Bayes
    likelihood of the roster -- a pick is 1/|category list| when the style
    lists it (x ``workable_weight`` when only workable), and 1/``miss_floor``
    when it does not. A style that misses the *defining* pick (the yeast in a
    Kölsch roster) falls far behind one that has it, even if both share the
    generic base malt and hop -- which is what "the style I'm building" means.

    ``mode="count"``: the older category-hit count (1.0 characteristic,
    ``workable_weight`` workable), retained for comparison.

    Normalised to sum to 1. Empty picks -> empty dict (no preference yet).
    """
    import math
    my_set = set(my_picks)
    if miss_floor is None:
        miss_floor = float(max(1, len({i for c in style_matrix.values()
                                       for l in c.values() for i in l})))
    focus = {}
    for style, cats in style_matrix.items():
        wcats = (workable or {}).get(style, {})
        if mode == "count":
            hits = 0.0
            for cat, ings in cats.items():
                if any(i in my_set for i in ings):
                    hits += 1.0
                elif any(i in my_set for i in wcats.get(cat, [])):
                    hits += workable_weight
            if hits:
                focus[style] = hits
            continue
        if not my_set:
            continue
        ll = 0.0
        for ing in my_set:
            hit = next((len(l) for l in cats.values() if ing in l), None)
            if hit:
                ll += math.log(1.0 / (hit + damp))
                continue
            whit = None
            if workable_weight:
                for cat, wl in wcats.items():
                    if ing in wl:
                        # union of both tiers for that category, so a short
                        # workable list cannot look "defining"
                        whit = len(cats.get(cat, [])) + len(wl)
                        break
            if whit:
                ll += math.log(workable_weight / (whit + damp))
            else:
                ll += math.log(1.0 / (miss_floor + damp))
        focus[style] = ll
    if mode != "count" and focus:
        top = max(focus.values())
        focus = {s: math.exp(v - top) for s, v in focus.items()}
    total = sum(focus.values())
    return {s: v / total for s, v in focus.items()} if total else {}


SIMILARITY_CLOSE_THRESHOLD = 0.3  # neighbour score that counts as a "close analog"


def _sim_name(rec):
    """Neighbour name from a similarity record (new ``ingredient`` key, legacy ``hop``)."""
    return rec.get("ingredient") or rec.get("hop")


def load_similarity(config=None, base_dir="."):
    """Merged ingredient-similarity lookup from the files named in league config.

    ``config["similarity_files"]`` maps a category to a JSON file of shape
    ``{name: [{"ingredient": other, "score": 0..1}, ...]}`` (see
    scripts/build_similarity.py). Files may be shared between categories (malt
    covers Base Malt + Specialty) and missing/malformed files are skipped, so
    the result is simply sparser. Returns ``{}`` when nothing is available.
    """
    if config is None:
        config = load_league_config()
    merged = {}
    seen_paths = set()
    for _cat, fname in (config.get("similarity_files") or {}).items():
        path = os.path.join(base_dir, fname)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        data = _load_optional_json(path)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def load_workable(config=None, base_dir="."):
    """Second-tier ("workable") style membership, same shape as the style matrix.

    Generated by scripts/build_style_matrix.py into the file named by
    ``config["style_matrix_workable_path"]``. Optional: ``{}`` when absent.
    """
    if config is None:
        config = load_league_config()
    path = os.path.join(base_dir, config.get("style_matrix_workable_path")
                        or "style_matrix_workable.json")
    data = _load_optional_json(path)
    return data if isinstance(data, dict) else {}


def compute_dynamic_scarcity(available_now, ingredient_to_category,
                             required_categories, num_players=8,
                             hop_similarity=None, similarity=None,
                             drafted=None, sub_strength=0.15):
    """Raw scarcity per available ingredient from live board supply vs. demand.

    pressure = demand / supply for the ingredient's category, where

      supply  = still-available ingredients in that category
      demand  = for a *required* category, the drafters who plausibly still
                need one: ``num_players`` minus the number already drafted from
                that category (when ``drafted`` is given -- a yeast run early
                makes yeast scarce; once everyone has one, the remaining yeasts
                are not scarce at all). Non-required (flex) categories carry a
                nominal demand of a third of the table.

    Close still-available analogs (similarity files: hops, yeasts, malts) cut
    the pressure -- you can wait, a substitute will still be there.

    Returned as ``1 - exp(-pressure)`` so it lives in [0,1), is ~pressure when
    small (plentiful board: 0.2 -> 0.18) and saturates smoothly instead of
    clipping (so substitutes keep mattering under heavy pressure). Caller does
    not min-max it: when nothing is scarce, nothing looks scarce.

    ``similarity`` is the merged lookup from ``load_similarity``; the legacy
    ``hop_similarity`` argument is still honoured for hops.
    """
    import math
    supply = Counter(ingredient_to_category.get(i, "Unknown") for i in available_now)
    taken = Counter(ingredient_to_category.get(i, "Unknown") for i in (drafted or []))
    req = set(required_categories)
    raw = {}
    for ing in available_now:
        c = ingredient_to_category.get(ing, "Unknown")
        if c in req:
            demand = max(1, num_players - taken.get(c, 0))
        else:
            demand = max(1, num_players // 3)
        pressure = demand / max(supply.get(c, 1), 1)
        sub_factor = 1.0
        neighbours = None
        if similarity and ing in similarity:
            neighbours = similarity[ing]
        elif hop_similarity and c == "Hop":
            neighbours = hop_similarity.get(ing, [])
        if neighbours:
            close = sum(1 for r in neighbours
                        if _sim_name(r) in available_now
                        and r.get("score", 0) >= SIMILARITY_CLOSE_THRESHOLD)
            sub_factor = 1.0 / (1.0 + sub_strength * close)
        raw[ing] = 1.0 - math.exp(-pressure * sub_factor)
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
    similarity=None,
    pair_lookup=None,
    num_players: int = 8,
    overall_pick: int = 1,
    your_seat_index: int = 0,
    weights=None,
    squash=None,
    available_set=None,
    single_pick_categories=None,
    workable=None,
):
    """Rank available ingredients by a normalized, weighted, explainable score.

    Backward compatible: legacy positional args are unchanged and the returned
    frame keeps the old columns (Style Coverage, Scarcity, Popularity, Pick
    Value) plus the new per-component columns and a human ``Why`` string. New
    signals (snake urgency, roster synergy, opponent denial, dynamic scarcity)
    are driven by the keyword-only args and degrade gracefully when absent.

    ``scarcity_df`` is accepted for backward compatibility and ignored: scarcity
    is computed live from the board (``compute_dynamic_scarcity``).

    ``workable`` (from ``load_workable``): second-tier style membership. A
    workable ingredient earns ``squash["workable_weight"]`` of a characteristic
    one in Fit, so it surfaces for a style once the defining options are gone
    (or as a twist on a finished build) and its ``Why`` says "workable for
    <style>". ``Style Coverage`` still counts characteristic styles only.

    ``single_pick_categories`` (default: league config, e.g. ``["Yeast"]``):
    once the roster satisfies such a category, further candidates from it are
    *redundant* -- their Pick Value is scaled by ``squash["redundant_mult"]``
    and their ``Why`` says so. They stay in the frame (filter to the category
    and they are still there), just far down.

    ``similarity`` (from ``load_similarity``) covers hops, yeasts and malts. It
    feeds the substitute discount in Scarcity for every category, but the
    hop-blend term in Synergy on purpose stays hops-only: a second yeast or
    base malt that resembles one you already hold is redundancy, not synergy.
    ``hop_similarity`` is the legacy hops-only lookup and is still honoured.
    """
    if similarity is None and hop_similarity:
        similarity = hop_similarity
    elif similarity and hop_similarity:
        similarity = {**hop_similarity, **similarity}
    weights = {**DEFAULT_PICK_WEIGHTS, **(weights or {})}
    squash = {**DEFAULT_SQUASH, **(squash or {})}
    if single_pick_categories is None:
        single_pick_categories = load_league_config().get("single_pick_categories") or []
    single_pick_buckets = {bucket_for_rules(c) for c in single_pick_categories}
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
    list_len = {}  # (style, ingredient) -> size of the category list it sits in
    for ing, cat, style in cand:
        coverage[ing].add(style)
        cat_of_cand[ing] = cat  # category as it appears on the board/matrix
        list_len[(style, ing)] = len(style_matrix[style][cat])

    # Second tier: styles where a candidate is merely *workable*.
    w_weight = squash["workable_weight"] if workable else 0.0
    workable_of = defaultdict(set)
    if w_weight > 0:
        for style, cats in workable.items():
            if style not in style_matrix:
                continue
            for cat, ings in cats.items():
                for ing in ings:
                    if ing in coverage and style not in coverage[ing]:
                        workable_of[ing].add(style)

    available_cands = set(coverage)
    idf = compute_style_idf(style_matrix)
    n_modeled = len({i for c in style_matrix.values() for l in c.values() for i in l})
    focus = compute_style_focus(
        my_picks, style_matrix, workable, min(w_weight, squash["focus_workable"]),
        mode="likelihood" if squash["focus_ll"] else "count",
        miss_floor=squash["focus_floor"] or float(max(n_modeled, 1)),
        damp=squash["focus_damp"],
    )
    focus_max = max(focus.values()) if focus else 0.0
    # Category map must also cover *drafted* ingredients so residual demand
    # can see what has already left the board.
    cat_lookup = {**ingredient_to_category, **cat_of_cand}
    scarcity_raw = compute_dynamic_scarcity(
        available_cands, cat_lookup, required, num_players, similarity=similarity,
        drafted=drafted if squash["scarce_residual"] else None,
        sub_strength=squash["scarce_sub"],
    )
    gap = picks_until_next_turn(overall_pick, num_players, your_seat_index)

    needed_buckets = set()
    have_counts = Counter(bucket_for_rules(cat_of_cand.get(i) or
                          ingredient_to_category.get(i, "Specialty")) for i in my_set)
    required_slots_left = 0
    filled_buckets = set()
    for cat, n in required.items():
        b = bucket_for_rules(cat)
        short = n - have_counts.get(b, 0)
        if short > 0:
            needed_buckets.add(b)
            required_slots_left += short
        else:
            filled_buckets.add(b)
    redundant_buckets = single_pick_buckets & filled_buckets
    total_picks = sum(required.values()) + flex_slots
    picks_remaining = max(0, total_picks - len(my_set))
    # Spare picks after reserving one for every unmet required slot. With
    # slack you can afford a flex pick now and fill the slot later; at zero
    # slack every remaining pick must go to a required category.
    slack = max(0, picks_remaining - required_slots_left)
    need_now = 1.0 / (1.0 + slack) if squash["need_slack"] else 1.0

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

        # Style fit = versatility x alignment x signature.
        #   breadth   - styles the ingredient serves, with diminishing returns
        #   align     - 1.0 if it belongs to the style your roster leads toward,
        #               proportionally less for styles you lean on less, 0 for
        #               styles you have nothing in (flat 1.0 on an empty roster)
        #   signature - few alternatives in its category within that style
        #               (a Belgian yeast in a Dubbel: 4 options; a base malt:
        #               10) -- the pick that *defines* the beer is worth more
        #               than one of many interchangeable ones.
        wstyles = workable_of.get(ing, set())
        # Breadth counts characteristic styles only: workable membership is a
        # fallback, not versatility (counting it re-inflates ubiquitous sugars).
        breadth = _squash(style_cov, squash["fit"])
        via_workable = None
        if focus:
            best = max(focus.get(s, 0.0) for s in styles)
            best_styles = [s for s in styles if focus.get(s, 0.0) == best]
            # A workable membership in a style the roster leads toward can beat
            # a characteristic membership in a style it barely touches.
            best_styles.sort(key=lambda s: (-focus.get(s, 0.0), s))
            if wstyles:
                wbest_style = max(wstyles, key=lambda s: (focus.get(s, 0.0), s))
                # Gate: characteristic options for this category in that style
                # still on the board (and not already covered by the roster)
                # keep the workable tier in reserve.
                char_list = style_matrix[wbest_style].get(cat, [])
                covered = any(p in char_list for p in my_set)
                n_char_avail = 0 if covered else sum(
                    1 for c in char_list if c in available_cands)
                gate = 1.0 / (1.0 + n_char_avail)
                wbest = focus.get(wbest_style, 0.0) * w_weight * gate
                if wbest > best:
                    best, via_workable = wbest, wbest_style
            align = best / focus_max
        else:
            align = 1.0
            best_styles = list(styles)
        # Signature applies to the categories whose lists *define* a style.
        # Adjunct lists are "what a brewer of this style would tolerate" and
        # are short precisely where adjuncts matter least (a Pils lists two
        # sugars), so for adjuncts the term is neutral instead of inverted.
        if bucket == "Adjunct" or via_workable:
            sig = 0.5
        else:
            sig = max(8.0 / (8.0 + list_len[(s, ing)]) for s in best_styles)
        a, g = squash["fit_align"], squash["fit_sig"]
        idf_tiebreak = 1.0 + squash["fit_idf"] * idf.get(ing, 0.0)
        fit = min(1.0, breadth * ((1 - a) + a * align) * ((1 - g) + g * sig) * idf_tiebreak)

        # Scarcity: absolute board pressure (already [0,1]).
        scarce = scarcity_raw.get(ing, 0.0)

        # Urgency: how much waiting costs. For an unmet required slot it grows
        # as spare picks run out (1/(1+slack): 0.25 with three spare picks,
        # 1.0 when every remaining pick is spoken for); for any candidate it
        # also grows with the chance this specific ingredient is sniped before
        # your next snake turn.
        need_term = need_now if bucket in needed_buckets else squash["need_flex_floor"]
        urgency = min(1.0, 0.7 * need_term + 0.3 * _survival_risk(popularity, gap))

        # Synergy: co-occurrence with your picks + hop-blend affinity.
        s_raw = 0.0
        if pair_lookup:
            for mine in my_set:
                s_raw += pair_lookup.get((mine, ing), 0)
        if similarity and bucket == "Hop" and my_hops:
            for h in my_hops:
                for rec in similarity.get(h, []):
                    if _sim_name(rec) == ing:
                        s_raw += float(rec.get("score", 0.0))
        synergy = _squash(s_raw, squash["syn"])

        # Denial: value to an opponent picking before your next turn.
        d_raw = 0.0
        if gap > 0 and pair_lookup:
            for r in opp_recent:
                d_raw += pair_lookup.get((r, ing), 0)
        denial = _squash(d_raw * popularity, squash["deny"])

        pop = _squash(popularity, squash["pop"])

        contrib = {
            "fit": weights["fit"] * fit,
            "scarce": weights["scarce"] * scarce,
            "need": weights["need"] * urgency,
            "syn": weights["syn"] * synergy,
            "deny": weights["deny"] * denial,
            "pop": weights["pop"] * pop,
        }
        pick_value = sum(contrib.values())
        why = explain_pick(contrib, style=(best_styles[0] if focus and best_styles else None))
        if via_workable:
            why = f"might work for {short_style(via_workable)}"
        if bucket in redundant_buckets:
            # Already holding one from a single-pick category: demote, don't hide.
            pick_value *= squash["redundant_mult"]
            why = f"redundant · already have a {bucket.lower()}"
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
            "Why": why,
        })

    df = pd.DataFrame(rows).sort_values(
        by=["Pick Value", "Fit", "Scarcity", "Style Coverage"],
        ascending=[False, False, False, False],
    )
    return df.head(top_k)


_WHY_LABELS = {
    "fit": "versatile / on-style",
    "scarce": "scarce now",
    "need": "fills a needed slot",
    "syn": "pairs with the roster",
    "deny": "denies an opponent",
    "pop": "popular pick",
}


def short_style(name):
    """'Kettle Sour / Fruit Sour (Berliner / Gose)' -> 'Kettle Sour'."""
    return str(name).split(" (")[0].split(" / ")[0].strip()


def explain_pick(contrib, threshold=0.02, top_n=2, style=None):
    """Human 'why' string: the 1-2 dominant weighted contributions.

    ``style`` names the style the Fit term aligned to (once the roster has a
    direction), so "on-style" says *which* style.
    """
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    labels = dict(_WHY_LABELS)
    if style:
        labels["fit"] = f"fits {short_style(style)}"
    parts = [labels.get(k, k) for k, v in ranked if v > threshold][:top_n]
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
                   top_k=15, similarity=None, weights=None, squash=None):
    """Rank the best ingredients left on the board, independent of any roster.

    A thin wrapper over next_best_picks with an empty roster and the board-value
    weight preset, so the result reflects intrinsic ingredient worth rather than
    a particular team's needs. Same DataFrame shape as next_best_picks.
    """
    return next_best_picks(
        [], drafted, ingredients, style_matrix, scarcity_df, required, flex_slots,
        ingredient_to_category, style_bias, early_signal=early_signal,
        hop_similarity=hop_similarity, similarity=similarity, num_players=num_players,
        weights={**BOARD_VALUE_WEIGHTS, **(weights or {})}, squash=squash, top_k=top_k,
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
        "feasible": status["feasible"],
    }


def _owner_map(teams):
    """{ingredient: player} from a teams projection (first owner wins)."""
    owner = {}
    for player, picks in (teams or {}).items():
        for ing in picks:
            owner.setdefault(ing, player)
    return owner


def style_build_plan(style, style_matrix, focus_picks, drafted, teams=None,
                     value_of=None, workable=None):
    """How a team could build ``style`` from the current board.

    Returns a list (in style-matrix category order) of dicts:
      category   - the style category ("Base Malt", "Hop", ...)
      have       - the focus team's own picks that this style uses
      available  - style options still on the board (not drafted), best-first
                   when a ``value_of`` {ingredient: score} map is supplied
      taken      - [(ingredient, player_or_None)] this style wants but that are
                   gone to someone else
      workable   - second-tier "might work" options still on the board, when a
                   ``workable`` matrix (``load_workable``) is supplied; excludes
                   anything already characteristic for the category, best-first
    Pure/read-only: pairs with show_style_dialog in the UI.
    """
    my = set(focus_picks)
    drafted_set = set(drafted)
    owner = _owner_map(teams)
    char = style_matrix.get(style, {})
    wk = (workable or {}).get(style, {})
    # Category order: characteristic first, then any workable-only categories.
    cats = list(char.keys()) + [c for c in wk if c not in char]
    plan = []
    for cat in cats:
        ings = char.get(cat, [])
        char_set = set(ings)
        have = [i for i in ings if i in my]
        available = [i for i in ings if i not in drafted_set]
        if value_of:
            available.sort(key=lambda i: value_of.get(i, 0.0), reverse=True)
        taken = [(i, owner.get(i)) for i in ings
                 if i in drafted_set and i not in my]
        workable_opts = [i for i in wk.get(cat, [])
                         if i not in char_set and i not in drafted_set]
        if value_of:
            workable_opts.sort(key=lambda i: value_of.get(i, 0.0), reverse=True)
        plan.append({"category": cat, "have": have,
                     "available": available, "taken": taken,
                     "workable": workable_opts})
    return plan


def ingredient_detail(ing, style_matrix, drafted, teams=None, similarity=None,
                      popularity=None, available_set=None, board_category=None,
                      workable=None):
    """Read-only detail view for a single ingredient.

    Returns {ingredient, category, styles, workable_styles, drafted_by,
    similar, popularity}:
      styles          - styles whose characteristic recipe lists this ingredient
      workable_styles - styles where it's only a second-tier "might work" fit
                        (from a ``workable`` matrix), excluding ``styles``
      drafted_by      - the player who took it, or None if still available
      similar         - [{ingredient, score, available}] neighbours from the
                        similarity data (available = still on the board)
      popularity      - the ingredient's opponent-model record (or None)
    """
    drafted_set = set(drafted)
    styles = [s for s, cats in style_matrix.items()
              if any(ing in lst for lst in cats.values())]
    char_styles = set(styles)
    workable_styles = [s for s, cats in (workable or {}).items()
                       if s not in char_styles
                       and any(ing in lst for lst in cats.values())]
    category = board_category
    if category is None:
        for cats in style_matrix.values():
            for cat, lst in cats.items():
                if ing in lst:
                    category = cat
                    break
            if category:
                break
    drafted_by = _owner_map(teams).get(ing) if teams else (
        ing if ing in drafted_set else None)
    if ing not in drafted_set:
        drafted_by = None
    similar = []
    for rec in (similarity or {}).get(ing, []):
        name = _sim_name(rec)
        if not name:
            continue
        avail = name not in drafted_set and (
            available_set is None or name in available_set)
        similar.append({"ingredient": name,
                        "score": float(rec.get("score", 0.0)),
                        "available": avail})
    return {
        "ingredient": ing,
        "category": category,
        "styles": styles,
        "workable_styles": workable_styles,
        "drafted_by": drafted_by,
        "similar": similar,
        "popularity": (popularity or {}).get(ing),
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

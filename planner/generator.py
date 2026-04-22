import pandas as pd
from .rules import (
    parse_activities,
    parse_limitations,
    RULES,
    BODY_FOCUS_ROTATION,
    select_exercises_for_day,
    select_light_exercises,
    filter_unstable,
    filter_high_intensity,
    count_complex,
    compute_plane_coverage,
    fill_plane_gaps,
)

WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPORT_TAGS = {"agility", "competition", "trial", "training"}


def _get_training_range(fitness_level: str) -> tuple[int, int]:
    """Return (min, max) training days per week based on fitness level."""
    ranges = {"low": (1, 2), "medium": (2, 3), "high": (3, 4)}
    return ranges.get(fitness_level, (2, 3))


def _find_sport_days(activities: dict[str, list[str]]) -> set[str]:
    """Identify days with heavy sport activities."""
    return {
        d for d, acts in activities.items()
        if any(a in SPORT_TAGS for a in acts)
    }


def _place_training_days(
    open_days: list[str], target: int, has_sport: bool
) -> list[str]:
    """Place training days with even spacing, respecting constraints.

    Strategy:
    1. Try ideal spacing (based on open slots / target)
    2. Fall back to minimum gap of 2 days
    3. Last resort (no sport only): allow consecutive
    """
    if not open_days or target <= 0:
        return []
    target = min(target, len(open_days))
    if target == len(open_days) and not has_sport:
        return list(open_days)

    ideal_gap = len(open_days) / target
    min_gap = max(2, int(ideal_gap))

    # First pass: place with ideal spacing
    placed = []
    for d in open_days:
        if len(placed) >= target:
            break
        if placed:
            gap = WEEK.index(d) - WEEK.index(placed[-1])
            if gap < min_gap:
                continue
        placed.append(d)

    # Second pass: relax to min_gap=2 if not enough
    if len(placed) < target:
        placed = []
        for d in open_days:
            if len(placed) >= target:
                break
            if placed:
                gap = WEEK.index(d) - WEEK.index(placed[-1])
                if gap < 2:
                    continue
            placed.append(d)

    # Third pass: allow consecutive only when no sport (last resort)
    if len(placed) < target and not has_sport:
        for d in open_days:
            if len(placed) >= target:
                break
            if d not in placed:
                placed.append(d)
        placed.sort(key=lambda x: WEEK.index(x))

    return placed


def _build_schedule(
    sport_days: set[str], training_range: tuple[int, int], has_sport: bool
) -> dict[str, str]:
    """Build weekly schedule respecting all placement rules.

    Rules enforced:
    - RULE-004: Rest/light after sport; no 3+ consecutive training days
    - RULE-005: At least 1 rest day per week
    - RULE-007/008: Light day (not full training) after sport
    - No 2 consecutive rest days (use 'light' instead)
    - No 2 consecutive training days when sport is in the week
    - Training count within fitness-level range
    """
    schedule = {d: None for d in WEEK}

    # Step 1: fix sport days
    for d in sport_days:
        schedule[d] = "sport"

    # Step 2: mandatory light after sport days (RULE-007/008)
    # (if next day is already sport, we accept it — user's fixed schedule)
    for i, d in enumerate(WEEK):
        if schedule[d] == "sport" and i + 1 < len(WEEK):
            next_d = WEEK[i + 1]
            if schedule[next_d] is None:
                schedule[next_d] = "light"

    # Step 3: place training days in remaining open slots
    open_days = [d for d in WEEK if schedule[d] is None]
    _, max_train = training_range
    training_days = _place_training_days(open_days, max_train, has_sport)

    for d in training_days:
        schedule[d] = "training"

    # Step 3b: RULE-004 — max 2 consecutive training days
    for i in range(len(WEEK) - 2):
        if all(schedule[WEEK[j]] == "training" for j in range(i, i + 3)):
            schedule[WEEK[i + 2]] = None  # will become rest in step 4

    # Step 4: fill remaining as rest
    for d in WEEK:
        if schedule[d] is None:
            schedule[d] = "rest"

    # Step 5: no 2 consecutive rest days -> convert second to light
    for i in range(len(WEEK) - 1):
        if schedule[WEEK[i]] == "rest" and schedule[WEEK[i + 1]] == "rest":
            schedule[WEEK[i + 1]] = "light"

    # Step 6: ensure at least 1 full rest day (RULE-005)
    rest_count = sum(1 for v in schedule.values() if v == "rest")
    if rest_count < 1:
        # Convert last non-sport day to rest
        for d in reversed(WEEK):
            if schedule[d] in ("training", "light"):
                schedule[d] = "rest"
                break

    return schedule


def _focus_label(body_focus: set[str]) -> str:
    """Convert body-focus set to readable label."""
    labels = {
        "rear": "rear end",
        "core": "core",
        "front": "front end",
        "flexibility": "flexibility",
        "full_body": "full body",
        "body_awareness": "body awareness",
    }
    return " + ".join(labels.get(f, f) for f in sorted(body_focus))


def _day_rules_rest(schedule: dict[str, str], day_index: int) -> list[str]:
    """Determine applied rules for a rest day."""
    rules = ["RULE-005"]
    # RULE-004: rest forced after 2 consecutive training days
    if day_index >= 2:
        prev1 = schedule[WEEK[day_index - 1]]
        prev2 = schedule[WEEK[day_index - 2]]
        if prev1 == "training" and prev2 == "training":
            rules.append("RULE-004")
    return rules


def make_week_plan(
    case_row: dict, allowed_exercises: pd.DataFrame
) -> dict:
    fitness = str(case_row.get("fitness_level", "medium")).lower()
    age_group = str(case_row.get("age_group", "adult")).lower()
    limitations = parse_limitations(case_row.get("limitations", ""))
    training_range = _get_training_range(fitness)

    activities = parse_activities(case_row.get("activities", ""))
    sport_days = _find_sport_days(activities)
    has_sport = len(sport_days) > 0

    # RULE-B: filter unstable equipment for non-high fitness
    df = filter_unstable(allowed_exercises, fitness)
    # RULE-C: filter high intensity for puppies / overweight
    df = filter_high_intensity(df, age_group, limitations)

    schedule = _build_schedule(sport_days, training_range, has_sport)

    plan = []
    training_day_num = 0
    applied_b = fitness != "high" and len(df) < len(allowed_exercises)
    applied_c = len(df) < len(filter_unstable(allowed_exercises, fitness))

    for d in WEEK:
        day_type = schedule[d]
        day_index = WEEK.index(d)

        if day_type == "sport":
            plan.append({
                "day": d,
                "type": "sport_only",
                "note": (
                    f"Planned activity: {', '.join(activities.get(d, []))}."
                    " No extra fitness today."
                ),
                "applied_rules": ["RULE-008"],
            })

        elif day_type == "training":
            body_focus = BODY_FOCUS_ROTATION[
                training_day_num % len(BODY_FOCUS_ROTATION)
            ]
            exercises = select_exercises_for_day(
                df, day_index, body_focus, count=3
            )
            training_day_num += 1
            rules = ["RULE-001", "RULE-009", "RULE-010", "RULE-012",
                      "RULE-A", "RULE-D"]
            if applied_b:
                rules.append("RULE-B")
            if applied_c:
                rules.append("RULE-C")
            plan.append({
                "day": d,
                "type": "training",
                "focus": _focus_label(body_focus),
                "warmup": "5 min easy walking + gentle mobility",
                "exercises": exercises,
                "new_exercises_count": count_complex(exercises),
                "cooldown": "2–5 min calm walking",
                "applied_rules": rules,
            })

        elif day_type == "light":
            exercises = select_light_exercises(
                df, day_index, count=4
            )
            rules = ["RULE-007", "RULE-008", "RULE-012", "RULE-A", "RULE-D"]
            if applied_b:
                rules.append("RULE-B")
            if applied_c:
                rules.append("RULE-C")
            plan.append({
                "day": d,
                "type": "light_training",
                "focus": "recovery + flexibility",
                "warmup": "5 min gentle walking",
                "exercises": exercises,
                "new_exercises_count": count_complex(exercises),
                "cooldown": "2–5 min calm walking",
                "applied_rules": rules,
            })

        else:  # rest
            plan.append({
                "day": d,
                "type": "rest",
                "note": "Rest day (walking is OK)",
                "applied_rules": _day_rules_rest(schedule, day_index),
            })

    # RULE-E: movement plane coverage
    coverage = compute_plane_coverage(plan, allowed_exercises)
    fill_plane_gaps(plan, allowed_exercises, coverage)
    # Recompute after gap filling
    coverage = compute_plane_coverage(plan, allowed_exercises)

    # Add RULE-E to training/light days
    for day in plan:
        if day.get("type") in ("training", "light_training"):
            if "RULE-E" not in day["applied_rules"]:
                day["applied_rules"].append("RULE-E")

    return {"plan": plan, "plane_coverage": coverage}

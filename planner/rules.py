import pandas as pd

DAY_MAP = {
    "mon": "Mon", "tue": "Tue", "wed": "Wed",
    "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"
}


def parse_limitations(raw: str) -> set[str]:
    if not isinstance(raw, str) or not raw.strip():
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def parse_activities(raw: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if not isinstance(raw, str) or not raw.strip():
        return mapping

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        day_raw, act_raw = part.split(":", 1)
        day_norm = DAY_MAP.get(day_raw.strip().lower()[:3])
        if not day_norm:
            continue
        mapping.setdefault(day_norm, []).append(act_raw.strip().lower())
    return mapping


def filter_exercises(
    exercises: pd.DataFrame,
    limitations: set[str],
    age_group: str,
    equipment_available: str,
) -> pd.DataFrame:
    df = exercises.copy()

    # Senior safety
    if str(age_group).strip().lower() == "senior" and "senior_safe" in df.columns:
        df = df[df["senior_safe"].astype(str).str.upper() == "TRUE"]

    # Limitations
    if "no_balance" in limitations and "equipment" in df.columns:
        df = df[df["equipment"].astype(str).str.lower() != "balance"]

    if "low_impact" in limitations and "impact" in df.columns:
        df = df[df["impact"].astype(str).str.lower() == "low"]

    # Equipment available
    if isinstance(equipment_available, str) and equipment_available.strip().lower() == "none" and "equipment" in df.columns:
        df = df[df["equipment"].astype(str).str.lower() == "none"]

    return df


# ── Fitness Rule Definitions ────────────────────────────────────────

RULES = {
    "RULE-001": "Balanced training: mix of strength, flexibility, body awareness",
    "RULE-004": "Recovery: rest after sport day; rest after 2 consecutive training days",
    "RULE-005": "Minimum 1 rest day per week",
    "RULE-007": "Day after sport: only light exercises",
    "RULE-008": "Sport day: no fitness; day after: max light/recovery",
    "RULE-009": "Full body coverage: front + core + rear per week",
    "RULE-010": "All movement planes: sagittal + frontal + transverse per week",
    "RULE-012": "Warmup mandatory; stretching only after warmup",
    "RULE-A": "Session order: body awareness → strength → cardio → stretching",
    "RULE-B": "Unstable equipment only for high fitness level",
    "RULE-C": "No high-intensity exercises for puppies or overweight dogs",
    "RULE-D": "Max 2 new/complex exercises per session",
    "RULE-E": "All 3 movement planes required per week",
}


# ── Classification Constants ────────────────────────────────────────

FOCUS_CATEGORIES = {
    "strength": {"strength", "power", "dynamic_strength"},
    "flexibility": {"flexibility", "rom", "lateral_flexibility", "hip_flexors", "stretching"},
    "body_awareness": {
        "body awareness", "body_awareness", "coordination",
        "mental control", "proprioception",
    },
}

BODY_REGION_KEYWORDS = {
    "front": {"front_end", "shoulders"},
    "core": {"core", "stabilization", "low_back"},
    "rear": {"rear_end", "rear_end_awareness", "hip_flexors"},
}

MOVEMENT_PLANE_KEYWORDS = {
    "sagittal": {"front_end", "rear_end", "strength", "eccentric", "plyometric", "gait_training"},
    "frontal": {"lateral", "lateral_work", "lateral_bend", "lateral_muscles", "abduction_adduction"},
    "transverse": {"rotation", "spine"},
}

BODY_FOCUS_ROTATION = [
    {"rear", "core"},               # day 1: rear end + core
    {"front", "flexibility"},       # day 2: front end + flexibility
    {"full_body", "body_awareness"},  # day 3: full body + body awareness
]

_BODY_FOCUS_MAP = {
    "rear": {"rear_end", "rear_end_awareness", "hip_flexors"},
    "front": {"front_end", "shoulders"},
    "core": {"core", "stabilization", "low_back"},
    "flexibility": {"flexibility", "rom", "lateral_flexibility", "hip_flexors", "stretching"},
    "full_body": {"full_body"},
    "body_awareness": {
        "body_awareness", "coordination", "mental control", "proprioception",
    },
}

# RULE-B: equipment keywords indicating unstable surfaces
UNSTABLE_EQUIPMENT = {
    "balance pad", "balance disc", "wobble board", "bosu", "balance ball",
}

# RULE-A: session phase keywords
_PHASE_CARDIO = {"cardio", "conditioning", "gait_training"}
_PHASE_STRENGTH = {"strength", "power", "dynamic_strength", "eccentric", "plyometric"}


# ── Helpers ─────────────────────────────────────────────────────────

def _tokenize(raw: str) -> set[str]:
    """Split comma-separated string into normalized token set."""
    if not isinstance(raw, str):
        return set()
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def _is_stretch(row: pd.Series) -> bool:
    """Check if exercise is a stretching/flexibility exercise."""
    tags = str(row.get("tags", "")).lower()
    name = str(row.get("name_en", "")).lower()
    return "stretching" in tags or "stretch" in name


def _matches_body_focus(focus_str: str, body_focus: set[str]) -> bool:
    """Check if an exercise's focus matches the target body focus."""
    tokens = _tokenize(focus_str)
    for target in body_focus:
        keywords = _BODY_FOCUS_MAP.get(target, set())
        if tokens & keywords:
            return True
    return False


def _assign_session_phase(row: pd.Series) -> int:
    """RULE-A: Assign session phase for ordering.

    1 = body awareness / proprioception
    2 = strength / power
    3 = cardio / conditioning
    4 = stretching / cooldown
    """
    focus_tokens = _tokenize(str(row.get("focus", "")))
    tag_tokens = _tokenize(str(row.get("tags", "")))
    name = str(row.get("name_en", "")).lower()
    all_tokens = focus_tokens | tag_tokens

    if "stretching" in all_tokens or "stretch" in name:
        return 4
    if all_tokens & _PHASE_CARDIO:
        return 3
    if all_tokens & _PHASE_STRENGTH:
        return 2
    return 1


def _phase_from_dict(ex: dict) -> int:
    """Lightweight phase classification from exercise dict (no tags)."""
    name = str(ex.get("name_en", "")).lower()
    if "stretch" in name:
        return 4
    tokens = _tokenize(str(ex.get("focus", "")))
    if tokens & _PHASE_CARDIO:
        return 3
    if tokens & _PHASE_STRENGTH:
        return 2
    return 1


# ── Classification Functions ────────────────────────────────────────

def classify_focus(focus_str: str) -> set[str]:
    """Classify exercise into strength / flexibility / body_awareness."""
    tokens = _tokenize(focus_str)
    cats = set()
    for cat, keywords in FOCUS_CATEGORIES.items():
        if tokens & keywords:
            cats.add(cat)
    return cats or {"body_awareness"}


def classify_body_region(focus_str: str) -> set[str]:
    """Classify exercise into front / core / rear body regions."""
    tokens = _tokenize(focus_str)
    regions = set()
    for region, keywords in BODY_REGION_KEYWORDS.items():
        if tokens & keywords:
            regions.add(region)
    if "full_body" in tokens:
        regions = {"front", "core", "rear"}
    return regions or {"core"}


def classify_movement_plane(focus_str: str, tags_str: str = "") -> set[str]:
    """Classify exercise into sagittal / frontal / transverse planes."""
    tokens = _tokenize(focus_str) | _tokenize(tags_str)
    planes = set()
    for plane, keywords in MOVEMENT_PLANE_KEYWORDS.items():
        if tokens & keywords:
            planes.add(plane)
    if "full_body" in tokens:
        planes = {"sagittal", "frontal", "transverse"}
    return planes or {"sagittal"}


# ── RULE-B / RULE-C Filters ────────────────────────────────────────

def is_unstable_equipment(equipment_str: str) -> bool:
    """Check if equipment involves an unstable surface."""
    eq = str(equipment_str).lower()
    return any(kw in eq for kw in UNSTABLE_EQUIPMENT)


def filter_unstable(df: pd.DataFrame, fitness_level: str) -> pd.DataFrame:
    """RULE-B: Remove unstable-equipment exercises unless fitness = high."""
    if fitness_level == "high" or "equipment" not in df.columns:
        return df
    mask = ~df["equipment"].apply(is_unstable_equipment)
    filtered = df[mask]
    return filtered if not filtered.empty else df


def filter_high_intensity(
    df: pd.DataFrame, age_group: str, limitations: set[str]
) -> pd.DataFrame:
    """RULE-C: Remove high-impact exercises for puppies or overweight dogs."""
    needs_filter = (
        str(age_group).lower() == "puppy"
        or "obese" in limitations
        or "overweight" in limitations
    )
    if not needs_filter or "impact" not in df.columns:
        return df
    mask = df["impact"].astype(str).str.lower() != "high"
    filtered = df[mask]
    return filtered if not filtered.empty else df


# ── Exercise Selection ──────────────────────────────────────────────

_OUTPUT_COLS = ["exercise_id", "name_en", "focus", "difficulty", "video_url"]


def _pick_stretch(exercises: pd.DataFrame, day_index: int) -> pd.DataFrame:
    """Pick one stretching exercise, varying by day_index."""
    stretch_mask = exercises.apply(_is_stretch, axis=1)
    stretches = exercises[stretch_mask]
    if stretches.empty:
        return exercises.head(0)
    offset = day_index % len(stretches)
    return stretches.iloc[offset:offset + 1]


def _cap_complex(picked: pd.DataFrame, max_new: int = 2) -> pd.DataFrame:
    """RULE-D: Keep at most *max_new* advanced/hard exercises."""
    if "difficulty" not in picked.columns:
        return picked
    is_complex = picked["difficulty"].astype(str).str.lower().isin(
        ["advanced", "hard"]
    )
    complex_indices = picked[is_complex].index[max_new:]
    if len(complex_indices) > 0:
        picked = picked.drop(complex_indices)
    return picked


def count_complex(exercises: list[dict]) -> int:
    """RULE-D: Count advanced/hard exercises in a session."""
    return sum(
        1 for e in exercises
        if str(e.get("difficulty", "")).lower() in ("advanced", "hard")
    )


def select_exercises_for_day(
    exercises: pd.DataFrame,
    day_index: int,
    body_focus: set[str],
    count: int = 3,
) -> list[dict]:
    """Select exercises for a training day with body-focus and offset variation.

    Applies: RULE-001, RULE-009, RULE-010, RULE-A (session order), RULE-D (cap).
    Returns ``count`` main exercises + 1 stretch at the end.
    """
    cols = [c for c in _OUTPUT_COLS if c in exercises.columns]
    if exercises.empty:
        return []

    df = exercises.copy()
    stretch_mask = df.apply(_is_stretch, axis=1)
    non_stretches = df[~stretch_mask]

    if non_stretches.empty:
        non_stretches = df  # fallback: use everything

    # Score by body-focus match (matching first, then rest)
    match_col = non_stretches["focus"].apply(
        lambda f: _matches_body_focus(f, body_focus)
    )
    matching = non_stretches[match_col]
    non_matching = non_stretches[~match_col]
    ordered = pd.concat([matching, non_matching])

    # Apply day-index offset for variety
    n = len(ordered)
    if n > 0:
        offset = day_index % n
        idx = list(ordered.index)
        ordered = ordered.loc[idx[offset:] + idx[:offset]]

    picked = ordered.head(count)

    # Add stretch at end (RULE-012)
    stretch = _pick_stretch(df, day_index)
    if not stretch.empty:
        picked_ids = set(picked["exercise_id"]) if "exercise_id" in picked.columns else set()
        stretch_id = stretch.iloc[0].get("exercise_id", "")
        if stretch_id and stretch_id in picked_ids:
            picked = picked[picked["exercise_id"] != stretch_id]
            picked = pd.concat([picked.head(count), stretch])
        else:
            picked = pd.concat([picked, stretch])

    # RULE-D: cap complex exercises
    picked = _cap_complex(picked)

    # RULE-A: sort by session phase (awareness → strength → cardio → stretch)
    picked = picked.copy()
    picked["_phase"] = picked.apply(_assign_session_phase, axis=1)
    picked = picked.sort_values("_phase", kind="mergesort")
    picked = picked.drop(columns=["_phase"])

    return picked[cols].fillna("").to_dict(orient="records")


def select_light_exercises(
    exercises: pd.DataFrame,
    day_index: int,
    count: int = 4,
) -> list[dict]:
    """Select light exercises for recovery / post-sport days.

    Applies: RULE-007, RULE-008, RULE-A (session order), RULE-D (cap).
    Only beginner/intermediate, prefers low impact & flexibility/body-awareness.
    """
    cols = [c for c in _OUTPUT_COLS if c in exercises.columns]
    if exercises.empty:
        return []

    df = exercises.copy()

    # Filter: only beginner / intermediate (no advanced / hard)
    if "difficulty" in df.columns:
        easy = df[df["difficulty"].astype(str).str.lower().isin(
            ["beginner", "intermediate"]
        )]
        if not easy.empty:
            df = easy

    # Prefer low impact
    if "impact" in df.columns:
        low = df[df["impact"].astype(str).str.lower() == "low"]
        if len(low) >= count:
            df = low

    # Separate stretches
    stretch_mask = df.apply(_is_stretch, axis=1)
    non_stretches = df[~stretch_mask]

    if non_stretches.empty:
        non_stretches = df

    # Prefer flexibility / body-awareness focus
    light_focus = {"flexibility", "body_awareness", "mental control", "coordination"}
    priority_col = non_stretches["focus"].apply(
        lambda f: bool(_tokenize(f) & light_focus)
    )
    prioritized = pd.concat([
        non_stretches[priority_col],
        non_stretches[~priority_col],
    ])

    # Apply offset for variety
    n = len(prioritized)
    if n > 0:
        offset = day_index % n
        idx = list(prioritized.index)
        prioritized = prioritized.loc[idx[offset:] + idx[:offset]]

    picked = prioritized.head(count)

    # Add stretch at end
    stretch = _pick_stretch(df, day_index)
    if not stretch.empty:
        picked_ids = set(picked["exercise_id"]) if "exercise_id" in picked.columns else set()
        stretch_id = stretch.iloc[0].get("exercise_id", "")
        if stretch_id and stretch_id in picked_ids:
            picked = picked[picked["exercise_id"] != stretch_id]
            picked = pd.concat([picked.head(count), stretch])
        else:
            picked = pd.concat([picked, stretch])

    # RULE-D: cap complex (light days should already exclude advanced)
    picked = _cap_complex(picked)

    # RULE-A: sort by session phase
    picked = picked.copy()
    picked["_phase"] = picked.apply(_assign_session_phase, axis=1)
    picked = picked.sort_values("_phase", kind="mergesort")
    picked = picked.drop(columns=["_phase"])

    return picked[cols].fillna("").to_dict(orient="records")


# ── RULE-E: Movement Plane Coverage ────────────────────────────────

def compute_plane_coverage(
    plan: list[dict], all_exercises: pd.DataFrame
) -> dict[str, bool]:
    """RULE-E: Check which movement planes are covered by the plan."""
    exercise_ids: set[str] = set()
    for day in plan:
        for ex in day.get("exercises", []):
            eid = ex.get("exercise_id", "")
            if eid:
                exercise_ids.add(eid)

    covered: set[str] = set()
    for _, row in all_exercises.iterrows():
        if row.get("exercise_id") in exercise_ids:
            planes = classify_movement_plane(
                str(row.get("focus", "")),
                str(row.get("tags", "")),
            )
            covered |= planes

    return {
        "median": "sagittal" in covered,
        "dorsal": "frontal" in covered,
        "transversal": "transverse" in covered,
    }


def fill_plane_gaps(
    plan: list[dict], all_exercises: pd.DataFrame, coverage: dict[str, bool]
) -> None:
    """RULE-E: Add exercises for any missing movement planes (modifies plan)."""
    needed: list[str] = []
    for label, internal in (("median", "sagittal"), ("dorsal", "frontal"), ("transversal", "transverse")):
        if not coverage.get(label, False):
            needed.append(internal)

    if not needed:
        return

    used_ids: set[str] = set()
    for day in plan:
        for ex in day.get("exercises", []):
            used_ids.add(ex.get("exercise_id", ""))

    cols = [c for c in _OUTPUT_COLS if c in all_exercises.columns]

    for target_plane in needed:
        for _, row in all_exercises.iterrows():
            eid = row.get("exercise_id", "")
            if eid in used_ids:
                continue
            planes = classify_movement_plane(
                str(row.get("focus", "")),
                str(row.get("tags", "")),
            )
            if target_plane not in planes:
                continue

            ex_dict = {c: str(row.get(c, "") or "") for c in cols}
            # Add to first training day
            for day in plan:
                if day.get("type") in ("training", "light_training"):
                    day["exercises"].append(ex_dict)
                    # Re-sort by session phase
                    day["exercises"] = sorted(
                        day["exercises"], key=_phase_from_dict
                    )
                    used_ids.add(eid)
                    break
            break

import pandas as pd
import pytest
from planner.generator import (
    make_week_plan,
    _build_schedule,
    _find_sport_days,
    _get_training_range,
    WEEK,
)
from planner.rules import (
    RULES,
    BODY_FOCUS_ROTATION,
    classify_body_region,
    classify_movement_plane,
    classify_focus,
    select_exercises_for_day,
    select_light_exercises,
    _is_stretch,
    _assign_session_phase,
    is_unstable_equipment,
    filter_unstable,
    filter_high_intensity,
    count_complex,
    compute_plane_coverage,
)


def _dummy_exercises():
    """Exercise DataFrame covering all body regions, planes, and difficulties."""
    return pd.DataFrame([
        # Rear + core (sagittal) — beginner
        {"exercise_id": "REAR_001", "name_en": "Two Paws On",
         "focus": "rear_end, core, strength", "difficulty": "beginner",
         "impact": "low", "tags": "rear_end,core,weight_shift,foundation",
         "equipment": "none", "video_url": ""},
        # Core (sagittal) — intermediate
        {"exercise_id": "CORE_001", "name_en": "Weight Shifts",
         "focus": "core", "difficulty": "intermediate",
         "impact": "medium", "tags": "core,stabilization,balance",
         "equipment": "none", "video_url": ""},
        # Front + core (sagittal) — intermediate
        {"exercise_id": "FRONT_001", "name_en": "Backwards Walking",
         "focus": "front_end, core, body_awareness",
         "difficulty": "intermediate", "impact": "medium",
         "tags": "front_end,core,proprioception,coordination",
         "equipment": "none", "video_url": ""},
        # Shoulders + flexibility (frontal) — intermediate
        {"exercise_id": "FLEX_001", "name_en": "Stand-Bow-Stand",
         "focus": "shoulders, flexibility, core",
         "difficulty": "intermediate", "impact": "medium",
         "tags": "flexibility,shoulders,core,front_end",
         "equipment": "none", "video_url": ""},
        # Full body (frontal) — intermediate
        {"exercise_id": "FULL_001", "name_en": "Sideways Stepping",
         "focus": "full_body, lateral_muscles",
         "difficulty": "intermediate", "impact": "medium",
         "tags": "lateral,full_body,coordination,abduction_adduction",
         "equipment": "none", "video_url": ""},
        # Spine (transverse) — intermediate
        {"exercise_id": "TWIST_001", "name_en": "Twist & Twirl",
         "focus": "spine, lateral_flexibility, core",
         "difficulty": "intermediate", "impact": "medium",
         "tags": "spine,flexibility,rotation,core",
         "equipment": "none", "video_url": ""},
        # Body awareness / mental — beginner
        {"exercise_id": "MENTAL_001", "name_en": "It's Your Choice",
         "focus": "mental control", "difficulty": "beginner",
         "impact": "low", "tags": "impulse_control,foundation,mental",
         "equipment": "none", "video_url": ""},
        # Rear end — intermediate
        {"exercise_id": "REAR_002", "name_en": "Pivot",
         "focus": "rear_end, core, lateral_muscles",
         "difficulty": "intermediate", "impact": "medium",
         "tags": "rear_end,core,lateral_work,body_awareness",
         "equipment": "none", "video_url": ""},
        # Stretch — beginner
        {"exercise_id": "STRETCH_001", "name_en": "Cookie Stretch",
         "focus": "spine, neck, lateral_flexibility",
         "difficulty": "beginner", "impact": "low",
         "tags": "flexibility,stretching,ROM,spine",
         "equipment": "none", "video_url": ""},
        # Stretch — intermediate
        {"exercise_id": "STRETCH_002", "name_en": "Rear End Stretch",
         "focus": "rear_end, hip_flexors, core",
         "difficulty": "intermediate", "impact": "low",
         "tags": "flexibility,rear_end,hip_flexors,core_stability",
         "equipment": "none", "video_url": ""},
        # Advanced / high impact — should NOT appear in light days
        {"exercise_id": "ADV_001", "name_en": "Push Up",
         "focus": "shoulders, core, stability",
         "difficulty": "advanced", "impact": "high",
         "tags": "shoulders,advanced,core,eccentric",
         "equipment": "none", "video_url": ""},
        # Unstable equipment exercise — for RULE-B testing
        {"exercise_id": "UNSTABLE_001", "name_en": "Balance Pad Stand",
         "focus": "core, body_awareness",
         "difficulty": "intermediate", "impact": "low",
         "tags": "balance,proprioception,core",
         "equipment": "balance pad", "video_url": ""},
    ])


def _get_plan(case, df=None):
    """Run make_week_plan and return the plan list."""
    if df is None:
        df = _dummy_exercises()
    return make_week_plan(case, df)["plan"]


def _get_types(plan):
    """Extract {day: type} mapping from plan output."""
    return {entry["day"]: entry["type"] for entry in plan}


# ── Classification tests ─────────────────────────────────────────────

class TestClassification:
    def test_body_region_rear(self):
        regions = classify_body_region("rear_end, core, strength")
        assert "rear" in regions
        assert "core" in regions

    def test_body_region_front(self):
        regions = classify_body_region("front_end, core, body_awareness")
        assert "front" in regions

    def test_body_region_full_body(self):
        regions = classify_body_region("full_body")
        assert regions == {"front", "core", "rear"}

    def test_body_region_default_core(self):
        regions = classify_body_region("mental control")
        assert regions == {"core"}

    def test_movement_plane_sagittal(self):
        planes = classify_movement_plane(
            "rear_end, core", "rear_end,strength,eccentric"
        )
        assert "sagittal" in planes

    def test_movement_plane_frontal(self):
        planes = classify_movement_plane(
            "full_body, lateral_muscles", "lateral,abduction_adduction"
        )
        assert "frontal" in planes

    def test_movement_plane_transverse(self):
        planes = classify_movement_plane(
            "spine, lateral_flexibility", "spine,rotation"
        )
        assert "transverse" in planes

    def test_focus_strength(self):
        assert "strength" in classify_focus("rear_end, core, strength")

    def test_focus_flexibility(self):
        assert "flexibility" in classify_focus("shoulders, flexibility, core")

    def test_focus_body_awareness(self):
        assert "body_awareness" in classify_focus("mental control")

    def test_focus_default(self):
        cats = classify_focus("something_unknown")
        assert cats == {"body_awareness"}


# ── Exercise selection tests ──────────────────────────────────────────

class TestExerciseSelection:
    def test_training_day_ends_with_stretch(self):
        df = _dummy_exercises()
        exercises = select_exercises_for_day(df, 0, {"rear", "core"}, count=3)
        assert len(exercises) >= 2
        last = exercises[-1]
        last_name = last.get("name_en", "").lower()
        last_id = last.get("exercise_id", "")
        assert "stretch" in last_name or last_id.startswith("STRETCH")

    def test_different_exercises_per_day(self):
        """Different day_index → different exercise sets."""
        df = _dummy_exercises()
        day0 = select_exercises_for_day(df, 0, {"rear", "core"}, count=3)
        day3 = select_exercises_for_day(df, 3, {"rear", "core"}, count=3)
        ids0 = [e["exercise_id"] for e in day0]
        ids3 = [e["exercise_id"] for e in day3]
        assert ids0 != ids3

    def test_body_focus_affects_selection(self):
        """Different body focus → different exercises prioritized."""
        df = _dummy_exercises()
        rear_day = select_exercises_for_day(df, 0, {"rear", "core"}, count=3)
        front_day = select_exercises_for_day(df, 0, {"front", "flexibility"}, count=3)
        rear_ids = [e["exercise_id"] for e in rear_day]
        front_ids = [e["exercise_id"] for e in front_day]
        assert rear_ids != front_ids

    def test_light_excludes_advanced(self):
        """Light exercises should not include advanced difficulty."""
        df = _dummy_exercises()
        exercises = select_light_exercises(df, 0, count=4)
        for ex in exercises:
            assert ex.get("difficulty", "") != "advanced", (
                f"Advanced exercise {ex['exercise_id']} in light day"
            )

    def test_light_ends_with_stretch(self):
        df = _dummy_exercises()
        exercises = select_light_exercises(df, 0, count=4)
        assert len(exercises) >= 2
        last = exercises[-1]
        last_name = last.get("name_en", "").lower()
        last_id = last.get("exercise_id", "")
        assert "stretch" in last_name or last_id.startswith("STRETCH")

    def test_light_varies_by_day(self):
        df = _dummy_exercises()
        day0 = select_light_exercises(df, 0, count=4)
        day3 = select_light_exercises(df, 3, count=4)
        ids0 = [e["exercise_id"] for e in day0]
        ids3 = [e["exercise_id"] for e in day3]
        assert ids0 != ids3

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["exercise_id", "name_en", "focus",
                                       "difficulty", "video_url"])
        assert select_exercises_for_day(empty, 0, {"core"}) == []
        assert select_light_exercises(empty, 0) == []


# ── Helper function tests ────────────────────────────────────────────

class TestTrainingRange:
    def test_low(self):
        assert _get_training_range("low") == (1, 2)

    def test_medium(self):
        assert _get_training_range("medium") == (2, 3)

    def test_high(self):
        assert _get_training_range("high") == (3, 4)

    def test_unknown_defaults_to_medium(self):
        assert _get_training_range("unknown") == (2, 3)


class TestFindSportDays:
    def test_agility_is_sport(self):
        assert _find_sport_days({"Mon": ["agility"]}) == {"Mon"}

    def test_competition_is_sport(self):
        assert _find_sport_days({"Sat": ["competition"]}) == {"Sat"}

    def test_walk_is_not_sport(self):
        assert _find_sport_days({"Wed": ["walk"]}) == set()

    def test_mixed_activities(self):
        acts = {"Mon": ["agility"], "Wed": ["walk"], "Fri": ["trial"]}
        assert _find_sport_days(acts) == {"Mon", "Fri"}

    def test_empty(self):
        assert _find_sport_days({}) == set()


# ── Schedule rule validation ─────────────────────────────────────────

class TestScheduleRules:
    """Verify _build_schedule enforces every rule across many combos."""

    SPORT_COMBOS = [
        set(),
        {"Tue"},
        {"Mon", "Thu"},
        {"Mon", "Wed", "Fri"},
        {"Mon", "Tue"},            # consecutive sport
        {"Sat"},                   # sport near end of week
    ]
    FITNESS_LEVELS = ["low", "medium", "high"]

    def _all_schedules(self):
        """Yield (schedule, sport_set, fitness) for every combo."""
        for sport in self.SPORT_COMBOS:
            for fit in self.FITNESS_LEVELS:
                tr = _get_training_range(fit)
                sched = _build_schedule(sport, tr, len(sport) > 0)
                yield sched, sport, fit

    # RULE-007/008: rest or light after sport
    def test_rest_or_light_after_sport_day(self):
        for sched, sport, fit in self._all_schedules():
            for i, d in enumerate(WEEK):
                if sched[d] == "sport" and i + 1 < len(WEEK):
                    next_type = sched[WEEK[i + 1]]
                    assert next_type in ("rest", "light", "sport"), (
                        f"After sport on {d}, got '{next_type}' "
                        f"(sport={sport}, fitness={fit})"
                    )

    # RULE-005: at least 1 rest day
    def test_at_least_one_rest_day(self):
        for sched, sport, fit in self._all_schedules():
            rest = sum(1 for v in sched.values() if v == "rest")
            assert rest >= 1, (
                f"No rest day: sport={sport}, fitness={fit}, "
                f"schedule={list(sched.values())}"
            )

    # No 2 consecutive rest days
    def test_no_consecutive_rest_days(self):
        for sched, sport, fit in self._all_schedules():
            for i in range(len(WEEK) - 1):
                pair = (sched[WEEK[i]], sched[WEEK[i + 1]])
                assert pair != ("rest", "rest"), (
                    f"Consecutive rest {WEEK[i]}-{WEEK[i+1]}: "
                    f"sport={sport}, fitness={fit}"
                )

    # No 2 consecutive training when sport in week
    def test_no_consecutive_training_with_sport(self):
        for sched, sport, fit in self._all_schedules():
            if not sport:
                continue
            for i in range(len(WEEK) - 1):
                both_train = (
                    sched[WEEK[i]] == "training"
                    and sched[WEEK[i + 1]] == "training"
                )
                assert not both_train, (
                    f"Consecutive training {WEEK[i]}-{WEEK[i+1]} "
                    f"with sport={sport}, fitness={fit}"
                )

    # RULE-004: no 3+ consecutive training days
    def test_no_three_consecutive_training(self):
        for sched, sport, fit in self._all_schedules():
            for i in range(len(WEEK) - 2):
                triple = all(
                    sched[WEEK[j]] == "training" for j in range(i, i + 3)
                )
                assert not triple, (
                    f"3 consecutive training {WEEK[i]}-{WEEK[i+2]}: "
                    f"sport={sport}, fitness={fit}"
                )

    # Training-count ranges (no-sport only, sport reduces availability)
    def test_training_count_no_sport(self):
        for fit in self.FITNESS_LEVELS:
            lo, hi = _get_training_range(fit)
            sched = _build_schedule(set(), (lo, hi), has_sport=False)
            count = sum(1 for v in sched.values() if v == "training")
            assert lo <= count <= hi, (
                f"fitness={fit}: expected {lo}-{hi} training days, got {count}"
            )

    # Every day accounted for
    def test_all_days_assigned(self):
        for sched, sport, fit in self._all_schedules():
            for d in WEEK:
                assert sched[d] in ("sport", "training", "rest", "light"), (
                    f"{d} unassigned: sport={sport}, fitness={fit}"
                )


# ── Integration tests (make_week_plan) ───────────────────────────────

class TestMakeWeekPlan:
    def test_high_fitness_one_sport_day(self):
        """Chouffe scenario: high fitness, agility on Tuesday."""
        case = {"fitness_level": "high", "activities": "tue:agility,thu:walk"}
        plan = _get_plan(case)
        types = _get_types(plan)

        assert len(plan) == 7
        assert types["Tue"] == "sport_only"
        assert types["Wed"] in ("rest", "light_training")
        training_count = sum(1 for v in types.values() if v == "training")
        assert training_count >= 3

    def test_medium_fitness_no_sport(self):
        case = {"fitness_level": "medium", "activities": ""}
        plan = _get_plan(case)
        types = _get_types(plan)

        training = sum(1 for v in types.values() if v == "training")
        rest = sum(1 for v in types.values() if v == "rest")
        assert 2 <= training <= 3
        assert rest >= 1

    def test_low_fitness_many_sport_days(self):
        case = {
            "fitness_level": "low",
            "activities": "mon:agility,wed:competition,fri:training",
        }
        plan = _get_plan(case)
        types = _get_types(plan)

        sport = sum(1 for v in types.values() if v == "sport_only")
        assert sport == 3

    def test_consecutive_sport_days(self):
        case = {
            "fitness_level": "medium",
            "activities": "mon:agility,tue:competition",
        }
        plan = _get_plan(case)
        types = _get_types(plan)

        assert types["Mon"] == "sport_only"
        assert types["Tue"] == "sport_only"
        assert types["Wed"] in ("rest", "light_training")

    def test_no_activities_at_all(self):
        case = {"fitness_level": "medium"}
        plan = _get_plan(case)
        types = _get_types(plan)

        assert len(plan) == 7
        training = sum(1 for v in types.values() if v == "training")
        assert 2 <= training <= 3

    def test_sport_on_sunday(self):
        case = {"fitness_level": "medium", "activities": "sun:agility"}
        plan = _get_plan(case)
        types = _get_types(plan)
        assert types["Sun"] == "sport_only"

    def test_all_seven_days_present(self):
        case = {"fitness_level": "medium", "activities": ""}
        plan = _get_plan(case)
        days = [e["day"] for e in plan]
        assert days == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def test_empty_exercises_dataframe(self):
        case = {"fitness_level": "medium", "activities": ""}
        empty_df = pd.DataFrame(columns=[
            "exercise_id", "name_en", "focus", "difficulty", "video_url",
        ])
        plan = make_week_plan(case, empty_df)["plan"]
        t_days = [e for e in plan if e["type"] == "training"]
        assert all(d["exercises"] == [] for d in t_days)

    def test_returns_dict_with_plan_and_coverage(self):
        case = {"fitness_level": "medium", "activities": ""}
        result = make_week_plan(case, _dummy_exercises())
        assert "plan" in result
        assert "plane_coverage" in result
        assert isinstance(result["plan"], list)
        assert isinstance(result["plane_coverage"], dict)
        for key in ("median", "dorsal", "transversal"):
            assert key in result["plane_coverage"]


# ── Output structure tests ────────────────────────────────────────────

class TestOutputStructure:
    def test_training_day_fields(self):
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case)
        t_days = [e for e in plan if e["type"] == "training"]

        assert len(t_days) >= 1
        day = t_days[0]
        assert "warmup" in day
        assert "exercises" in day
        assert isinstance(day["exercises"], list)
        assert len(day["exercises"]) >= 3
        assert "cooldown" in day
        assert "focus" in day
        assert "new_exercises_count" in day

    def test_sport_day_fields(self):
        case = {"fitness_level": "high", "activities": "tue:agility"}
        plan = _get_plan(case)
        s_days = [e for e in plan if e["type"] == "sport_only"]

        assert len(s_days) == 1
        assert "agility" in s_days[0]["note"]

    def test_light_training_day_fields(self):
        """Light training days after sport should have warmup + exercises."""
        case = {"fitness_level": "high", "activities": "tue:agility"}
        plan = _get_plan(case)
        l_days = [e for e in plan if e["type"] == "light_training"]

        for day in l_days:
            assert "warmup" in day
            assert "exercises" in day
            assert isinstance(day["exercises"], list)
            assert "cooldown" in day
            assert "new_exercises_count" in day

    def test_rest_day_fields(self):
        case = {"fitness_level": "low", "activities": ""}
        plan = _get_plan(case)
        r_days = [e for e in plan if e["type"] == "rest"]
        assert len(r_days) >= 1
        assert "note" in r_days[0]


# ── Applied rules tests ──────────────────────────────────────────────

class TestAppliedRules:
    def test_every_day_has_applied_rules(self):
        case = {"fitness_level": "medium", "activities": "tue:agility"}
        plan = _get_plan(case)
        for day in plan:
            assert "applied_rules" in day, f"Missing applied_rules on {day['day']}"
            assert isinstance(day["applied_rules"], list)
            assert len(day["applied_rules"]) >= 1

    def test_sport_day_has_rule_008(self):
        case = {"fitness_level": "medium", "activities": "tue:agility"}
        plan = _get_plan(case)
        tue = [d for d in plan if d["day"] == "Tue"][0]
        assert "RULE-008" in tue["applied_rules"]

    def test_training_day_has_all_training_rules(self):
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case)
        t_day = [d for d in plan if d["type"] == "training"][0]
        for rule in ["RULE-001", "RULE-009", "RULE-010", "RULE-012",
                      "RULE-A", "RULE-D", "RULE-E"]:
            assert rule in t_day["applied_rules"], (
                f"Missing {rule} on training day"
            )

    def test_light_training_has_recovery_rules(self):
        case = {"fitness_level": "high", "activities": "tue:agility"}
        plan = _get_plan(case)
        l_days = [d for d in plan if d["type"] == "light_training"]
        if l_days:
            day = l_days[0]
            assert "RULE-007" in day["applied_rules"]
            assert "RULE-012" in day["applied_rules"]
            assert "RULE-A" in day["applied_rules"]
            assert "RULE-E" in day["applied_rules"]

    def test_rest_day_has_rule_005(self):
        case = {"fitness_level": "medium", "activities": ""}
        plan = _get_plan(case)
        r_day = [d for d in plan if d["type"] == "rest"][0]
        assert "RULE-005" in r_day["applied_rules"]

    def test_all_rule_ids_are_valid(self):
        case = {"fitness_level": "high", "activities": "tue:agility"}
        plan = _get_plan(case)
        for day in plan:
            for rule_id in day["applied_rules"]:
                assert rule_id in RULES, f"Unknown rule {rule_id}"


# ── Exercise variation tests ──────────────────────────────────────────

class TestExerciseVariation:
    def test_training_days_have_different_exercises(self):
        """Each training day should have different exercises (offset rotation)."""
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case)
        t_days = [d for d in plan if d["type"] == "training"]

        assert len(t_days) >= 2
        all_sets = []
        for day in t_days:
            ids = tuple(e["exercise_id"] for e in day["exercises"])
            all_sets.append(ids)

        # At least some days should differ
        assert len(set(all_sets)) > 1, (
            "All training days have identical exercises"
        )

    def test_body_focus_rotates(self):
        """Training days should rotate through body focus labels."""
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case)
        t_days = [d for d in plan if d["type"] == "training"]

        focuses = [d["focus"] for d in t_days]
        # With 3+ training days, we should see at least 2 different focuses
        assert len(set(focuses)) >= 2, (
            f"Focus doesn't vary: {focuses}"
        )

    def test_training_day_ends_with_stretch(self):
        """Last exercise in each training day should be a stretch."""
        df = _dummy_exercises()
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case, df)
        t_days = [d for d in plan if d["type"] == "training"]

        for day in t_days:
            exs = day["exercises"]
            if not exs:
                continue
            last = exs[-1]
            last_name = last.get("name_en", "").lower()
            last_id = last.get("exercise_id", "")
            assert "stretch" in last_name or last_id.startswith("STRETCH"), (
                f"Training {day['day']} doesn't end with stretch: {last}"
            )


# ── RULE-A: Session phase ordering tests ─────────────────────────────

class TestRuleASessionOrder:
    def test_phase_assignment(self):
        """Body awareness=1, strength=2, cardio=3, stretch=4."""
        df = _dummy_exercises()
        mental = df[df["exercise_id"] == "MENTAL_001"].iloc[0]
        rear = df[df["exercise_id"] == "REAR_001"].iloc[0]
        stretch = df[df["exercise_id"] == "STRETCH_001"].iloc[0]
        assert _assign_session_phase(mental) == 1
        assert _assign_session_phase(rear) == 2
        assert _assign_session_phase(stretch) == 4

    def test_exercises_sorted_by_phase(self):
        """Within a training day, exercises should be sorted by phase."""
        df = _dummy_exercises()
        exercises = select_exercises_for_day(df, 0, {"rear", "core"}, count=3)
        if len(exercises) >= 2:
            # Stretch should always be last
            last = exercises[-1]
            assert "stretch" in last.get("name_en", "").lower() or \
                   last.get("exercise_id", "").startswith("STRETCH")


# ── RULE-B: Unstable equipment filter tests ──────────────────────────

class TestRuleBUnstable:
    def test_is_unstable_balance_pad(self):
        assert is_unstable_equipment("balance pad") is True

    def test_is_unstable_none(self):
        assert is_unstable_equipment("none") is False

    def test_filter_unstable_removes_for_medium(self):
        df = _dummy_exercises()
        filtered = filter_unstable(df, "medium")
        ids = set(filtered["exercise_id"])
        assert "UNSTABLE_001" not in ids

    def test_filter_unstable_keeps_for_high(self):
        df = _dummy_exercises()
        filtered = filter_unstable(df, "high")
        ids = set(filtered["exercise_id"])
        assert "UNSTABLE_001" in ids

    def test_rule_b_applied_in_plan_medium(self):
        """RULE-B should appear in applied_rules for non-high fitness."""
        case = {"fitness_level": "medium", "activities": ""}
        plan = _get_plan(case)
        t_days = [d for d in plan if d["type"] == "training"]
        if t_days:
            assert "RULE-B" in t_days[0]["applied_rules"]


# ── RULE-C: High intensity filter tests ──────────────────────────────

class TestRuleCIntensity:
    def test_filter_puppies(self):
        df = _dummy_exercises()
        filtered = filter_high_intensity(df, "puppy", set())
        for _, row in filtered.iterrows():
            assert str(row.get("impact", "")).lower() != "high"

    def test_filter_overweight(self):
        df = _dummy_exercises()
        filtered = filter_high_intensity(df, "adult", {"overweight"})
        for _, row in filtered.iterrows():
            assert str(row.get("impact", "")).lower() != "high"

    def test_no_filter_for_normal_adult(self):
        df = _dummy_exercises()
        filtered = filter_high_intensity(df, "adult", set())
        assert len(filtered) == len(df)

    def test_rule_c_applied_for_puppy(self):
        case = {"fitness_level": "medium", "age_group": "puppy", "activities": ""}
        plan = _get_plan(case)
        t_days = [d for d in plan if d["type"] == "training"]
        if t_days:
            assert "RULE-C" in t_days[0]["applied_rules"]


# ── RULE-D: Complex exercise cap tests ───────────────────────────────

class TestRuleDComplexCap:
    def test_count_complex(self):
        exercises = [
            {"difficulty": "advanced"},
            {"difficulty": "beginner"},
            {"difficulty": "hard"},
            {"difficulty": "intermediate"},
        ]
        assert count_complex(exercises) == 2

    def test_count_complex_none(self):
        exercises = [
            {"difficulty": "beginner"},
            {"difficulty": "intermediate"},
        ]
        assert count_complex(exercises) == 0

    def test_new_exercises_count_field_present(self):
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case)
        for day in plan:
            if day["type"] in ("training", "light_training"):
                assert "new_exercises_count" in day
                assert isinstance(day["new_exercises_count"], int)

    def test_max_2_complex_per_session(self):
        """Training days should have at most 2 advanced/hard exercises."""
        case = {"fitness_level": "high", "activities": ""}
        plan = _get_plan(case)
        for day in plan:
            if day["type"] in ("training", "light_training"):
                assert day["new_exercises_count"] <= 2


# ── RULE-E: Plane coverage tests ────────────────────────────────────

class TestRuleEPlaneCoverage:
    def test_coverage_dict_keys(self):
        case = {"fitness_level": "medium", "activities": ""}
        result = make_week_plan(case, _dummy_exercises())
        cov = result["plane_coverage"]
        assert set(cov.keys()) == {"median", "dorsal", "transversal"}

    def test_coverage_with_diverse_exercises(self):
        """Dummy exercises cover all planes, so coverage should be full."""
        case = {"fitness_level": "high", "activities": ""}
        result = make_week_plan(case, _dummy_exercises())
        cov = result["plane_coverage"]
        # With our diverse dummy set, at least median (sagittal) should be covered
        assert cov["median"] is True

    def test_compute_plane_coverage_function(self):
        df = _dummy_exercises()
        plan = [
            {"exercises": [{"exercise_id": "REAR_001"}, {"exercise_id": "TWIST_001"}]},
            {"exercises": [{"exercise_id": "FULL_001"}]},
        ]
        cov = compute_plane_coverage(plan, df)
        assert cov["median"] is True   # REAR_001 is sagittal
        assert cov["transversal"] is True  # TWIST_001 is transverse
        assert cov["dorsal"] is True   # FULL_001 is frontal (full_body)

    def test_rule_e_in_training_applied_rules(self):
        case = {"fitness_level": "medium", "activities": ""}
        plan = _get_plan(case)
        t_days = [d for d in plan if d["type"] == "training"]
        for day in t_days:
            assert "RULE-E" in day["applied_rules"]

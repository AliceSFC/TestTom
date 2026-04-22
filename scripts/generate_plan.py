import argparse
import json

from planner.io import load_data
from planner.rules import parse_limitations, filter_exercises
from planner.generator import make_week_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", required=True)
    args = parser.parse_args()

    exercises, cases = load_data()

    case = cases[cases["case_id"] == args.case_id]
    if case.empty:
        raise SystemExit(f"Unknown case_id: {args.case_id}")

    row = case.iloc[0].to_dict()

    limitations = parse_limitations(row.get("limitations", ""))
    age_group = str(row.get("age_group", "adult"))
    equipment_available = str(row.get("equipment_available", "none"))

    allowed = filter_exercises(exercises, limitations, age_group, equipment_available)
    week_plan = make_week_plan(row, allowed)

    output = {
        "case_id": args.case_id,
        "dog_name": row.get("dog_name"),
        "plan": week_plan,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

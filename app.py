from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from planner.io import load_data
from planner.rules import parse_limitations, filter_exercises
from planner.generator import make_week_plan
from pydantic import BaseModel

app = FastAPI(title="GoDoggity Fitness Planner Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data model voor POST request
class PlanRequest(BaseModel):
    dog_name: str
    age_group: str
    fitness_level: str
    limitations: str = ""
    activities: str = ""
    equipment: str = "none"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/generate-plan")
def generate_plan(case_id: str):
    exercises, cases = load_data()

    case = cases[cases["case_id"] == case_id]
    if case.empty:
        raise HTTPException(status_code=404, detail="Unknown case_id")

    row = case.iloc[0].to_dict()

    limitations = parse_limitations(row.get("limitations", ""))
    age_group = str(row.get("age_group", "adult"))
    equipment_available = str(row.get("equipment_available", "none"))

    allowed = filter_exercises(exercises, limitations, age_group, equipment_available)
    result = make_week_plan(row, allowed)

    return {
        "case_id": case_id,
        "dog_name": row.get("dog_name"),
        "plan": result["plan"],
        "plane_coverage": result["plane_coverage"],
    }

@app.post("/generate-plan")
def generate_plan_post(request: PlanRequest):
    exercises, _ = load_data()

    row = {
        "dog_name": request.dog_name,
        "age_group": request.age_group,
        "fitness_level": request.fitness_level,
        "limitations": request.limitations,
        "activities": request.activities,
        "equipment_available": request.equipment,
    }

    limitations = parse_limitations(row.get("limitations", ""))
    age_group = str(row.get("age_group", "adult"))
    equipment_available = str(row.get("equipment_available", "none"))

    allowed = filter_exercises(exercises, limitations, age_group, equipment_available)
    result = make_week_plan(row, allowed)

    return {
        "dog_name": row.get("dog_name"),
        "fitness_level": row.get("fitness_level"),
        "week_plan": result["plan"],
        "plane_coverage": result["plane_coverage"],
    }

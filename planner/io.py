from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def load_data():
    exercises = pd.read_csv(DATA_DIR / "exercises.csv")
    cases = pd.read_csv(DATA_DIR / "test_cases.csv")
    return exercises, cases

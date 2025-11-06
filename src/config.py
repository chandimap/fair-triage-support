
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for p in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

TARGET_COL = "deteriorated_6w"
GROUP_COLS = ["sex", "ethnicity", "deprivation_quintile", "age_band"]
NUM_COLS = ["age", "bmi", "resting_hr", "tug_sec", "walk_6m", "pain_score", "mobility_score", "comorb_count"]
CAT_COLS = ["sex", "ethnicity", "deprivation_quintile", "age_band"]

RANDOM_STATE = 42

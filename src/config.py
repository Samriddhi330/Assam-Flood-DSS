"""
config.py
---------
Central configuration for the Assam Flood DSS pipeline.
Every other script imports paths and constants from here so there is
a single place to change them.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATASET_PATH = os.path.join(DATA_DIR, "baseline_modeling_dataset_with_metadata.csv")
CENTROIDS_PATH = os.path.join(DATA_DIR, "circle_centroids.csv")
BASES_PATH = os.path.join(DATA_DIR, "relief_bases.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "flood_impact_model.joblib")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions_latest.csv")
SHAP_GLOBAL_PATH = os.path.join(OUTPUT_DIR, "shap_global_importance.csv")
SHAP_LOCAL_PATH = os.path.join(OUTPUT_DIR, "shap_local_explanations.csv")
ALLOCATION_PATH = os.path.join(OUTPUT_DIR, "relief_allocation.csv")
ALLOCATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "relief_allocation_summary.csv")

for _dir in (MODEL_DIR, OUTPUT_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Modelling constants (Stage 1 — Predictive Engine)
# ---------------------------------------------------------------------------
ID_COLUMN = "object_id"
TIME_COLUMN = "timeperiod"
TARGET_COLUMN = "next_month_impact"

# The 20 baseline prediction features documented in docs/Modeling_Findings.md
FEATURE_COLUMNS = [
    "max_rain",
    "mean_rain",
    "sum_rain",
    "mean_ndvi",
    "mean_ndbi",
    "sum_population",
    "sum_aged_population",
    "sum_young_population",
    "mean_sex_ratio",
    "total_hhd",
    "net_sown_area_in_hac",
    "schools_count",
    "health_centres_count",
    "road_length",
    "distance_from_river",
    "drainage_density",
    "elevation_mean",
    "slope_mean",
    "mean_cn",
    "impact",
]

# Fraction of the chronologically-ordered months used for training.
# The remaining, most-recent months are held out for testing (temporal split,
# never a random split — see docs/Modeling_Findings.md section 7).
TRAIN_MONTH_FRACTION = 0.80

RANDOM_STATE = 42

# Probability -> impact-category thresholds used for reporting / relief
# prioritisation. These are cut points on the model's predicted probability
# of next-month flood impact, not on the raw hazard signal.
IMPACT_CATEGORY_THRESHOLDS = {
    "Low": 0.0,
    "Medium": 0.33,
    "High": 0.66,
}

# ---------------------------------------------------------------------------
# Relief-demand heuristic (turns a predicted probability into a resource
# need). This is intentionally simple and documented so it can be replaced
# with ASDMA ground-truth relief-camp / affected-population figures later
# without touching any other part of the pipeline.
# ---------------------------------------------------------------------------
PEOPLE_PER_BOAT = 250          # 1 boat unit assumed to serve this many people
PEOPLE_PER_FOOD_PACKET_BATCH = 50   # 1 "food unit" = supplies for 50 people/day
PEOPLE_PER_MEDICAL_TEAM = 2000  # 1 medical team assumed to serve this many people

# Minimum predicted probability for a circle to be considered "in need" and
# therefore included in the Stage-2 optimisation.
DEMAND_PROBABILITY_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Optimisation constants (Stage 2 — Prescriptive Engine)
# ---------------------------------------------------------------------------
MAX_TRAVEL_KM = 200  # a base cannot practically serve a circle beyond this

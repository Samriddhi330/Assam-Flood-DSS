"""
predict_impact.py
------------------
Bridges Stage 1 (prediction) and Stage 2 (optimisation).

For the most recent month in the dataset, this script:
1. Loads the trained model and scores every Revenue Circle.
2. Converts the predicted probability into an impact category
   (Low / Medium / High) using config.IMPACT_CATEGORY_THRESHOLDS.
3. Converts the probability into a simple relief-demand estimate
   (predicted affected population, boats/food/medical teams needed) using
   the heuristic documented in config.py.
4. Attaches the top-3 SHAP reasons (if outputs/shap_local_explanations.csv
   already exists) so the final table matches the "Predicted Impact /
   Priority / Main Reasons" table in the project write-up.

Output: outputs/predictions_latest.csv — one row per Revenue Circle, this
is the direct input to optimize_relief.py.

Run (after train_model.py, and optionally explain_shap.py):
    python src/predict_impact.py
"""

import os

import joblib
import numpy as np
import pandas as pd

import config


def categorise(prob: float) -> str:
    thresholds = config.IMPACT_CATEGORY_THRESHOLDS
    if prob >= thresholds["High"]:
        return "High"
    if prob >= thresholds["Medium"]:
        return "Medium"
    return "Low"


def demand_from_probability(prob: float, population: float) -> dict:
    """Simple, documented heuristic: expected affected population scales
    with predicted probability. See config.py for the assumptions and how
    to replace this with real ASDMA relief-camp data."""
    affected_population = prob * population
    return {
        "predicted_affected_population": affected_population,
        "boats_needed": int(np.ceil(affected_population / config.PEOPLE_PER_BOAT)),
        "food_units_needed": int(np.ceil(affected_population / config.PEOPLE_PER_FOOD_PACKET_BATCH)),
        "medical_teams_needed": int(np.ceil(affected_population / config.PEOPLE_PER_MEDICAL_TEAM)),
    }


def main():
    model = joblib.load(config.MODEL_PATH)
    df = pd.read_csv(config.RAW_DATASET_PATH)
    df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], format="%Y_%m")

    latest_month = df[config.TIME_COLUMN].max()
    latest_df = df[df[config.TIME_COLUMN] == latest_month].reset_index(drop=True)

    X_latest = latest_df[config.FEATURE_COLUMNS]
    probabilities = model.predict_proba(X_latest)[:, 1]

    out = latest_df[[config.ID_COLUMN, config.TIME_COLUMN, "sum_population"]].copy()
    out["predicted_probability"] = probabilities
    out["impact_category"] = out["predicted_probability"].apply(categorise)

    demand_records = [
        demand_from_probability(p, pop)
        for p, pop in zip(out["predicted_probability"], out["sum_population"])
    ]
    demand_df = pd.DataFrame(demand_records)
    out = pd.concat([out, demand_df], axis=1)

    # Priority rank: 1 = most urgent. Ties broken by predicted affected population.
    out = out.sort_values(
        ["predicted_probability", "predicted_affected_population"], ascending=False
    ).reset_index(drop=True)
    out["priority_rank"] = out.index + 1

    if os.path.exists(config.SHAP_LOCAL_PATH):
        shap_df = pd.read_csv(config.SHAP_LOCAL_PATH)[[config.ID_COLUMN, "top_reasons"]]
        out = out.merge(shap_df, on=config.ID_COLUMN, how="left")
    else:
        out["top_reasons"] = "Run explain_shap.py to populate this column."

    out.to_csv(config.PREDICTIONS_PATH, index=False)

    print(f"Scored {len(out)} circles for {latest_month.strftime('%Y-%m')} (predicting next month's impact).")
    print(out["impact_category"].value_counts().to_string())
    print(f"\nSaved predictions to {config.PREDICTIONS_PATH}")
    print("\nTop 5 highest-priority circles:")
    print(
        out[[config.ID_COLUMN, "predicted_probability", "impact_category", "priority_rank", "top_reasons"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

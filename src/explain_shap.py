"""
explain_shap.py
----------------
Stage 1 — Explainable AI component (TreeSHAP), per docs/Modeling_Findings.md
section 15.

Loads the trained model from models/flood_impact_model.joblib and computes:

1. Global explanation
   Mean absolute SHAP value per feature across the test set -> which
   features generally drive next-month flood-impact predictions.
   Saved to outputs/shap_global_importance.csv

2. Local explanation
   Per-Revenue-Circle, per-latest-month SHAP contributions, plus a
   human-readable "top reasons" string of the form used in the project
   master plan, e.g.:
       "Circle X needs priority because: sum_rain contributed +35%,
        sum_population contributed +25%, road_length contributed +20%"
   Saved to outputs/shap_local_explanations.csv

Run (after train_model.py):
    python src/explain_shap.py
"""

import joblib
import numpy as np
import pandas as pd
import shap

import config


def load_model_and_data():
    model = joblib.load(config.MODEL_PATH)
    df = pd.read_csv(config.RAW_DATASET_PATH)
    df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], format="%Y_%m")
    return model, df


def compute_shap_values(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Some tree explainers return a list (per class) for classifiers.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def global_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["importance_share_pct"] = (100 * out["mean_abs_shap"] / out["mean_abs_shap"].sum()).round(2)
    return out


def top_reasons_string(row_shap: np.ndarray, feature_names: list, top_n: int = 3) -> str:
    """Build a human-readable explanation string from one row's SHAP values,
    e.g. 'sum_rain contributed +35%, sum_population contributed +25%'."""
    total_abs = np.abs(row_shap).sum()
    if total_abs == 0:
        return "No strong contributing factors."

    order = np.argsort(-np.abs(row_shap))[:top_n]
    parts = []
    for idx in order:
        share_pct = 100 * row_shap[idx] / total_abs
        sign = "+" if share_pct >= 0 else ""
        parts.append(f"{feature_names[idx]} contributed {sign}{share_pct:.0f}%")
    return "; ".join(parts)


def local_explanations(df: pd.DataFrame, shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    reasons = [top_reasons_string(shap_values[i], feature_names) for i in range(len(df))]

    out = df[[config.ID_COLUMN, config.TIME_COLUMN]].copy()
    out["top_reasons"] = reasons

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_names], index=df.index)
    return pd.concat([out, shap_df], axis=1)


def main():
    model, df = load_model_and_data()

    # Explain predictions for the most recent available month only — this is
    # what an authority actually needs ("why is Circle X flagged right now?").
    latest_month = df[config.TIME_COLUMN].max()
    latest_df = df[df[config.TIME_COLUMN] == latest_month].reset_index(drop=True)
    X_latest = latest_df[config.FEATURE_COLUMNS]

    print(f"Computing SHAP values for {len(X_latest)} circles in {latest_month.strftime('%Y-%m')}...")
    shap_values = compute_shap_values(model, X_latest)

    global_df = global_importance(shap_values, config.FEATURE_COLUMNS)
    global_df.to_csv(config.SHAP_GLOBAL_PATH, index=False)
    print(f"\nTop 5 globally important features:\n{global_df.head(5).to_string(index=False)}")
    print(f"\nSaved global SHAP importance to {config.SHAP_GLOBAL_PATH}")

    local_df = local_explanations(latest_df, shap_values, config.FEATURE_COLUMNS)
    local_df.to_csv(config.SHAP_LOCAL_PATH, index=False)
    print(f"Saved per-circle SHAP explanations to {config.SHAP_LOCAL_PATH}")


if __name__ == "__main__":
    main()

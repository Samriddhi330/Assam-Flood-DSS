"""
pipeline_run.py
----------------
Runs the complete Assam Flood DSS pipeline end to end:

    1. generate_geo_reference.py  (only if the reference CSVs are missing)
    2. train_model.py             (only if a saved model is missing)
    3. explain_shap.py
    4. predict_impact.py
    5. optimize_relief.py

This is the single command needed to go from the raw modelling dataset to a
final relief allocation plan.

Run:
    python src/pipeline_run.py

Use --retrain to force Stage 1 to retrain even if a saved model exists:
    python src/pipeline_run.py --retrain
"""

import argparse
import os

import config
import explain_shap
import generate_geo_reference
import optimize_relief
import predict_impact
import train_model


def main():
    parser = argparse.ArgumentParser(description="Run the full Assam Flood DSS pipeline.")
    parser.add_argument("--retrain", action="store_true", help="Force Stage 1 model retraining.")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 0 — Geographic reference data")
    print("=" * 70)
    if os.path.exists(config.CENTROIDS_PATH) and os.path.exists(config.BASES_PATH):
        print("Reference data already present, skipping generation.")
    else:
        generate_geo_reference.main()

    print("\n" + "=" * 70)
    print("STEP 1 — Train / load predictive model (Stage 1)")
    print("=" * 70)
    if args.retrain or not os.path.exists(config.MODEL_PATH):
        train_model.main()
    else:
        print(f"Using existing model at {config.MODEL_PATH} (pass --retrain to rebuild).")

    print("\n" + "=" * 70)
    print("STEP 2 — Explainable AI (SHAP)")
    print("=" * 70)
    explain_shap.main()

    print("\n" + "=" * 70)
    print("STEP 3 — Score circles & estimate relief demand")
    print("=" * 70)
    predict_impact.main()

    print("\n" + "=" * 70)
    print("STEP 4 — Optimise relief allocation (Stage 2)")
    print("=" * 70)
    optimize_relief.main()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Predictions:        {config.PREDICTIONS_PATH}")
    print(f"SHAP explanations:  {config.SHAP_LOCAL_PATH}")
    print(f"Relief allocation:  {config.ALLOCATION_PATH}")
    print(f"Allocation summary: {config.ALLOCATION_SUMMARY_PATH}")
    print("\nRun the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

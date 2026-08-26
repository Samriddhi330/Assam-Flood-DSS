"""
train_model.py
---------------
Stage 1 — Predictive Engine.

This is the script version of notebooks/Modeling_final (2).ipynb: it loads
the baseline modelling dataset, performs the temporal (never random)
train/test split described in docs/Modeling_Findings.md, trains the four
baseline models (Logistic Regression, Random Forest, XGBoost, CatBoost),
compares them on Accuracy / Precision / Recall / F1 / ROC-AUC, and saves the
best model (selected on F1 Score, per section 20 of the findings doc) plus
its chosen classification threshold to models/.

Run:
    python src/train_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

import config


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATASET_PATH)
    df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], format="%Y_%m")
    df = df.sort_values([config.TIME_COLUMN, config.ID_COLUMN]).reset_index(drop=True)
    return df


def temporal_train_test_split(df: pd.DataFrame):
    """Split chronologically: earliest months -> train, latest months -> test."""
    unique_months = sorted(df[config.TIME_COLUMN].dt.to_period("M").unique())
    split_position = int(len(unique_months) * config.TRAIN_MONTH_FRACTION)
    train_months = unique_months[:split_position]
    test_months = unique_months[split_position:]

    train_df = df[df[config.TIME_COLUMN].dt.to_period("M").isin(train_months)].copy()
    test_df = df[df[config.TIME_COLUMN].dt.to_period("M").isin(test_months)].copy()
    return train_df, test_df


def build_models(scale_pos_weight: float):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=config.RANDOM_STATE,
            verbose=False,
        ),
    }


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def best_threshold_for_f1(y_true, y_prob) -> float:
    """Scan thresholds and return the one maximising F1 (see notebook section
    'Threshold Analysis'). Falls back to 0.5 if nothing beats it."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.05):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main():
    df = load_dataset()
    train_df, test_df = temporal_train_test_split(df)

    X_train = train_df[config.FEATURE_COLUMNS].copy()
    X_test = test_df[config.FEATURE_COLUMNS].copy()
    y_train = train_df[config.TARGET_COLUMN].astype(int).copy()
    y_test = test_df[config.TARGET_COLUMN].astype(int).copy()

    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print(f"Training positive rate: {y_train.mean():.4f} | Testing positive rate: {y_test.mean():.4f}")

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    models = build_models(scale_pos_weight)

    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)
        fitted_models[name] = model
        print(f"{name:22s} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "Model"))

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    results_df.to_csv(config.OUTPUT_DIR + "/baseline_model_results.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    best_prob = best_model.predict_proba(X_test)[:, 1]
    best_threshold = best_threshold_for_f1(y_test, best_prob)

    print(f"\nBest model (by F1 Score): {best_model_name}")
    print(f"Chosen classification threshold: {best_threshold:.2f}")

    joblib.dump(best_model, config.MODEL_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "threshold": best_threshold,
        "features": config.FEATURE_COLUMNS,
        "test_metrics": results_df.iloc[0].drop("Model").to_dict(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {config.MODEL_PATH}")
    print(f"Saved metadata to {config.MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()

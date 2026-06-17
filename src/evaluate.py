"""
TalentMatch AI — Model Evaluation Module
===========================================
Loads the trained model and held-out test set, computes:
  - Classification report (precision, recall, f1 per class)
  - Confusion matrix
  - ROC curve + AUC
  - Precision-Recall curve + average precision
  - Threshold sensitivity analysis (since recall showed val drift in training)

All metrics are computed strictly on data the model never saw —
not even indirectly via early stopping (that used the val set).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras # type: ignore
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

ROOT_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"


def load_test_artifacts():
    """Load trained model and held-out test set."""
    model = keras.models.load_model(MODEL_DIR / "talentmatch_ann_v1.keras")
    X_test = np.load(MODEL_DIR / "X_test_v1.npy")
    y_test = np.load(MODEL_DIR / "y_test_v1.npy")

    with open(MODEL_DIR / "training_config_v1.json") as f:
        config = json.load(f)

    print(f"  Loaded model: {model.name}")
    print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    return model, X_test, y_test, config


def get_predictions(model, X_test, threshold: float = 0.5):
    """Get predicted probabilities and binary predictions at given threshold."""
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_proba >= threshold).astype(int)
    return y_proba, y_pred


def compute_classification_metrics(y_test, y_pred) -> dict:
    """Return classification report as a dict, plus confusion matrix."""
    y_test_int = y_test.astype(int)
    report = classification_report(y_test, y_pred, target_names=["Poor Fit", "Good Fit"], output_dict=True)
    cm = confusion_matrix(y_test_int, y_pred)
    return report, cm


def compute_roc(y_test, y_proba):
    """ROC curve points and AUC."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def compute_pr_curve(y_test, y_proba):
    """Precision-Recall curve points and average precision."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    return precision, recall, thresholds, avg_precision


def threshold_sensitivity(y_test, y_proba, thresholds=None) -> pd.DataFrame:
    """
    Sweep decision thresholds and report precision/recall/f1 at each.

    Default 0.5 isn't always optimal — given the val recall drift seen
    in training, this lets us check whether a lower threshold recovers
    recall without destroying precision.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)

    y_test_int = y_test.astype(int)

    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        report = classification_report(y_test_int, y_pred_t, output_dict=True, zero_division=0)
        rows.append({
            "threshold": round(t, 2),
            "precision_good_fit": report["1"]["precision"],
            "recall_good_fit":    report["1"]["recall"],
            "f1_good_fit":        report["1"]["f1-score"],
            "accuracy":           report["accuracy"],
        })

    return pd.DataFrame(rows)


def save_evaluation_report(report, cm, roc_auc, avg_precision, config):
    """Save all evaluation metrics to models/evaluation_report_v1.json."""
    output = {
        "model_version":      config["model_version"],
        "test_samples":       config["test_samples"],
        "classification_report": report,
        "confusion_matrix":   cm.tolist(),
        "roc_auc":            float(roc_auc),
        "average_precision":  float(avg_precision),
    }

    path = MODEL_DIR / "evaluation_report_v1.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {path}")


def main():
    print("TalentMatch AI — Model Evaluation")
    print("=" * 38)

    print("\n[1/4] Loading model and test set...")
    model, X_test, y_test, config = load_test_artifacts()

    print("\n[2/4] Generating predictions (threshold=0.5)...")
    y_proba, y_pred = get_predictions(model, X_test, threshold=0.5)

    print("\n[3/4] Computing metrics...")
    report, cm = compute_classification_metrics(y_test, y_pred)
    fpr, tpr, _, roc_auc = compute_roc(y_test, y_proba)
    precision, recall, _, avg_precision = compute_pr_curve(y_test, y_proba)
    threshold_df = threshold_sensitivity(y_test, y_proba)

    print("\n── Test Set Results (threshold=0.5) ─────────────")
    print(f"   Accuracy        : {report['accuracy']:.4f}")
    print(f"   Precision (Good): {report['Good Fit']['precision']:.4f}")
    print(f"   Recall (Good)   : {report['Good Fit']['recall']:.4f}")
    print(f"   F1 (Good)       : {report['Good Fit']['f1-score']:.4f}")
    print(f"   ROC AUC         : {roc_auc:.4f}")
    print(f"   Avg Precision   : {avg_precision:.4f}")
    print("\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:>4}  FP={cm[0][1]:>4}]")
    print(f"    [FN={cm[1][0]:>4}  TP={cm[1][1]:>4}]]")

    print("\n── Threshold Sensitivity ─────────────────────────")
    print(threshold_df.to_string(index=False))

    print("\n[4/4] Saving evaluation report...")
    save_evaluation_report(report, cm, roc_auc, avg_precision, config)

    # Save threshold sweep + curve data for the notebook
    threshold_df.to_csv(MODEL_DIR / "threshold_sensitivity_v1.csv", index=False)
    np.savez(
        MODEL_DIR / "roc_pr_curves_v1.npz",
        fpr=fpr, tpr=tpr,
        precision=precision, recall=recall,
        y_test=y_test, y_proba=y_proba
    )
    print(f"  Saved → models/threshold_sensitivity_v1.csv")
    print(f"  Saved → models/roc_pr_curves_v1.npz")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
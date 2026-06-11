"""
TalentMatch AI — Feature Pipeline Runner
=========================================
Runs preprocessing + feature engineering on the full dataset.
Saves:
  - models/feature_engineer_v1.joblib   (fitted scaler)
  - data/processed/model_training_dataset.csv  (overwritten with pipeline output)

Run once after data generation. Re-run if raw data changes.
"""

import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

import sys
sys.path.append(str(ROOT_DIR))

from src.preprocessing import load_candidates, load_jobs, load_applications, build_raw_pairs
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS

RAW_DIR   = ROOT_DIR / "data" / "raw"
PROC_DIR  = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

def main():
    print("TalentMatch AI — Feature Pipeline Runner")
    print("=" * 45)

    print("\n[1/4] Loading raw data...")
    candidates   = load_candidates(RAW_DIR / "candidates.csv")
    jobs         = load_jobs(RAW_DIR / "jobs.csv")
    applications = load_applications(RAW_DIR / "applications.csv")
    print(f"      Candidates: {len(candidates)} | Jobs: {len(jobs)} | Applications: {len(applications)}")

    print("\n[2/4] Building raw pairs...")
    pairs = build_raw_pairs(applications, candidates, jobs)
    print(f"      Pairs shape: {pairs.shape}")

    print("\n[3/4] Running feature engineering (fit + transform)...")
    fe = FeatureEngineer()
    X, y_label, y_score = fe.fit_transform(pairs)

    print("\n[4/4] Saving artifacts...")
    fe.save(MODEL_DIR / "feature_engineer_v1.joblib")

    # Save processed dataset
    out = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    out.insert(0, "application_id", applications["application_id"].values)
    out["fit_score"] = y_score
    out["fit_label"] = y_label
    out.to_csv(PROC_DIR / "model_training_dataset.csv", index=False)
    print(f"      Saved → {PROC_DIR / 'model_training_dataset.csv'}")

    print("\n── Pipeline Summary ─────────────────────────────")
    print(f"   Features   : {len(FEATURE_COLUMNS)}")
    print(f"   Samples    : {X.shape[0]}")
    print(f"   Good Fit   : {y_label.sum()} ({y_label.mean()*100:.1f}%)")
    print(f"   Poor Fit   : {(1-y_label).sum()} ({(1-y_label).mean()*100:.1f}%)")
    print("─" * 50)
    print("\nFeature pipeline complete.")

if __name__ == "__main__":
    main()
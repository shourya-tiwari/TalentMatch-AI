"""
TalentMatch AI — Prediction & Batch Scoring Module
=====================================================
Loads the trained model + fitted FeatureEngineer and exposes:
  - predict_single(candidate, job)      → one candidate vs one job
  - predict_batch(candidates_df, job)   → many candidates vs one job, ranked
  - skill_gap_report(candidate, job)    → missing skills, matched skills

Decision threshold: 0.40 (see docs/feature_schema.md / README "Key Model
Decisions" — chosen over 0.50 because false negatives (missing a good
candidate) are costlier than false positives in a recruiting context).

This module is the inference-time entry point. It does NOT retrain
anything — it loads what train.py and run_feature_pipeline.py produced.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras  # type: ignore

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.preprocessing import (
    load_candidates, load_jobs, build_raw_pairs,
    EDUCATION_ORDINAL, SENIORITY_BANDS
)
from src.feature_engineering import FeatureEngineer

MODEL_DIR = ROOT_DIR / "models"

DEFAULT_THRESHOLD = 0.40

# Guardrail: skill_match_score below this floor caps the verdict at "Poor Fit"
# regardless of model output. See README "Key Model Decisions" for rationale —
# the ANN is well-calibrated on aggregate test metrics (verified via bottom-decile
# analysis) but showed unreliable extrapolation on rare out-of-distribution
# feature combinations (very low skill match + maxed-out secondary signals like
# GPA, certifications, experience). This rule prevents that failure mode from
# reaching production output.
SKILL_MATCH_FLOOR = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loading (cached at module level — loaded once, reused)
# ─────────────────────────────────────────────────────────────────────────────

class TalentMatchPredictor:
    """
    Wraps the trained model and fitted FeatureEngineer.
    Instantiate once, reuse across many predictions (avoids reloading
    the model/scaler on every call — important for the Streamlit app
    in Phase 7).
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.model = keras.models.load_model(MODEL_DIR / "talentmatch_ann_v1.keras")
        self.fe = FeatureEngineer.load(MODEL_DIR / "feature_engineer_v1.joblib")

        with open(MODEL_DIR / "training_config_v1.json") as f:
            self.config = json.load(f)

        print(f"  [TalentMatchPredictor] Model loaded, threshold={self.threshold}")

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _build_pair_row(self, candidate: pd.Series, job: pd.Series) -> pd.DataFrame:
        """
        Build a single-row pairs DataFrame matching the structure
        build_raw_pairs() produces, so FeatureEngineer.transform() works
        identically to training time.
        """
        pair = {}
        for col in candidate.index:
            pair[f"c_{col}"] = candidate[col]
        for col in job.index:
            pair[f"j_{col}"] = job[col]

        pair["c_skills_set"] = set(str(candidate["skills"]).split("|"))
        pair["j_required_skills_set"] = set(str(job["required_skills"]).split("|"))
        pair["c_education_ordinal"] = EDUCATION_ORDINAL.get(candidate["education_level"], 0)
        pair["j_education_req_ordinal"] = EDUCATION_ORDINAL.get(job["education_requirement"], 2)

        return pd.DataFrame([pair])

    def score_pair(self, candidate: pd.Series, job: pd.Series) -> dict:
        """
        Score a single (candidate, job) pair.
        Returns fit_probability, fit_label, and skill diagnostics.

        Applies a skill_match_score floor as a production safeguard: candidates
        matching less than SKILL_MATCH_FLOOR of required skills are capped at
        "Poor Fit" regardless of model output, since the ANN showed unreliable
        extrapolation on this rare combination during evaluation (see README).
        """
        pair_row = self._build_pair_row(candidate, job)
        X = self.fe.transform(pair_row)

        fit_probability = float(self.model.predict(X, verbose=0).flatten()[0])

        candidate_skills = set(str(candidate["skills"]).split("|"))
        required_skills  = set(str(job["required_skills"]).split("|"))
        matched = candidate_skills & required_skills
        missing = required_skills - candidate_skills
        skill_match_score = len(matched) / len(required_skills) if required_skills else 0.0

        # ── Guardrail ────────────────────────────────────────────────────────────
        guardrail_applied = False
        if skill_match_score < SKILL_MATCH_FLOOR:
            fit_label = 0
            guardrail_applied = fit_probability >= self.threshold  # only flag if it would have flipped the verdict
        else:
            fit_label = int(fit_probability >= self.threshold)

        return {
            "candidate_id":         candidate.get("candidate_id", "N/A"),
            "candidate_name":       candidate.get("name", "N/A"),
            "job_id":               job.get("job_id", "N/A"),
            "fit_probability":      round(fit_probability, 4),
            "fit_percentage":       round(fit_probability * 100, 1),
            "fit_label":            fit_label,
            "fit_verdict":          "Good Fit" if fit_label == 1 else "Poor Fit",
            "confidence_band":      self._confidence_band(fit_probability, fit_label, guardrail_applied),
            "guardrail_applied":    guardrail_applied,
            "matched_skills":       sorted(matched),
            "missing_skills":       sorted(missing),
            "matched_skill_count":  len(matched),
            "missing_skill_count":  len(missing),
            "required_skill_count": len(required_skills),
        }

    @staticmethod
    def _confidence_band(probability: float, fit_label: int, guardrail_applied: bool) -> str:
        """
        Translate raw probability into a recruiter-friendly band.
        If the guardrail overrode the model's verdict, label it explicitly
        so recruiters understand why a high-probability candidate was capped.
        """
        if guardrail_applied:
            return "Skill Gap Override (Low Match)"

        if probability >= 0.75:
            return "Strong Fit"
        elif probability >= 0.55:
            return "Likely Fit"
        elif probability >= 0.40:
            return "Borderline Fit"
        elif probability >= 0.25:
            return "Likely Poor Fit"
        else:
            return "Strong Poor Fit"

    # ── Batch scoring ─────────────────────────────────────────────────────────

    def score_batch(self, candidates_df: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """
        Score every candidate in candidates_df against a single job.
        Returns a ranked DataFrame, sorted by fit_probability descending.

        This is the core "recruiter screens 200 resumes" use case.
        """
        results = []
        for _, candidate in candidates_df.iterrows():
            result = self.score_pair(candidate, job)
            results.append(result)

        ranked = pd.DataFrame(results)
        ranked = ranked.sort_values("fit_probability", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))

        return ranked

    # ── Skill gap report (recruiter-friendly text) ───────────────────────────

    def skill_gap_report(self, candidate: pd.Series, job: pd.Series) -> str:
        """
        Generate a human-readable skill gap summary for a single candidate.
        This is the 'recruiter-friendly insight' the original brief asked for.
        """
        result = self.score_pair(candidate, job)

        lines = [
            f"Candidate: {result['candidate_name']} ({result['candidate_id']})",
            f"Job: {result['job_id']}",
            f"Fit Score: {result['fit_percentage']}% — {result['confidence_band']}",
            f"Skills matched: {result['matched_skill_count']}/{result['required_skill_count']}",
        ]

        if result["missing_skills"]:
            lines.append(f"Missing skills ({result['missing_skill_count']}): " +
                         ", ".join(result["missing_skills"]))
        else:
            lines.append("No missing skills — candidate covers all requirements.")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Demo run: load a real candidate and job from the dataset,
    score them, and run a batch scoring example.
    """
    print("TalentMatch AI — Prediction Demo")
    print("=" * 36)

    candidates = load_candidates(ROOT_DIR / "data" / "raw" / "candidates.csv")
    jobs       = load_jobs(ROOT_DIR / "data" / "raw" / "jobs.csv")

    predictor = TalentMatchPredictor(threshold=DEFAULT_THRESHOLD)

    # ── Single prediction demo ───────────────────────────────────────────────
    sample_candidate = candidates.iloc[0]
    sample_job        = jobs.iloc[0]

    print("\n── Single Prediction ─────────────────────────────")
    print(predictor.skill_gap_report(sample_candidate, sample_job))

    # ── Batch scoring demo ───────────────────────────────────────────────────
    print("\n── Batch Scoring (top 20 candidates vs job) ──────")
    sample_pool = candidates.sample(n=50, random_state=42)
    ranked = predictor.score_batch(sample_pool, sample_job)

    display_cols = ["rank", "candidate_name", "fit_percentage", "fit_verdict",
                     "confidence_band", "matched_skill_count", "missing_skill_count"]
    print(ranked[display_cols].head(20).to_string(index=False))

    # Save demo output
    output_path = ROOT_DIR / "data" / "processed" / "sample_batch_ranking.csv"
    ranked.to_csv(output_path, index=False)
    print(f"\nSaved full ranking → {output_path}")


if __name__ == "__main__":
    main()
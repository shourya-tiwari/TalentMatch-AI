"""
TalentMatch AI — Feature Engineering Module
============================================
Responsibilities:
  - Compute all engineered features from raw candidate-job pairs
  - Fit MinMaxScaler on continuous features (training mode)
  - Apply fitted scaler (inference mode)
  - Save and load fitted scaler via joblib
  - Output a model-ready feature matrix (X) and target vectors (y)

Usage:
  Training:
    fe = FeatureEngineer()
    X, y_label, y_score = fe.fit_transform(pairs_df)
    fe.save(path)

  Inference:
    fe = FeatureEngineer.load(path)
    X = fe.transform(pairs_df)
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "metadata" / "skills_config.json"
MODELS_DIR  = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

EDUCATION_ORDINAL = {
    lvl: i for i, lvl in enumerate(CONFIG["education_levels"])
}
SENIORITY_BANDS   = CONFIG["seniority_experience_bands"]
ALL_SKILLS        = [s for pool in CONFIG["skill_categories"].values() for s in pool]
MASTER_POOL_SIZE  = len(ALL_SKILLS)

# ── Feature column definitions ────────────────────────────────────────────────
# Columns that will be scaled by MinMaxScaler
CONTINUOUS_FEATURES = [
    "skill_match_score",
    "experience_gap",
    "skill_coverage_ratio",
    "gpa_normalized",
    "projects_normalized",
    "internships_normalized",
    "hackathons_normalized",
    "research_normalized",
    "certification_gap",
]

# Columns that are already binary/ordinal — no scaling needed
PASSTHROUGH_FEATURES = [
    "education_match",
    "specialization_match",
    "seniority_match",
    "leadership_experience",
]

# Final ordered feature list fed to the model
FEATURE_COLUMNS = CONTINUOUS_FEATURES + PASSTHROUGH_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Raw Feature Computation (unscaled)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_raw_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute unscaled feature values from a raw pairs DataFrame.
    Expects c_ and j_ prefixed columns from preprocessing.build_raw_pairs().

    Returns a DataFrame with one column per feature in FEATURE_COLUMNS,
    plus 'fit_score' and 'fit_label' if present.
    """
    feats = pd.DataFrame(index=pairs.index)

    # ── Skill match score ─────────────────────────────────────────────────────
    def skill_match(row):
        c_skills   = row.get("c_skills_set", set())
        j_skills   = row.get("j_required_skills_set", set())
        if not j_skills:
            return 0.0
        return len(c_skills & j_skills) / len(j_skills)

    feats["skill_match_score"] = pairs.apply(skill_match, axis=1).round(4)

    # ── Experience gap (clipped) ──────────────────────────────────────────────
    feats["experience_gap"] = (
        pairs["c_years_experience"] - pairs["j_min_experience"]
    ).clip(-5, 10).round(2)

    # ── Skill coverage ratio ──────────────────────────────────────────────────
    feats["skill_coverage_ratio"] = (
        pairs["c_skills"].apply(lambda s: len(str(s).split("|"))) / MASTER_POOL_SIZE
    ).round(4)

    # ── GPA normalised ────────────────────────────────────────────────────────
    feats["gpa_normalized"] = (pairs["c_gpa"] / 4.0).clip(0, 1).round(4)

    # ── Projects normalised ───────────────────────────────────────────────────
    max_projects = pairs["c_projects_count"].max()
    feats["projects_normalized"] = (
        pairs["c_projects_count"] / max_projects if max_projects > 0 else 0.0
    ).round(4)

    # ── Internships normalised ────────────────────────────────────────────────
    max_internships = pairs["c_internships_count"].max()
    feats["internships_normalized"] = (
        pairs["c_internships_count"] / max_internships if max_internships > 0 else 0.0
    ).round(4)

    # ── Hackathons normalised ─────────────────────────────────────────────────
    max_hackathons = pairs["c_hackathons"].max()
    feats["hackathons_normalized"] = (
        pairs["c_hackathons"] / max_hackathons if max_hackathons > 0 else 0.0
    ).round(4)

    # ── Research papers normalised ────────────────────────────────────────────
    max_research = pairs["c_research_papers"].max()
    feats["research_normalized"] = (
        pairs["c_research_papers"] / max_research if max_research > 0 else 0.0
    ).round(4)

    # ── Certification gap (clipped 0–4) ───────────────────────────────────────
    feats["certification_gap"] = (
        pairs["c_certifications"] - pairs["j_required_certifications"]
    ).clip(0, 4)

    # ── Education match ───────────────────────────────────────────────────────
    feats["education_match"] = (
        pairs["c_education_ordinal"] >= pairs["j_education_req_ordinal"]
    ).astype(int)

    # ── Specialization match ──────────────────────────────────────────────────
    feats["specialization_match"] = (
        pairs["c_specialization"] == pairs["j_preferred_specialization"]
    ).astype(int)

    # ── Seniority match ───────────────────────────────────────────────────────
    def seniority_match(row):
        band = SENIORITY_BANDS.get(row["j_seniority_level"])
        if band is None:
            return 0
        return 1 if band[0] <= row["c_years_experience"] <= band[1] else 0

    feats["seniority_match"]       = pairs.apply(seniority_match, axis=1)
    feats["leadership_experience"] = pairs["c_leadership_experience"].astype(int)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer Class
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Stateful feature engineering pipeline.

    fit_transform() — use on training data (fits scaler)
    transform()     — use on validation/test/inference data (applies fitted scaler)
    save() / load() — persist the fitted scaler to disk
    """

    def __init__(self):
        self.scaler: Optional[MinMaxScaler] = None
        self._is_fitted: bool = False

    # ── Training path ─────────────────────────────────────────────────────────
    def fit_transform(
        self, pairs: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute features, fit scaler on continuous features,
        return (X, y_label, y_score).

        X shape: (n_samples, len(FEATURE_COLUMNS))
        """
        raw = _compute_raw_features(pairs)

        self.scaler = MinMaxScaler()
        raw[CONTINUOUS_FEATURES] = self.scaler.fit_transform(
            raw[CONTINUOUS_FEATURES]
        )
        self._is_fitted = True

        X       = raw[FEATURE_COLUMNS].values.astype(np.float32)
        y_label = pairs["fit_label"].values.astype(np.int32)
        y_score = pairs["fit_score"].values.astype(np.float32)

        print(f"  [FeatureEngineer] fit_transform complete.")
        print(f"  X shape : {X.shape}")
        print(f"  Features: {FEATURE_COLUMNS}")

        return X, y_label, y_score

    # ── Inference path ────────────────────────────────────────────────────────
    def transform(self, pairs: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted scaler to new data.
        Raises RuntimeError if called before fit_transform or load.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer is not fitted. "
                "Call fit_transform() on training data first, or load a saved instance."
            )

        raw = _compute_raw_features(pairs)
        raw[CONTINUOUS_FEATURES] = self.scaler.transform(raw[CONTINUOUS_FEATURES])

        return raw[FEATURE_COLUMNS].values.astype(np.float32)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted FeatureEngineer (scaler state) to disk."""
        path = Path(path)
        joblib.dump(self, path)
        print(f"  [FeatureEngineer] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureEngineer":
        """Load a previously saved FeatureEngineer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved FeatureEngineer found at: {path}")
        instance = joblib.load(path)
        print(f"  [FeatureEngineer] Loaded ← {path}")
        return instance

    # ── Metadata ──────────────────────────────────────────────────────────────
    @property
    def feature_names(self) -> list:
        return FEATURE_COLUMNS

    @property
    def n_features(self) -> int:
        return len(FEATURE_COLUMNS)
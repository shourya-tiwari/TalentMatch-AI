"""
TalentMatch AI — Preprocessing Module
======================================
Responsibilities:
  - Load raw CSVs with schema validation
  - Parse pipe-separated skill fields into Python sets
  - Ordinal-encode education level and seniority
  - Validate required columns are present
  - Expose a clean API for feature_engineering.py to consume

This module contains NO model logic and NO scaling.
Scaling lives in feature_engineering.py so it can be fitted
once on training data and reused at inference time.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "metadata" / "skills_config.json"

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

EDUCATION_LEVELS  = CONFIG["education_levels"]
EDUCATION_ORDINAL = {lvl: i for i, lvl in enumerate(EDUCATION_LEVELS)}
SENIORITY_BANDS   = CONFIG["seniority_experience_bands"]

# ── Required columns per schema ───────────────────────────────────────────────
CANDIDATE_REQUIRED_COLS = [
    "candidate_id", "specialization", "years_experience", "education_level",
    "skills", "certifications", "internships_count", "projects_count",
    "hackathons", "research_papers", "leadership_experience", "gpa"
]

JOB_REQUIRED_COLS = [
    "job_id", "preferred_specialization", "seniority_level",
    "min_experience", "education_requirement",
    "required_skills", "required_certifications"
]


# ─────────────────────────────────────────────────────────────────────────────
# Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required_cols: list, name: str) -> None:
    """
    Raise ValueError if any required column is missing from df.
    Called at load time to catch schema drift early.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load candidates CSV, validate schema, parse skills field.

    Returns a DataFrame with:
      - All original columns intact
      - 'skills_set' column: Python set of skills (for feature engineering)
      - 'education_ordinal': integer encoding of education_level
    """
    df = pd.read_csv(path)
    validate_schema(df, CANDIDATE_REQUIRED_COLS, "candidates")

    # Parse pipe-separated skills into sets
    df["skills_set"] = df["skills"].apply(
        lambda s: set(str(s).split("|")) if pd.notna(s) else set()
    )

    # Ordinal encode education
    df["education_ordinal"] = df["education_level"].map(EDUCATION_ORDINAL)

    # Guard: fill any unmapped education values with 0
    unmapped = df["education_ordinal"].isna().sum()
    if unmapped > 0:
        print(f"  [WARN] {unmapped} candidates have unrecognised education_level — defaulting to 0")
        df["education_ordinal"] = df["education_ordinal"].fillna(0).astype(int)
    else:
        df["education_ordinal"] = df["education_ordinal"].astype(int)

    # Type enforcement
    numeric_cols = [
        "years_experience", "certifications", "internships_count",
        "projects_count", "hackathons", "research_papers",
        "leadership_experience", "gpa"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def load_jobs(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load jobs CSV, validate schema, parse required_skills field.

    Returns a DataFrame with:
      - All original columns intact
      - 'required_skills_set': Python set of required skills
      - 'education_req_ordinal': integer encoding of education_requirement
    """
    df = pd.read_csv(path)
    validate_schema(df, JOB_REQUIRED_COLS, "jobs")

    df["required_skills_set"] = df["required_skills"].apply(
        lambda s: set(str(s).split("|")) if pd.notna(s) else set()
    )

    df["education_req_ordinal"] = df["education_requirement"].map(EDUCATION_ORDINAL)

    unmapped = df["education_req_ordinal"].isna().sum()
    if unmapped > 0:
        print(f"  [WARN] {unmapped} jobs have unrecognised education_requirement — defaulting to 2")
        df["education_req_ordinal"] = df["education_req_ordinal"].fillna(2).astype(int)
    else:
        df["education_req_ordinal"] = df["education_req_ordinal"].astype(int)

    numeric_cols = ["min_experience", "required_certifications"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def load_applications(path: Union[str, Path]) -> pd.DataFrame:
    """Load applications CSV with basic type enforcement."""
    df = pd.read_csv(path)
    required = ["application_id", "candidate_id", "job_id", "fit_score", "fit_label"]
    validate_schema(df, required, "applications")
    df["fit_label"] = df["fit_label"].astype(int)
    df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce").fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Seniority Helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_seniority_match(years_experience: float, seniority_level: str) -> int:
    """
    Return 1 if the candidate's experience falls within the job's
    seniority band, 0 otherwise.
    """
    band = SENIORITY_BANDS.get(seniority_level)
    if band is None:
        return 0
    return 1 if band[0] <= years_experience <= band[1] else 0


# ─────────────────────────────────────────────────────────────────────────────
# Pair Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_pairs(
    applications: pd.DataFrame,
    candidates: pd.DataFrame,
    jobs: pd.DataFrame
) -> pd.DataFrame:
    """
    Join applications with candidate and job data into a flat pair DataFrame.
    This is the input to feature_engineering.build_features().

    Returns one row per application with all candidate and job fields
    accessible via c_ and j_ prefixes.
    """
    pairs = applications.merge(
        candidates.add_prefix("c_"),
        left_on="candidate_id",
        right_on="c_candidate_id",
        how="left"
    ).merge(
        jobs.add_prefix("j_"),
        left_on="job_id",
        right_on="j_job_id",
        how="left"
    )

    missing_candidates = pairs["c_candidate_id"].isna().sum()
    missing_jobs       = pairs["j_job_id"].isna().sum()

    if missing_candidates > 0:
        print(f"  [WARN] {missing_candidates} applications have no matching candidate record")
    if missing_jobs > 0:
        print(f"  [WARN] {missing_jobs} applications have no matching job record")

    return pairs
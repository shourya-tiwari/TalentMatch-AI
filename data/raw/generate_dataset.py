"""
TalentMatch AI — Synthetic Dataset Generator V1.1
==================================================
Generates:
  - candidates.csv         : 1000 structured candidate profiles
  - jobs.csv               : 100 structured job descriptions
  - applications.csv       : application pairs with skill diagnostics,
                             fit_score, and fit_label
  - model_training_dataset.csv : fully engineered, model-ready feature matrix

Design principles:
  - Fully seeded for reproducibility
  - Skill pools loaded from metadata/skills_config.json
  - fit_score is a weighted composite (0-100)
  - fit_label derived from fit_score threshold (default: 50)
  - 5-10% synthetic label noise to prevent deterministic overfitting
  - Normalised features output directly in model_training_dataset.csv
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "metadata" / "skills_config.json"
RAW_DIR     = ROOT_DIR / "data" / "raw"
PROC_DIR    = ROOT_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

SKILL_CATEGORIES    = CONFIG["skill_categories"]
ALL_SKILLS          = [s for pool in SKILL_CATEGORIES.values() for s in pool]
SKILL_SET           = set(ALL_SKILLS)
MASTER_POOL_SIZE    = len(ALL_SKILLS)

SPECIALIZATIONS     = CONFIG["specializations"]
JOB_FAMILIES        = CONFIG["job_families"]
JOB_TITLES          = CONFIG["job_titles"]
EDUCATION_LEVELS    = CONFIG["education_levels"]
EDUCATION_ORDINAL   = {lvl: i for i, lvl in enumerate(EDUCATION_LEVELS)}
SENIORITY_LEVELS    = CONFIG["seniority_levels"]
SENIORITY_BANDS     = CONFIG["seniority_experience_bands"]
FIT_WEIGHTS         = CONFIG["fit_score_weights"]
FIT_THRESHOLD       = CONFIG["fit_label_threshold"]
NOISE_MIN           = CONFIG["noise_rate_min"]
NOISE_MAX           = CONFIG["noise_rate_max"]
N_CANDIDATES        = CONFIG["dataset_sizes"]["candidates"]
N_JOBS              = CONFIG["dataset_sizes"]["jobs"]
APPS_PER_JOB        = CONFIG["dataset_sizes"]["applications_per_job"]

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Name pools ────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun",
    "Neha", "Karan", "Pooja", "Rahul", "Divya", "Amit", "Shruti",
    "Deepak", "Meera", "Suresh", "Kavya", "Nikhil", "Isha",
    "James", "Sarah", "Michael", "Emily", "David", "Jessica",
    "Daniel", "Ashley", "Chris", "Amanda", "Matthew", "Stephanie",
    "Wei", "Lin", "Zhang", "Fatima", "Omar", "Aisha", "Carlos", "Maria",
    "Lena", "Kai", "Yuki", "Hassan", "Zara", "Ivan", "Sofia", "Lucas", "Nina"
]
LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Mehta", "Gupta", "Joshi",
    "Verma", "Nair", "Reddy", "Smith", "Johnson", "Williams", "Brown",
    "Jones", "Garcia", "Martinez", "Davis", "Wilson", "Anderson",
    "Chen", "Wang", "Liu", "Ahmed", "Hassan", "Ali", "Rodriguez",
    "Müller", "Tanaka", "Ivanov", "Okonkwo", "Mensah", "Castillo"
]

# ── Specialization → relevant skill categories ────────────────────────────────
SPEC_SKILL_AFFINITY = {
    "Machine Learning":       ["Machine Learning", "Programming", "Data"],
    "Data Science":           ["Data", "Machine Learning", "Programming"],
    "Data Engineering":       ["Data", "Cloud", "Programming"],
    "Business Intelligence":  ["BI", "Data", "Soft Skills"],
    "Cloud Engineering":      ["Cloud", "Programming", "Data"],
    "NLP":                    ["Machine Learning", "Programming", "Data"],
    "Computer Vision":        ["Machine Learning", "Programming", "Data"],
    "MLOps":                  ["Cloud", "Machine Learning", "Programming"],
    "Quantitative Finance":   ["Data", "Programming", "Machine Learning"],
    "Software Engineering":   ["Programming", "Cloud", "Data"],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_skills_for_specialization(specialization: str, n: int) -> list:
    """
    Sample n skills biased toward the candidate's specialization.
    70% drawn from affinity categories, 30% from remaining pool.
    """
    affinity_cats = SPEC_SKILL_AFFINITY.get(specialization, list(SKILL_CATEGORIES.keys()))
    affinity_skills = [
        s for cat in affinity_cats
        for s in SKILL_CATEGORIES.get(cat, [])
    ]
    other_skills = [s for s in ALL_SKILLS if s not in affinity_skills]

    n_affinity = min(int(n * 0.7), len(affinity_skills))
    n_other    = min(n - n_affinity, len(other_skills))

    selected = (
        random.sample(affinity_skills, n_affinity) +
        random.sample(other_skills, n_other)
    )
    random.shuffle(selected)
    return selected[:n]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Candidate Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(n: int) -> pd.DataFrame:
    """Generate n candidate profiles with student-oriented features."""

    edu_experience_ceiling = {
        "High School": 10.0,
        "Associate's": 8.0,
        "Bachelor's":  8.0,
        "Master's":    6.0,
        "PhD":         5.0,
    }

    records = []
    for i in range(n):
        candidate_id   = f"C{i+1:04d}"
        name           = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        specialization = random.choice(SPECIALIZATIONS)

        edu_level = random.choices(
            EDUCATION_LEVELS,
            weights=[0.04, 0.07, 0.50, 0.30, 0.09],
            k=1
        )[0]

        exp_ceil        = edu_experience_ceiling[edu_level]
        years_experience = round(random.uniform(0.0, exp_ceil), 1)

        num_skills = random.randint(5, 18)
        skills     = sample_skills_for_specialization(specialization, num_skills)

        certifications = random.choices(
            [0, 1, 2, 3, 4],
            weights=[0.25, 0.35, 0.22, 0.12, 0.06]
        )[0]

        internships_count = random.choices(
            [0, 1, 2, 3, 4],
            weights=[0.20, 0.35, 0.25, 0.12, 0.08]
        )[0]

        projects_count = random.choices(
            list(range(9)),
            weights=[0.04, 0.09, 0.15, 0.20, 0.20, 0.15, 0.09, 0.05, 0.03]
        )[0]

        hackathons = random.choices(
            [0, 1, 2, 3, 4, 5],
            weights=[0.25, 0.30, 0.22, 0.12, 0.07, 0.04]
        )[0]

        research_papers = random.choices(
            [0, 1, 2, 3],
            weights=[0.60, 0.25, 0.10, 0.05]
        )[0]

        leadership_experience = random.choices([0, 1], weights=[0.55, 0.45])[0]

        gpa = round(random.uniform(2.0, 4.0), 2)

        records.append({
            "candidate_id":         candidate_id,
            "name":                 name,
            "specialization":       specialization,
            "years_experience":     years_experience,
            "education_level":      edu_level,
            "skills":               "|".join(skills),
            "certifications":       certifications,
            "internships_count":    internships_count,
            "projects_count":       projects_count,
            "hackathons":           hackathons,
            "research_papers":      research_papers,
            "leadership_experience": leadership_experience,
            "gpa":                  gpa,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Job Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_jobs(n: int) -> pd.DataFrame:
    """Generate n job descriptions with family and specialization fields."""

    edu_weights_by_seniority = {
        "Junior": [0.05, 0.10, 0.60, 0.20, 0.05],
        "Mid":    [0.02, 0.05, 0.45, 0.40, 0.08],
        "Senior": [0.01, 0.02, 0.25, 0.50, 0.22],
    }

    # Map title → natural job family
    title_family_map = {
        "ML Engineer":                "Engineering",
        "Data Scientist":             "Research",
        "Data Analyst":               "Analytics",
        "AI Research Engineer":       "Research",
        "MLOps Engineer":             "Engineering",
        "Business Intelligence Analyst": "Analytics",
        "Data Engineer":              "Engineering",
        "Computer Vision Engineer":   "Research",
        "NLP Engineer":               "Research",
        "Quantitative Analyst":       "Finance",
        "AI Product Analyst":         "Analytics",
        "Research Scientist":         "Research",
    }

    title_specialization_map = {
        "ML Engineer":                "Machine Learning",
        "Data Scientist":             "Data Science",
        "Data Analyst":               "Data Science",
        "AI Research Engineer":       "Machine Learning",
        "MLOps Engineer":             "MLOps",
        "Business Intelligence Analyst": "Business Intelligence",
        "Data Engineer":              "Data Engineering",
        "Computer Vision Engineer":   "Computer Vision",
        "NLP Engineer":               "NLP",
        "Quantitative Analyst":       "Quantitative Finance",
        "AI Product Analyst":         "Data Science",
        "Research Scientist":         "Machine Learning",
    }

    records = []
    for i in range(n):
        job_id    = f"J{i+1:03d}"
        title     = random.choice(JOB_TITLES)
        seniority = random.choice(SENIORITY_LEVELS)

        job_family             = title_family_map.get(title, random.choice(JOB_FAMILIES))
        preferred_specialization = title_specialization_map.get(title, random.choice(SPECIALIZATIONS))

        exp_min, exp_max = SENIORITY_BANDS[seniority]
        min_experience   = round(random.uniform(exp_min, exp_max), 1)

        education_requirement = random.choices(
            EDUCATION_LEVELS,
            weights=edu_weights_by_seniority[seniority],
            k=1
        )[0]

        # Required skills biased toward job's preferred specialization
        num_required   = random.randint(4, 12)
        required_skills = sample_skills_for_specialization(preferred_specialization, num_required)

        required_certifications = random.choices(
            [0, 1, 2],
            weights=[0.50, 0.35, 0.15]
        )[0]

        records.append({
            "job_id":                   job_id,
            "title":                    title,
            "job_family":               job_family,
            "preferred_specialization": preferred_specialization,
            "seniority_level":          seniority,
            "min_experience":           min_experience,
            "education_requirement":    education_requirement,
            "required_skills":          "|".join(required_skills),
            "required_certifications":  required_certifications,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Fit Score Engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_fit_score(candidate: pd.Series, job: pd.Series,
                      max_projects: int, max_internships: int,
                      max_hackathons: int, max_research: int) -> dict:
    """
    Compute continuous fit_score (0–100) as a weighted composite.

    Returns a dict with the score and intermediate diagnostics
    (matched_skill_count, missing_skill_count, required_skill_count).
    """
    candidate_skills = set(candidate["skills"].split("|"))
    required_skills  = set(job["required_skills"].split("|"))

    matched  = candidate_skills & required_skills
    missing  = required_skills - candidate_skills

    required_skill_count = len(required_skills)
    matched_skill_count  = len(matched)
    missing_skill_count  = len(missing)

    # ── Component scores (each normalised to its max weight * 100) ─────────
    skill_match_score = matched_skill_count / required_skill_count if required_skill_count else 0.0

    exp_gap = candidate["years_experience"] - job["min_experience"]
    if exp_gap >= 0:
        experience_score = 1.0
    elif exp_gap >= -1.0:
        experience_score = 0.5 + (exp_gap + 1.0) * 0.5   # linear ramp 0.5–1.0
    elif exp_gap >= -3.0:
        experience_score = clamp((exp_gap + 3.0) / 2.0 * 0.5, 0.0, 0.5)
    else:
        experience_score = 0.0

    candidate_edu = EDUCATION_ORDINAL.get(candidate["education_level"], 0)
    required_edu  = EDUCATION_ORDINAL.get(job["education_requirement"], 0)
    edu_gap       = candidate_edu - required_edu
    if edu_gap >= 0:
        education_score = 1.0
    elif edu_gap == -1:
        education_score = 0.5
    else:
        education_score = 0.0

    gpa_score   = clamp((candidate["gpa"] - 2.0) / 2.0, 0.0, 1.0)

    cert_score  = clamp(
        candidate["certifications"] / max(job["required_certifications"], 1),
        0.0, 1.0
    ) if job["required_certifications"] > 0 else (
        min(candidate["certifications"] / 2.0, 1.0)
    )

    proj_score  = candidate["projects_count"] / max_projects if max_projects > 0 else 0.0
    intern_score = candidate["internships_count"] / max_internships if max_internships > 0 else 0.0

    spec_match  = 1.0 if candidate["specialization"] == job["preferred_specialization"] else 0.0

    # ── Weighted composite ────────────────────────────────────────────────────
    w = FIT_WEIGHTS
    raw_score = (
        w["skill_match"]          * skill_match_score  +
        w["experience_adequacy"]  * experience_score   +
        w["education_match"]      * education_score    +
        w["gpa"]                  * gpa_score          +
        w["certifications"]       * cert_score         +
        w["projects"]             * proj_score         +
        w["internships"]          * intern_score       +
        w["specialization_match"] * spec_match
    )

    fit_score = round(raw_score * 100, 2)

    return {
        "required_skill_count": required_skill_count,
        "matched_skill_count":  matched_skill_count,
        "missing_skill_count":  missing_skill_count,
        "fit_score":            fit_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Application Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_applications(candidates: pd.DataFrame,
                           jobs: pd.DataFrame,
                           apps_per_job: int) -> pd.DataFrame:
    """
    Generate application pairs.
    Each job samples apps_per_job candidates.
    fit_score computed per pair; fit_label derived from threshold.
    5–10% label noise injected post-scoring.
    """
    max_projects    = int(candidates["projects_count"].max())
    max_internships = int(candidates["internships_count"].max())
    max_hackathons  = int(candidates["hackathons"].max())
    max_research    = int(candidates["research_papers"].max())

    records     = []
    app_counter = 1

    for _, job in jobs.iterrows():
        seed_val = int(job["job_id"][1:])
        sampled  = candidates.sample(
            n=min(apps_per_job, len(candidates)),
            random_state=seed_val
        )

        for _, candidate in sampled.iterrows():
            diagnostics = compute_fit_score(
                candidate, job,
                max_projects, max_internships, max_hackathons, max_research
            )

            fit_label = 1 if diagnostics["fit_score"] >= FIT_THRESHOLD else 0

            records.append({
                "application_id":       f"A{app_counter:06d}",
                "candidate_id":         candidate["candidate_id"],
                "job_id":               job["job_id"],
                "required_skill_count": diagnostics["required_skill_count"],
                "matched_skill_count":  diagnostics["matched_skill_count"],
                "missing_skill_count":  diagnostics["missing_skill_count"],
                "fit_score":            diagnostics["fit_score"],
                "fit_label":            fit_label,
            })
            app_counter += 1

    df = pd.DataFrame(records)

    # ── Noise injection ───────────────────────────────────────────────────────
    noise_rate = random.uniform(NOISE_MIN, NOISE_MAX)
    n_noise    = int(len(df) * noise_rate)
    noise_idx  = df.sample(n=n_noise, random_state=SEED).index
    df.loc[noise_idx, "fit_label"] = 1 - df.loc[noise_idx, "fit_label"]

    print(f"      Noise injected: {n_noise} labels flipped ({noise_rate*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Feature Engineering (model_training_dataset.csv)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_training_dataset(candidates: pd.DataFrame,
                                  jobs: pd.DataFrame,
                                  applications: pd.DataFrame) -> pd.DataFrame:
    """
    Join application pairs with candidate and job data.
    Engineer and normalise all features into a model-ready matrix.
    """
    df = applications.merge(
        candidates.add_prefix("c_"), left_on="candidate_id", right_on="c_candidate_id"
    ).merge(
        jobs.add_prefix("j_"), left_on="job_id", right_on="j_job_id"
    )

    # ── Normalisation denominators ────────────────────────────────────────────
    max_projects    = candidates["projects_count"].max()
    max_internships = candidates["internships_count"].max()
    max_hackathons  = candidates["hackathons"].max()
    max_research    = candidates["research_papers"].max()

    # ── Feature construction ──────────────────────────────────────────────────
    out = pd.DataFrame()
    out["application_id"] = df["application_id"]
    out["candidate_id"]   = df["candidate_id"]
    out["job_id"]         = df["job_id"]

    out["skill_match_score"] = (
        df["matched_skill_count"] / df["required_skill_count"].replace(0, 1)
    ).round(4)

    out["experience_gap"] = (
        df["c_years_experience"] - df["j_min_experience"]
    ).clip(-5, 10).round(2)

    out["education_match"] = (
        df["c_education_level"].map(EDUCATION_ORDINAL) >=
        df["j_education_requirement"].map(EDUCATION_ORDINAL)
    ).astype(int)

    out["certification_gap"] = (
        df["c_certifications"] - df["j_required_certifications"]
    ).clip(0, 4)

    out["skill_coverage_ratio"] = (
        df["c_skills"].apply(lambda s: len(s.split("|"))) / MASTER_POOL_SIZE
    ).round(4)

    out["projects_normalized"]    = (df["c_projects_count"]    / max_projects).round(4)
    out["internships_normalized"] = (df["c_internships_count"] / max_internships).round(4)
    out["gpa_normalized"]         = (df["c_gpa"] / 4.0).round(4)
    out["hackathons_normalized"]  = (df["c_hackathons"]        / max_hackathons).round(4)
    out["research_normalized"]    = (df["c_research_papers"]   / max_research).round(4)
    out["leadership_experience"]  = df["c_leadership_experience"]

    out["specialization_match"] = (
        df["c_specialization"] == df["j_preferred_specialization"]
    ).astype(int)

    out["seniority_match"] = df.apply(
        lambda row: 1 if (
            SENIORITY_BANDS[row["j_seniority_level"]][0] <=
            row["c_years_experience"] <=
            SENIORITY_BANDS[row["j_seniority_level"]][1]
        ) else 0,
        axis=1
    )

    # ── Targets ───────────────────────────────────────────────────────────────
    out["fit_score"] = df["fit_score"]
    out["fit_label"] = df["fit_label"]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("TalentMatch AI — Synthetic Dataset Generator V1.1")
    print("=" * 52)

    print(f"\n[1/4] Generating {N_CANDIDATES} candidate profiles...")
    candidates = generate_candidates(N_CANDIDATES)
    candidates.to_csv(RAW_DIR / "candidates.csv", index=False)
    print(f"      Saved → {RAW_DIR / 'candidates.csv'}")

    print(f"\n[2/4] Generating {N_JOBS} job descriptions...")
    jobs = generate_jobs(N_JOBS)
    jobs.to_csv(RAW_DIR / "jobs.csv", index=False)
    print(f"      Saved → {RAW_DIR / 'jobs.csv'}")

    print(f"\n[3/4] Generating application pairs ({N_JOBS} jobs × {APPS_PER_JOB} candidates)...")
    applications = generate_applications(candidates, jobs, APPS_PER_JOB)
    applications.to_csv(RAW_DIR / "applications.csv", index=False)
    print(f"      Saved → {RAW_DIR / 'applications.csv'}")

    print("\n[4/4] Building model training dataset...")
    model_df = build_model_training_dataset(candidates, jobs, applications)
    model_df.to_csv(PROC_DIR / "model_training_dataset.csv", index=False)
    print(f"      Saved → {PROC_DIR / 'model_training_dataset.csv'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total    = len(applications)
    positive = applications["fit_label"].sum()
    negative = total - positive
    balance  = positive / total * 100

    print("\n── Dataset Summary ───────────────────────────────────")
    print(f"   Candidates              : {len(candidates)}")
    print(f"   Jobs                    : {len(jobs)}")
    print(f"   Applications            : {total}")
    print(f"   Good Fit  (1)           : {positive} ({balance:.1f}%)")
    print(f"   Poor Fit  (0)           : {negative} ({100-balance:.1f}%)")
    print(f"   Model training features : {model_df.shape[1] - 3} features + 2 targets")
    print(f"   Master skill pool size  : {MASTER_POOL_SIZE} skills")
    balance_flag = "Balanced ✓" if 38 <= balance <= 62 else "Imbalanced — review label logic"
    print(f"   Class balance           : {balance_flag}")
    print("─" * 54)
    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
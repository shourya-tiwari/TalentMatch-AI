"""
TalentMatch AI — Recruiter Dashboard
======================================
Streamlit app for candidate-job fit scoring and ranking.

Workflow:
  1. Recruiter selects a job from the existing job pool
  2. Recruiter selects candidates from the pool (multi-select or "all")
     OR uploads a CSV of new candidates matching the candidate schema
  3. App scores all selected candidates against the job
  4. Ranked table + per-candidate skill gap detail displayed

Run with:
  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st
import plotly.express as px

from src.preprocessing import load_candidates, load_jobs
from src.predict import TalentMatchPredictor, DEFAULT_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TalentMatch AI",
    page_icon="🎯",
    layout="wide"
)

DATA_RAW = ROOT_DIR / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# Cached Loaders (avoid reloading model/data on every interaction)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_predictor(threshold: float) -> TalentMatchPredictor:
    return TalentMatchPredictor(threshold=threshold)

@st.cache_data
def get_candidates() -> pd.DataFrame:
    return load_candidates(DATA_RAW / "candidates.csv")

@st.cache_data
def get_jobs() -> pd.DataFrame:
    return load_jobs(DATA_RAW / "jobs.csv")

REQUIRED_UPLOAD_COLS = [
    "candidate_id", "name", "specialization", "years_experience",
    "education_level", "skills", "certifications", "internships_count",
    "projects_count", "hackathons", "research_papers",
    "leadership_experience", "gpa"
]


def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Check uploaded CSV matches the candidate schema. Returns (is_valid, message)."""
    missing = [c for c in REQUIRED_UPLOAD_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "Schema valid."


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Job Selection & Settings
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🎯 TalentMatch AI")
st.sidebar.caption("Intelligent Resume Ranking & Candidate Fit Analysis")

jobs_df = get_jobs()

job_display = jobs_df.apply(
    lambda r: f"{r['job_id']} — {r['title']} ({r['seniority_level']})", axis=1
)
job_selection = st.sidebar.selectbox(
    "Select Job Posting",
    options=job_display,
    index=0
)
selected_job_id = job_selection.split(" — ")[0]
selected_job = jobs_df[jobs_df["job_id"] == selected_job_id].iloc[0]

st.sidebar.divider()

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10, max_value=0.90, value=DEFAULT_THRESHOLD, step=0.05,
    help="Probability above this value is classified as 'Good Fit'. "
         "Default (0.40) was chosen via threshold sensitivity analysis "
         "to favor recall — minimizing missed qualified candidates."
)

predictor = get_predictor(threshold)

st.sidebar.divider()
st.sidebar.caption("V1 — Structured Feature Engineering + ANN")
st.sidebar.caption("Future: TF-IDF → Word2Vec → BERT → LLM/RAG → MLOps")


# ─────────────────────────────────────────────────────────────────────────────
# Header — Job Detail Card
# ─────────────────────────────────────────────────────────────────────────────

st.title("Candidate Fit Ranking")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Job Title", selected_job["title"])
    c2.metric("Seniority", selected_job["seniority_level"])
    c3.metric("Min. Experience", f"{selected_job['min_experience']} yrs")
    c4.metric("Education Req.", selected_job["education_requirement"])

    st.markdown(f"**Job Family:** {selected_job['job_family']}  |  "
                f"**Preferred Specialization:** {selected_job['preferred_specialization']}")

    required_skills = selected_job["required_skills"].split("|")
    st.markdown("**Required Skills:**")
    st.markdown(" ".join([f"`{s}`" for s in required_skills]))


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Source Selection
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Select Candidates")

source_tab1, source_tab2 = st.tabs(["📋 From Candidate Pool", "📤 Upload CSV"])

candidates_to_score = None

with source_tab1:
    candidates_df = get_candidates()

    col1, col2 = st.columns([3, 1])
    with col1:
        specialization_filter = st.multiselect(
            "Filter by specialization (optional)",
            options=sorted(candidates_df["specialization"].unique()),
            default=[]
        )
    with col2:
        sample_size = st.number_input(
            "Max candidates to score",
            min_value=10, max_value=len(candidates_df), value=100, step=10
        )

    pool = candidates_df.copy()
    if specialization_filter:
        pool = pool[pool["specialization"].isin(specialization_filter)]

    pool = pool.head(int(sample_size))
    st.caption(f"{len(pool)} candidates selected from pool "
               f"({'filtered by: ' + ', '.join(specialization_filter) if specialization_filter else 'no filter'})")

    if st.button("Score Candidates from Pool", type="primary", key="score_pool"):
        candidates_to_score = pool

with source_tab2:
    st.caption("CSV must contain columns: " + ", ".join(REQUIRED_UPLOAD_COLS))
    uploaded_file = st.file_uploader("Upload candidates CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            is_valid, message = validate_uploaded_csv(uploaded_df)

            if is_valid:
                st.success(f"✅ {message} ({len(uploaded_df)} candidates loaded)")
                if st.button("Score Uploaded Candidates", type="primary", key="score_upload"):
                    candidates_to_score = uploaded_df
            else:
                st.error(f"❌ {message}")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring & Results
# ─────────────────────────────────────────────────────────────────────────────

if candidates_to_score is not None and len(candidates_to_score) > 0:
    st.divider()
    st.subheader("Ranked Results")

    with st.spinner(f"Scoring {len(candidates_to_score)} candidates..."):
        ranked = predictor.score_batch(candidates_to_score, selected_job)

    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scored", len(ranked))
    m2.metric("Good Fit", int((ranked["fit_label"] == 1).sum()))
    m3.metric("Poor Fit", int((ranked["fit_label"] == 0).sum()))
    m4.metric("Avg Fit Score", f"{ranked['fit_percentage'].mean():.1f}%")

    # ── Distribution chart ───────────────────────────────────────────────────
    fig = px.histogram(
        ranked, x="fit_percentage", color="fit_verdict",
        nbins=30, title="Fit Score Distribution",
        color_discrete_map={"Good Fit": "#2ecc71", "Poor Fit": "#e74c3c"}
    )
    fig.add_vline(x=threshold * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"Threshold ({threshold*100:.0f}%)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Ranked table ──────────────────────────────────────────────────────────
    st.markdown("### Candidate Leaderboard")

    display_df = ranked[[
        "rank", "candidate_name", "fit_percentage", "fit_verdict",
        "confidence_band", "matched_skill_count", "missing_skill_count"
    ]].rename(columns={
        "rank": "Rank", "candidate_name": "Candidate",
        "fit_percentage": "Fit %", "fit_verdict": "Verdict",
        "confidence_band": "Confidence", "matched_skill_count": "Skills Matched",
        "missing_skill_count": "Skills Missing"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fit %": st.column_config.ProgressColumn(
                "Fit %", min_value=0, max_value=100, format="%.1f%%"
            )
        }
    )

    # ── Per-candidate skill gap detail ──────────────────────────────────────
    st.markdown("### Skill Gap Detail")
    selected_candidate_name = st.selectbox(
        "Select a candidate to view detailed skill gap analysis",
        options=ranked["candidate_name"]
    )

    detail_row = ranked[ranked["candidate_name"] == selected_candidate_name].iloc[0]

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**Fit Score:** {detail_row['fit_percentage']}% — {detail_row['confidence_band']}")
        st.markdown(f"**Matched Skills ({detail_row['matched_skill_count']}/{detail_row['required_skill_count']}):**")
        if detail_row["matched_skills"]:
            st.markdown(" ".join([f"`{s}` ✅" for s in detail_row["matched_skills"]]))
        else:
            st.caption("No matched skills.")

    with d2:
        st.markdown(f"**Missing Skills ({detail_row['missing_skill_count']}):**")
        if detail_row["missing_skills"]:
            st.markdown(" ".join([f"`{s}` ❌" for s in detail_row["missing_skills"]]))
        else:
            st.success("No missing skills — full requirement coverage.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    csv_export = ranked.drop(columns=["matched_skills", "missing_skills"]).to_csv(index=False)
    st.download_button(
        "📥 Download Full Ranking (CSV)",
        data=csv_export,
        file_name=f"talentmatch_ranking_{selected_job_id}.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Select candidates from the pool or upload a CSV, then click the score button to see ranked results.")
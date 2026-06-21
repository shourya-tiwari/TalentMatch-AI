# TalentMatch AI

### Intelligent Resume Ranking and Candidate Fit Analysis Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.4-blue?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1%20Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

An AI-powered recruitment intelligence platform that analyzes candidate profiles and job requirements to predict candidate-job compatibility, identify skill gaps, and rank applicants based on suitability.

---

## Problem Statement

Recruiters often receive hundreds of applications for a single job opening.

Traditional Applicant Tracking Systems (ATS) primarily rely on keyword matching and frequently fail to evaluate the overall compatibility between a candidate and a job role.

**TalentMatch AI** addresses this challenge by:

* Engineering meaningful candidate-job matching features
* Training a Neural Network to predict compatibility scores
* Identifying missing skills and strengths
* Ranking candidates automatically
* Providing recruiter-friendly insights through an interactive dashboard

---

## Key Features

### Candidate Analysis

* Candidate–Job compatibility scoring (0–100 Fit Score)
* Binary hiring recommendation (Fit Label)
* Skill gap identification
* Experience, education, and specialization matching
* Internship, project, hackathon, and research profile analysis

### Recruiter Dashboard

* Job posting selection with full requirement breakdown
* Candidate sourcing from existing pool (with specialization filtering) or CSV upload
* Ranked candidate leaderboard with progress-bar fit scores
* Adjustable decision threshold (sidebar slider)
* Interactive fit score distribution chart
* Per-candidate skill gap detail view
* Downloadable ranking export (CSV)

### Explainability

* Matched skill count
* Missing skill count
* Required skill diagnostics
* Confidence band per candidate (Strong Fit / Likely Fit / Borderline / etc.)
* Skill-match guardrail override flag (see "Key Model Decisions" below)

### Synthetic Data Pipeline

* Config-driven synthetic candidate generation
* Config-driven job description generation
* Noise injection for realistic labels (5–10%)
* Automated model-training dataset generation
* Reproducible dataset creation using random seeds

---

## Tech Stack

| Layer              | Technologies                    |
| ------------------ | -------------------------------- |
| Language           | Python 3.11                     |
| Machine Learning    | TensorFlow, Keras, Scikit-Learn  |
| Data Processing     | Pandas, NumPy                   |
| Visualization       | Matplotlib, Seaborn, Plotly      |
| Frontend            | Streamlit                       |
| Model Persistence   | Joblib                          |
| Version Control     | Git, GitHub                     |

---

## System Workflow

```text
Candidate Profiles + Job Descriptions
                    │
                    ▼
        Synthetic Dataset Generator
                    │
                    ▼
        Feature Engineering Pipeline
                    │
                    ▼
       Compatibility Feature Matrix
                    │
                    ▼
        Artificial Neural Network
                    │
                    ▼
             Fit Score (0–100)
                    │
                    ▼
       Fit Label (0/1) + Skill-Match Guardrail
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Skill Gaps   Candidate Rank   Recruiter Dashboard
```

---

## Project Structure

```text
TalentMatch-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── candidates.csv
│   │   ├── jobs.csv
│   │   ├── applications.csv
│   │   └── generate_dataset.py
│   │
│   └── processed/
│       ├── model_training_dataset.csv
│       └── sample_batch_ranking.csv
│
├── docs/
│   └── feature_schema.md
│
├── metadata/
│   ├── __init__.py
│   └── skills_config.json
│
├── models/
│   ├── feature_engineer_v1.joblib
│   ├── talentmatch_ann_v1.keras
│   ├── training_history_v1.json
│   ├── training_config_v1.json
│   ├── evaluation_report_v1.json
│   ├── threshold_sensitivity_v1.csv
│   ├── roc_pr_curves_v1.npz
│   ├── X_test_v1.npy
│   └── y_test_v1.npy
│
├── notebooks/
│   ├── 01_data_generation_overview.ipynb
│   ├── 02_feature_engineering_verification.ipynb
│   ├── 03_training_curves.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_prediction_verification.ipynb
│
├── screenshots/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── run_feature_pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset Overview

### Generated Datasets

| Dataset                | Records                    |
| ----------------------- | --------------------------- |
| Candidates              | 1,000                      |
| Jobs                    | 100                         |
| Applications            | 10,000                      |
| Model Training Dataset  | 10,000 Candidate–Job Pairs   |

### Candidate Features

* Specialization
* Years of Experience
* Education Level
* Skills (categorized: Programming, ML, Data, Cloud, BI, Soft Skills)
* Certifications
* Internships
* Projects
* Hackathons
* Research Papers
* Leadership Experience
* GPA

### Job Features

* Job Family
* Preferred Specialization
* Required Skills
* Seniority Level
* Education Requirement
* Required Certifications

### Targets

* Fit Score (0–100) — continuous, weighted composite
* Fit Label (0 / 1) — derived from Fit Score threshold, with 5–10% synthetic noise

### Engineered Model Features (13 total)

skill_match_score, experience_gap, skill_coverage_ratio, gpa_normalized, projects_normalized, internships_normalized, hackathons_normalized, research_normalized, certification_gap, education_match, specialization_match, seniority_match, leadership_experience

Full derivations documented in `docs/feature_schema.md`.

---

## Model Performance (Test Set)

| Metric              | Score   |
| --------------------- | ------- |
| Accuracy              | 0.850   |
| Precision (Good Fit)   | 0.817   |
| Recall (Good Fit)      | 0.778   |
| F1 (Good Fit)          | 0.797   |
| ROC AUC                | 0.895   |
| Average Precision      | 0.850   |

Architecture: `Input(13) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dense(1, Sigmoid)`, trained with Adam, early stopping, and learning-rate reduction on plateau. Full training and evaluation artifacts are in `models/`.

---

## Key Model Decisions

**Decision threshold (V1): 0.40, not the default 0.50**

Threshold sensitivity analysis on the held-out test set showed F1 peaks at 0.40 (0.807) versus 0.50 (0.797). More importantly, recall for "Good Fit" candidates improves from 0.78 to 0.83 at 0.40, at a modest precision cost (0.82 → 0.78). In a recruiting context, missing a qualified candidate (false negative) is generally more costly than surfacing an extra candidate for human review (false positive), so 0.40 was chosen as the production default. It is exposed as a configurable slider in the dashboard.

**Guardrail: skill_match_score floor (0.45)**

Manual testing of the live dashboard surfaced a real ANN failure mode: candidates with very low skill match (e.g., 1 of 8 required skills) but maxed-out secondary signals (GPA, certifications, experience, education, leadership) received spuriously high fit probabilities (~98%).

Bottom-decile analysis on the test set confirmed this was *not* a systematic miscalibration — on the lowest 10% of skill_match_score values, the model's predicted positive rate (0.193) tracked reasonably close to the true positive rate (0.245). The issue is unreliable extrapolation on rare, out-of-distribution joint feature combinations that were underrepresented during training, not a broad bias.

Rather than silently trust the model in these edge cases, a production safeguard was added to `predict.py`: any candidate with `skill_match_score` below 0.45 is capped at "Poor Fit" regardless of model output, with an explicit "Skill Gap Override (Low Match)" confidence label so the override is transparent rather than silent. This reflects standard practice of layering deterministic business rules over ML output to guard against known model failure modes — and is documented here rather than hidden, since the underlying model probability is still shown alongside the overridden verdict.

---

## Known Limitations

* **Synthetic ground truth.** Fit labels are derived from a deterministic, weighted rule-based formula applied during data generation, not real recruiter decisions. The model is learning to approximate that formula (with noise), not real-world hiring outcomes.
* **ANN extrapolation behavior.** As described above, the network can behave unpredictably on rare feature combinations far from the training distribution, even when aggregate test metrics look strong. The skill-match guardrail mitigates the one identified instance of this; other untested edge cases may exist.
* **No real resume text.** V1 uses fully structured fields (skills as discrete lists, not free text). Real-world resume parsing, formatting variance, and ambiguous skill phrasing are not modeled yet — planned for V2 onward.
* **Tabular data, ANN architecture.** Gradient-boosted trees often outperform ANNs on structured tabular data of this size. The ANN was a deliberate choice here to support future extensibility toward embedding-based inputs (V3+), not necessarily the optimal model for V1 alone.

---

## Development Roadmap

| Phase   | Task                          | Status      |
| ------- | ------------------------------ | ----------- |
| Phase 0 | Project Initialization         | ✅ Complete |
| Phase 1 | Dataset Design & Generation    | ✅ Complete |
| Phase 2 | Exploratory Data Analysis       | ✅ Complete |
| Phase 3 | Feature Engineering Pipeline    | ✅ Complete |
| Phase 4 | ANN Training Pipeline           | ✅ Complete |
| Phase 5 | Model Evaluation                | ✅ Complete |
| Phase 6 | Prediction & Batch Scoring      | ✅ Complete |
| Phase 7 | Streamlit Dashboard             | ✅ Complete |

**V1 is feature-complete and tagged as `v1.0`.**

---

## Version Roadmap

| Version | Planned Upgrade                                   |
| ------- | --------------------------------------------------- |
| V1      | Structured Features + ANN (current — complete)     |
| V2      | TF-IDF Feature Integration                          |
| V3      | Word2Vec Embeddings                                 |
| V4      | Transformer-Based Encoding                          |
| V5      | BERT Resume Embeddings                              |
| V6      | LLM-Powered Recruiter Insights                      |
| V7      | RAG-Based Dynamic JD Matching                       |
| V8      | Complete MLOps Pipeline                             |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/shourya-tiwari/TalentMatch-AI.git
cd TalentMatch-AI
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Generate Synthetic Dataset

```bash
python data/raw/generate_dataset.py
```

### 2. Run Feature Engineering Pipeline

```bash
python src/run_feature_pipeline.py
```

### 3. Train the Model

```bash
python src/train.py
```

### 4. Evaluate on the Test Set

```bash
python src/evaluate.py
```

### 5. Run a Prediction Demo

```bash
python src/predict.py
```

### 6. Launch the Recruiter Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Explore the Notebooks

```bash
jupyter notebook
```

Recommended order: `01_data_generation_overview.ipynb` → `02_feature_engineering_verification.ipynb` → `03_training_curves.ipynb` → `04_model_evaluation.ipynb` → `05_prediction_verification.ipynb`

---

## Live Demo

![Job Selection](screenshots/app_job_selection.png)

Recruiters select a job posting, choose candidates from the existing pool or upload a new CSV, and get an instantly ranked leaderboard.

![Ranked Leaderboard](screenshots/app_ranked_leaderboard.png)

Each candidate includes a fit score, confidence band, and matched/missing skill counts. Candidates with strong model scores but very low skill overlap are automatically flagged with a "Skill Gap Override" label rather than silently trusted.

![Skill Gap Detail](screenshots/app_skill_gap_detail.png)

Clicking into a candidate shows the full skill gap breakdown — matched and missing skills listed explicitly.

---

## Screenshots

```text
screenshots/
├── correlation_matrix.png
├── education_distribution.png
├── experience_distribution.png
├── fit_score_distribution.png
├── job_family_distribution.png
├── label_distribution.png
├── leadership_distribution.png
├── specialization_distribution.png
├── student_features.png
├── feature_correlation_with_label.png
├── feature_distributions.png
├── training_curves.png
├── training_metrics.png
├── confusion_matrix.png
├── roc_pr_curves.png
├── threshold_sensitivity.png
├── batch_scoring_distribution.png
├── confidence_band_distribution.png
├── app_job_selection.png
├── app_ranked_leaderboard.png
└── app_skill_gap_detail.png
```

---

## Future Enhancements

* Resume PDF parsing and real-world text ingestion
* TF-IDF resume features (V2)
* Root-cause fix for the skill-match extrapolation issue via rebalanced synthetic data and retraining
* Transformer-based resume encoding (V4)
* BERT resume embeddings (V5)
* SHAP-based feature contribution explainability
* Multi-job candidate matching (one candidate ranked across many jobs)
* LLM-powered recruiter insight generation (V6)
* RAG-based dynamic job matching (V7)
* Unit tests and CI pipeline
* Cloud deployment
* Complete MLOps pipeline (V8)

---

## Status

**Current Version:** V1.0 — Complete

All eight phases of the V1 roadmap are finished: synthetic data generation, feature engineering, ANN training, evaluation with a documented threshold decision, an inference pipeline with a debugged and documented skill-match guardrail, and a working Streamlit recruiter dashboard.

Next steps are open — see "Future Enhancements" above.

---

## Author

**Shourya Tiwari**
B.Tech Artificial Intelligence & Machine Learning
Symbiosis Institute of Technology, Pune

---

## License

This project is licensed under the MIT License.
See the `LICENSE` file for details.
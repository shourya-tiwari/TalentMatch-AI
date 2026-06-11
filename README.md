# TalentMatch AI

### Intelligent Resume Ranking and Candidate Fit Analysis Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.4-blue?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

# TalentMatch AI

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

* Ranked candidate leaderboard
* Candidate comparison
* Batch application processing
* Interactive visual analytics
* Candidate filtering and sorting

### Explainability

* Matched skill count
* Missing skill count
* Required skill diagnostics
* Feature contribution analysis
* Recruiter-friendly recommendations

### Synthetic Data Pipeline

* Config-driven synthetic candidate generation
* Config-driven job description generation
* Noise injection for realistic labels
* Automated model-training dataset generation
* Reproducible dataset creation using random seeds

---

## Tech Stack

| Layer             | Technologies                    |
| ----------------- | ------------------------------- |
| Language          | Python 3.11                     |
| Machine Learning  | TensorFlow, Keras, Scikit-Learn |
| Data Processing   | Pandas, NumPy                   |
| Visualization     | Matplotlib, Seaborn, Plotly     |
| Frontend          | Streamlit                       |
| Model Persistence | Joblib                          |
| Version Control   | Git, GitHub                     |

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
             Fit Label (0/1)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Skill Gaps   Candidate Rank   Insights
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
│       └── model_training_dataset.csv
│
├── docs/
│   └── feature_schema.md
│
├── metadata/
│   ├── __init__.py
│   └── skills_config.json
│
├── models/
│
├── notebooks/
│   └── 01_data_generation_overview.ipynb
│
├── screenshots/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
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
| ---------------------- | -------------------------- |
| Candidates             | 1,000                      |
| Jobs                   | 100                        |
| Applications           | 10,000                     |
| Model Training Dataset | 10,000 Candidate–Job Pairs |

### Candidate Features

* Specialization
* Years of Experience
* Education Level
* Skills
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

* Fit Score (0–100)
* Fit Label (0 / 1)

---

## Development Roadmap

| Phase   | Task                         | Status     |
| ------- | ---------------------------- | ---------- |
| Phase 0 | Project Initialization       | ✅ Complete |
| Phase 1 | Dataset Design & Generation  | ✅ Complete |
| Phase 2 | Exploratory Data Analysis    | ✅ Complete |
| Phase 3 | Feature Engineering Pipeline | ✅ Complete |
| Phase 4 | ANN Training Pipeline        | ⏳ Pending  |
| Phase 5 | Model Evaluation             | ⏳ Pending  |
| Phase 6 | Prediction & Batch Scoring   | ⏳ Pending  |
| Phase 7 | Streamlit Dashboard          | ⏳ Pending  |
| Phase 8 | Documentation & Deployment   | ⏳ Pending  |

---

## Version Roadmap

| Version | Planned Upgrade                                  |
| ------- | ------------------------------------------------ |
| V1      | Structured Features + ANN                        |
| V1.1    | Synthetic Recruitment Dataset + Fit Score Engine |
| V2      | TF-IDF Feature Integration                       |
| V3      | Word2Vec Embeddings                              |
| V4      | Transformer-Based Encoding                       |
| V5      | BERT Resume Embeddings                           |
| V6      | LLM-Powered Recruiter Insights                   |
| V7      | RAG-Based Dynamic JD Matching                    |
| V8      | Complete MLOps Pipeline                          |

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

### Generate Dataset

```bash
python data/raw/generate_dataset.py
```

### Run Exploratory Analysis

```bash
jupyter notebook
```

Open:

```text
notebooks/01_data_generation_overview.ipynb
```

### Train Model

```bash
python src/train.py
```

### Run Prediction

```bash
python src/predict.py
```

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Screenshots

### Dataset Analysis

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
└── student_features.png
```

EDA visualizations generated during Phase 1 validation are stored in the screenshots directory.

---

## Future Enhancements

* Resume PDF Parsing
* TF-IDF Resume Features
* Transformer-Based Resume Encoding
* BERT Resume Embeddings
* SHAP Explainability
* Multi-Job Candidate Matching
* LLM-Powered Recruiter Insights
* RAG-Based Dynamic Job Matching
* Cloud Deployment
* Complete MLOps Pipeline

---

## Status

**Current Version:** V1.1

Phase 1 (Dataset Design & Generation) has been completed successfully.

Current focus:

* Exploratory Data Analysis (EDA)
* Feature Engineering Validation
* ANN Training Pipeline

---

## Author

**Shourya Tiwari**

B.Tech Artificial Intelligence & Machine Learning
Symbiosis Institute of Technology, Pune

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

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

- Engineering meaningful candidate-job matching features
- Training a Neural Network to predict compatibility scores
- Identifying missing skills and strengths
- Ranking candidates automatically
- Providing recruiter-friendly insights through an interactive dashboard

---

## Key Features

### Candidate Analysis
- Resume-based feature extraction
- Candidate-job compatibility scoring
- Skill gap identification
- Experience and education matching

### Recruiter Dashboard
- Ranked candidate leaderboard
- Candidate comparison
- Batch resume processing
- Interactive visual analytics

### Explainability
- Feature contribution analysis
- Model confidence visualization
- Recruiter-friendly recommendations

---

## Tech Stack

| Layer | Technologies |
|---------|-------------|
| Language | Python 3.11 |
| Machine Learning | TensorFlow, Keras, Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Frontend | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git, GitHub |

---

## System Workflow

```text
Job Description
        │
        ▼
Feature Engineering
        │
        ▼
Candidate Profiles
        │
        ▼
Compatibility Feature Matrix
        │
        ▼
Artificial Neural Network
        │
        ▼
Compatibility Score
        │
        ├── Skill Gap Analysis
        ├── Candidate Ranking
        └── Recruiter Insights
```

---

## Project Structure

```text
TalentMatch-AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── app/
│   └── streamlit_app.py
│
├── docs/
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Development Roadmap

| Phase | Task | Status |
|---------|---------|---------|
| Phase 0 | Project Initialization | ✅ Complete |
| Phase 1 | Dataset Design & Generation | 🔄 In Progress |
| Phase 2 | Exploratory Data Analysis | ⏳ Pending |
| Phase 3 | Feature Engineering Pipeline | ⏳ Pending |
| Phase 4 | ANN Training Pipeline | ⏳ Pending |
| Phase 5 | Model Evaluation | ⏳ Pending |
| Phase 6 | Prediction & Batch Scoring | ⏳ Pending |
| Phase 7 | Streamlit Dashboard | ⏳ Pending |
| Phase 8 | Documentation & Deployment | ⏳ Pending |

---

## Version Roadmap

| Version | Planned Upgrade |
|---------|----------------|
| V1 | Structured Features + ANN |
| V2 | TF-IDF Feature Integration |
| V3 | Word2Vec Embeddings |
| V4 | Transformer-Based Encoding |
| V5 | BERT Resume Embeddings |
| V6 | LLM-Powered Recruiter Insights |
| V7 | RAG-Based Dynamic JD Matching |
| V8 | Complete MLOps Pipeline |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/TalentMatch-AI.git

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

## Running the Application

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

## Future Enhancements

- Resume PDF Parsing
- Real-Time Job Description Analysis
- SHAP-Based Explainability
- Multi-Job Candidate Matching
- Resume Recommendation Engine
- Cloud Deployment
- MLOps Integration

---

## Screenshots

Screenshots will be added as development progresses.

```text
screenshots/
├── dashboard.png
├── ranking_view.png
└── candidate_analysis.png
```

---

## Author

**Shourya Tiwari**

B.Tech Artificial Intelligence & Machine Learning  
Symbiosis Institute of Technology, Pune

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
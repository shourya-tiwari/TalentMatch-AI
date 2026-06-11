# TalentMatch AI — Feature Schema (V1.1)

## Design Philosophy

V1 uses structured, engineered features derived from candidate profiles
and job descriptions. No raw text processing. Every feature is numeric
or ordinally encoded.

The schema covers three layers:
1. Raw profile data (candidates.csv, jobs.csv)
2. Application pair data with skill diagnostics (applications.csv)
3. Fully engineered model-ready features (model_training_dataset.csv)

---

## Candidate Profile Schema

| Field | Type | Description | Example |
|---|---|---|---|
| candidate_id | string | Unique identifier | C0001 |
| name | string | Display name | Priya Sharma |
| specialization | string | Domain focus | Machine Learning |
| years_experience | float | Total years of experience | 3.5 |
| education_level | ordinal | Highest education completed | Bachelor's |
| skills | list[str] | Pipe-separated skill list | Python\|SQL\|ML |
| certifications | int | Number of certifications | 2 |
| internships_count | int | Number of internships completed | 1 |
| projects_count | int | Number of technical projects | 4 |
| hackathons | int | Hackathons participated in | 2 |
| research_papers | int | Published or submitted papers | 0 |
| leadership_experience | int | 0 or 1 flag | 1 |
| gpa | float | GPA on 4.0 scale | 3.6 |

---

## Job Description Schema

| Field | Type | Description | Example |
|---|---|---|---|
| job_id | string | Unique identifier | J001 |
| title | string | Job title | ML Engineer |
| job_family | string | Functional family | Engineering |
| preferred_specialization | string | Preferred candidate domain | Machine Learning |
| seniority_level | ordinal | Junior / Mid / Senior | Mid |
| min_experience | float | Minimum years required | 2.0 |
| education_requirement | ordinal | Minimum education level | Bachelor's |
| required_skills | list[str] | Pipe-separated required skills | Python\|TensorFlow\|SQL |
| required_certifications | int | Minimum certifications preferred | 1 |

---

## Application Schema

| Field | Type | Description |
|---|---|---|
| application_id | string | Unique identifier |
| candidate_id | string | FK to candidates |
| job_id | string | FK to jobs |
| required_skill_count | int | Total skills the job requires |
| matched_skill_count | int | Skills candidate has that job requires |
| missing_skill_count | int | Required skills candidate lacks |
| fit_score | float | Continuous score 0–100 (pre-noise) |
| fit_label | int | 1 if fit_score >= 50, else 0 (post-noise) |

---

## Model Training Dataset Schema (model_training_dataset.csv)

This is the direct model input. All features are numeric.

| Feature | Derivation |
|---|---|
| skill_match_score | matched_skill_count / required_skill_count |
| experience_gap | candidate_experience - job_min_experience (clipped -5 to +10) |
| education_match | 1 if candidate meets or exceeds requirement, else 0 |
| certification_gap | candidate_certs - required_certs (clipped 0 to 4) |
| skill_coverage_ratio | candidate total skills / master skill pool size |
| projects_normalized | projects_count / max in dataset |
| internships_normalized | internships_count / max in dataset |
| gpa_normalized | gpa / 4.0 |
| hackathons_normalized | hackathons / max in dataset |
| research_papers_normalized | research_papers / max in dataset |
| leadership_experience | raw flag (0 or 1) |
| specialization_match | 1 if candidate specialization matches job preferred, else 0 |
| seniority_match | 1 if experience aligns with seniority band, else 0 |
| fit_score | target — continuous (regression, future use) |
| fit_label | target — binary (classification, V1) |

---

## Fit Score Formula

fit_score is computed as a weighted sum across 8 components (0–100):

| Component | Weight | Max Points |
|---|---|---|
| Skill match score | 35% | 35 |
| Experience adequacy | 20% | 20 |
| Education match | 15% | 15 |
| GPA | 10% | 10 |
| Certifications | 8% | 8 |
| Projects | 5% | 5 |
| Internships | 4% | 4 |
| Specialization match | 3% | 3 |

fit_label = 1 if fit_score >= 50, else 0

---

## Noise Injection

To prevent the model from learning a perfect deterministic function
(which would cause overfitting and fail to generalize),
5–10% of fit_label values are randomly flipped after generation.

This simulates real-world recruiter disagreement and label noise.

---

## Education Encoding

| Level | Ordinal |
|---|---|
| High School | 0 |
| Associate's | 1 |
| Bachelor's | 2 |
| Master's | 3 |
| PhD | 4 |

---

## Seniority Bands

| Level | Experience Range |
|---|---|
| Junior | 0–2 years |
| Mid | 2–5 years |
| Senior | 5–12 years |

---

## Candidate Specializations

Machine Learning, Data Science, Data Engineering, Business Intelligence,
Cloud Engineering, NLP, Computer Vision, MLOps, Quantitative Finance,
Software Engineering

---

## Job Families

Engineering, Research, Analytics, Operations, Finance
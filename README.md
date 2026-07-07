# 🤖 AI Resume Analyzer

> An AI-powered Resume Analyzer built with **Python**, **Streamlit**, and **Natural Language Processing (NLP)** that evaluates resumes against job descriptions, calculates ATS compatibility, identifies skill gaps, and generates actionable recommendations.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

# 📌 Project Overview

Recruiters use Applicant Tracking Systems (ATS) to filter resumes before they reach hiring managers.

This project helps job seekers evaluate how well their resume matches a Job Description by analyzing:

- ATS Compatibility Score
- Skill Matching
- Missing Skills
- Keyword Analysis
- Resume Parsing
- Job Description Parsing
- Candidate Profile
- Interactive Dashboard
- Downloadable Report

The application provides intelligent suggestions to improve resumes and increase interview opportunities.

---

# ✨ Features

## Resume Analysis

- PDF Resume Parsing
- DOCX Resume Parsing
- Resume Information Extraction
- Experience Detection
- Education Detection
- Certification Detection

---

## Job Description Analysis

- Parse Job Description
- Extract Required Skills
- Extract Preferred Skills
- Detect Experience Requirements
- Identify Keywords

---

## ATS Scoring

- Overall ATS Score
- Resume Quality Score
- Skills Match Score
- Keyword Match Score
- Experience Score
- Education Score
- Resume Completeness
- Overall Compatibility

---

## Skill Gap Analysis

- Matched Skills
- Missing Skills
- Additional Skills
- Skills Categorization
- AI Recommendations

---

## Interactive Dashboard

- ATS Score Gauge
- Radar Charts
- Keyword Charts
- Skill Distribution
- Progress Indicators
- Resume Statistics

---

## Report Generation

Generate a professional HTML report containing:

- ATS Score
- Skills Analysis
- Resume Summary
- Improvement Suggestions
- Candidate Information

---

# 🖥️ Screenshots

## Dashboard

<img width="1568" height="864" alt="Screenshot 2026-07-07 171518" src="https://github.com/user-attachments/assets/b078ba4c-0148-40ca-b074-2f492a375be2" />

---

## Skill Analysis

<img width="1579" height="833" alt="Screenshot 2026-07-07 163510" src="https://github.com/user-attachments/assets/9bda250b-2059-4f5b-8d0c-f15c9eee168d" />


---

## Keyword Analysis

<img width="1568" height="787" alt="Screenshot 2026-07-07 163313" src="https://github.com/user-attachments/assets/39b92146-181a-4ecf-a544-93a10bb82d86" />

---

## Resume Details

<img width="1588" height="859" alt="Screenshot 2026-07-07 163435" src="https://github.com/user-attachments/assets/1f7635e1-6e28-4e8b-a12a-19d7ea93afed" />

---

## HTML Report

<img width="1594" height="857" alt="Screenshot 2026-07-07 163425" src="https://github.com/user-attachments/assets/27f21dc2-565b-41b8-9898-493cae2d33ab" />


---

# 🛠️ Technology Stack

### Programming Language

- Python

### Framework

- Streamlit

### NLP

- spaCy
- Regular Expressions

### Data Processing

- Pandas
- NumPy

### Resume Processing

- pdfplumber
- python-docx

### Visualization

- Plotly

### Machine Learning

- scikit-learn

### Version Control

- Git
- GitHub

---

# 📂 Project Structure

```
AI-Resume-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
│
├── data/
│   ├── skills.csv
│   └── sample_jd.txt
│
├── models/
│
├── reports/
│
├── tests/
│
└── utils/
    ├── ats_score.py
    ├── charts.py
    ├── docx_reader.py
    ├── jd_parser.py
    ├── matcher.py
    ├── pdf_reader.py
    ├── report_generator.py
    ├── resume_parser.py
    ├── skill_extractor.py
    └── text_cleaner.py
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/saideepakchittithula/AI-Resume-Analyzer.git
```

Go into the project

```bash
cd AI-Resume-Analyzer
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🚀 How to Use

1. Upload Resume (PDF or DOCX)

2. Paste Job Description

3. Click Analyze

4. Review

- ATS Score
- Skill Gap
- Keywords
- Resume Details

5. Download HTML Report

---

# 🎯 Future Improvements

- AI Resume Rewriting
- Resume Ranking
- AI Cover Letter Generator
- LinkedIn Profile Analyzer
- Interview Question Generator
- Multi-language Resume Support
- PDF Report Export
- OpenAI Integration
- Recruiter Dashboard
- Candidate Comparison

---

# 💼 Skills Demonstrated

- Python
- Streamlit
- NLP
- Resume Parsing
- ATS Scoring
- Prompt Engineering
- Data Visualization
- Modular Architecture
- Git & GitHub
- Software Development

---

# 👨‍💻 Author

**Sai Deepak**

AI Automation Engineer

Hyderabad, India

GitHub

https://github.com/saideepakchittithula

LinkedIn

www.linkedin.com/in/sai-deepak-chittithula-b51a70247

---

# ⭐ If you like this project

Please consider giving this repository a ⭐ on GitHub.

# 🤖 AI Resume Analyzer

> An AI-powered Resume Analyzer built with **Python**, **Streamlit**, and **Natural Language Processing (NLP)** that evaluates resumes against job descriptions, calculates ATS compatibility, identifies skill gaps, and generates actionable recommendations.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Project Overview

Recruiters use Applicant Tracking Systems (ATS) to filter resumes before they reach hiring managers.

This project helps job seekers evaluate how well their resume matches a job description by analyzing ATS compatibility, skill overlap, keyword usage, and resume structure — then generates an interactive dashboard and a downloadable report with clear, actionable suggestions to improve interview chances.

---

## ✨ Features

### 📄 Resume & Job Description Parsing
- Parses resumes from **PDF** and **DOCX** formats
- Extracts experience, education, and certifications from the resume
- Parses the job description to identify required skills, preferred skills, experience requirements, and key terms

### 📊 ATS Scoring
- Overall ATS Score combining multiple sub-metrics
- Resume Quality, Skills Match, Keyword Match, Experience, and Education scores
- Resume Completeness and Overall Compatibility ratings

### 🎯 Skill Gap Analysis
- Highlights **matched**, **missing**, and **additional** skills relative to the job description
- Categorizes skills (e.g., technical, soft, tools) for easier review
- Provides AI-generated recommendations to close identified gaps

### 📈 Interactive Dashboard
- ATS Score Gauge and Radar Charts
- Keyword frequency and skill distribution charts
- Progress indicators and resume statistics, all built with Plotly

### 📥 Report Generation
- Generates a professional, downloadable **HTML report** containing the ATS score, skills analysis, resume summary, candidate info, and improvement suggestions

---

## 🖥️ Screenshots

### Dashboard
<img width="1568" height="864" alt="Dashboard screenshot" src="https://github.com/user-attachments/assets/b078ba4c-0148-40ca-b074-2f492a375be2" />

### Skill Analysis
<img width="1579" height="833" alt="Skill analysis screenshot" src="https://github.com/user-attachments/assets/9bda250b-2059-4f5b-8d0c-f15c9eee168d" />

### Keyword Analysis
<img width="1568" height="787" alt="Keyword analysis screenshot" src="https://github.com/user-attachments/assets/39b92146-181a-4ecf-a544-93a10bb82d86" />

### Resume Details
<img width="1588" height="859" alt="Resume details screenshot" src="https://github.com/user-attachments/assets/1f7635e1-6e28-4e8b-a12a-19d7ea93afed" />

### HTML Report
<img width="1594" height="857" alt="HTML report screenshot" src="https://github.com/user-attachments/assets/27f21dc2-565b-41b8-9898-493cae2d33ab" />

---

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| Programming Language | Python |
| Framework | Streamlit |
| NLP | spaCy, Regular Expressions |
| Data Processing | Pandas, NumPy |
| Resume Processing | pdfplumber, python-docx |
| Visualization | Plotly |
| Machine Learning | scikit-learn |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

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

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/saideepakchittithula/AI-Resume-Analyzer.git
```

**2. Go into the project**
```bash
cd AI-Resume-Analyzer
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the spaCy language model** *(required for NLP parsing)*
```bash
python -m spacy download en_core_web_sm
```

**5. Run the application**
```bash
streamlit run app.py
```

---

## 🚀 How to Use

1. Upload your resume (PDF or DOCX)
2. Paste the job description
3. Click **Analyze**
4. Review your ATS score, skill gaps, keyword analysis, and resume details
5. Download the HTML report

---

## 🎯 Future Improvements

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

## 💼 Skills Demonstrated

Python · NLP (spaCy) · Resume/JD Parsing · ATS Scoring Algorithms · Data Visualization (Plotly) · Modular Software Architecture · Git & GitHub

---

## 👨‍💻 Author

**Sai Deepak**
AI Automation Engineer — Hyderabad, India

[GitHub](https://github.com/saideepakchittithula) · [LinkedIn](https://www.linkedin.com/in/sai-deepak-chittithula-b51a70247)

---

## ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub!

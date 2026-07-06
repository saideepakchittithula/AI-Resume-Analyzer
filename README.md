# AI Resume Analyzer

A local, privacy-first resume analysis tool that scores resumes against job descriptions using NLP, skill matching, and ATS simulation — all running on your machine with no data sent to any external service.

---

## Features

- **ATS Scoring** — 9-category weighted score (skills, experience, education, projects, certifications, keyword density, formatting, length, summary)
- **Skill Gap Analysis** — 341 skills / 948 aliases matched via spaCy PhraseMatcher + regex
- **Semantic Matching** — TF-IDF cosine similarity with Jaccard fallback
- **Keyword Comparison** — side-by-side resume vs JD keyword frequency
- **Interactive Charts** — gauge, radar, donut, bar, and timeline charts (Plotly)
- **HTML Report** — downloadable self-contained dark-theme report
- **File Support** — PDF (pdfplumber + PyPDF2 fallback) and DOCX (body + tables + headers + textboxes)

---

## Quick Start

### 1. Clone / download

```bash
git clone https://github.com/your-username/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_md
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Usage

1. **Upload Resume** — drag and drop a PDF or DOCX file in the sidebar
2. **Add Job Description** — paste the JD text or upload a TXT / PDF / DOCX file
3. **Click Analyze** — the full pipeline runs in a few seconds
4. **Explore tabs**:
   - **Dashboard** — overall ATS score, radar chart, category breakdown
   - **Skill Analysis** — matched / missing / extra skills, skills by category
   - **Keywords** — keyword frequency comparison between resume and JD
   - **Resume Details** — parsed candidate info, experience timeline, JD summary
   - **Report** — download a self-contained HTML report

---

## Project Structure

```
AI-Resume-Analyzer/
│
├── app.py                    # Streamlit entry-point
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
│
├── data/
│   ├── skills.csv            # 341 skills with categories and aliases
│   └── sample_jd.txt         # Sample job description for testing
│
└── utils/
    ├── __init__.py           # Lazy import module
    ├── constants.py          # ATS weights, regex patterns, config
    ├── logger.py             # Rotating file + colored console logger
    ├── helpers.py            # Pure utility functions
    ├── pdf_reader.py         # PDF text extraction (pdfplumber + PyPDF2)
    ├── docx_reader.py        # DOCX text extraction (body + tables + XML)
    ├── text_cleaner.py       # 8-stage NLP cleaning pipeline
    ├── resume_parser.py      # Resume field extractor (17 methods)
    ├── jd_parser.py          # Job description parser (16 methods)
    ├── skill_extractor.py    # Skill matching engine (PhraseMatcher + regex)
    ├── matcher.py            # TF-IDF + semantic similarity matcher
    ├── ats_score.py          # 9-category ATS scorer
    ├── charts.py             # Plotly chart builders
    └── report_generator.py   # HTML report generator
```

---

## ATS Scoring Breakdown

| Category        | Weight | What it measures |
|-----------------|--------|-----------------|
| Skills Match    | 30     | Overlap between resume skills and JD required/preferred skills |
| Experience      | 20     | Years of experience vs JD requirement |
| Education       | 10     | Degree level vs JD education requirement |
| Projects        | 10     | Presence and relevance of project entries |
| Certifications  | 5      | Relevant certifications listed |
| Keyword Density | 10     | JD keyword coverage in resume |
| Formatting      | 5      | Section structure, contact info, length balance |
| Resume Length   | 5      | Word count in optimal range (400–800 words) |
| Summary         | 5      | Presence and quality of professional summary |

---

## Tech Stack

| Layer        | Library |
|--------------|---------|
| UI           | Streamlit |
| NLP          | spaCy `en_core_web_md`, NLTK |
| ML / Similarity | scikit-learn (TF-IDF, cosine similarity) |
| PDF parsing  | pdfplumber, PyPDF2 |
| DOCX parsing | python-docx |
| Charts       | Plotly |
| Data         | pandas |
| Logging      | Python `logging` + rotating file handler |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

---

## Privacy

All processing runs locally. No resume text, scores, or personal data is sent to any external server or API.

---

## License

MIT License — free to use, modify, and distribute.

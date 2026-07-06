# utils/constants.py
"""
constants.py — Central configuration and constants for AI Resume Analyzer.

All magic numbers, strings, thresholds, weights, color palettes, and
section definitions live here so every other module imports from one
single source of truth.  Change a value here and it propagates
everywhere automatically.
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Root directory of the project (one level above utils/)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
REPORTS_DIR: Path = ROOT_DIR / "reports"
ASSETS_DIR: Path = ROOT_DIR / "assets"
MODELS_DIR: Path = ROOT_DIR / "models"
LOGS_DIR: Path = ROOT_DIR / "logs"

# Key data files
SKILLS_CSV: Path = DATA_DIR / "skills.csv"
SAMPLE_JD: Path = DATA_DIR / "sample_jd.txt"

# Create directories if they don't exist yet
for _dir in (REPORTS_DIR, ASSETS_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# APPLICATION METADATA
# ============================================================

APP_NAME: str = "AI Resume Analyzer"
APP_VERSION: str = "1.0.0"
APP_AUTHOR: str = "AI Resume Analyzer Team"
APP_DESCRIPTION: str = (
    "An intelligent ATS-powered resume analyzer that scores, "
    "parses, and matches resumes against job descriptions."
)
APP_ICON: str = "📄"
APP_TAGLINE: str = "Analyze. Optimize. Get Hired."


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_RESUME_TYPES: list[str] = ["pdf", "docx"]
SUPPORTED_JD_TYPES: list[str] = ["txt"]
MAX_FILE_SIZE_MB: int = 10  # Maximum upload size in megabytes
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024


# ============================================================
# SPACY MODEL
# ============================================================

SPACY_MODEL: str = "en_core_web_md"
SPACY_MODEL_FALLBACK: str = "en_core_web_sm"


# ============================================================
# ATS SCORING WEIGHTS
# Each key maps to a category; values must sum to 100
# ============================================================

ATS_WEIGHTS: dict[str, int] = {
    "skills_match":     30,   # Skills found in resume vs JD
    "experience":       20,   # Work experience relevance & length
    "education":        10,   # Education qualifications
    "projects":         10,   # Projects listed
    "certifications":    5,   # Certifications mentioned
    "keyword_density":  10,   # JD keyword frequency in resume
    "formatting":        5,   # Resume formatting quality signals
    "resume_length":     5,   # Optimal page/word length
    "summary":           5,   # Professional summary present
}

# Sanity check — weights must sum to 100
assert sum(ATS_WEIGHTS.values()) == 100, (
    f"ATS_WEIGHTS must sum to 100, got {sum(ATS_WEIGHTS.values())}"
)


# ============================================================
# ATS SCORE THRESHOLDS
# ============================================================

ATS_SCORE_EXCELLENT: int = 85   # ≥ 85 → Excellent
ATS_SCORE_GOOD: int = 70        # ≥ 70 → Good
ATS_SCORE_AVERAGE: int = 50     # ≥ 50 → Average
ATS_SCORE_POOR: int = 0         # <  50 → Poor

ATS_SCORE_LABELS: dict[str, tuple[int, int, str]] = {
    # label: (min_score, max_score, emoji)
    "Excellent": (ATS_SCORE_EXCELLENT, 100, "🏆"),
    "Good":      (ATS_SCORE_GOOD,       84, "✅"),
    "Average":   (ATS_SCORE_AVERAGE,    69, "⚠️"),
    "Poor":      (ATS_SCORE_POOR,       49, "❌"),
}


# ============================================================
# RESUME SECTIONS
# Labels used for section detection in raw resume text
# ============================================================

RESUME_SECTIONS: dict[str, list[str]] = {
    "summary": [
        "summary", "professional summary", "career summary",
        "about me", "profile", "objective", "career objective",
        "professional profile", "overview",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "career history",
        "positions held", "relevant experience", "internship",
        "internships",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "qualifications", "educational background", "schooling",
        "degrees", "academic history",
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "competencies", "technologies", "tech stack",
        "programming languages", "tools", "tools & technologies",
        "key skills", "areas of expertise", "expertise",
    ],
    "projects": [
        "projects", "personal projects", "key projects",
        "notable projects", "academic projects", "side projects",
        "portfolio", "open source", "contributions",
    ],
    "certifications": [
        "certifications", "certificates", "certified",
        "professional certifications", "licenses", "accreditations",
        "credentials", "courses",
    ],
    "languages": [
        "languages", "spoken languages", "language proficiency",
        "foreign languages",
    ],
    "awards": [
        "awards", "honors", "achievements", "accomplishments",
        "recognitions", "accolades",
    ],
    "publications": [
        "publications", "papers", "research", "articles",
        "journals", "conference papers",
    ],
    "volunteer": [
        "volunteer", "volunteering", "community service",
        "social work",
    ],
    "references": [
        "references", "referees",
    ],
}


# ============================================================
# REGEX PATTERNS
# Pre-compiled pattern strings used by resume_parser.py
# ============================================================

REGEX_EMAIL: str = (
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

REGEX_PHONE: str = (
    r"(?:\+?\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{3}\)?[\s\-.]?)"
    r"\d{3}[\s\-.]?\d{4}"
)

REGEX_LINKEDIN: str = (
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-_%]+"
)

REGEX_GITHUB: str = (
    r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+"
)

REGEX_PORTFOLIO: str = (
    r"(?:https?://)?(?:www\.)?"
    r"(?!linkedin|github)"
    r"[\w\-]+\.(?:com|io|dev|me|co|net|org|app)"
    r"(?:/[\w\-./]*)?"
)

REGEX_URL: str = (
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
)

REGEX_YEAR: str = r"\b(19|20)\d{2}\b"

REGEX_DATE_RANGE: str = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)?"
    r"\.?\s*(?:19|20)\d{2}"
    r"\s*[-–—to]+\s*"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)?"
    r"\.?\s*(?:(?:19|20)\d{2}|[Pp]resent|[Cc]urrent|[Nn]ow)"
)

REGEX_GPA: str = (
    r"(?:GPA|CGPA|gpa|cgpa)\s*[:=]?\s*(\d+\.?\d*)\s*/?\s*(\d+\.?\d*)?"
)


# ============================================================
# DEGREE KEYWORDS
# Used by education parser to identify degree level
# ============================================================

DEGREE_KEYWORDS: dict[str, list[str]] = {
    "PhD": [
        "ph.d", "phd", "doctor of philosophy", "doctorate",
        "d.phil",
    ],
    "Master's": [
        "master", "m.s.", "ms", "m.sc", "msc", "m.tech", "mtech",
        "m.e.", "me", "mba", "m.b.a", "m.eng", "meng",
        "master of science", "master of arts", "master of engineering",
        "master of business",
    ],
    "Bachelor's": [
        "bachelor", "b.s.", "bs", "b.sc", "bsc", "b.tech", "btech",
        "b.e.", "be", "b.a.", "ba", "b.eng", "beng",
        "bachelor of science", "bachelor of arts",
        "bachelor of engineering", "bachelor of technology",
        "undergraduate",
    ],
    "Associate": [
        "associate", "a.s.", "a.a.", "associate degree",
        "associate of science", "associate of arts",
    ],
    "High School": [
        "high school", "secondary school", "12th grade",
        "hsc", "ssc", "ged", "diploma",
    ],
}


# ============================================================
# EXPERIENCE LEVEL THRESHOLDS (years)
# ============================================================

EXPERIENCE_LEVELS: dict[str, tuple[int, int]] = {
    "Entry Level":  (0, 1),
    "Junior":       (1, 3),
    "Mid Level":    (3, 5),
    "Senior":       (5, 8),
    "Lead":         (8, 12),
    "Principal":    (12, 100),
}


# ============================================================
# RESUME LENGTH BENCHMARKS (word count)
# ============================================================

RESUME_LENGTH: dict[str, tuple[int, int]] = {
    "Too Short":  (0,   200),
    "Short":      (200, 400),
    "Optimal":    (400, 800),
    "Good":       (800, 1200),
    "Too Long":   (1200, 99999),
}

RESUME_OPTIMAL_MIN_WORDS: int = 400
RESUME_OPTIMAL_MAX_WORDS: int = 1200


# ============================================================
# SUGGESTION TEMPLATES
# Reusable improvement messages shown in the UI and PDF report
# ============================================================

SUGGESTIONS: dict[str, str] = {
    "add_linkedin":
        "Add your LinkedIn profile URL to increase recruiter visibility.",
    "add_github":
        "Add your GitHub profile URL to showcase your code portfolio.",
    "add_portfolio":
        "Add a portfolio or personal website URL.",
    "add_summary":
        "Write a concise professional summary (3–5 sentences) at the top.",
    "add_certifications":
        "Add relevant certifications to strengthen your profile.",
    "add_projects":
        "List 2–4 projects with tech stack, impact, and GitHub links.",
    "improve_experience":
        "Quantify achievements in experience (e.g., 'Reduced API latency by 40%').",
    "use_action_verbs":
        "Start bullet points with strong action verbs: Built, Designed, Led, Optimized.",
    "improve_formatting":
        "Use consistent fonts, clear section headers, and proper spacing.",
    "remove_graphics":
        "Remove images, graphics, and tables — ATS systems cannot parse them.",
    "add_keywords":
        "Include more keywords from the job description in your resume.",
    "resume_too_short":
        "Your resume appears too short. Expand your experience and projects sections.",
    "resume_too_long":
        "Your resume is too long. Keep it to 1–2 pages for best ATS results.",
    "add_education":
        "Add your education details including degree, institution, and year.",
    "learn_docker":
        "Consider learning Docker — it is listed as required in this job description.",
    "learn_kubernetes":
        "Kubernetes is a highly sought-after skill for this role. Consider learning it.",
    "learn_cloud":
        "Cloud skills (AWS/Azure/GCP) are required. Add relevant cloud experience.",
    "learn_fastapi":
        "FastAPI is listed in the JD. It is quick to learn and widely used.",
    "improve_skills_section":
        "Organize your skills section clearly by category (e.g., Languages, Tools, Cloud).",
    "add_measurable_achievements":
        "Add measurable achievements and metrics to stand out from other candidates.",
    "tailor_resume":
        "Tailor your resume specifically to this job description for a higher ATS score.",
}


# ============================================================
# ACTION VERBS
# Used to check if experience bullets start with action verbs
# ============================================================

ACTION_VERBS: list[str] = [
    "achieved", "architected", "automated", "built", "collaborated",
    "contributed", "created", "debugged", "delivered", "deployed",
    "designed", "developed", "drove", "engineered", "enhanced",
    "established", "executed", "generated", "implemented", "improved",
    "integrated", "launched", "led", "maintained", "managed",
    "mentored", "migrated", "modeled", "monitored", "optimized",
    "orchestrated", "owned", "performed", "published", "reduced",
    "refactored", "researched", "resolved", "reviewed", "scaled",
    "shipped", "streamlined", "tested", "trained", "transformed",
    "upgraded", "utilized", "validated", "wrote",
]


# ============================================================
# UI COLOR THEME
# ============================================================

COLORS: dict[str, str] = {
    # Brand
    "primary":       "#4F8BF9",
    "secondary":     "#0E1117",
    "accent":        "#FF6B6B",

    # Score colors
    "excellent":     "#00C896",
    "good":          "#4F8BF9",
    "average":       "#FFB347",
    "poor":          "#FF6B6B",

    # Chart palette
    "chart_1":       "#4F8BF9",
    "chart_2":       "#00C896",
    "chart_3":       "#FFB347",
    "chart_4":       "#FF6B6B",
    "chart_5":       "#A78BFA",
    "chart_6":       "#F472B6",
    "chart_7":       "#34D399",
    "chart_8":       "#FBBF24",

    # Text
    "text_primary":  "#FAFAFA",
    "text_muted":    "#9CA3AF",

    # Backgrounds
    "card_bg":       "#1E2130",
    "sidebar_bg":    "#0E1117",
    "success_bg":    "#064E3B",
    "warning_bg":    "#78350F",
    "error_bg":      "#7F1D1D",
}

# Chart color sequence (ordered list for plotly/matplotlib)
CHART_COLORS: list[str] = [
    COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
    COLORS["chart_4"], COLORS["chart_5"], COLORS["chart_6"],
    COLORS["chart_7"], COLORS["chart_8"],
]


# ============================================================
# REPORT SETTINGS
# ============================================================

REPORT_FONT_FAMILY: str = "Helvetica"
REPORT_TITLE_SIZE: int = 24
REPORT_HEADING_SIZE: int = 14
REPORT_BODY_SIZE: int = 10
REPORT_SMALL_SIZE: int = 8
REPORT_PAGE_WIDTH: int = 595    # A4 width in points
REPORT_PAGE_HEIGHT: int = 842   # A4 height in points
REPORT_MARGIN: int = 50


# ============================================================
# LOGGING
# ============================================================

LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_MAX_BYTES: int = 5 * 1024 * 1024   # 5 MB per log file
LOG_BACKUP_COUNT: int = 3               # Keep last 3 rotated files


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

PAGE_CONFIG: dict = {
    "page_title":   APP_NAME,
    "page_icon":    APP_ICON,
    "layout":       "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": f"{APP_NAME} v{APP_VERSION} — {APP_TAGLINE}",
    },
}


# ============================================================
# SKILL CATEGORIES (display order in UI)
# ============================================================

SKILL_CATEGORY_ORDER: list[str] = [
    "Programming",
    "AI",
    "Frontend",
    "Backend",
    "Database",
    "Cloud",
    "DevOps",
    "Data",
    "Automation",
    "Mobile",
    "Security",
    "Messaging",
    "Design",
    "Architecture",
    "Methodology",
    "Infrastructure",
    "Web3",
    "Embedded",
    "Game",
    "Communication",
    "Payment",
    "Soft Skills",
]


# ============================================================
# SEMANTIC SIMILARITY THRESHOLDS
# ============================================================

SIMILARITY_HIGH: float = 0.75
SIMILARITY_MEDIUM: float = 0.50
SIMILARITY_LOW: float = 0.25


# ============================================================
# KEYWORD DENSITY SETTINGS
# ============================================================

MIN_KEYWORD_DENSITY: float = 0.01   # 1% minimum
MAX_KEYWORD_DENSITY: float = 0.05   # 5% maximum (avoid stuffing)


# ============================================================
# FUTURE AI FEATURE FLAGS
# Set to True when the respective API key is configured
# ============================================================

FEATURE_FLAGS: dict[str, bool] = {
    "openai_chat":           False,   # Resume chatbot (OpenAI)
    "gemini_analysis":       False,   # Gemini AI deep analysis
    "cover_letter_gen":      False,   # Cover letter generator
    "interview_questions":   False,   # Interview question generator
    "resume_ranking":        False,   # Multi-resume ranking
    "ai_improvement":        False,   # AI-powered resume rewrite
}

# utils/helpers.py
"""
helpers.py — General-purpose utility functions for AI Resume Analyzer.

This module provides reusable, stateless helper functions that are
shared across multiple modules.  Every function is:
  - Fully type-hinted
  - Documented with a docstring
  - Defensively coded with try/except
  - Free of side effects (pure functions where possible)
"""

import re
import os
import json
import hashlib
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.constants import (
    REGEX_EMAIL,
    REGEX_PHONE,
    REGEX_LINKEDIN,
    REGEX_GITHUB,
    REGEX_PORTFOLIO,
    REGEX_YEAR,
    RESUME_OPTIMAL_MIN_WORDS,
    RESUME_OPTIMAL_MAX_WORDS,
    RESUME_LENGTH,
    EXPERIENCE_LEVELS,
    ATS_SCORE_LABELS,
    ATS_SCORE_EXCELLENT,
    ATS_SCORE_GOOD,
    ATS_SCORE_AVERAGE,
    ACTION_VERBS,
    COLORS,
    REPORTS_DIR,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into a single space
    and strip leading/trailing whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string with normalised whitespace.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """
    Normalize unicode characters to ASCII equivalents and
    lower-case the result.  Useful for consistent comparisons.

    Args:
        text: Input string (may contain unicode).

    Returns:
        ASCII lower-cased string.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to max_length characters, appending suffix if cut.

    Args:
        text:       Input string.
        max_length: Maximum allowed character count.
        suffix:     String appended when truncation occurs.

    Returns:
        Truncated string.
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rstrip() + suffix


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.

    Args:
        text: Input string.

    Returns:
        Word count as integer.
    """
    if not text:
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """
    Estimate the number of sentences using punctuation splitting.

    Args:
        text: Input string.

    Returns:
        Sentence count as integer.
    """
    if not text:
        return 0
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def extract_bullet_points(text: str) -> list[str]:
    """
    Extract lines that look like bullet points from resume text.
    Detects •, -, *, >, →, and numbered lists.

    Args:
        text: Multi-line resume text.

    Returns:
        List of bullet point strings (stripped).
    """
    bullets: list[str] = []
    pattern = re.compile(
        r"^[\s]*(?:[•\-\*\>→◆▪▸●]|\d+[\.\)])\s+(.+)$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        line = match.group(1).strip()
        if line:
            bullets.append(line)
    return bullets


def check_action_verbs(text: str) -> dict[str, Any]:
    """
    Check whether bullet points in the text start with strong action verbs.

    Args:
        text: Resume text block.

    Returns:
        Dictionary with keys:
          - found_verbs (list[str])
          - missing_count (int)
          - score (float 0–1)
    """
    bullets = extract_bullet_points(text)
    if not bullets:
        return {"found_verbs": [], "missing_count": 0, "score": 0.5}

    found: list[str] = []
    missing = 0
    for bullet in bullets:
        first_word = bullet.split()[0].lower().rstrip(".,;:") if bullet.split() else ""
        if first_word in ACTION_VERBS:
            found.append(first_word)
        else:
            missing += 1

    total = len(bullets)
    score = len(found) / total if total > 0 else 0.5
    return {"found_verbs": list(set(found)), "missing_count": missing, "score": score}


# ============================================================
# CONTACT EXTRACTION HELPERS
# ============================================================

def extract_emails(text: str) -> list[str]:
    """
    Extract all email addresses from text using regex.

    Args:
        text: Raw text to search.

    Returns:
        List of unique email strings (lower-cased).
    """
    if not text:
        return []
    matches = re.findall(REGEX_EMAIL, text, re.IGNORECASE)
    return list({m.lower() for m in matches})


def extract_phones(text: str) -> list[str]:
    """
    Extract all phone numbers from text.

    Args:
        text: Raw text to search.

    Returns:
        List of unique phone strings.
    """
    if not text:
        return []
    matches = re.findall(REGEX_PHONE, text)
    # Filter out very short matches (false positives like years)
    return list({m.strip() for m in matches if len(re.sub(r"\D", "", m)) >= 10})


def extract_linkedin(text: str) -> str:
    """
    Extract the first LinkedIn profile URL found in text.

    Args:
        text: Raw text to search.

    Returns:
        LinkedIn URL string, or empty string if not found.
    """
    if not text:
        return ""
    match = re.search(REGEX_LINKEDIN, text, re.IGNORECASE)
    return match.group(0).strip() if match else ""


def extract_github(text: str) -> str:
    """
    Extract the first GitHub profile URL found in text.

    Args:
        text: Raw text to search.

    Returns:
        GitHub URL string, or empty string if not found.
    """
    if not text:
        return ""
    match = re.search(REGEX_GITHUB, text, re.IGNORECASE)
    return match.group(0).strip() if match else ""


def extract_portfolio(text: str) -> str:
    """
    Extract a portfolio / personal website URL from text,
    excluding LinkedIn and GitHub URLs.

    Args:
        text: Raw text to search.

    Returns:
        Portfolio URL string, or empty string if not found.
    """
    if not text:
        return ""
    # Remove known social URLs first to avoid false positives
    cleaned = re.sub(REGEX_LINKEDIN, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(REGEX_GITHUB, "", cleaned, flags=re.IGNORECASE)
    match = re.search(REGEX_PORTFOLIO, cleaned, re.IGNORECASE)
    return match.group(0).strip() if match else ""


# ============================================================
# DATE & YEAR HELPERS
# ============================================================

def extract_years(text: str) -> list[int]:
    """
    Extract all 4-digit years (1900–2099) from text.

    Args:
        text: Input string.

    Returns:
        Sorted list of unique year integers.
    """
    if not text:
        return []
    matches = re.findall(REGEX_YEAR, text)
    return sorted(set(int(y) for y in matches))


def calculate_experience_years(text: str) -> float:
    """
    Estimate total years of experience from year mentions in text.
    Uses the span between the earliest and latest year found.

    Args:
        text: Resume text block.

    Returns:
        Estimated years of experience as a float.
    """
    years = extract_years(text)
    current_year = datetime.now().year
    if not years:
        return 0.0
    # Exclude future years
    past_years = [y for y in years if y <= current_year]
    if len(past_years) < 2:
        return 0.0
    return float(current_year - min(past_years))


def get_experience_level(years: float) -> str:
    """
    Map a number of years to an experience level label.

    Args:
        years: Years of professional experience.

    Returns:
        Experience level string (e.g., 'Senior', 'Junior').
    """
    for level, (min_y, max_y) in EXPERIENCE_LEVELS.items():
        if min_y <= years < max_y:
            return level
    return "Principal"


def get_resume_length_label(word_count: int) -> str:
    """
    Classify resume word count into a human-readable length label.

    Args:
        word_count: Number of words in the resume.

    Returns:
        Length label string (e.g., 'Optimal', 'Too Short').
    """
    for label, (min_w, max_w) in RESUME_LENGTH.items():
        if min_w <= word_count < max_w:
            return label
    return "Too Long"


def is_optimal_length(word_count: int) -> bool:
    """
    Return True if the resume word count is within the optimal range.

    Args:
        word_count: Number of words in the resume.

    Returns:
        Boolean indicating whether length is optimal.
    """
    return RESUME_OPTIMAL_MIN_WORDS <= word_count <= RESUME_OPTIMAL_MAX_WORDS


# ============================================================
# SCORE HELPERS
# ============================================================

def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Clamp a numeric value between min_val and max_val.

    Args:
        value:   Input number.
        min_val: Minimum bound (default 0).
        max_val: Maximum bound (default 100).

    Returns:
        Clamped float value.
    """
    return max(min_val, min(max_val, value))


def get_score_label(score: float) -> tuple[str, str]:
    """
    Return a human-readable label and emoji for an ATS score.

    Args:
        score: Numeric ATS score (0–100).

    Returns:
        Tuple of (label, emoji), e.g., ('Excellent', '🏆').
    """
    if score >= ATS_SCORE_EXCELLENT:
        return "Excellent", "🏆"
    elif score >= ATS_SCORE_GOOD:
        return "Good", "✅"
    elif score >= ATS_SCORE_AVERAGE:
        return "Average", "⚠️"
    return "Poor", "❌"


def get_score_color(score: float) -> str:
    """
    Return the hex color string associated with an ATS score level.

    Args:
        score: Numeric ATS score (0–100).

    Returns:
        Hex color string.
    """
    if score >= ATS_SCORE_EXCELLENT:
        return COLORS["excellent"]
    elif score >= ATS_SCORE_GOOD:
        return COLORS["good"]
    elif score >= ATS_SCORE_AVERAGE:
        return COLORS["average"]
    return COLORS["poor"]


def percentage(part: int | float, total: int | float, decimals: int = 1) -> float:
    """
    Safely compute a percentage, returning 0.0 if total is zero.

    Args:
        part:     Numerator value.
        total:    Denominator value.
        decimals: Decimal places to round to.

    Returns:
        Percentage float.
    """
    if not total:
        return 0.0
    return round((part / total) * 100, decimals)


# ============================================================
# FILE HELPERS
# ============================================================

def get_file_extension(filename: str) -> str:
    """
    Return the lower-cased file extension without the leading dot.

    Args:
        filename: File name or full path string.

    Returns:
        Extension string (e.g., 'pdf', 'docx').
    """
    return Path(filename).suffix.lstrip(".").lower()


def get_file_size_mb(file_bytes: bytes) -> float:
    """
    Convert raw bytes to megabytes, rounded to 2 decimal places.

    Args:
        file_bytes: File content as bytes.

    Returns:
        File size in MB.
    """
    return round(len(file_bytes) / (1024 * 1024), 2)


def generate_filename(candidate_name: str, suffix: str = "report") -> str:
    """
    Generate a safe, timestamped filename for the PDF report.

    Args:
        candidate_name: Name of the candidate (used as prefix).
        suffix:         File suffix label (default 'report').

    Returns:
        Filename string, e.g., 'john_doe_report_20240101_153045.pdf'.
    """
    safe_name = re.sub(r"[^\w\s]", "", candidate_name or "candidate")
    safe_name = safe_name.strip().replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{suffix}_{timestamp}.pdf"


def get_report_path(filename: str) -> Path:
    """
    Resolve the full path to a report file inside the reports/ directory.

    Args:
        filename: Report filename (without directory).

    Returns:
        Full Path object pointing to reports/<filename>.
    """
    return REPORTS_DIR / filename


def hash_file(file_bytes: bytes) -> str:
    """
    Compute an MD5 hash of file content for deduplication / caching.

    Args:
        file_bytes: File content as bytes.

    Returns:
        MD5 hex-digest string.
    """
    return hashlib.md5(file_bytes).hexdigest()


# ============================================================
# LIST / DICT HELPERS
# ============================================================

def deduplicate(items: list[Any]) -> list[Any]:
    """
    Remove duplicates from a list while preserving insertion order.

    Args:
        items: Input list (items must be hashable).

    Returns:
        De-duplicated list.
    """
    seen: set = set()
    result: list = []
    for item in items:
        key = item.lower() if isinstance(item, str) else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def flatten(nested: list[list[Any]]) -> list[Any]:
    """
    Flatten a list of lists into a single flat list.

    Args:
        nested: List of lists.

    Returns:
        Flat list.
    """
    return [item for sublist in nested for item in sublist]


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested dictionary value using a chain of keys.

    Args:
        d:       Input dictionary.
        *keys:   Sequence of keys forming the path.
        default: Value returned if any key is missing.

    Returns:
        Retrieved value or default.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries into one (later dicts override earlier).

    Args:
        *dicts: Dictionaries to merge.

    Returns:
        Merged dictionary.
    """
    result: dict = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


# ============================================================
# JSON HELPERS
# ============================================================

def save_json(data: dict | list, path: str | Path) -> bool:
    """
    Save a dictionary or list to a JSON file with pretty formatting.

    Args:
        data: Serialisable Python object.
        path: Output file path.

    Returns:
        True on success, False on failure.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.debug("Saved JSON to %s", path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Failed to save JSON to %s: %s", path, exc)
        return False


def load_json(path: str | Path) -> dict | list | None:
    """
    Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object, or None if loading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load JSON from %s: %s", path, exc)
        return None


# ============================================================
# VALIDATION HELPERS
# ============================================================

def is_valid_email(email: str) -> bool:
    """
    Validate an email address format.

    Args:
        email: Email string to validate.

    Returns:
        True if valid format, False otherwise.
    """
    if not email:
        return False
    return bool(re.fullmatch(REGEX_EMAIL, email.strip(), re.IGNORECASE))


def is_valid_url(url: str) -> bool:
    """
    Validate a URL format (http/https only).

    Args:
        url: URL string to validate.

    Returns:
        True if valid URL format, False otherwise.
    """
    if not url:
        return False
    pattern = re.compile(
        r"^https?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
        r"localhost|"
        r"\d{1,3}(?:\.\d{1,3}){3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return bool(pattern.match(url.strip()))


def is_empty_text(text: str | None) -> bool:
    """
    Return True if text is None, empty, or only whitespace.

    Args:
        text: Input string.

    Returns:
        Boolean.
    """
    return not text or not text.strip()


def validate_file_size(file_bytes: bytes, max_mb: int = 10) -> tuple[bool, str]:
    """
    Validate that a file does not exceed the maximum allowed size.

    Args:
        file_bytes: File content as bytes.
        max_mb:     Maximum allowed size in megabytes.

    Returns:
        Tuple of (is_valid: bool, message: str).
    """
    size_mb = get_file_size_mb(file_bytes)
    if size_mb > max_mb:
        return False, f"File size {size_mb} MB exceeds the {max_mb} MB limit."
    return True, f"File size {size_mb} MB is within the allowed limit."


# ============================================================
# FORMAT HELPERS
# ============================================================

def format_score(score: float) -> str:
    """
    Format a score float as a clean integer string (e.g., '87').

    Args:
        score: Numeric score.

    Returns:
        String representation of the rounded integer score.
    """
    return str(int(round(score)))


def format_percentage(value: float) -> str:
    """
    Format a float as a percentage string (e.g., '87.5%').

    Args:
        value: Float value (0–100).

    Returns:
        Formatted percentage string.
    """
    return f"{value:.1f}%"


def format_list_as_string(items: list[str], separator: str = ", ") -> str:
    """
    Join a list of strings into a single readable string.

    Args:
        items:     List of strings.
        separator: Delimiter between items.

    Returns:
        Joined string.
    """
    if not items:
        return "None"
    return separator.join(items)


def format_candidate_name(name: str) -> str:
    """
    Title-case a candidate name and strip extra whitespace.

    Args:
        name: Raw name string.

    Returns:
        Title-cased, cleaned name string.
    """
    if not name:
        return "Unknown Candidate"
    return " ".join(word.capitalize() for word in name.strip().split())


def pluralize(word: str, count: int) -> str:
    """
    Return singular or plural form of a word based on count.

    Args:
        word:  Base word (e.g., 'skill').
        count: Numeric count.

    Returns:
        Pluralized string (e.g., '1 skill', '3 skills').
    """
    if count == 1:
        return f"{count} {word}"
    return f"{count} {word}s"


# ============================================================
# STREAMLIT DISPLAY HELPERS
# ============================================================

def build_metric_delta(current: float, benchmark: float) -> tuple[float, str]:
    """
    Compute the delta and direction string for a Streamlit metric card.

    Args:
        current:   Current score value.
        benchmark: Baseline or benchmark value to compare against.

    Returns:
        Tuple of (delta_value, delta_color) where delta_color is
        'normal', 'inverse', or 'off'.
    """
    delta = round(current - benchmark, 1)
    return delta, "normal"


def skills_to_badge_html(skills: list[str], color: str = "#4F8BF9") -> str:
    """
    Convert a list of skill strings into styled HTML badge elements
    suitable for Streamlit's st.markdown(unsafe_allow_html=True).

    Args:
        skills: List of skill name strings.
        color:  Background hex color for the badges.

    Returns:
        HTML string containing one badge per skill.
    """
    if not skills:
        return "<p>None found</p>"
    badges = "".join(
        f'<span style="'
        f'background-color:{color};'
        f'color:white;'
        f'padding:3px 10px;'
        f'border-radius:12px;'
        f'margin:3px;'
        f'display:inline-block;'
        f'font-size:13px;'
        f'font-weight:500;'
        f'">{skill}</span>'
        for skill in skills
    )
    return f'<div style="line-height:2.2;">{badges}</div>'


def section_divider() -> str:
    """
    Return an HTML horizontal rule string for Streamlit markdown.

    Returns:
        HTML <hr> string with custom styling.
    """
    return (
        '<hr style="border:none;'
        'border-top:1px solid #2D3748;'
        'margin:20px 0;">'
    )

# utils/jd_parser.py
"""
jd_parser.py — Job Description parser for AI Resume Analyzer.

Extracts structured information from raw job description text including:
  - Job title, company, location
  - Required and preferred skills
  - Experience requirements
  - Education requirements
  - Responsibilities
  - Keywords for ATS matching
  - Salary range (if present)
  - Employment type (full-time, remote, etc.)

Design:
  - Single JDParser class with one public method: parse()
  - Each field has its own private extractor method
  - Returns a fully typed Python dictionary
  - All extractors are defensive — missing data returns empty value
"""

import re
from typing import Any

from utils.helpers import (
    clean_whitespace,
    count_words,
    deduplicate,
    extract_years,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class JDParser:
    """
    Extracts structured information from raw job description text.

    Usage:
        parser = JDParser()
        data   = parser.parse(jd_text)

        print(data["job_title"])
        print(data["required_skills"])
        print(data["experience_years"])

    Returns a dictionary with keys:
        job_title, company, location, employment_type,
        required_skills, preferred_skills, all_skills,
        responsibilities, requirements, education,
        experience_years, experience_level, salary,
        keywords, word_count, is_remote
    """

    # ------------------------------------------------------------------
    # Section header keywords specific to job descriptions
    # ------------------------------------------------------------------
    _JD_SECTIONS: dict[str, list[str]] = {
        "responsibilities": [
            "responsibilities", "what you will do", "what you'll do",
            "role responsibilities", "key responsibilities", "duties",
            "your role", "the role", "about the role", "job duties",
            "day to day", "day-to-day", "what we expect",
        ],
        "requirements": [
            "requirements", "required qualifications", "what we need",
            "what you need", "must have", "must-have", "required skills",
            "basic qualifications", "minimum qualifications",
            "you must have", "you should have", "we require",
        ],
        "preferred": [
            "preferred", "nice to have", "nice-to-have", "bonus",
            "preferred qualifications", "plus", "good to have",
            "desired", "desirable", "advantageous", "optional",
            "would be great", "ideally",
        ],
        "education": [
            "education", "educational requirements", "academic",
            "qualifications", "degree", "academic background",
        ],
        "benefits": [
            "benefits", "what we offer", "perks", "compensation",
            "we offer", "our offer", "package", "salary",
        ],
        "about_company": [
            "about us", "about the company", "who we are",
            "company overview", "our company", "the company",
        ],
    }

    # Experience level keywords
    _EXP_LEVEL_KEYWORDS: dict[str, list[str]] = {
        "Entry Level":  ["entry level", "entry-level", "junior", "0-1", "0-2", "fresh"],
        "Mid Level":    ["mid level", "mid-level", "intermediate", "2-4", "3-5"],
        "Senior":       ["senior", "sr.", "lead", "5+", "5-7", "6+", "7+"],
        "Principal":    ["principal", "staff", "architect", "10+", "8+"],
        "Manager":      ["manager", "director", "head of", "vp", "vice president"],
    }

    # Employment type keywords
    _EMPLOYMENT_TYPES: list[str] = [
        "full-time", "full time", "part-time", "part time",
        "contract", "freelance", "internship", "temporary",
        "permanent", "remote", "hybrid", "on-site", "onsite",
    ]

    def __init__(self) -> None:
        logger.debug("JDParser initialised.")

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse raw job description text into structured fields.

        Args:
            text: Raw JD text (pasted or from .txt file).

        Returns:
            Dictionary containing all extracted JD information.
            All fields always present; missing data uses empty
            string, empty list, or 0 as appropriate.
        """
        if not text or not text.strip():
            logger.warning("JDParser received empty text.")
            return self._empty_result()

        logger.info("Parsing JD — %d characters", len(text))

        # Split into labelled sections first
        sections = self._split_sections(text)

        # Extract all skills from full text for maximum coverage
        all_skills = self._extract_all_skills(text, sections)
        required   = self._extract_required_skills(sections, text)
        preferred  = self._extract_preferred_skills(sections)

        result: dict[str, Any] = {
            "job_title":        self._extract_job_title(text, sections),
            "company":          self._extract_company(text, sections),
            "location":         self._extract_location(text),
            "employment_type":  self._extract_employment_type(text),
            "is_remote":        self._detect_remote(text),

            "required_skills":  required,
            "preferred_skills": preferred,
            "all_skills":       deduplicate(required + preferred + all_skills),

            "responsibilities": self._extract_responsibilities(sections),
            "requirements":     self._extract_requirements(sections),
            "education":        self._extract_education(sections, text),

            "experience_years": self._extract_experience_years(text),
            "experience_level": self._extract_experience_level(text),
            "salary":           self._extract_salary(text),

            "keywords":         self._extract_keywords(text),
            "word_count":       count_words(text),
            "sections_found":   list(sections.keys()),
        }

        logger.info(
            "JD parsed — title='%s', company='%s', "
            "required_skills=%d, all_skills=%d",
            result["job_title"],
            result["company"],
            len(result["required_skills"]),
            len(result["all_skills"]),
        )

        return result

    # ------------------------------------------------------------------
    # SECTION SPLITTER
    # ------------------------------------------------------------------

    def _split_sections(self, text: str) -> dict[str, str]:
        """
        Split JD text into labelled sections using keyword detection.

        Args:
            text: Full JD text.

        Returns:
            Dictionary mapping section_name → section_text.
        """
        lines = text.splitlines()
        sections: dict[str, str] = {}
        current_section = "header"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            detected = self._detect_section(stripped)

            if detected and len(stripped) < 80:
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = detected
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        logger.debug("JD sections detected: %s", list(sections.keys()))
        return sections

    def _detect_section(self, line: str) -> str | None:
        """
        Check if a line matches any known JD section header.

        Args:
            line: A single stripped line of text.

        Returns:
            Section name string if matched, None otherwise.
        """
        normalised = line.lower().strip(":-#*_ \t")
        for section_name, keywords in self._JD_SECTIONS.items():
            for keyword in keywords:
                if normalised == keyword or normalised.startswith(keyword):
                    return section_name
        return None

    # ------------------------------------------------------------------
    # JOB TITLE EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_job_title(
        self, text: str, sections: dict[str, str]
    ) -> str:
        """
        Extract the job title using multiple strategies.

        Strategy 1: Look for explicit "Job Title:" label.
        Strategy 2: Look for title-like line in header section.
        Strategy 3: Scan for common title patterns in first 10 lines.

        Args:
            text:     Full JD text.
            sections: Pre-split section dictionary.

        Returns:
            Job title string or empty string.
        """
        # Strategy 1: explicit label
        label_pattern = re.compile(
            r"(?:job\s+title|position|role|title)\s*[:\-]\s*(.+)",
            re.IGNORECASE,
        )
        match = label_pattern.search(text[:500])
        if match:
            return clean_whitespace(match.group(1).strip()[:80])

        # Strategy 2: header section first meaningful line
        header = sections.get("header", "")
        for line in header.splitlines():
            line = line.strip()
            if not line or len(line) < 3:
                continue
            # Skip lines that look like company/location info
            if re.search(r"[@|,]|http|www|\d{5}", line):
                continue
            words = line.split()
            if 1 <= len(words) <= 8:
                return clean_whitespace(line[:80])

        # Strategy 3: first 10 lines scan for title patterns
        title_keywords = [
            "engineer", "developer", "scientist", "analyst", "manager",
            "architect", "designer", "lead", "specialist", "consultant",
            "director", "officer", "head", "intern", "associate",
        ]
        for line in text.splitlines()[:10]:
            line = line.strip()
            if any(kw in line.lower() for kw in title_keywords):
                if len(line.split()) <= 8:
                    return clean_whitespace(line[:80])

        return ""

    # ------------------------------------------------------------------
    # COMPANY EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_company(
        self, text: str, sections: dict[str, str]
    ) -> str:
        """
        Extract the company name from the JD.

        Args:
            text:     Full JD text.
            sections: Pre-split section dictionary.

        Returns:
            Company name string or empty string.
        """
        # Explicit label
        label_pattern = re.compile(
            r"(?:company|employer|organization|organisation|firm)\s*[:\-]\s*(.+)",
            re.IGNORECASE,
        )
        match = label_pattern.search(text[:600])
        if match:
            return clean_whitespace(match.group(1).strip()[:80])

        # "About Us" / "About the Company" section first line
        about = sections.get("about_company", "")
        if about:
            first_line = about.splitlines()[0].strip()
            if first_line and len(first_line) < 80:
                return clean_whitespace(first_line)

        # Pattern: "at <Company>" or "join <Company>"
        at_pattern = re.compile(
            r"(?:at|join|joining|with)\s+([A-Z][A-Za-z0-9\s&.,'-]{2,40})"
            r"(?:\s+(?:as|to|for|is|are|we|our)|\.|,|$)",
            re.MULTILINE,
        )
        match = at_pattern.search(text[:800])
        if match:
            return clean_whitespace(match.group(1).strip())

        return ""

    # ------------------------------------------------------------------
    # LOCATION EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_location(self, text: str) -> str:
        """
        Extract job location from the JD text.

        Args:
            text: Full JD text.

        Returns:
            Location string or empty string.
        """
        label_pattern = re.compile(
            r"(?:location|based in|office|city|country)\s*[:\-]\s*(.+)",
            re.IGNORECASE,
        )
        match = label_pattern.search(text[:600])
        if match:
            loc = clean_whitespace(match.group(1).strip()[:80])
            # Remove trailing noise
            loc = re.split(r"[|\n\r]", loc)[0].strip()
            return loc

        # Common city/country patterns
        city_pattern = re.compile(
            r"\b(?:New York|San Francisco|London|Berlin|Toronto|Sydney|"
            r"Austin|Seattle|Boston|Chicago|Los Angeles|Remote|Hybrid|"
            r"Singapore|Dubai|Mumbai|Bangalore|Hyderabad|Delhi)\b",
            re.IGNORECASE,
        )
        match = city_pattern.search(text[:500])
        if match:
            return match.group(0)

        return ""

    # ------------------------------------------------------------------
    # EMPLOYMENT TYPE EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_employment_type(self, text: str) -> str:
        """
        Extract employment type (full-time, remote, contract, etc.).

        Args:
            text: Full JD text.

        Returns:
            Employment type string or empty string.
        """
        found: list[str] = []
        text_lower = text.lower()

        for emp_type in self._EMPLOYMENT_TYPES:
            if emp_type in text_lower:
                found.append(emp_type.title())

        return ", ".join(deduplicate(found)) if found else ""

    def _detect_remote(self, text: str) -> bool:
        """
        Detect whether the job is remote or hybrid.

        Args:
            text: Full JD text.

        Returns:
            True if remote/hybrid indicators found.
        """
        remote_keywords = ["remote", "work from home", "wfh", "hybrid", "distributed"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in remote_keywords)

    # ------------------------------------------------------------------
    # SKILLS EXTRACTORS
    # ------------------------------------------------------------------

    def _extract_all_skills(
        self, text: str, sections: dict[str, str]
    ) -> list[str]:
        """
        Extract all skills mentioned anywhere in the JD.

        Combines skills from requirements, preferred, and full text
        using comma/bullet/newline parsing.

        Args:
            text:     Full JD text.
            sections: Pre-split section dictionary.

        Returns:
            Deduplicated list of skill strings.
        """
        skills: list[str] = []

        # Scan requirements and preferred sections
        for section_key in ("requirements", "preferred", "header"):
            section_text = sections.get(section_key, "")
            if section_text:
                skills.extend(self._parse_skill_list(section_text))

        # Also scan full text for inline skill mentions
        skills.extend(self._parse_skill_list(text))

        return deduplicate(skills)[:60]

    def _extract_required_skills(
        self, sections: dict[str, str], full_text: str
    ) -> list[str]:
        """
        Extract skills from the required/must-have section.

        Args:
            sections:  Pre-split section dictionary.
            full_text: Full JD text for fallback.

        Returns:
            List of required skill strings.
        """
        req_text = sections.get("requirements", "")
        if not req_text:
            # Fallback: scan full text for "required:" patterns
            req_pattern = re.compile(
                r"(?:required|must have|must-have)[:\s]+([^\n]+)",
                re.IGNORECASE,
            )
            matches = req_pattern.findall(full_text)
            req_text = "\n".join(matches)

        return deduplicate(self._parse_skill_list(req_text))[:40]

    def _extract_preferred_skills(
        self, sections: dict[str, str]
    ) -> list[str]:
        """
        Extract skills from the preferred/nice-to-have section.

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of preferred skill strings.
        """
        pref_text = sections.get("preferred", "")
        if not pref_text:
            return []
        return deduplicate(self._parse_skill_list(pref_text))[:30]

    def _parse_skill_list(self, text: str) -> list[str]:
        """
        Parse a block of text into individual skill tokens.

        Handles comma, pipe, slash, bullet, and newline separators.
        Filters out non-skill noise (very long phrases, pure numbers).

        Args:
            text: Text block to parse.

        Returns:
            List of individual skill strings.
        """
        if not text:
            return []

        # Replace common separators with commas
        text = re.sub(r"[•\-\*►▪▸●]", ",", text)
        text = re.sub(r"[|\n\r]", ",", text)
        text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)

        raw = text.split(",")
        skills: list[str] = []

        for item in raw:
            # Strip bullets, numbers, punctuation
            item = re.sub(r"^[\s\d.)\-•*►]+", "", item)
            item = clean_whitespace(item.strip(".,;:- \t()"))

            # Keep items that look like skills (2–40 chars, not pure digits)
            if 2 <= len(item) <= 40 and not item.isdigit():
                # Skip generic filler phrases
                filler = {
                    "experience", "years", "knowledge", "understanding",
                    "familiarity", "proficiency", "ability", "skills",
                    "strong", "excellent", "good", "required", "preferred",
                    "must", "have", "with", "using", "the", "and", "or",
                    "in", "of", "a", "an", "to", "for", "is", "are",
                }
                if item.lower() not in filler:
                    skills.append(item)

        return skills

    # ------------------------------------------------------------------
    # RESPONSIBILITIES EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_responsibilities(
        self, sections: dict[str, str]
    ) -> list[str]:
        """
        Extract job responsibilities as a list of bullet points.

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of responsibility strings.
        """
        resp_text = sections.get("responsibilities", "")
        if not resp_text:
            return []

        items: list[str] = []
        for line in resp_text.splitlines():
            line = re.sub(r"^[\s\-•*►▪▸●\d.)\s]+", "", line).strip()
            if len(line) > 10:
                items.append(clean_whitespace(line[:200]))

        return items[:20]  # Cap at 20 bullets

    # ------------------------------------------------------------------
    # REQUIREMENTS EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_requirements(
        self, sections: dict[str, str]
    ) -> list[str]:
        """
        Extract job requirements as a list of bullet points.

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of requirement strings.
        """
        req_text = sections.get("requirements", "")
        if not req_text:
            return []

        items: list[str] = []
        for line in req_text.splitlines():
            line = re.sub(r"^[\s\-•*►▪▸●\d.)\s]+", "", line).strip()
            if len(line) > 10:
                items.append(clean_whitespace(line[:200]))

        return items[:20]

    # ------------------------------------------------------------------
    # EDUCATION EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_education(
        self, sections: dict[str, str], text: str
    ) -> str:
        """
        Extract education requirements from the JD.

        Args:
            sections: Pre-split section dictionary.
            text:     Full JD text for fallback.

        Returns:
            Education requirement string.
        """
        edu_text = sections.get("education", "")
        if edu_text:
            lines = [
                l.strip() for l in edu_text.splitlines()
                if l.strip() and len(l.strip()) > 5
            ]
            if lines:
                return clean_whitespace(" ".join(lines[:3])[:200])

        # Fallback: scan for degree keywords in full text
        degree_pattern = re.compile(
            r"(?:bachelor|master|phd|ph\.d|degree|b\.s\.|m\.s\.|b\.tech|m\.tech)"
            r"[^\n]{0,100}",
            re.IGNORECASE,
        )
        match = degree_pattern.search(text)
        if match:
            return clean_whitespace(match.group(0)[:150])

        return ""

    # ------------------------------------------------------------------
    # EXPERIENCE EXTRACTORS
    # ------------------------------------------------------------------

    def _extract_experience_years(self, text: str) -> float:
        """
        Extract the minimum years of experience required.

        Looks for patterns like:
          - "5+ years", "3-5 years", "minimum 4 years"
          - "at least 3 years", "5 years of experience"

        Args:
            text: Full JD text.

        Returns:
            Minimum required years as float, or 0.0 if not found.
        """
        patterns = [
            r"(\d+)\+\s*years?",
            r"(\d+)\s*[-–]\s*\d+\s*years?",
            r"(?:minimum|at least|min\.?)\s*(\d+)\s*years?",
            r"(\d+)\s*years?\s*(?:of\s+)?(?:experience|exp)",
            r"(\d+)\s*\+\s*years?\s*(?:of\s+)?(?:experience|exp)",
        ]

        found_years: list[int] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                try:
                    yr = int(m)
                    if 0 < yr <= 30:  # Sanity check
                        found_years.append(yr)
                except (ValueError, TypeError):
                    continue

        return float(min(found_years)) if found_years else 0.0

    def _extract_experience_level(self, text: str) -> str:
        """
        Detect the seniority level required for the role.

        Args:
            text: Full JD text.

        Returns:
            Experience level string (e.g., 'Senior', 'Mid Level').
        """
        text_lower = text.lower()
        for level, keywords in self._EXP_LEVEL_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return level
        return "Not Specified"

    # ------------------------------------------------------------------
    # SALARY EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_salary(self, text: str) -> str:
        """
        Extract salary range from the JD if mentioned.

        Handles formats like:
          - "$120,000 - $160,000"
          - "£80k - £100k"
          - "$130k/year"
          - "USD 100,000"

        Args:
            text: Full JD text.

        Returns:
            Salary range string or empty string.
        """
        salary_pattern = re.compile(
            r"(?:USD|GBP|EUR|INR|CAD|AUD)?\s*"
            r"[\$£€₹]?\s*"
            r"\d{2,3}[,.]?\d{0,3}\s*[kK]?"
            r"\s*(?:[-–—]|to)\s*"
            r"[\$£€₹]?\s*"
            r"\d{2,3}[,.]?\d{0,3}\s*[kK]?"
            r"(?:\s*(?:per year|/year|annually|pa|p\.a\.))?",
            re.IGNORECASE,
        )
        match = salary_pattern.search(text)
        if match:
            return clean_whitespace(match.group(0))

        # Single salary mention
        single_pattern = re.compile(
            r"[\$£€₹]\s*\d{2,3}[,.]?\d{0,3}\s*[kK]?"
            r"(?:\s*(?:per year|/year|annually))?",
            re.IGNORECASE,
        )
        match = single_pattern.search(text)
        if match:
            return clean_whitespace(match.group(0))

        return ""

    # ------------------------------------------------------------------
    # KEYWORD EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract the most important keywords from the JD for ATS matching.

        Uses frequency-based ranking after removing stop words.
        Includes both single words and common tech bigrams.

        Args:
            text: Full JD text.

        Returns:
            List of top keyword strings (most frequent first).
        """
        # Simple stop words for JD context
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "our",
            "your", "their", "we", "you", "they", "it", "this", "that",
            "these", "those", "as", "if", "not", "no", "so", "up",
            "out", "about", "into", "through", "during", "including",
            "also", "both", "each", "more", "most", "other", "some",
            "such", "than", "then", "there", "when", "where", "which",
            "while", "who", "whom", "how", "all", "any", "few", "own",
            "same", "too", "very", "just", "because", "well", "new",
            "work", "working", "team", "role", "position", "job",
            "company", "candidate", "looking", "seeking", "join",
            "opportunity", "strong", "excellent", "good", "great",
            "experience", "years", "ability", "skills", "knowledge",
        }

        # Tokenise and count
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./\-]{1,30}\b", text)
        freq: dict[str, int] = {}
        for word in words:
            w = word.lower()
            if w not in stop_words and len(w) >= 2:
                freq[w] = freq.get(w, 0) + 1

        # Sort by frequency
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:40]]

    # ------------------------------------------------------------------
    # EMPTY RESULT
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return a fully structured empty result dictionary."""
        return {
            "job_title":        "",
            "company":          "",
            "location":         "",
            "employment_type":  "",
            "is_remote":        False,
            "required_skills":  [],
            "preferred_skills": [],
            "all_skills":       [],
            "responsibilities": [],
            "requirements":     [],
            "education":        "",
            "experience_years": 0.0,
            "experience_level": "Not Specified",
            "salary":           "",
            "keywords":         [],
            "word_count":       0,
            "sections_found":   [],
        }

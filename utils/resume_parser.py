# utils/resume_parser.py
"""
resume_parser.py — Structured information extraction from resume text.

Extracts all candidate details from raw resume text using a combination
of regex patterns, spaCy NER, and heuristic section parsing.

Extracted fields:
  - Name, Email, Phone, LinkedIn, GitHub, Portfolio
  - Skills, Education, Experience, Projects
  - Certifications, Languages, Tools, Summary

Design:
  - Single ResumeParser class with one public method: parse()
  - Each field has its own private extractor method
  - All extractors are defensive — missing data returns empty value
  - Returns a fully typed Python dictionary
"""

import re
from typing import Any

from utils.constants import (
    RESUME_SECTIONS,
    DEGREE_KEYWORDS,
    REGEX_EMAIL,
    REGEX_PHONE,
    REGEX_LINKEDIN,
    REGEX_GITHUB,
    REGEX_PORTFOLIO,
    REGEX_DATE_RANGE,
    REGEX_GPA,
)
from utils.helpers import (
    extract_emails,
    extract_phones,
    extract_linkedin,
    extract_github,
    extract_portfolio,
    extract_years,
    calculate_experience_years,
    clean_whitespace,
    count_words,
    deduplicate,
    format_candidate_name,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ResumeParser:
    """
    Extracts structured candidate information from raw resume text.

    Usage:
        parser = ResumeParser()
        data   = parser.parse(resume_text)

        # Access fields
        print(data["name"])
        print(data["email"])
        print(data["skills"])

    Returns a dictionary with the following keys:
        name, email, phone, linkedin, github, portfolio,
        summary, skills, education, experience, projects,
        certifications, languages, tools, word_count,
        experience_years, sections_found
    """

    def __init__(self) -> None:
        # Load spaCy lazily to avoid import-time cost
        self._nlp = None
        logger.debug("ResumeParser initialised.")

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse raw resume text and extract all structured fields.

        Args:
            text: Raw resume text from PDFReader or DOCXReader.

        Returns:
            Dictionary containing all extracted candidate information.
            All fields are always present; missing data uses empty
            string, empty list, or 0 as appropriate.
        """
        if not text or not text.strip():
            logger.warning("ResumeParser received empty text.")
            return self._empty_result()

        logger.info("Parsing resume — %d characters", len(text))

        # Split text into labelled sections first — most extractors
        # work better when they operate on the relevant section only
        sections = self._split_sections(text)

        result: dict[str, Any] = {
            # Contact info
            "name":             self._extract_name(text, sections),
            "email":            self._extract_email(text),
            "phone":            self._extract_phone(text),
            "linkedin":         self._extract_linkedin(text),
            "github":           self._extract_github(text),
            "portfolio":        self._extract_portfolio(text),

            # Content sections
            "summary":          self._extract_summary(sections),
            "skills":           self._extract_skills(sections, text),
            "education":        self._extract_education(sections),
            "experience":       self._extract_experience(sections),
            "projects":         self._extract_projects(sections),
            "certifications":   self._extract_certifications(sections),
            "languages":        self._extract_languages(sections),
            "tools":            self._extract_tools(sections, text),

            # Metadata
            "word_count":       count_words(text),
            "experience_years": self._calc_experience_years(text, sections),
            "sections_found":   list(sections.keys()),
        }

        logger.info(
            "Resume parsed — name='%s', email='%s', skills=%d, "
            "experience=%d entries, education=%d entries",
            result["name"],
            result["email"],
            len(result["skills"]),
            len(result["experience"]),
            len(result["education"]),
        )

        return result

    # ------------------------------------------------------------------
    # SECTION SPLITTER
    # ------------------------------------------------------------------

    def _split_sections(self, text: str) -> dict[str, str]:
        """
        Split resume text into labelled sections using keyword detection.

        Scans each line for known section header keywords defined in
        RESUME_SECTIONS constant. When a header is found, all following
        lines are collected under that section label until the next
        header is detected.

        Args:
            text: Full resume text.

        Returns:
            Dictionary mapping section_name → section_text.
            Always includes a 'header' key for the top portion of the
            resume (before the first detected section).
        """
        lines = text.splitlines()
        sections: dict[str, str] = {}
        current_section = "header"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            detected = self._detect_section(stripped)

            if detected and len(stripped) < 60:
                # Save previous section
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = detected
                current_lines = []
            else:
                current_lines.append(line)

        # Save the last section
        if current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        logger.debug("Sections detected: %s", list(sections.keys()))
        return sections

    def _detect_section(self, line: str) -> str | None:
        """
        Check if a line matches any known section header keyword.

        Args:
            line: A single line of text (already stripped).

        Returns:
            Section name string if matched, None otherwise.
        """
        normalised = line.lower().strip(":-#*_ \t")
        for section_name, keywords in RESUME_SECTIONS.items():
            for keyword in keywords:
                if normalised == keyword or normalised.startswith(keyword):
                    return section_name
        return None

    # ------------------------------------------------------------------
    # CONTACT EXTRACTORS
    # ------------------------------------------------------------------

    def _extract_name(self, text: str, sections: dict[str, str]) -> str:
        """
        Extract candidate name using multiple strategies:

        Strategy 1 — Header section heuristic:
            The first non-empty, non-contact line in the header section
            is very likely the candidate's name.

        Strategy 2 — spaCy PERSON entity:
            Use NER to find PERSON entities near the top of the resume.

        Strategy 3 — Capitalised line heuristic:
            Find the first short line (2–4 words) that is title-cased
            and contains no digits or special characters.

        Args:
            text:     Full resume text.
            sections: Pre-split section dictionary.

        Returns:
            Candidate name string, or 'Unknown Candidate' if not found.
        """
        # Strategy 1: header section first non-contact line
        header_text = sections.get("header", "")
        if header_text:
            for line in header_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Skip lines that look like contact info
                if re.search(r"[@|/\\]|http|www|\d{5,}", line):
                    continue
                # Skip lines that are too long (likely a summary)
                words = line.split()
                if 1 < len(words) <= 5:
                    # Check it looks like a name (mostly alpha, title case)
                    if all(w[0].isupper() for w in words if w.isalpha()):
                        return format_candidate_name(line)

        # Strategy 2: spaCy NER on first 500 chars
        nlp = self._get_nlp()
        if nlp is not None:
            try:
                doc = nlp(text[:500])
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        name = ent.text.strip()
                        words = name.split()
                        if 1 < len(words) <= 4:
                            return format_candidate_name(name)
            except Exception as exc:
                logger.warning("spaCy NER name extraction failed: %s", exc)

        # Strategy 3: first title-cased short line in full text
        for line in text.splitlines()[:20]:
            line = line.strip()
            if not line or re.search(r"[@\d|/\\]", line):
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(
                w[0].isupper() for w in words if w.isalpha()
            ):
                return format_candidate_name(line)

        return "Unknown Candidate"

    def _extract_email(self, text: str) -> str:
        """Extract the first email address found in the resume."""
        emails = extract_emails(text)
        return emails[0] if emails else ""

    def _extract_phone(self, text: str) -> str:
        """Extract the first phone number found in the resume."""
        phones = extract_phones(text)
        return phones[0] if phones else ""

    def _extract_linkedin(self, text: str) -> str:
        """Extract LinkedIn profile URL."""
        return extract_linkedin(text)

    def _extract_github(self, text: str) -> str:
        """Extract GitHub profile URL."""
        return extract_github(text)

    def _extract_portfolio(self, text: str) -> str:
        """Extract portfolio / personal website URL."""
        return extract_portfolio(text)

    # ------------------------------------------------------------------
    # SUMMARY EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_summary(self, sections: dict[str, str]) -> str:
        """
        Extract the professional summary / objective section.

        Args:
            sections: Pre-split section dictionary.

        Returns:
            Summary text string, or empty string if not found.
        """
        summary_text = sections.get("summary", "")
        if not summary_text:
            return ""

        # Clean up and return first 500 chars max
        lines = [l.strip() for l in summary_text.splitlines() if l.strip()]
        summary = " ".join(lines)
        return clean_whitespace(summary[:600])

    # ------------------------------------------------------------------
    # SKILLS EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_skills(
        self, sections: dict[str, str], full_text: str
    ) -> list[str]:
        """
        Extract skills from the skills section and full text.

        Combines:
          1. Skills section text (comma/pipe/newline separated)
          2. Inline skill mentions throughout the resume

        Args:
            sections:  Pre-split section dictionary.
            full_text: Full resume text for fallback scanning.

        Returns:
            Deduplicated list of skill strings.
        """
        skills: list[str] = []

        # Primary: skills section
        skills_text = sections.get("skills", "")
        if skills_text:
            skills.extend(self._parse_skill_list(skills_text))

        # Fallback: scan full text if skills section is empty
        if not skills:
            skills.extend(self._parse_skill_list(full_text))

        return deduplicate(skills)[:50]  # Cap at 50 to avoid noise

    def _parse_skill_list(self, text: str) -> list[str]:
        """
        Parse a block of text into individual skill tokens.

        Handles comma, pipe, slash, bullet, and newline separators.

        Args:
            text: Skills section text.

        Returns:
            List of individual skill strings.
        """
        # Replace common separators with commas
        text = re.sub(r"[•\-\*►▪▸●]", ",", text)
        text = re.sub(r"[|\n\r/]", ",", text)

        # Split and clean
        raw_skills = text.split(",")
        skills: list[str] = []

        for skill in raw_skills:
            skill = clean_whitespace(skill.strip(".,;:- \t"))
            # Keep skills that are 2–40 chars and not pure numbers
            if 2 <= len(skill) <= 40 and not skill.isdigit():
                skills.append(skill)

        return skills

    # ------------------------------------------------------------------
    # EDUCATION EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_education(self, sections: dict[str, str]) -> list[dict]:
        """
        Extract education entries from the education section.

        Each entry attempts to capture:
          - degree:      Degree level (Bachelor's, Master's, PhD, etc.)
          - field:       Field of study
          - institution: University / college name
          - year:        Graduation year
          - gpa:         GPA if mentioned

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of education entry dictionaries.
        """
        edu_text = sections.get("education", "")
        if not edu_text:
            return []

        entries: list[dict] = []
        # Split into blocks by blank lines or year patterns
        blocks = re.split(r"\n{2,}", edu_text.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            entry = self._parse_education_block(block)
            if entry:
                entries.append(entry)

        # If block splitting found nothing, treat whole section as one entry
        if not entries and edu_text.strip():
            entry = self._parse_education_block(edu_text)
            if entry:
                entries.append(entry)

        return entries

    def _parse_education_block(self, text: str) -> dict | None:
        """
        Parse a single education block into a structured dictionary.

        Args:
            text: A block of text representing one education entry.

        Returns:
            Education entry dict or None if block seems invalid.
        """
        if not text.strip():
            return None

        entry: dict[str, str] = {
            "degree":      "",
            "field":       "",
            "institution": "",
            "year":        "",
            "gpa":         "",
            "raw":         clean_whitespace(text[:200]),
        }

        text_lower = text.lower()

        # Detect degree level
        for degree_label, keywords in DEGREE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    entry["degree"] = degree_label
                    break
            if entry["degree"]:
                break

        # Extract year (last 4-digit year found)
        years = extract_years(text)
        if years:
            entry["year"] = str(max(years))

        # Extract GPA
        gpa_match = re.search(REGEX_GPA, text, re.IGNORECASE)
        if gpa_match:
            gpa_val = gpa_match.group(1)
            gpa_max = gpa_match.group(2) or "4.0"
            entry["gpa"] = f"{gpa_val}/{gpa_max}"

        # Extract institution — look for University/College/Institute keywords
        inst_pattern = re.compile(
            r"([A-Z][A-Za-z\s&,.'()-]{2,40}?"
            r"(?:University|College|Institute|School|Academy))"
            r"|\b(MIT|IIT|IIM|NIT|BITS|UCLA|NYU|CMU|Stanford|Harvard|Oxford|Cambridge)\b",
            re.IGNORECASE,
        )
        inst_match = inst_pattern.search(text)
        if inst_match:
            entry["institution"] = clean_whitespace(
                inst_match.group(1) or inst_match.group(2) or ""
            )

        # Extract field of study — text after degree keyword
        field_pattern = re.compile(
            r"(?:bachelor|master|b\.?s\.?|m\.?s\.?|b\.?tech|m\.?tech|"
            r"b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc|phd|ph\.?d)"
            r"[\s.,]*(?:of|in|-)?\s*([A-Za-z\s&]{3,50})",
            re.IGNORECASE,
        )
        field_match = field_pattern.search(text)
        if field_match:
            field = clean_whitespace(field_match.group(1))
            # Avoid capturing institution names as field
            if len(field.split()) <= 6:
                entry["field"] = field

        return entry

    # ------------------------------------------------------------------
    # EXPERIENCE EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_experience(self, sections: dict[str, str]) -> list[dict]:
        """
        Extract work experience entries from the experience section.

        Each entry attempts to capture:
          - title:       Job title
          - company:     Company name
          - duration:    Date range string
          - description: Bullet points / responsibilities
          - years:       Numeric years in this role

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of experience entry dictionaries.
        """
        exp_text = sections.get("experience", "")
        if not exp_text:
            return []

        entries: list[dict] = []

        # Split on date range patterns or double newlines
        # Each job entry typically starts with a title/company line
        blocks = self._split_experience_blocks(exp_text)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue
            entry = self._parse_experience_block(block)
            if entry:
                entries.append(entry)

        return entries

    def _split_experience_blocks(self, text: str) -> list[str]:
        """
        Split experience section into individual job blocks.

        Uses date range patterns and blank lines as delimiters.

        Args:
            text: Experience section text.

        Returns:
            List of text blocks, one per job entry.
        """
        # Split on blank lines first
        blocks = re.split(r"\n{2,}", text.strip())

        # If only one block, try splitting on date range lines
        if len(blocks) <= 1:
            lines = text.splitlines()
            current: list[str] = []
            result: list[str] = []
            date_re = re.compile(REGEX_DATE_RANGE, re.IGNORECASE)

            for line in lines:
                if date_re.search(line) and current:
                    result.append("\n".join(current))
                    current = [line]
                else:
                    current.append(line)

            if current:
                result.append("\n".join(current))

            return result if result else blocks

        return blocks

    def _parse_experience_block(self, text: str) -> dict | None:
        """
        Parse a single experience block into a structured dictionary.

        Args:
            text: A block of text representing one job entry.

        Returns:
            Experience entry dict or None if block seems invalid.
        """
        if not text.strip():
            return None

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return None

        entry: dict[str, Any] = {
            "title":       "",
            "company":     "",
            "duration":    "",
            "description": [],
            "years":       0.0,
            "raw":         clean_whitespace(text[:300]),
        }

        # Extract date range
        date_re = re.compile(REGEX_DATE_RANGE, re.IGNORECASE)
        for line in lines:
            date_match = date_re.search(line)
            if date_match:
                entry["duration"] = clean_whitespace(date_match.group(0))
                # Estimate years from date range
                years_found = extract_years(entry["duration"])
                if len(years_found) >= 2:
                    entry["years"] = float(years_found[-1] - years_found[0])
                elif len(years_found) == 1:
                    entry["years"] = 1.0
                break

        # First line is usually "Title — Company" or "Title at Company"
        first_line = lines[0]
        separators = [" at ", " - ", " – ", " — ", " | ", ", "]
        for sep in separators:
            if sep.lower() in first_line.lower():
                parts = re.split(re.escape(sep), first_line, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    entry["title"]   = clean_whitespace(parts[0])
                    # Strip date from company name
                    company = date_re.sub("", parts[1]).strip(" -()")
                    entry["company"] = clean_whitespace(company)
                    break

        # If no separator found, first line is the title
        if not entry["title"]:
            entry["title"] = clean_whitespace(first_line[:80])

        # Remaining lines are description bullets
        bullet_re = re.compile(r"^[-•*►▪▸●>]?\s*(.+)$")
        for line in lines[1:]:
            match = bullet_re.match(line)
            if match:
                desc = clean_whitespace(match.group(1))
                if len(desc) > 10:
                    entry["description"].append(desc)

        return entry

    # ------------------------------------------------------------------
    # PROJECTS EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_projects(self, sections: dict[str, str]) -> list[dict]:
        """
        Extract project entries from the projects section.

        Each entry captures:
          - name:        Project name
          - description: Project description
          - technologies: Tech stack mentioned
          - link:        GitHub or URL if present

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of project entry dictionaries.
        """
        proj_text = sections.get("projects", "")
        if not proj_text:
            return []

        entries: list[dict] = []
        blocks = re.split(r"\n{2,}", proj_text.strip())

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 5:
                continue

            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue

            entry: dict[str, Any] = {
                "name":         "",
                "description":  "",
                "technologies": [],
                "link":         "",
                "raw":          clean_whitespace(block[:200]),
            }

            # First line is the project name
            entry["name"] = clean_whitespace(lines[0][:80])

            # Extract URL / GitHub link
            url_match = re.search(
                r"(?:https?://)?(?:www\.)?github\.com/[\w\-/]+|"
                r"https?://[\w\-./]+",
                block,
                re.IGNORECASE,
            )
            if url_match:
                entry["link"] = url_match.group(0)

            # Extract technologies (text after | or : or "Tech:" / "Stack:")
            tech_pattern = re.compile(
                r"(?:tech(?:nologies)?|stack|built with|tools?)[:\s]+([^\n]+)",
                re.IGNORECASE,
            )
            tech_match = tech_pattern.search(block)
            if tech_match:
                techs = re.split(r"[,|/]", tech_match.group(1))
                entry["technologies"] = [
                    clean_whitespace(t) for t in techs if t.strip()
                ]
            else:
                # Try pipe-separated tech on first line
                if "|" in lines[0]:
                    parts = lines[0].split("|")
                    entry["name"] = clean_whitespace(parts[0])
                    entry["technologies"] = [
                        clean_whitespace(p) for p in parts[1:] if p.strip()
                    ]

            # Description: remaining lines joined
            desc_lines = lines[1:] if len(lines) > 1 else []
            entry["description"] = clean_whitespace(
                " ".join(desc_lines)[:300]
            )

            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # CERTIFICATIONS EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_certifications(self, sections: dict[str, str]) -> list[str]:
        """
        Extract certification names from the certifications section.

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of certification name strings.
        """
        cert_text = sections.get("certifications", "")
        if not cert_text:
            return []

        certs: list[str] = []
        for line in cert_text.splitlines():
            line = re.sub(r"^[-•*►▪▸●>\d.)\s]+", "", line).strip()
            if len(line) > 5:
                certs.append(clean_whitespace(line[:120]))

        return deduplicate(certs)

    # ------------------------------------------------------------------
    # LANGUAGES EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_languages(self, sections: dict[str, str]) -> list[str]:
        """
        Extract spoken/written languages from the languages section.

        Args:
            sections: Pre-split section dictionary.

        Returns:
            List of language strings with optional proficiency level.
        """
        lang_text = sections.get("languages", "")
        if not lang_text:
            return []

        # Common spoken languages to look for
        known_languages = {
            "english", "spanish", "french", "german", "chinese",
            "mandarin", "japanese", "korean", "arabic", "portuguese",
            "italian", "russian", "hindi", "dutch", "swedish",
            "turkish", "polish", "vietnamese", "thai", "indonesian",
        }

        found: list[str] = []
        text_lower = lang_text.lower()

        for lang in known_languages:
            if lang in text_lower:
                # Try to find proficiency level near the language
                pattern = re.compile(
                    rf"{lang}"
                    r"[\s\-:]*"
                    r"(native|fluent|professional|conversational|"
                    r"intermediate|beginner|basic|advanced)?",
                    re.IGNORECASE,
                )
                match = pattern.search(lang_text)
                if match:
                    level = match.group(1)
                    entry = lang.capitalize()
                    if level:
                        entry += f" ({level.capitalize()})"
                    found.append(entry)

        # Also parse comma/newline separated list
        if not found:
            for line in lang_text.splitlines():
                line = re.sub(r"^[-•*►\s]+", "", line).strip()
                if 2 < len(line) < 50:
                    found.append(clean_whitespace(line))

        return deduplicate(found)

    # ------------------------------------------------------------------
    # TOOLS EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_tools(
        self, sections: dict[str, str], full_text: str
    ) -> list[str]:
        """
        Extract tools and technologies mentioned in the resume.

        Scans the skills section and full text for known tool patterns
        (IDEs, editors, platforms, services).

        Args:
            sections:  Pre-split section dictionary.
            full_text: Full resume text.

        Returns:
            List of tool name strings.
        """
        # Known tools to scan for
        known_tools = [
            "VS Code", "Visual Studio", "PyCharm", "IntelliJ", "Eclipse",
            "Jupyter", "Postman", "Insomnia", "Swagger", "Jira", "Confluence",
            "Trello", "Notion", "Slack", "Teams", "Zoom", "Figma", "Sketch",
            "GitHub", "GitLab", "Bitbucket", "Jenkins", "CircleCI", "Travis",
            "Heroku", "Vercel", "Netlify", "AWS", "Azure", "GCP",
            "Terraform", "Ansible", "Helm", "Grafana", "Prometheus",
            "Datadog", "Sentry", "New Relic", "Splunk",
            "Excel", "Google Sheets", "Power BI", "Tableau", "Looker",
            "Photoshop", "Illustrator", "Canva", "XD",
        ]

        scan_text = sections.get("skills", "") + "\n" + full_text
        found: list[str] = []

        for tool in known_tools:
            if re.search(re.escape(tool), scan_text, re.IGNORECASE):
                found.append(tool)

        return deduplicate(found)

    # ------------------------------------------------------------------
    # EXPERIENCE YEARS CALCULATOR
    # ------------------------------------------------------------------

    def _calc_experience_years(self, text: str, sections: dict[str, str]) -> float:
        """
        Calculate total years of experience.

        Tries three strategies in order:
          1. Span between earliest year and current year when "present"
             or "current" is found in the experience section.
          2. Span between earliest and latest year in experience section.
          3. Span across full resume text.

        Args:
            text:     Full resume text.
            sections: Pre-split section dictionary.

        Returns:
            Estimated years of experience as a float.
        """
        from datetime import datetime
        current_year = datetime.now().year

        exp_text = sections.get("experience", "") or text

        # Strategy 1: if "present" / "current" found, span to current year
        if re.search(r"\b(present|current|now)\b", exp_text, re.IGNORECASE):
            years = extract_years(exp_text)
            past = [y for y in years if y <= current_year]
            if past:
                return float(current_year - min(past))

        # Strategy 2: span between earliest and latest year
        years = extract_years(exp_text)
        past = [y for y in years if y <= current_year]
        if len(past) >= 2:
            return float(max(past) - min(past))

        # Strategy 3: full text year span
        return calculate_experience_years(text)

    # ------------------------------------------------------------------
    # SPACY LOADER
    # ------------------------------------------------------------------

    def _get_nlp(self):
        """
        Lazily load and cache the spaCy NLP model.

        Returns:
            spaCy Language object or None if unavailable.
        """
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy
            from utils.constants import SPACY_MODEL, SPACY_MODEL_FALLBACK
            for model in (SPACY_MODEL, SPACY_MODEL_FALLBACK):
                try:
                    self._nlp = spacy.load(model)
                    logger.debug("spaCy model loaded: %s", model)
                    return self._nlp
                except OSError:
                    continue
        except ImportError:
            logger.warning("spaCy not installed — NER-based name extraction disabled.")

        return None

    # ------------------------------------------------------------------
    # EMPTY RESULT
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return a fully structured empty result dictionary."""
        return {
            "name":             "Unknown Candidate",
            "email":            "",
            "phone":            "",
            "linkedin":         "",
            "github":           "",
            "portfolio":        "",
            "summary":          "",
            "skills":           [],
            "education":        [],
            "experience":       [],
            "projects":         [],
            "certifications":   [],
            "languages":        [],
            "tools":            [],
            "word_count":       0,
            "experience_years": 0.0,
            "sections_found":   [],
        }

# utils/skill_extractor.py
"""
skill_extractor.py — Skills extraction engine for AI Resume Analyzer.

Extracts technical and soft skills from resume/JD text by matching
against a curated CSV database of 500+ skills with aliases.

Matching strategies (applied in order):
  1. Direct string match       — exact skill name in text
  2. Alias match               — any alias for the skill found in text
  3. N-gram match              — bigram/trigram skill names matched
  4. spaCy PhraseMatcher       — fast, case-insensitive bulk matching
  5. Fuzzy normalisation       — normalised text comparison

Returns skills grouped by category with match confidence.
"""

import re
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from utils.constants import SKILLS_CSV, SKILL_CATEGORY_ORDER
from utils.helpers import normalize_text, deduplicate, clean_whitespace
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillExtractor:
    """
    Extracts skills from text using a CSV-backed skill database.

    The skills database (data/skills.csv) contains:
      - skill:     canonical skill name  (e.g., "Machine Learning")
      - category:  skill category        (e.g., "AI")
      - aliases:   comma-separated aliases (e.g., "ml,supervised learning")

    Usage:
        extractor = SkillExtractor()

        # Extract from resume text
        result = extractor.extract(resume_text)
        print(result["skills"])           # list of skill names
        print(result["by_category"])      # dict grouped by category
        print(result["skill_count"])      # total count

        # Extract from JD text
        jd_skills = extractor.extract(jd_text)
    """

    def __init__(self) -> None:
        self._skills_df: pd.DataFrame = pd.DataFrame()
        self._skill_map: dict[str, dict] = {}   # normalised_name → skill info
        self._alias_map: dict[str, str]  = {}   # normalised_alias → skill name
        self._nlp        = None
        self._matcher    = None
        self._loaded     = False
        self._load_skills()
        logger.debug("SkillExtractor initialised with %d skills.", len(self._skill_map))

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict[str, Any]:
        """
        Extract all skills found in the given text.

        Args:
            text: Raw or cleaned resume / JD text.

        Returns:
            Dictionary with keys:
              - skills       (list[str])  canonical skill names found
              - by_category  (dict)       skills grouped by category
              - skill_count  (int)        total number of skills found
              - categories   (list[str])  categories present
        """
        if not text or not text.strip():
            return self._empty_result()

        try:
            found_skills = self._match_skills(text)
            by_category  = self._group_by_category(found_skills)

            return {
                "skills":      found_skills,
                "by_category": by_category,
                "skill_count": len(found_skills),
                "categories":  [c for c in SKILL_CATEGORY_ORDER if c in by_category],
            }
        except Exception as exc:
            logger.error("SkillExtractor.extract failed: %s", exc)
            return self._empty_result()

    def extract_from_list(self, skill_list: list[str]) -> dict[str, Any]:
        """
        Resolve a pre-parsed list of skill strings against the database.

        Useful when skills have already been extracted by ResumeParser
        and need to be normalised and categorised.

        Args:
            skill_list: List of raw skill strings.

        Returns:
            Same structure as extract().
        """
        if not skill_list:
            return self._empty_result()

        text = ", ".join(skill_list)
        return self.extract(text)

    def get_all_skill_names(self) -> list[str]:
        """
        Return all canonical skill names in the database.

        Returns:
            Sorted list of skill name strings.
        """
        return sorted(self._skill_map.keys())

    def get_skills_by_category(self, skills):

        categorized = {}

        if not skills:
            return categorized

        for skill in skills:

            norm = normalize_text(skill)
            info = self._skill_map.get(norm) or self._skill_map.get(skill.lower(), {})

            category = info.get("category", "Other")

            if isinstance(category, list):
                category = category[0] if category else "Other"

            category = str(category)

            categorized.setdefault(category, []).append(skill)

        return categorized

    def get_categories(self) -> list[str]:
        """
        Return all unique skill categories in the database.

        Returns:
            List of category name strings.
        """
        cats = {info["category"] for info in self._skill_map.values()}
        return [c for c in SKILL_CATEGORY_ORDER if c in cats]

    # ------------------------------------------------------------------
    # SKILL DATABASE LOADER
    # ------------------------------------------------------------------

    def _load_skills(self) -> None:
        """
        Load the skills CSV into memory and build lookup maps.

        Builds two dictionaries:
          _skill_map:  normalised_skill_name → {skill, category, aliases}
          _alias_map:  normalised_alias      → canonical_skill_name

        Falls back to a minimal built-in skill set if the CSV is missing.
        """
        if self._loaded:
            return

        csv_path = Path(SKILLS_CSV)

        if not csv_path.exists():
            logger.warning(
                "Skills CSV not found at %s — using built-in fallback.", csv_path
            )
            self._load_fallback_skills()
            self._loaded = True
            return

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            required_cols = {"skill", "category", "aliases"}
            if not required_cols.issubset(set(df.columns)):
                logger.error(
                    "Skills CSV missing columns. Expected: %s, Got: %s",
                    required_cols, set(df.columns),
                )
                self._load_fallback_skills()
                self._loaded = True
                return

            self._skills_df = df

            for _, row in df.iterrows():
                skill_name = str(row["skill"]).strip()
                category   = str(row["category"]).strip()
                aliases_raw = str(row.get("aliases", ""))

                if not skill_name or skill_name == "nan":
                    continue

                # Parse aliases
                aliases: list[str] = []
                if aliases_raw and aliases_raw != "nan":
                    aliases = [
                        a.strip() for a in aliases_raw.split(",")
                        if a.strip()
                    ]

                norm_name = normalize_text(skill_name)
                self._skill_map[norm_name] = {
                    "skill":    skill_name,
                    "category": category,
                    "aliases":  aliases,
                }

                # Register all aliases → canonical name
                self._alias_map[norm_name] = skill_name
                for alias in aliases:
                    norm_alias = normalize_text(alias)
                    if norm_alias:
                        self._alias_map[norm_alias] = skill_name

            logger.info(
                "Skills loaded: %d skills, %d aliases from %s",
                len(self._skill_map),
                len(self._alias_map),
                csv_path.name,
            )
            self._loaded = True

        except Exception as exc:
            logger.error("Failed to load skills CSV: %s", exc)
            self._load_fallback_skills()
            self._loaded = True

    def _load_fallback_skills(self) -> None:
        """
        Load a minimal built-in skill set when the CSV is unavailable.
        Covers the most common technical skills across all categories.
        """
        fallback = [
            ("Python",          "Programming"),
            ("JavaScript",      "Programming"),
            ("TypeScript",      "Programming"),
            ("Java",            "Programming"),
            ("C++",             "Programming"),
            ("Go",              "Programming"),
            ("Rust",            "Programming"),
            ("React",           "Frontend"),
            ("Angular",         "Frontend"),
            ("Vue",             "Frontend"),
            ("FastAPI",         "Backend"),
            ("Flask",           "Backend"),
            ("Django",          "Backend"),
            ("NodeJS",          "Backend"),
            ("SQL",             "Database"),
            ("PostgreSQL",      "Database"),
            ("MongoDB",         "Database"),
            ("Redis",           "Database"),
            ("AWS",             "Cloud"),
            ("Azure",           "Cloud"),
            ("GCP",             "Cloud"),
            ("Docker",          "DevOps"),
            ("Kubernetes",      "DevOps"),
            ("Git",             "DevOps"),
            ("CI/CD",           "DevOps"),
            ("Machine Learning","AI"),
            ("Deep Learning",   "AI"),
            ("NLP",             "AI"),
            ("LangChain",       "AI"),
            ("OpenAI",          "AI"),
            ("Pandas",          "Data"),
            ("NumPy",           "Data"),
            ("Scikit-learn",    "AI"),
            ("TensorFlow",      "AI"),
            ("PyTorch",         "AI"),
        ]
        for skill_name, category in fallback:
            norm = normalize_text(skill_name)
            self._skill_map[norm] = {
                "skill":    skill_name,
                "category": category,
                "aliases":  [],
            }
            self._alias_map[norm] = skill_name

        logger.info("Fallback skills loaded: %d skills.", len(self._skill_map))

    # ------------------------------------------------------------------
    # SPACY PHRASE MATCHER (lazy load)
    # ------------------------------------------------------------------

    def _get_matcher(self):
        """
        Lazily build and cache a spaCy PhraseMatcher for fast bulk matching.

        The PhraseMatcher is significantly faster than iterating all
        skill patterns manually for large texts.

        Returns:
            Tuple of (nlp, matcher) or (None, None) if spaCy unavailable.
        """
        if self._matcher is not None:
            return self._nlp, self._matcher

        try:
            import spacy
            from spacy.matcher import PhraseMatcher
            from utils.constants import SPACY_MODEL, SPACY_MODEL_FALLBACK

            nlp = None
            for model in (SPACY_MODEL, SPACY_MODEL_FALLBACK):
                try:
                    nlp = spacy.load(model, disable=["parser", "ner"])
                    break
                except OSError:
                    continue

            if nlp is None:
                return None, None

            matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

            # Add all skill names and aliases as patterns
            for norm_name, info in self._skill_map.items():
                patterns = [info["skill"]] + info["aliases"]
                valid_patterns = []
                for p in patterns:
                    if p and len(p.split()) <= 4:  # Max 4-word phrases
                        try:
                            valid_patterns.append(nlp.make_doc(p.lower()))
                        except Exception:
                            continue
                if valid_patterns:
                    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", norm_name)[:50]
                    try:
                        matcher.add(safe_key, valid_patterns)
                    except Exception:
                        continue

            self._nlp     = nlp
            self._matcher = matcher
            logger.debug("spaCy PhraseMatcher built with %d patterns.", len(self._skill_map))
            return nlp, matcher

        except ImportError:
            logger.warning("spaCy not available — using regex matching only.")
            return None, None
        except Exception as exc:
            logger.warning("PhraseMatcher build failed: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # CORE MATCHING ENGINE
    # ------------------------------------------------------------------

    def _match_skills(self, text: str) -> list[str]:
        """
        Run all matching strategies and return deduplicated skill names.

        Strategies applied:
          1. spaCy PhraseMatcher  (fast, handles multi-word)
          2. Regex direct match   (fallback, handles edge cases)

        Args:
            text: Input text to scan.

        Returns:
            Deduplicated list of canonical skill name strings.
        """
        found: set[str] = set()

        # Strategy 1: spaCy PhraseMatcher
        spacy_skills = self._match_with_spacy(text)
        found.update(spacy_skills)

        # Strategy 2: Regex / string matching (catches what spaCy misses)
        regex_skills = self._match_with_regex(text)
        found.update(regex_skills)

        result = sorted(found)
        logger.debug("Skills matched: %d total", len(result))
        return result

    def _match_with_spacy(self, text: str) -> list[str]:
        """
        Use spaCy PhraseMatcher to find skill mentions in text.

        Args:
            text: Input text.

        Returns:
            List of canonical skill name strings found.
        """
        nlp, matcher = self._get_matcher()
        if nlp is None or matcher is None:
            return []

        try:
            # Cap text length for performance
            doc = nlp(text[:50000].lower())
            matches = matcher(doc)

            found: list[str] = []
            for match_id, start, end in matches:
                matched_text = doc[start:end].text.lower()
                norm = normalize_text(matched_text)

                # Resolve to canonical name
                canonical = self._alias_map.get(norm)
                if canonical:
                    found.append(canonical)

            return deduplicate(found)

        except Exception as exc:
            logger.warning("spaCy matching failed: %s", exc)
            return []

    def _match_with_regex(self, text: str) -> list[str]:
        """
        Match skills using word-boundary regex patterns.

        Handles:
          - Direct skill name matches
          - Alias matches
          - Case-insensitive matching
          - Word boundary enforcement (avoids partial matches)

        Args:
            text: Input text.

        Returns:
            List of canonical skill name strings found.
        """
        found: list[str] = []
        text_norm = normalize_text(text)

        for norm_alias, canonical_name in self._alias_map.items():
            if not norm_alias or len(norm_alias) < 2:
                continue

            # Build a word-boundary aware pattern
            # Escape special regex chars in the skill name
            escaped = re.escape(norm_alias)

            # Use word boundaries for single words, flexible for multi-word
            if " " in norm_alias:
                pattern = escaped.replace(r"\ ", r"[\s\-/]+")
            else:
                pattern = r"\b" + escaped + r"\b"

            try:
                if re.search(pattern, text_norm):
                    found.append(canonical_name)
            except re.error:
                # Skip malformed patterns
                continue

        return deduplicate(found)

    # ------------------------------------------------------------------
    # CATEGORY GROUPER
    # ------------------------------------------------------------------

    def _group_by_category(
        self, skills: list[str]
    ) -> dict[str, list[str]]:
        """
        Group a list of skill names by their category.

        Args:
            skills: List of canonical skill name strings.

        Returns:
            Dictionary mapping category → list of skill names,
            ordered by SKILL_CATEGORY_ORDER.
        """
        grouped: dict[str, list[str]] = {}

        for skill_name in skills:
            norm = normalize_text(skill_name)
            info = self._skill_map.get(norm)

            if info:
                category = info["category"]
            else:
                # Try to find via alias map
                canonical = self._alias_map.get(norm)
                if canonical:
                    norm_canonical = normalize_text(canonical)
                    info = self._skill_map.get(norm_canonical)
                    category = info["category"] if info else "Other"
                else:
                    category = "Other"

            grouped.setdefault(category, [])
            if skill_name not in grouped[category]:
                grouped[category].append(skill_name)

        # Sort categories by preferred display order
        ordered: dict[str, list[str]] = {}
        for cat in SKILL_CATEGORY_ORDER:
            if cat in grouped:
                ordered[cat] = sorted(grouped[cat])
        # Append any remaining categories not in the order list
        for cat, skills_list in grouped.items():
            if cat not in ordered:
                ordered[cat] = sorted(skills_list)

        return ordered

    # ------------------------------------------------------------------
    # COMPARISON UTILITIES
    # ------------------------------------------------------------------

    def compare(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict[str, list[str]]:
        """
        Compare resume skills against JD skills and classify each.

        Args:
            resume_skills: Skills extracted from the resume.
            jd_skills:     Skills extracted from the JD.

        Returns:
            Dictionary with keys:
              - matched:   skills in both resume and JD
              - missing:   skills in JD but not in resume
              - additional: skills in resume but not in JD
        """
        # Normalise both lists for comparison
        norm_resume = {normalize_text(s): s for s in resume_skills}
        norm_jd     = {normalize_text(s): s for s in jd_skills}

        # Also check aliases — a resume skill might be an alias of a JD skill
        matched_norm:    set[str] = set()
        missing_norm:    set[str] = set()
        additional_norm: set[str] = set()

        for norm_jd_skill in norm_jd:
            # Direct match
            if norm_jd_skill in norm_resume:
                matched_norm.add(norm_jd_skill)
                continue

            # Alias match — check if any resume skill is an alias of this JD skill
            jd_canonical = self._alias_map.get(norm_jd_skill, "")
            jd_norm_canonical = normalize_text(jd_canonical)

            alias_found = False
            for norm_res_skill in norm_resume:
                res_canonical = self._alias_map.get(norm_res_skill, "")
                if (
                    normalize_text(res_canonical) == jd_norm_canonical
                    and jd_norm_canonical
                ):
                    matched_norm.add(norm_jd_skill)
                    alias_found = True
                    break

            if not alias_found:
                missing_norm.add(norm_jd_skill)

        for norm_res_skill in norm_resume:
            if norm_res_skill not in norm_jd:
                # Check if it's an alias of any JD skill
                res_canonical = self._alias_map.get(norm_res_skill, "")
                norm_res_canonical = normalize_text(res_canonical)
                is_alias_of_jd = any(
                    normalize_text(self._alias_map.get(nj, "")) == norm_res_canonical
                    and norm_res_canonical
                    for nj in norm_jd
                )
                if not is_alias_of_jd:
                    additional_norm.add(norm_res_skill)

        return {
            "matched":    [norm_jd[n]    for n in matched_norm    if n in norm_jd],
            "missing":    [norm_jd[n]    for n in missing_norm    if n in norm_jd],
            "additional": [norm_resume[n] for n in additional_norm if n in norm_resume],
        }

    # ------------------------------------------------------------------
    # EMPTY RESULT
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return a fully structured empty result dictionary."""
        return {
            "skills":      [],
            "by_category": {},
            "skill_count": 0,
            "categories":  [],
        }

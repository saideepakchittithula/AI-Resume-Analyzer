# utils/matcher.py
"""
matcher.py — Resume vs Job Description matching engine.

Computes multiple similarity and compatibility metrics between
a resume and a job description:

  1. Skill Match %         — matched skills / total JD skills
  2. Keyword Match %       — JD keywords found in resume
  3. Semantic Similarity % — TF-IDF cosine similarity
  4. Overall Compatibility % — weighted combination of all metrics

Also generates:
  - Matched skills list
  - Missing skills list
  - Additional skills list
  - Improvement suggestions based on gaps
"""

import re
import math
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import (
    normalize_text,
    deduplicate,
    percentage,
    clamp,
)
from utils.constants import (
    SUGGESTIONS,
    SIMILARITY_HIGH,
    SIMILARITY_MEDIUM,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ResumeMatcher:
    """
    Computes compatibility metrics between a resume and a job description.

    Usage:
        matcher = ResumeMatcher()
        result  = matcher.match(resume_data, jd_data)

        print(result["skill_match_pct"])
        print(result["semantic_similarity_pct"])
        print(result["overall_compatibility_pct"])
        print(result["missing_skills"])
        print(result["suggestions"])

    Args to match():
        resume_data: dict from ResumeParser.parse()
        jd_data:     dict from JDParser.parse()

    Returns a dict with all matching metrics and suggestions.
    """

    # Weights for overall compatibility score
    _WEIGHTS: dict[str, float] = {
        "skill_match":   0.45,   # Skills are the strongest signal
        "keyword_match": 0.25,   # Keyword density in resume
        "semantic_sim":  0.30,   # Semantic text similarity
    }

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),      # Unigrams, bigrams, trigrams
            stop_words="english",
            min_df=1,
            max_features=5000,
            sublinear_tf=True,       # Apply log normalisation to TF
        )
        logger.debug("ResumeMatcher initialised.")

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def match(
        self,
        resume_data: dict[str, Any],
        jd_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute all matching metrics between resume and JD.

        Args:
            resume_data: Parsed resume dictionary from ResumeParser.
            jd_data:     Parsed JD dictionary from JDParser.

        Returns:
            Dictionary containing all match metrics and suggestions.
        """
        if not resume_data or not jd_data:
            logger.warning("ResumeMatcher received empty input.")
            return self._empty_result()

        try:
            # Extract text and skills from both documents
            resume_text   = self._build_resume_text(resume_data)
            jd_text       = self._build_jd_text(jd_data)
            resume_skills = self._normalise_skills(resume_data.get("skills", []))
            jd_skills     = self._normalise_skills(jd_data.get("all_skills", []))
            jd_keywords   = jd_data.get("keywords", [])

            # Compute individual metrics
            skill_result      = self._compute_skill_match(resume_skills, jd_skills)
            keyword_match_pct = self._compute_keyword_match(resume_text, jd_keywords)
            semantic_sim_pct  = self._compute_semantic_similarity(resume_text, jd_text)

            # Compute weighted overall compatibility
            overall_pct = self._compute_overall(
                skill_result["match_pct"],
                keyword_match_pct,
                semantic_sim_pct,
            )

            # Generate suggestions based on gaps
            suggestions = self._generate_suggestions(
                resume_data, jd_data, skill_result
            )

            result = {
                # Skill matching
                "matched_skills":       skill_result["matched"],
                "missing_skills":       skill_result["missing"],
                "additional_skills":    skill_result["additional"],
                "skill_match_pct":      skill_result["match_pct"],

                # Text matching
                "keyword_match_pct":    keyword_match_pct,
                "semantic_similarity_pct": semantic_sim_pct,

                # Overall
                "overall_compatibility_pct": overall_pct,

                # Suggestions
                "suggestions": suggestions,

                # Counts
                "total_resume_skills":  len(resume_skills),
                "total_jd_skills":      len(jd_skills),
                "matched_count":        len(skill_result["matched"]),
                "missing_count":        len(skill_result["missing"]),
            }

            logger.info(
                "Match complete — skill=%.1f%%, keyword=%.1f%%, "
                "semantic=%.1f%%, overall=%.1f%%",
                skill_result["match_pct"],
                keyword_match_pct,
                semantic_sim_pct,
                overall_pct,
            )

            return result

        except Exception as exc:
            logger.exception("ResumeMatcher.match failed: %s", exc)
            return self._empty_result()

    # ------------------------------------------------------------------
    # TEXT BUILDERS
    # ------------------------------------------------------------------

    def _build_resume_text(self, resume_data: dict) -> str:
        """
        Concatenate all resume text fields into one string for NLP.

        Args:
            resume_data: Parsed resume dictionary.

        Returns:
            Combined resume text string.
        """
        parts: list[str] = []

        # Weight skills and summary more heavily by repeating them
        summary = resume_data.get("summary", "")
        if summary:
            parts.append(summary)
            parts.append(summary)   # Double weight

        skills = resume_data.get("skills", [])
        if skills:
            skill_text = " ".join(skills)
            parts.append(skill_text)
            parts.append(skill_text)  # Double weight

        # Experience descriptions
        for exp in resume_data.get("experience", []):
            parts.append(exp.get("title", ""))
            parts.append(exp.get("company", ""))
            parts.extend(exp.get("description", []))

        # Education
        for edu in resume_data.get("education", []):
            parts.append(edu.get("degree", ""))
            parts.append(edu.get("field", ""))
            parts.append(edu.get("institution", ""))

        # Projects
        for proj in resume_data.get("projects", []):
            parts.append(proj.get("name", ""))
            parts.append(proj.get("description", ""))
            parts.extend(proj.get("technologies", []))

        # Certifications
        parts.extend(resume_data.get("certifications", []))

        return " ".join(p for p in parts if p)

    def _build_jd_text(self, jd_data: dict) -> str:
        """
        Concatenate all JD text fields into one string for NLP.

        Args:
            jd_data: Parsed JD dictionary.

        Returns:
            Combined JD text string.
        """
        parts: list[str] = []

        parts.append(jd_data.get("job_title", ""))

        # Weight required skills heavily
        req_skills = jd_data.get("required_skills", [])
        if req_skills:
            skill_text = " ".join(req_skills)
            parts.append(skill_text)
            parts.append(skill_text)  # Double weight

        parts.extend(jd_data.get("preferred_skills", []))
        parts.extend(jd_data.get("responsibilities", []))
        parts.extend(jd_data.get("requirements", []))
        parts.append(jd_data.get("education", ""))
        parts.extend(jd_data.get("keywords", []))

        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # SKILL MATCHING
    # ------------------------------------------------------------------

    def _normalise_skills(self, skills: list[str]) -> list[str]:
        """
        Normalise a list of skill strings for comparison.

        Lowercases, strips punctuation, and deduplicates.

        Args:
            skills: Raw skill name list.

        Returns:
            Normalised, deduplicated skill list.
        """
        normalised: list[str] = []
        for skill in skills:
            norm = normalize_text(skill).strip()
            if norm and len(norm) >= 2:
                normalised.append(norm)
        return deduplicate(normalised)

    def _compute_skill_match(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict[str, Any]:
        """
        Compute skill overlap between resume and JD.

        Uses normalised string matching with partial alias awareness.
        A skill is considered matched if:
          - Exact normalised match, OR
          - One is a substring of the other (handles abbreviations)

        Args:
            resume_skills: Normalised resume skill list.
            jd_skills:     Normalised JD skill list.

        Returns:
            Dict with matched, missing, additional lists and match_pct.
        """
        if not jd_skills:
            return {
                "matched":   [],
                "missing":   [],
                "additional": resume_skills,
                "match_pct": 0.0,
            }

        matched:    list[str] = []
        missing:    list[str] = []
        additional: list[str] = []

        resume_set = set(resume_skills)

        for jd_skill in jd_skills:
            if self._skills_match(jd_skill, resume_set):
                matched.append(jd_skill)
            else:
                missing.append(jd_skill)

        for res_skill in resume_skills:
            jd_set = set(jd_skills)
            if not self._skills_match(res_skill, jd_set):
                additional.append(res_skill)

        match_pct = percentage(len(matched), len(jd_skills))

        return {
            "matched":    deduplicate(matched),
            "missing":    deduplicate(missing),
            "additional": deduplicate(additional),
            "match_pct":  match_pct,
        }

    def _skills_match(self, skill: str, skill_set: set[str]) -> bool:
        """
        Check if a skill matches any skill in a set.

        Matching rules (in order):
          1. Exact match
          2. Skill is a substring of a set member
          3. A set member is a substring of the skill

        Args:
            skill:     Normalised skill string to look up.
            skill_set: Set of normalised skill strings to search.

        Returns:
            True if a match is found.
        """
        if skill in skill_set:
            return True

        # Substring matching for abbreviations and partial names
        for s in skill_set:
            if len(skill) >= 3 and len(s) >= 3:
                if skill in s or s in skill:
                    return True

        return False

    # ------------------------------------------------------------------
    # KEYWORD MATCHING
    # ------------------------------------------------------------------

    def _compute_keyword_match(
        self,
        resume_text: str,
        jd_keywords: list[str],
    ) -> float:
        """
        Compute what percentage of JD keywords appear in the resume.

        Args:
            resume_text: Combined resume text string.
            jd_keywords: List of important JD keyword strings.

        Returns:
            Keyword match percentage (0–100).
        """
        if not jd_keywords or not resume_text:
            return 0.0

        resume_norm = normalize_text(resume_text)
        matched = 0

        for keyword in jd_keywords:
            kw_norm = normalize_text(keyword)
            if not kw_norm:
                continue
            # Use word boundary for single words, flexible for multi-word
            if " " in kw_norm:
                pattern = re.escape(kw_norm).replace(r"\ ", r"[\s\-]+")
            else:
                pattern = r"\b" + re.escape(kw_norm) + r"\b"

            try:
                if re.search(pattern, resume_norm):
                    matched += 1
            except re.error:
                if kw_norm in resume_norm:
                    matched += 1

        return percentage(matched, len(jd_keywords))

    # ------------------------------------------------------------------
    # SEMANTIC SIMILARITY (TF-IDF COSINE)
    # ------------------------------------------------------------------

    def _compute_semantic_similarity(
        self,
        resume_text: str,
        jd_text: str,
    ) -> float:
        """
        Compute semantic similarity using TF-IDF cosine similarity.

        TF-IDF captures the importance of terms relative to both
        documents, and cosine similarity measures the angle between
        the two document vectors — a robust proxy for semantic overlap.

        Args:
            resume_text: Combined resume text.
            jd_text:     Combined JD text.

        Returns:
            Similarity percentage (0–100).
        """
        if not resume_text or not jd_text:
            return 0.0

        try:
            # Fit vectorizer on both documents together
            tfidf_matrix = self._vectorizer.fit_transform(
                [resume_text, jd_text]
            )

            # Cosine similarity between the two document vectors
            sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity = float(sim_matrix[0][0])

            # Convert to percentage and clamp
            return clamp(round(similarity * 100, 1), 0.0, 100.0)

        except Exception as exc:
            logger.warning("TF-IDF similarity failed: %s", exc)
            return self._fallback_similarity(resume_text, jd_text)

    def _fallback_similarity(
        self,
        resume_text: str,
        jd_text: str,
    ) -> float:
        """
        Jaccard similarity fallback when TF-IDF fails.

        Jaccard = |intersection| / |union| of word sets.

        Args:
            resume_text: Resume text string.
            jd_text:     JD text string.

        Returns:
            Similarity percentage (0–100).
        """
        try:
            resume_words = set(normalize_text(resume_text).split())
            jd_words     = set(normalize_text(jd_text).split())

            if not resume_words or not jd_words:
                return 0.0

            intersection = resume_words & jd_words
            union        = resume_words | jd_words

            jaccard = len(intersection) / len(union)
            return clamp(round(jaccard * 100, 1), 0.0, 100.0)

        except Exception as exc:
            logger.error("Fallback similarity failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # OVERALL COMPATIBILITY
    # ------------------------------------------------------------------

    def _compute_overall(
        self,
        skill_pct: float,
        keyword_pct: float,
        semantic_pct: float,
    ) -> float:
        """
        Compute weighted overall compatibility score.

        Weights (from _WEIGHTS):
          - Skill match:   45%
          - Keyword match: 25%
          - Semantic sim:  30%

        Args:
            skill_pct:    Skill match percentage.
            keyword_pct:  Keyword match percentage.
            semantic_pct: Semantic similarity percentage.

        Returns:
            Overall compatibility percentage (0–100).
        """
        w = self._WEIGHTS
        overall = (
            skill_pct    * w["skill_match"]   +
            keyword_pct  * w["keyword_match"] +
            semantic_pct * w["semantic_sim"]
        )
        return clamp(round(overall, 1), 0.0, 100.0)

    # ------------------------------------------------------------------
    # SUGGESTIONS GENERATOR
    # ------------------------------------------------------------------

    def _generate_suggestions(
        self,
        resume_data: dict,
        jd_data: dict,
        skill_result: dict,
    ) -> list[str]:
        """
        Generate actionable improvement suggestions based on match gaps.

        Checks for:
          - Missing critical skills
          - Missing contact info (LinkedIn, GitHub)
          - Missing resume sections (summary, projects, certifications)
          - Resume length issues
          - Formatting signals

        Args:
            resume_data:  Parsed resume dictionary.
            jd_data:      Parsed JD dictionary.
            skill_result: Skill matching result dictionary.

        Returns:
            List of suggestion strings, ordered by priority.
        """
        suggestions: list[str] = []
        missing_skills = skill_result.get("missing", [])
        missing_lower  = [s.lower() for s in missing_skills]

        # --- Contact info suggestions ---
        if not resume_data.get("linkedin"):
            suggestions.append(SUGGESTIONS["add_linkedin"])
        if not resume_data.get("github"):
            suggestions.append(SUGGESTIONS["add_github"])

        # --- Section suggestions ---
        if not resume_data.get("summary"):
            suggestions.append(SUGGESTIONS["add_summary"])
        if not resume_data.get("certifications"):
            suggestions.append(SUGGESTIONS["add_certifications"])
        if not resume_data.get("projects"):
            suggestions.append(SUGGESTIONS["add_projects"])

        # --- Resume length ---
        word_count = resume_data.get("word_count", 0)
        if word_count < 200:
            suggestions.append(SUGGESTIONS["resume_too_short"])
        elif word_count > 1200:
            suggestions.append(SUGGESTIONS["resume_too_long"])

        # --- Missing skill suggestions ---
        if any(s in missing_lower for s in ["docker", "container"]):
            suggestions.append(SUGGESTIONS["learn_docker"])
        if any(s in missing_lower for s in ["kubernetes", "k8s"]):
            suggestions.append(SUGGESTIONS["learn_kubernetes"])
        if any(s in missing_lower for s in ["aws", "azure", "gcp", "cloud"]):
            suggestions.append(SUGGESTIONS["learn_cloud"])
        if any(s in missing_lower for s in ["fastapi", "fast api"]):
            suggestions.append(SUGGESTIONS["learn_fastapi"])

        # --- General quality suggestions ---
        if len(missing_skills) > 5:
            suggestions.append(SUGGESTIONS["add_keywords"])
            suggestions.append(SUGGESTIONS["tailor_resume"])

        if not resume_data.get("education"):
            suggestions.append(SUGGESTIONS["add_education"])

        suggestions.append(SUGGESTIONS["improve_experience"])
        suggestions.append(SUGGESTIONS["use_action_verbs"])
        suggestions.append(SUGGESTIONS["improve_formatting"])

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:12]   # Return top 12 suggestions max

    # ------------------------------------------------------------------
    # ANALYSIS HELPERS
    # ------------------------------------------------------------------

    def get_match_summary(self, match_result: dict) -> dict[str, str]:
        """
        Generate a human-readable summary of the match result.

        Args:
            match_result: Dictionary returned by match().

        Returns:
            Dictionary with summary strings for each metric.
        """
        skill_pct    = match_result.get("skill_match_pct", 0)
        keyword_pct  = match_result.get("keyword_match_pct", 0)
        semantic_pct = match_result.get("semantic_similarity_pct", 0)
        overall_pct  = match_result.get("overall_compatibility_pct", 0)

        def _label(pct: float) -> str:
            if pct >= 75:  return "Excellent"
            if pct >= 55:  return "Good"
            if pct >= 35:  return "Average"
            return "Poor"

        return {
            "skill_match":    f"{skill_pct:.1f}% — {_label(skill_pct)}",
            "keyword_match":  f"{keyword_pct:.1f}% — {_label(keyword_pct)}",
            "semantic_sim":   f"{semantic_pct:.1f}% — {_label(semantic_pct)}",
            "overall":        f"{overall_pct:.1f}% — {_label(overall_pct)}",
        }

    def get_top_missing_skills(
        self,
        match_result: dict,
        top_n: int = 10,
    ) -> list[str]:
        """
        Return the top N missing skills sorted by importance.

        Currently returns the first N missing skills as they appear
        in the JD (JD order implies priority).

        Args:
            match_result: Dictionary returned by match().
            top_n:        Maximum number of skills to return.

        Returns:
            List of missing skill name strings.
        """
        return match_result.get("missing_skills", [])[:top_n]

    # ------------------------------------------------------------------
    # EMPTY RESULT
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return a fully structured empty result dictionary."""
        return {
            "matched_skills":            [],
            "missing_skills":            [],
            "additional_skills":         [],
            "skill_match_pct":           0.0,
            "keyword_match_pct":         0.0,
            "semantic_similarity_pct":   0.0,
            "overall_compatibility_pct": 0.0,
            "suggestions":               [],
            "total_resume_skills":       0,
            "total_jd_skills":           0,
            "matched_count":             0,
            "missing_count":             0,
        }

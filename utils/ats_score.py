# utils/ats_score.py
"""
ats_score.py — ATS (Applicant Tracking System) score calculator.

Computes a comprehensive ATS score out of 100 across 9 weighted
categories that mirror how real ATS systems evaluate resumes.

Scoring categories (weights from constants.ATS_WEIGHTS):
  1. skills_match     (30) — Skills overlap with JD
  2. experience       (20) — Work experience relevance & years
  3. education        (10) — Education qualifications
  4. projects         (10) — Projects listed
  5. certifications   ( 5) — Certifications mentioned
  6. keyword_density  (10) — JD keyword frequency in resume
  7. formatting       ( 5) — Resume formatting quality signals
  8. resume_length    ( 5) — Optimal word count range
  9. summary          ( 5) — Professional summary present

Returns a detailed breakdown with score, max, percentage, and
feedback for each category plus an overall score with label.
"""

from typing import Any

from utils.constants import (
    ATS_WEIGHTS,
    ATS_SCORE_EXCELLENT,
    ATS_SCORE_GOOD,
    ATS_SCORE_AVERAGE,
    RESUME_OPTIMAL_MIN_WORDS,
    RESUME_OPTIMAL_MAX_WORDS,
    DEGREE_KEYWORDS,
    ACTION_VERBS,
)
from utils.helpers import (
    clamp,
    get_score_label,
    get_score_color,
    percentage,
    count_words,
    is_optimal_length,
    get_resume_length_label,
    get_experience_level,
    check_action_verbs,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ATSScorer:
    """
    Calculates a comprehensive ATS score for a resume against a JD.

    Usage:
        scorer = ATSScorer()
        result = scorer.score(resume_data, jd_data, match_result)

        print(result["overall_score"])      # e.g. 78
        print(result["label"])              # e.g. "Good"
        print(result["breakdown"])          # per-category scores
        print(result["strengths"])          # what's working well
        print(result["weaknesses"])         # what needs improvement
    """

    def __init__(self) -> None:
        logger.debug("ATSScorer initialised.")

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def score(
        self,
        resume_data: dict[str, Any],
        jd_data:     dict[str, Any],
        match_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute the full ATS score across all 9 categories.

        Args:
            resume_data:  Parsed resume dict from ResumeParser.
            jd_data:      Parsed JD dict from JDParser.
            match_result: Match result dict from ResumeMatcher.

        Returns:
            Dictionary containing:
              - overall_score   (float 0–100)
              - label           (str: Excellent/Good/Average/Poor)
              - color           (str: hex color)
              - breakdown       (dict: per-category details)
              - strengths       (list[str])
              - weaknesses      (list[str])
              - category_scores (dict: category → score out of weight)
        """
        if not resume_data:
            logger.warning("ATSScorer received empty resume_data.")
            return self._empty_result()

        try:
            breakdown = self._compute_breakdown(
                resume_data, jd_data, match_result
            )

            overall = self._compute_overall(breakdown)
            label, emoji = get_score_label(overall)
            color = get_score_color(overall)

            strengths, weaknesses = self._analyse_breakdown(breakdown)

            category_scores = {
                cat: round(info["score"], 1)
                for cat, info in breakdown.items()
            }

            result = {
                "overall_score":   round(overall, 1),
                "label":           label,
                "emoji":           emoji,
                "color":           color,
                "breakdown":       breakdown,
                "category_scores": category_scores,
                "strengths":       strengths,
                "weaknesses":      weaknesses,
                "max_score":       100,
            }

            logger.info(
                "ATS score: %.1f/100 (%s) | breakdown: %s",
                overall, label,
                {k: round(v["score"], 1) for k, v in breakdown.items()},
            )

            return result

        except Exception as exc:
            logger.exception("ATSScorer.score failed: %s", exc)
            return self._empty_result()

    # ------------------------------------------------------------------
    # BREAKDOWN CALCULATOR
    # ------------------------------------------------------------------

    def _compute_breakdown(
        self,
        resume_data:  dict,
        jd_data:      dict,
        match_result: dict,
    ) -> dict[str, dict]:
        """
        Compute score and feedback for each of the 9 ATS categories.

        Args:
            resume_data:  Parsed resume dictionary.
            jd_data:      Parsed JD dictionary.
            match_result: Match result dictionary.

        Returns:
            Dictionary mapping category_name → {
                score:    float (0 – weight),
                max:      int   (weight),
                pct:      float (0–100),
                feedback: str,
            }
        """
        return {
            "skills_match":   self._score_skills(resume_data, match_result),
            "experience":     self._score_experience(resume_data, jd_data),
            "education":      self._score_education(resume_data, jd_data),
            "projects":       self._score_projects(resume_data),
            "certifications": self._score_certifications(resume_data),
            "keyword_density":self._score_keywords(resume_data, match_result),
            "formatting":     self._score_formatting(resume_data),
            "resume_length":  self._score_length(resume_data),
            "summary":        self._score_summary(resume_data),
        }

    # ------------------------------------------------------------------
    # CATEGORY SCORERS
    # ------------------------------------------------------------------

    def _score_skills(
        self,
        resume_data:  dict,
        match_result: dict,
    ) -> dict:
        """
        Score skills match category (weight: 30).

        Based on skill_match_pct from the matcher result.
        Full marks at 90%+ match, scaled linearly below.

        Args:
            resume_data:  Parsed resume dictionary.
            match_result: Match result dictionary.

        Returns:
            Category score dictionary.
        """
        weight = ATS_WEIGHTS["skills_match"]
        skill_pct = match_result.get("skill_match_pct", 0.0)
        matched   = len(match_result.get("matched_skills", []))
        missing   = len(match_result.get("missing_skills", []))
        total_jd  = match_result.get("total_jd_skills", 0)

        # Scale: 90%+ match = full marks, linear below
        raw_score = (skill_pct / 100) * weight
        score = clamp(raw_score, 0, weight)

        if skill_pct >= 85:
            feedback = (
                f"Excellent skills match! {matched}/{total_jd} required "
                f"skills found in your resume."
            )
        elif skill_pct >= 65:
            feedback = (
                f"Good skills match. {matched}/{total_jd} skills matched. "
                f"Add {missing} more skills to improve."
            )
        elif skill_pct >= 40:
            feedback = (
                f"Average skills match. Only {matched}/{total_jd} skills "
                f"found. Focus on adding missing skills."
            )
        else:
            feedback = (
                f"Poor skills match. Only {matched}/{total_jd} JD skills "
                f"found. Significantly update your skills section."
            )

        return self._make_entry(score, weight, feedback)

    def _score_experience(
        self,
        resume_data: dict,
        jd_data:     dict,
    ) -> dict:
        """
        Score work experience category (weight: 20).

        Evaluates:
          - Number of experience entries
          - Years of experience vs JD requirement
          - Presence of bullet point descriptions
          - Use of action verbs in descriptions

        Args:
            resume_data: Parsed resume dictionary.
            jd_data:     Parsed JD dictionary.

        Returns:
            Category score dictionary.
        """
        weight     = ATS_WEIGHTS["experience"]
        experience = resume_data.get("experience", [])
        exp_years  = resume_data.get("experience_years", 0.0)
        req_years  = jd_data.get("experience_years", 0.0) if jd_data else 0.0

        score = 0.0
        notes: list[str] = []

        # Sub-score 1: Has experience entries (up to 8 pts)
        if len(experience) >= 3:
            score += 8
            notes.append(f"{len(experience)} experience entries found.")
        elif len(experience) == 2:
            score += 6
            notes.append("2 experience entries found.")
        elif len(experience) == 1:
            score += 4
            notes.append("Only 1 experience entry found.")
        else:
            notes.append("No work experience found.")

        # Sub-score 2: Years of experience vs requirement (up to 6 pts)
        if req_years > 0 and exp_years >= req_years:
            score += 6
            notes.append(
                f"Experience ({exp_years:.0f} yrs) meets requirement ({req_years:.0f} yrs)."
            )
        elif req_years > 0 and exp_years >= req_years * 0.7:
            score += 4
            notes.append(
                f"Experience ({exp_years:.0f} yrs) slightly below requirement ({req_years:.0f} yrs)."
            )
        elif exp_years > 0:
            score += 2
            notes.append(f"{exp_years:.0f} years of experience detected.")
        else:
            notes.append("Could not determine years of experience.")

        # Sub-score 3: Descriptions with bullet points (up to 4 pts)
        entries_with_desc = sum(
            1 for e in experience if e.get("description")
        )
        if entries_with_desc >= 2:
            score += 4
            notes.append("Experience entries have detailed descriptions.")
        elif entries_with_desc == 1:
            score += 2
            notes.append("Add more detail to experience descriptions.")
        else:
            notes.append("Add bullet point descriptions to experience entries.")

        # Sub-score 4: Action verbs in descriptions (up to 2 pts)
        all_desc = " ".join(
            " ".join(e.get("description", []))
            for e in experience
        )
        verb_result = check_action_verbs(all_desc)
        if verb_result["score"] >= 0.5:
            score += 2
            notes.append("Good use of action verbs in experience.")
        else:
            notes.append("Use more action verbs (Built, Led, Designed, etc.).")

        score = clamp(score, 0, weight)
        feedback = " ".join(notes)
        return self._make_entry(score, weight, feedback)

    def _score_education(
        self,
        resume_data: dict,
        jd_data:     dict,
    ) -> dict:
        """
        Score education category (weight: 10).

        Evaluates:
          - Presence of education entries
          - Degree level vs JD requirement
          - GPA mention

        Args:
            resume_data: Parsed resume dictionary.
            jd_data:     Parsed JD dictionary.

        Returns:
            Category score dictionary.
        """
        weight    = ATS_WEIGHTS["education"]
        education = resume_data.get("education", [])

        if not education:
            return self._make_entry(
                0, weight,
                "No education information found. Add your degree details."
            )

        score = 0.0
        notes: list[str] = []
        edu   = education[0]  # Use highest/first degree

        # Sub-score 1: Has education entry (4 pts)
        score += 4
        notes.append("Education section found.")

        # Sub-score 2: Degree level (up to 4 pts)
        degree = edu.get("degree", "")
        degree_scores = {
            "PhD":        4,
            "Master's":   4,
            "Bachelor's": 3,
            "Associate":  2,
            "High School":1,
        }
        deg_score = degree_scores.get(degree, 2)
        score += deg_score
        if degree:
            notes.append(f"{degree} degree detected.")
        else:
            notes.append("Degree level not clearly identified.")

        # Sub-score 3: GPA mentioned (2 pts)
        if edu.get("gpa"):
            score += 2
            notes.append(f"GPA listed: {edu['gpa']}.")

        score = clamp(score, 0, weight)
        feedback = " ".join(notes)
        return self._make_entry(score, weight, feedback)

    def _score_projects(self, resume_data: dict) -> dict:
        """
        Score projects category (weight: 10).

        Evaluates:
          - Number of projects
          - Projects have descriptions
          - Projects mention technologies
          - Projects have links

        Args:
            resume_data: Parsed resume dictionary.

        Returns:
            Category score dictionary.
        """
        weight   = ATS_WEIGHTS["projects"]
        projects = resume_data.get("projects", [])

        if not projects:
            return self._make_entry(
                0, weight,
                "No projects found. Add 2-4 projects with tech stack and links."
            )

        score = 0.0
        notes: list[str] = []

        # Sub-score 1: Number of projects (up to 4 pts)
        if len(projects) >= 3:
            score += 4
            notes.append(f"{len(projects)} projects listed.")
        elif len(projects) == 2:
            score += 3
            notes.append("2 projects listed.")
        else:
            score += 2
            notes.append("1 project listed. Add more projects.")

        # Sub-score 2: Projects have descriptions (up to 3 pts)
        with_desc = sum(1 for p in projects if p.get("description"))
        if with_desc >= 2:
            score += 3
            notes.append("Projects have descriptions.")
        elif with_desc == 1:
            score += 1
            notes.append("Add descriptions to all projects.")

        # Sub-score 3: Projects mention technologies (up to 2 pts)
        with_tech = sum(1 for p in projects if p.get("technologies"))
        if with_tech >= 1:
            score += 2
            notes.append("Tech stack mentioned in projects.")
        else:
            notes.append("Add tech stack to project descriptions.")

        # Sub-score 4: Projects have links (1 pt)
        with_link = sum(1 for p in projects if p.get("link"))
        if with_link >= 1:
            score += 1
            notes.append("Project links included.")
        else:
            notes.append("Add GitHub/demo links to projects.")

        score = clamp(score, 0, weight)
        feedback = " ".join(notes)
        return self._make_entry(score, weight, feedback)

    def _score_certifications(self, resume_data: dict) -> dict:
        """
        Score certifications category (weight: 5).

        Args:
            resume_data: Parsed resume dictionary.

        Returns:
            Category score dictionary.
        """
        weight = ATS_WEIGHTS["certifications"]
        certs  = resume_data.get("certifications", [])

        if not certs:
            return self._make_entry(
                0, weight,
                "No certifications found. Add relevant certifications."
            )

        if len(certs) >= 3:
            score    = weight
            feedback = f"Excellent! {len(certs)} certifications listed."
        elif len(certs) == 2:
            score    = weight * 0.8
            feedback = f"2 certifications found. Consider adding more."
        else:
            score    = weight * 0.6
            feedback = f"1 certification found. Add more relevant certifications."

        return self._make_entry(clamp(score, 0, weight), weight, feedback)

    def _score_keywords(
        self,
        resume_data:  dict,
        match_result: dict,
    ) -> dict:
        """
        Score keyword density category (weight: 10).

        Based on keyword_match_pct from the matcher result.

        Args:
            resume_data:  Parsed resume dictionary.
            match_result: Match result dictionary.

        Returns:
            Category score dictionary.
        """
        weight      = ATS_WEIGHTS["keyword_density"]
        keyword_pct = match_result.get("keyword_match_pct", 0.0)

        raw_score = (keyword_pct / 100) * weight
        score     = clamp(raw_score, 0, weight)

        if keyword_pct >= 75:
            feedback = (
                f"Excellent keyword coverage ({keyword_pct:.0f}%). "
                "Resume is well-aligned with the JD."
            )
        elif keyword_pct >= 50:
            feedback = (
                f"Good keyword coverage ({keyword_pct:.0f}%). "
                "Add more JD-specific terms."
            )
        elif keyword_pct >= 25:
            feedback = (
                f"Average keyword coverage ({keyword_pct:.0f}%). "
                "Incorporate more keywords from the job description."
            )
        else:
            feedback = (
                f"Low keyword coverage ({keyword_pct:.0f}%). "
                "Tailor your resume to match the job description language."
            )

        return self._make_entry(score, weight, feedback)

    def _score_formatting(self, resume_data: dict) -> dict:
        """
        Score formatting quality category (weight: 5).

        Evaluates formatting signals:
          - All key sections present
          - Contact info complete
          - LinkedIn and GitHub present
          - No excessively long lines (graphics/tables)

        Args:
            resume_data: Parsed resume dictionary.

        Returns:
            Category score dictionary.
        """
        weight = ATS_WEIGHTS["formatting"]
        score  = 0.0
        notes: list[str] = []

        # Check key sections present
        sections = resume_data.get("sections_found", [])
        key_sections = {"experience", "education", "skills"}
        present = key_sections & set(sections)

        if len(present) == 3:
            score += 2
            notes.append("All key sections present.")
        elif len(present) == 2:
            score += 1
            notes.append("Some key sections missing.")
        else:
            notes.append("Add Experience, Education, and Skills sections.")

        # Contact info completeness
        has_email    = bool(resume_data.get("email"))
        has_phone    = bool(resume_data.get("phone"))
        has_linkedin = bool(resume_data.get("linkedin"))
        has_github   = bool(resume_data.get("github"))

        contact_score = sum([has_email, has_phone, has_linkedin, has_github])
        if contact_score >= 3:
            score += 2
            notes.append("Contact information is complete.")
        elif contact_score == 2:
            score += 1
            notes.append("Add LinkedIn and/or GitHub to contact info.")
        else:
            notes.append("Contact information is incomplete.")

        # Name present
        if resume_data.get("name") and resume_data["name"] != "Unknown Candidate":
            score += 1
            notes.append("Name clearly identified.")
        else:
            notes.append("Ensure your name is at the top of the resume.")

        score = clamp(score, 0, weight)
        feedback = " ".join(notes)
        return self._make_entry(score, weight, feedback)

    def _score_length(self, resume_data: dict) -> dict:
        """
        Score resume length category (weight: 5).

        Optimal range: 400–1200 words.

        Args:
            resume_data: Parsed resume dictionary.

        Returns:
            Category score dictionary.
        """
        weight     = ATS_WEIGHTS["resume_length"]
        word_count = resume_data.get("word_count", 0)
        label      = get_resume_length_label(word_count)

        length_scores = {
            "Optimal":   weight,
            "Good":      weight * 0.8,
            "Short":     weight * 0.6,
            "Too Short": weight * 0.2,
            "Too Long":  weight * 0.4,
        }

        length_feedback = {
            "Optimal":   f"Resume length is optimal ({word_count} words).",
            "Good":      f"Resume length is good ({word_count} words).",
            "Short":     f"Resume is a bit short ({word_count} words). Expand your experience.",
            "Too Short": f"Resume is too short ({word_count} words). Add more detail.",
            "Too Long":  f"Resume is too long ({word_count} words). Aim for 400–1200 words.",
        }

        score    = clamp(length_scores.get(label, weight * 0.5), 0, weight)
        feedback = length_feedback.get(
            label, f"Resume has {word_count} words."
        )
        return self._make_entry(score, weight, feedback)

    def _score_summary(self, resume_data: dict) -> dict:
        """
        Score professional summary category (weight: 5).

        Evaluates:
          - Summary present
          - Summary length (not too short, not too long)

        Args:
            resume_data: Parsed resume dictionary.

        Returns:
            Category score dictionary.
        """
        weight  = ATS_WEIGHTS["summary"]
        summary = resume_data.get("summary", "")

        if not summary:
            return self._make_entry(
                0, weight,
                "No professional summary found. Add a 3-5 sentence summary."
            )

        word_count = count_words(summary)

        if 30 <= word_count <= 100:
            score    = weight
            feedback = f"Professional summary is well-written ({word_count} words)."
        elif 15 <= word_count < 30:
            score    = weight * 0.7
            feedback = (
                f"Summary is a bit short ({word_count} words). "
                "Expand to 3-5 sentences."
            )
        elif word_count > 100:
            score    = weight * 0.6
            feedback = (
                f"Summary is too long ({word_count} words). "
                "Keep it concise (3-5 sentences)."
            )
        else:
            score    = weight * 0.4
            feedback = "Summary is too brief. Write 3-5 impactful sentences."

        return self._make_entry(clamp(score, 0, weight), weight, feedback)

    # ------------------------------------------------------------------
    # OVERALL SCORE
    # ------------------------------------------------------------------

    def _compute_overall(self, breakdown: dict[str, dict]) -> float:
        """
        Sum all category scores to produce the overall ATS score.

        Args:
            breakdown: Per-category score dictionary.

        Returns:
            Overall ATS score (0–100).
        """
        total = sum(info["score"] for info in breakdown.values())
        return clamp(round(total, 1), 0.0, 100.0)

    # ------------------------------------------------------------------
    # STRENGTHS & WEAKNESSES ANALYSER
    # ------------------------------------------------------------------

    def _analyse_breakdown(
        self,
        breakdown: dict[str, dict],
    ) -> tuple[list[str], list[str]]:
        """
        Identify strengths and weaknesses from the category breakdown.

        A category is a strength  if its pct >= 75%.
        A category is a weakness  if its pct <  50%.

        Args:
            breakdown: Per-category score dictionary.

        Returns:
            Tuple of (strengths, weaknesses) as lists of strings.
        """
        strengths:  list[str] = []
        weaknesses: list[str] = []

        # Human-readable category labels
        labels = {
            "skills_match":    "Skills Match",
            "experience":      "Work Experience",
            "education":       "Education",
            "projects":        "Projects",
            "certifications":  "Certifications",
            "keyword_density": "Keyword Density",
            "formatting":      "Resume Formatting",
            "resume_length":   "Resume Length",
            "summary":         "Professional Summary",
        }

        for category, info in breakdown.items():
            pct   = info["pct"]
            label = labels.get(category, category.replace("_", " ").title())

            if pct >= 75:
                strengths.append(f"{label} ({pct:.0f}%)")
            elif pct < 50:
                weaknesses.append(f"{label} ({pct:.0f}%)")

        return strengths, weaknesses

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _make_entry(score: float, weight: int, feedback: str) -> dict:
        """
        Build a standardised category score entry.

        Args:
            score:    Achieved score (0 – weight).
            weight:   Maximum possible score for this category.
            feedback: Human-readable feedback string.

        Returns:
            Dictionary with score, max, pct, and feedback.
        """
        pct = percentage(score, weight) if weight > 0 else 0.0
        return {
            "score":    round(score, 1),
            "max":      weight,
            "pct":      round(pct, 1),
            "feedback": feedback,
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return a fully structured empty result dictionary."""
        return {
            "overall_score":   0.0,
            "label":           "Poor",
            "emoji":           "❌",
            "color":           "#FF6B6B",
            "breakdown":       {},
            "category_scores": {},
            "strengths":       [],
            "weaknesses":      [],
            "max_score":       100,
        }

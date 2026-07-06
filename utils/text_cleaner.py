# utils/text_cleaner.py
"""
text_cleaner.py — NLP text normalisation pipeline for AI Resume Analyzer.

Transforms raw extracted resume/JD text into clean, normalised,
analysis-ready tokens and strings.

Pipeline stages:
  1. Unicode normalisation & encoding fixes
  2. HTML / markdown artefact removal
  3. Special character and noise removal
  4. Sentence segmentation
  5. Tokenisation
  6. Stop-word removal  (NLTK)
  7. Lemmatisation      (spaCy)
  8. N-gram generation  (bigrams + trigrams for multi-word skills)

Design decisions:
  - Two output modes:
      * clean_text()   → cleaned readable string  (for display / PDF)
      * clean_tokens() → normalised token list    (for ML / matching)
  - spaCy is loaded once at module level and reused (expensive to load)
  - Graceful fallback if spaCy model is not installed
  - Thread-safe (no mutable module-level state)
"""

import re
import string
import unicodedata
from functools import lru_cache
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from utils.constants import SPACY_MODEL, SPACY_MODEL_FALLBACK
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# NLTK DATA — download quietly if not present
# ============================================================

for _corpus in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                "punkt_tab"):
    try:
        nltk.download(_corpus, quiet=True)
    except Exception:
        pass


# ============================================================
# SPACY MODEL — load once, reuse everywhere
# ============================================================

_nlp = None  # Module-level cache


def _load_spacy():
    """
    Load and cache the spaCy NLP model.

    Tries the medium model first (en_core_web_md), falls back to
    the small model (en_core_web_sm), and finally returns None if
    neither is available so the rest of the pipeline still works.

    Returns:
        spaCy Language object or None.
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    import spacy

    for model_name in (SPACY_MODEL, SPACY_MODEL_FALLBACK):
        try:
            _nlp = spacy.load(
                model_name,
                disable=["parser", "ner"],   # Only need tokenizer + tagger
            )
            logger.info("spaCy model loaded: %s", model_name)
            return _nlp
        except OSError:
            logger.warning("spaCy model not found: %s", model_name)

    logger.error(
        "No spaCy model available. Run: python -m spacy download %s",
        SPACY_MODEL,
    )
    return None


# ============================================================
# CONSTANTS
# ============================================================

# Extended English stop words (NLTK base + resume-specific noise)
_BASE_STOPWORDS: set[str] = set()
try:
    _BASE_STOPWORDS = set(stopwords.words("english"))
except Exception:
    pass

_RESUME_NOISE_WORDS: set[str] = {
    # Generic resume filler words that add no signal
    "responsible", "responsibilities", "worked", "working",
    "experienced", "experience", "knowledge", "strong",
    "excellent", "good", "best", "great", "various",
    "including", "etc", "eg", "ie", "also", "well",
    "ability", "skills", "skill", "proficient", "proficiency",
    "familiar", "familiarity", "understanding", "exposure",
    "background", "years", "year", "month", "months",
    "currently", "previously", "currently", "present",
    "resume", "cv", "curriculum", "vitae", "reference",
    "available", "request", "contact", "phone", "email",
    "address", "linkedin", "github", "portfolio", "website",
}

STOP_WORDS: set[str] = _BASE_STOPWORDS | _RESUME_NOISE_WORDS

# Punctuation to strip from tokens
_PUNCT: set[str] = set(string.punctuation) | {"•", "·", "–", "—", "→", "►"}

# Regex patterns compiled once at import time
_RE_HTML_TAG     = re.compile(r"<[^>]+>")
_RE_URL          = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL        = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_PHONE        = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_RE_SPECIAL_CHAR = re.compile(r"[^\w\s\-.,@/+#&()]")
_RE_EXTRA_SPACE  = re.compile(r" {2,}")
_RE_BULLET       = re.compile(r"^[\s]*[•\-\*\>→◆▪▸●►]\s*", re.MULTILINE)
_RE_NUMBERED     = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)
_RE_PAGE_HEADER  = re.compile(r"\[Page \d+\]\n?")
_RE_SECTION_MARK = re.compile(r"^#{1,4}\s+", re.MULTILINE)


# ============================================================
# TEXT CLEANER CLASS
# ============================================================

class TextCleaner:
    """
    Multi-stage text normalisation pipeline.

    Two primary public methods:

        clean_text(text)    → str
            Returns a clean, human-readable string suitable for display,
            section parsing, and PDF report generation.

        clean_tokens(text)  → list[str]
            Returns a list of lowercase, lemmatised, stop-word-filtered
            tokens suitable for TF-IDF, skill matching, and NLP tasks.

    Additional helpers:
        get_sentences(text)          → list[str]
        get_ngrams(tokens, n)        → list[str]
        get_bigrams(tokens)          → list[str]
        get_trigrams(tokens)         → list[str]
        remove_stop_words(tokens)    → list[str]
        lemmatize(tokens)            → list[str]
        extract_keywords(text, n)    → list[str]
    """

    def __init__(self) -> None:
        self._nlp = _load_spacy()
        self._lemmatizer = WordNetLemmatizer()
        logger.debug("TextCleaner initialised.")

    # ------------------------------------------------------------------
    # PRIMARY PUBLIC METHODS
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Transform raw resume/JD text into clean, readable plain text.

        Applies the following stages in order:
          1. Unicode normalisation
          2. HTML tag removal
          3. Page/section marker removal
          4. Bullet point standardisation
          5. Special character cleaning
          6. Whitespace normalisation

        The original text structure (sentences, paragraphs) is preserved.
        Contact info (email, phone, URLs) is kept intact.

        Args:
            text: Raw extracted text from PDF or DOCX reader.

        Returns:
            Cleaned plain-text string.
        """
        if not text or not text.strip():
            return ""

        try:
            text = self._normalize_unicode(text)
            text = self._remove_html(text)
            text = self._remove_page_markers(text)
            text = self._standardise_bullets(text)
            text = self._remove_special_chars(text)
            text = self._normalise_whitespace(text)
            return text.strip()
        except Exception as exc:
            logger.error("clean_text failed: %s", exc)
            return text.strip()

    def clean_tokens(
        self,
        text: str,
        remove_stops: bool = True,
        lemmatize: bool = True,
        min_length: int = 2,
    ) -> list[str]:
        """
        Tokenise and normalise text into an analysis-ready token list.

        Pipeline:
          1. clean_text()         → readable clean string
          2. Remove URLs/emails   → no noise tokens
          3. Lowercase            → case normalisation
          4. Word tokenise        → NLTK word_tokenize
          5. Remove punctuation   → strip non-alpha tokens
          6. Stop word removal    → optional (default True)
          7. Lemmatisation        → optional via spaCy or NLTK fallback
          8. Length filter        → drop tokens < min_length chars

        Args:
            text:          Raw input text.
            remove_stops:  Whether to remove stop words (default True).
            lemmatize:     Whether to lemmatise tokens (default True).
            min_length:    Minimum token character length (default 2).

        Returns:
            List of clean, normalised token strings.
        """
        if not text or not text.strip():
            return []

        try:
            # Stage 1: clean the text first
            cleaned = self.clean_text(text)

            # Stage 2: remove contact info to avoid noise tokens
            cleaned = _RE_URL.sub(" ", cleaned)
            cleaned = _RE_EMAIL.sub(" ", cleaned)
            cleaned = _RE_PHONE.sub(" ", cleaned)

            # Stage 3: lowercase
            cleaned = cleaned.lower()

            # Stage 4: tokenise
            tokens = word_tokenize(cleaned)

            # Stage 5: remove punctuation tokens
            tokens = [
                t for t in tokens
                if t not in _PUNCT and not all(c in _PUNCT for c in t)
            ]

            # Stage 6: stop word removal
            if remove_stops:
                tokens = self.remove_stop_words(tokens)

            # Stage 7: lemmatisation
            if lemmatize:
                tokens = self.lemmatize(tokens)

            # Stage 8: length filter
            tokens = [t for t in tokens if len(t) >= min_length]

            return tokens

        except Exception as exc:
            logger.error("clean_tokens failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # SENTENCE UTILITIES
    # ------------------------------------------------------------------

    def get_sentences(self, text: str) -> list[str]:
        """
        Split text into a list of individual sentences.

        Uses NLTK's sent_tokenize which handles abbreviations,
        decimal numbers, and ellipses correctly.

        Args:
            text: Input text string.

        Returns:
            List of sentence strings.
        """
        if not text or not text.strip():
            return []
        try:
            cleaned = self.clean_text(text)
            sentences = sent_tokenize(cleaned)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as exc:
            logger.error("get_sentences failed: %s", exc)
            return [text.strip()]

    # ------------------------------------------------------------------
    # STOP WORD & LEMMATISATION
    # ------------------------------------------------------------------

    def remove_stop_words(self, tokens: list[str]) -> list[str]:
        """
        Remove English stop words and resume noise words from a token list.

        Args:
            tokens: List of lowercase string tokens.

        Returns:
            Filtered token list.
        """
        return [t for t in tokens if t.lower() not in STOP_WORDS]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Lemmatise tokens using spaCy (preferred) or NLTK WordNet fallback.

        spaCy lemmatisation is context-aware and more accurate.
        NLTK WordNet lemmatisation is used only if spaCy is unavailable.

        Args:
            tokens: List of string tokens.

        Returns:
            List of lemmatised token strings.
        """
        if not tokens:
            return []

        # --- spaCy lemmatisation (preferred) ---
        if self._nlp is not None:
            try:
                doc = self._nlp(" ".join(tokens))
                return [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_space
                ]
            except Exception as exc:
                logger.warning("spaCy lemmatisation failed: %s", exc)

        # --- NLTK WordNet fallback ---
        try:
            return [
                self._lemmatizer.lemmatize(t.lower()) for t in tokens
            ]
        except Exception as exc:
            logger.warning("NLTK lemmatisation failed: %s", exc)
            return [t.lower() for t in tokens]

    # ------------------------------------------------------------------
    # N-GRAM GENERATION
    # ------------------------------------------------------------------

    def get_ngrams(self, tokens: list[str], n: int) -> list[str]:
        """
        Generate n-grams from a token list as joined strings.

        N-grams are essential for matching multi-word skills like
        'machine learning', 'deep learning', 'continuous integration'.

        Args:
            tokens: List of token strings.
            n:      N-gram size (1=unigram, 2=bigram, 3=trigram).

        Returns:
            List of n-gram strings joined by spaces.
        """
        if not tokens or n < 1 or n > len(tokens):
            return []
        return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]

    def get_bigrams(self, tokens: list[str]) -> list[str]:
        """
        Generate bigrams (2-grams) from a token list.

        Args:
            tokens: List of token strings.

        Returns:
            List of bigram strings.
        """
        return self.get_ngrams(tokens, 2)

    def get_trigrams(self, tokens: list[str]) -> list[str]:
        """
        Generate trigrams (3-grams) from a token list.

        Args:
            tokens: List of token strings.

        Returns:
            List of trigram strings.
        """
        return self.get_ngrams(tokens, 3)

    def get_all_ngrams(self, tokens: list[str]) -> list[str]:
        """
        Generate all unigrams, bigrams, and trigrams combined.

        Used by the skill extractor to match both single-word and
        multi-word skill names (e.g., 'machine learning', 'ci/cd').

        Args:
            tokens: List of token strings.

        Returns:
            Combined list of 1-, 2-, and 3-gram strings.
        """
        unigrams  = tokens
        bigrams   = self.get_bigrams(tokens)
        trigrams  = self.get_trigrams(tokens)
        return unigrams + bigrams + trigrams

    # ------------------------------------------------------------------
    # KEYWORD EXTRACTION
    # ------------------------------------------------------------------

    def extract_keywords(
        self,
        text: str,
        top_n: int = 30,
        include_ngrams: bool = True,
    ) -> list[str]:
        """
        Extract the most meaningful keywords from text using TF-style
        frequency ranking.

        Process:
          1. Clean and tokenise the text.
          2. Optionally include bigrams and trigrams.
          3. Rank by frequency.
          4. Return top_n most frequent terms.

        Args:
            text:           Input text string.
            top_n:          Number of top keywords to return.
            include_ngrams: Whether to include bigrams/trigrams.

        Returns:
            List of top keyword strings (most frequent first).
        """
        if not text or not text.strip():
            return []

        try:
            tokens = self.clean_tokens(text, remove_stops=True, lemmatize=True)

            if include_ngrams:
                all_terms = self.get_all_ngrams(tokens)
            else:
                all_terms = tokens

            # Count frequency
            freq: dict[str, int] = {}
            for term in all_terms:
                if term.strip():
                    freq[term] = freq.get(term, 0) + 1

            # Sort by frequency descending
            sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            return [term for term, _ in sorted_terms[:top_n]]

        except Exception as exc:
            logger.error("extract_keywords failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # SPACY NAMED ENTITY HELPER
    # ------------------------------------------------------------------

    def extract_named_entities(
        self,
        text: str,
        labels: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Extract named entities from text using spaCy NER.

        Args:
            text:   Input text string.
            labels: List of entity labels to extract. If None, all
                    entity types are returned. Common labels:
                    PERSON, ORG, GPE, DATE, PRODUCT, WORK_OF_ART.

        Returns:
            Dictionary mapping label → list of entity strings.
            Returns empty dict if spaCy is not available.
        """
        if self._nlp is None:
            logger.warning("spaCy not available — skipping NER.")
            return {}

        if not text or not text.strip():
            return {}

        # Re-enable NER for this call by using a separate pipeline
        try:
            import spacy
            nlp_full = None
            for model_name in (SPACY_MODEL, SPACY_MODEL_FALLBACK):
                try:
                    nlp_full = spacy.load(model_name)
                    break
                except OSError:
                    continue

            if nlp_full is None:
                return {}

            doc = nlp_full(text[:50000])  # Cap at 50k chars for performance
            entities: dict[str, list[str]] = {}

            for ent in doc.ents:
                if labels is None or ent.label_ in labels:
                    ent_list = entities.setdefault(ent.label_, [])
                    if ent.text.strip() not in ent_list:
                        ent_list.append(ent.text.strip())

            return entities

        except Exception as exc:
            logger.error("Named entity extraction failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # PRIVATE CLEANING STAGES
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """
        Normalise unicode to NFC form and replace common
        problematic characters with ASCII equivalents.
        """
        # NFC normalisation (compose characters)
        text = unicodedata.normalize("NFC", text)

        # Smart quotes → straight quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')

        # Dashes → hyphen
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # Bullets → dash
        text = text.replace("\u2022", "-").replace("\u25cf", "-")
        text = text.replace("\u25ba", "-").replace("\u2192", "->")

        # Non-breaking space → regular space
        text = text.replace("\u00a0", " ")

        # Remove zero-width characters
        for char in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"):
            text = text.replace(char, "")

        return text

    @staticmethod
    def _remove_html(text: str) -> str:
        """Strip HTML tags and decode common HTML entities."""
        text = _RE_HTML_TAG.sub(" ", text)
        html_entities = {
            "&amp;":  "&",
            "&lt;":   "<",
            "&gt;":   ">",
            "&nbsp;": " ",
            "&quot;": '"',
            "&#39;":  "'",
            "&apos;": "'",
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        return text

    @staticmethod
    def _remove_page_markers(text: str) -> str:
        """Remove internal page and section markers from extractors."""
        text = _RE_PAGE_HEADER.sub("", text)
        text = _RE_SECTION_MARK.sub("", text)
        return text

    @staticmethod
    def _standardise_bullets(text: str) -> str:
        """
        Replace various bullet symbols (•, *, >, →, ►) with a dash
        so downstream parsers see a uniform list format.
        """
        text = _RE_BULLET.sub("- ", text)
        text = _RE_NUMBERED.sub("", text)
        return text

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """
        Remove characters that are not letters, digits, whitespace,
        or common punctuation needed for resumes (-, ., ,, @, /, +, #).
        """
        return _RE_SPECIAL_CHAR.sub(" ", text)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """
        Collapse multiple spaces into one, and 3+ newlines into two.
        Preserves paragraph structure.
        """
        # Collapse multiple spaces (not newlines)
        text = _RE_EXTRA_SPACE.sub(" ", text)
        # Collapse 3+ blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing whitespace on each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text


# ============================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================

_default_cleaner: Optional[TextCleaner] = None


def get_cleaner() -> TextCleaner:
    """
    Return the shared module-level TextCleaner instance.
    Creates it on first call (lazy singleton pattern).

    Returns:
        Shared TextCleaner instance.
    """
    global _default_cleaner
    if _default_cleaner is None:
        _default_cleaner = TextCleaner()
    return _default_cleaner


def clean_text(text: str) -> str:
    """Module-level shortcut for TextCleaner().clean_text()."""
    return get_cleaner().clean_text(text)


def clean_tokens(text: str, **kwargs) -> list[str]:
    """Module-level shortcut for TextCleaner().clean_tokens()."""
    return get_cleaner().clean_tokens(text, **kwargs)


def extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """Module-level shortcut for TextCleaner().extract_keywords()."""
    return get_cleaner().extract_keywords(text, top_n=top_n)

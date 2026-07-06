# utils/pdf_reader.py
"""
pdf_reader.py — Robust PDF text extraction for AI Resume Analyzer.

Strategy:
  1. Primary  — pdfplumber  (best for structured/tabular PDFs)
  2. Fallback — PyPDF2      (handles more edge-case PDFs)
  3. Both extractors are tried; the one returning more text wins.

Handles:
  - Password-protected PDFs        → friendly error
  - Corrupted / unreadable PDFs    → friendly error
  - Scanned image-only PDFs        → detected and warned
  - Multi-page PDFs                → all pages concatenated
  - PDFs with tables               → table text preserved
  - Empty / blank PDFs             → detected and warned
  - Oversized files                → rejected before processing
"""

import io
import re
from pathlib import Path
from typing import Optional

import pdfplumber
import PyPDF2

from utils.constants import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
from utils.helpers import (
    clean_whitespace,
    count_words,
    get_file_size_mb,
    validate_file_size,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFReadError(Exception):
    """Raised when a PDF cannot be read or parsed."""


class PDFReader:
    """
    Extracts plain text from PDF resume files.

    Supports both file-path-based and bytes-based reading so it works
    seamlessly with Streamlit's UploadedFile (bytes) and local files.

    Usage:
        reader = PDFReader()

        # From an uploaded file (Streamlit)
        result = reader.extract(file_bytes=uploaded_file.read())

        # From a local file path
        result = reader.extract(file_path="data/resume.pdf")

    Returns a dict:
        {
            "text":       str,   # full extracted text
            "pages":      int,   # number of pages
            "word_count": int,   # total word count
            "method":     str,   # "pdfplumber" | "PyPDF2" | "combined"
            "warnings":   list,  # any non-fatal warnings
            "success":    bool,
            "error":      str,   # populated only when success=False
        }
    """

    # Minimum characters to consider extraction successful
    MIN_CHARS: int = 50

    def __init__(self) -> None:
        self._warnings: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        file_bytes: Optional[bytes] = None,
        file_path: Optional[str | Path] = None,
    ) -> dict:
        """
        Extract text from a PDF supplied as raw bytes or a file path.

        Exactly one of file_bytes or file_path must be provided.

        Args:
            file_bytes: Raw PDF bytes (e.g. from Streamlit uploader).
            file_path:  Path to a local PDF file.

        Returns:
            Extraction result dictionary (see class docstring).
        """
        self._warnings = []

        try:
            raw_bytes = self._load_bytes(file_bytes, file_path)
            return self._process(raw_bytes)
        except PDFReadError as exc:
            logger.error("PDFReadError: %s", exc)
            return self._error_result(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error reading PDF: %s", exc)
            return self._error_result(
                f"An unexpected error occurred while reading the PDF: {exc}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_bytes(
        self,
        file_bytes: Optional[bytes],
        file_path: Optional[str | Path],
    ) -> bytes:
        """
        Resolve the PDF source into raw bytes.

        Args:
            file_bytes: Optional raw bytes.
            file_path:  Optional file path.

        Returns:
            Raw PDF bytes.

        Raises:
            PDFReadError: If neither or invalid source is provided.
        """
        if file_bytes is not None:
            return file_bytes

        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise PDFReadError(f"File not found: {path}")
            if not path.is_file():
                raise PDFReadError(f"Path is not a file: {path}")
            if path.suffix.lower() != ".pdf":
                raise PDFReadError(
                    f"File '{path.name}' is not a PDF. "
                    f"Got extension: '{path.suffix}'"
                )
            return path.read_bytes()

        raise PDFReadError(
            "No PDF source provided. Supply either file_bytes or file_path."
        )

    def _process(self, raw_bytes: bytes) -> dict:
        """
        Validate then extract text from raw PDF bytes.

        Args:
            raw_bytes: PDF file content.

        Returns:
            Extraction result dictionary.

        Raises:
            PDFReadError: On validation failures.
        """
        # --- Size validation ---
        is_valid, size_msg = validate_file_size(raw_bytes, MAX_FILE_SIZE_MB)
        if not is_valid:
            raise PDFReadError(size_msg)

        logger.info(
            "Processing PDF — size: %.2f MB", get_file_size_mb(raw_bytes)
        )

        # --- Try pdfplumber (primary) ---
        plumber_result = self._extract_with_pdfplumber(raw_bytes)

        # --- Try PyPDF2 (fallback / comparison) ---
        pypdf_result = self._extract_with_pypdf2(raw_bytes)

        # --- Choose the better result ---
        text, method, page_count = self._select_best(
            plumber_result, pypdf_result
        )

        # --- Post-process ---
        text = self._post_process(text)
        word_count = count_words(text)

        # --- Detect image-only (scanned) PDFs ---
        if len(text.strip()) < self.MIN_CHARS:
            self._warnings.append(
                "Very little text was extracted. This PDF may be a scanned "
                "image. Please use a text-based PDF for best results."
            )
            logger.warning("Low text extraction — possibly a scanned PDF.")

        logger.info(
            "PDF extracted via %s — %d pages, %d words",
            method, page_count, word_count,
        )

        return {
            "text":       text,
            "pages":      page_count,
            "word_count": word_count,
            "method":     method,
            "warnings":   self._warnings,
            "success":    True,
            "error":      "",
        }

    # ------------------------------------------------------------------
    # Extractor: pdfplumber
    # ------------------------------------------------------------------

    def _extract_with_pdfplumber(self, raw_bytes: bytes) -> dict:
        """
        Extract text using pdfplumber.

        pdfplumber is excellent at preserving layout, extracting text
        from tables, and handling complex multi-column PDFs.

        Args:
            raw_bytes: PDF file content.

        Returns:
            Dict with keys: text, pages, success.
        """
        try:
            pages_text: list[str] = []

            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                # Detect password protection
                if pdf.pages is None:
                    raise PDFReadError(
                        "PDF appears to be password-protected or unreadable."
                    )

                page_count = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = ""

                    # 1. Extract standard text
                    raw_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13,
                    )
                    if raw_text:
                        page_text += raw_text + "\n"

                    # 2. Extract text from tables separately
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            row_text = " | ".join(
                                cell.strip() if cell else ""
                                for cell in row
                                if cell
                            )
                            if row_text.strip():
                                page_text += row_text + "\n"

                    pages_text.append(page_text)
                    logger.debug(
                        "pdfplumber — page %d: %d chars",
                        page_num, len(page_text),
                    )

            combined = "\n\n".join(
                f"[Page {i+1}]\n{t}" for i, t in enumerate(pages_text) if t.strip()
            )
            return {"text": combined, "pages": page_count, "success": True}

        except PDFReadError:
            raise
        except Exception as exc:
            logger.warning("pdfplumber extraction failed: %s", exc)
            return {"text": "", "pages": 0, "success": False}

    # ------------------------------------------------------------------
    # Extractor: PyPDF2
    # ------------------------------------------------------------------

    def _extract_with_pypdf2(self, raw_bytes: bytes) -> dict:
        """
        Extract text using PyPDF2 as a fallback extractor.

        PyPDF2 handles some PDFs that pdfplumber struggles with,
        particularly older or non-standard PDF formats.

        Args:
            raw_bytes: PDF file content.

        Returns:
            Dict with keys: text, pages, success.
        """
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))

            # Detect encryption
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise PDFReadError(
                        "This PDF is password-protected. "
                        "Please provide an unprotected PDF."
                    )

            page_count = len(reader.pages)
            pages_text: list[str] = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                    logger.debug(
                        "PyPDF2 — page %d: %d chars",
                        page_num, len(page_text),
                    )
                except Exception as page_exc:
                    logger.warning(
                        "PyPDF2 — failed to extract page %d: %s",
                        page_num, page_exc,
                    )
                    pages_text.append("")

            combined = "\n\n".join(
                f"[Page {i+1}]\n{t}" for i, t in enumerate(pages_text) if t.strip()
            )
            return {"text": combined, "pages": page_count, "success": True}

        except PDFReadError:
            raise
        except PyPDF2.errors.PdfReadError as exc:
            logger.warning("PyPDF2 PdfReadError: %s", exc)
            return {"text": "", "pages": 0, "success": False}
        except Exception as exc:
            logger.warning("PyPDF2 extraction failed: %s", exc)
            return {"text": "", "pages": 0, "success": False}

    # ------------------------------------------------------------------
    # Selector: choose the richer extraction
    # ------------------------------------------------------------------

    def _select_best(
        self,
        plumber: dict,
        pypdf: dict,
    ) -> tuple[str, str, int]:
        """
        Compare both extraction results and return the richer one.

        Decision rule: whichever extractor produced more characters wins.
        If both succeed, we also try combining them for maximum coverage.

        Args:
            plumber: pdfplumber result dict.
            pypdf:   PyPDF2 result dict.

        Returns:
            Tuple of (text, method_name, page_count).

        Raises:
            PDFReadError: If both extractors completely failed.
        """
        plumber_len = len(plumber.get("text", ""))
        pypdf_len   = len(pypdf.get("text", ""))

        logger.debug(
            "Extraction comparison — pdfplumber: %d chars | PyPDF2: %d chars",
            plumber_len, pypdf_len,
        )

        if not plumber["success"] and not pypdf["success"]:
            raise PDFReadError(
                "Failed to extract text from the PDF. "
                "The file may be corrupted, encrypted, or image-based."
            )

        # Both succeeded — return richer result and note the other
        if plumber["success"] and pypdf["success"]:
            if plumber_len >= pypdf_len:
                self._warnings.append(
                    f"Extracted {plumber_len} characters via pdfplumber."
                )
                return (
                    plumber["text"],
                    "pdfplumber",
                    plumber["pages"],
                )
            else:
                self._warnings.append(
                    f"Extracted {pypdf_len} characters via PyPDF2."
                )
                return (
                    pypdf["text"],
                    "PyPDF2",
                    pypdf["pages"],
                )

        # Only one succeeded
        if plumber["success"]:
            return plumber["text"], "pdfplumber", plumber["pages"]

        return pypdf["text"], "PyPDF2", pypdf["pages"]

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, text: str) -> str:
        """
        Clean and normalise extracted PDF text.

        Steps:
          1. Remove page markers added during extraction.
          2. Collapse excessive blank lines (max 2 consecutive).
          3. Fix common PDF extraction artefacts (ligatures, hyphens).
          4. Strip leading/trailing whitespace.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        # Remove internal page markers
        text = re.sub(r"\[Page \d+\]\n?", "", text)

        # Fix common PDF ligature issues (fi, fl, ff, ffi, ffl)
        ligature_map = {
            "\ufb00": "ff",
            "\ufb01": "fi",
            "\ufb02": "fl",
            "\ufb03": "ffi",
            "\ufb04": "ffl",
            "\ufb05": "st",
            "\ufb06": "st",
        }
        for ligature, replacement in ligature_map.items():
            text = text.replace(ligature, replacement)

        # Remove soft hyphens and zero-width characters
        text = text.replace("\u00ad", "")   # soft hyphen
        text = text.replace("\u200b", "")   # zero-width space
        text = text.replace("\u200c", "")   # zero-width non-joiner
        text = text.replace("\u200d", "")   # zero-width joiner
        text = text.replace("\ufeff", "")   # BOM

        # Rejoin hyphenated line-break words (e.g., "develop-\nment")
        text = re.sub(r"-\n(\w)", r"\1", text)

        # Collapse 3+ consecutive blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove non-printable characters except standard whitespace
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(message: str) -> dict:
        """Build a standardised error result dictionary."""
        return {
            "text":       "",
            "pages":      0,
            "word_count": 0,
            "method":     "none",
            "warnings":   [],
            "success":    False,
            "error":      message,
        }

    # ------------------------------------------------------------------
    # Convenience class method
    # ------------------------------------------------------------------

    @classmethod
    def read(
        cls,
        file_bytes: Optional[bytes] = None,
        file_path: Optional[str | Path] = None,
    ) -> dict:
        """
        One-liner convenience wrapper.

        Args:
            file_bytes: Raw PDF bytes.
            file_path:  Path to a PDF file.

        Returns:
            Extraction result dictionary.

        Example:
            result = PDFReader.read(file_bytes=uploaded_file.read())
            if result["success"]:
                text = result["text"]
        """
        return cls().extract(file_bytes=file_bytes, file_path=file_path)

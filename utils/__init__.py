# utils/__init__.py
"""
Lazy module registry for the AI Resume Analyzer utility package.

Imports are deferred until each module file exists, which allows the
project to be built incrementally (one file at a time) without raising
ImportError on modules that haven't been created yet.

Once all modules are present, you can import directly:
    from utils import ATSScorer, ResumeParser, ChartGenerator
"""

import importlib
from typing import Any


def _lazy_import(module_name: str, class_name: str) -> Any:
    """Attempt to import a class from a submodule; return None if not ready."""
    try:
        module = importlib.import_module(f"utils.{module_name}")
        return getattr(module, class_name, None)
    except (ImportError, ModuleNotFoundError):
        return None


# --- Lazily expose all public classes ---
PDFReader         = _lazy_import("pdf_reader",       "PDFReader")
DOCXReader        = _lazy_import("docx_reader",      "DOCXReader")
TextCleaner       = _lazy_import("text_cleaner",     "TextCleaner")
ResumeParser      = _lazy_import("resume_parser",    "ResumeParser")
JDParser          = _lazy_import("jd_parser",        "JDParser")
SkillExtractor    = _lazy_import("skill_extractor",  "SkillExtractor")
ATSScorer         = _lazy_import("ats_score",        "ATSScorer")
ResumeMatcher     = _lazy_import("matcher",          "ResumeMatcher")
ChartGenerator    = _lazy_import("charts",           "ChartGenerator")
ReportGenerator   = _lazy_import("report_generator", "ReportGenerator")
get_logger        = _lazy_import("logger",           "get_logger")

__all__ = [
    "PDFReader",
    "DOCXReader",
    "TextCleaner",
    "ResumeParser",
    "JDParser",
    "SkillExtractor",
    "ATSScorer",
    "ResumeMatcher",
    "ChartGenerator",
    "ReportGenerator",
    "get_logger",
]

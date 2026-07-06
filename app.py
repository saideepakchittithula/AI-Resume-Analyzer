"""
app.py — AI Resume Analyzer  |  Streamlit entry-point
"""

import streamlit as st
import os, sys

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy imports (after path setup) ──────────────────────────────────────────
from utils.pdf_reader        import PDFReader
from utils.docx_reader       import DOCXReader
from utils.text_cleaner      import TextCleaner
from utils.resume_parser     import ResumeParser
from utils.jd_parser         import JDParser
from utils.skill_extractor   import SkillExtractor
from utils.matcher           import ResumeMatcher
from utils.ats_score         import ATSScorer
from utils.charts            import (gauge_chart, category_bar_chart,
                                     skill_match_donut, keyword_bar_chart,
                                     radar_chart, experience_timeline)
from utils.report_generator  import ReportGenerator

# ── global singletons (cached) ────────────────────────────────────────────────
@st.cache_resource
def _load_resources():
    return {
        "cleaner":   TextCleaner(),
        "r_parser":  ResumeParser(),
        "jd_parser": JDParser(),
        "extractor": SkillExtractor(),
        "matcher":   ResumeMatcher(),
        "scorer":    ATSScorer(),
        "reporter":  ReportGenerator(),
    }

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* light background */
  .stApp { background-color: #ffffff; color: #1a1a1a; }
  section[data-testid="stSidebar"] { background-color: #f5f7fa; }

  /* metric cards */
  div[data-testid="metric-container"] {
    background: #f5f7fa;
    border: 1px solid #dde1e7;
    border-radius: 10px;
    padding: 14px 18px;
  }

  /* tab styling */
  button[data-baseweb="tab"] {
    font-size: 14px !important;
    color: #555 !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a73e8 !important;
    border-bottom: 2px solid #1a73e8 !important;
  }

  /* pill tags */
  .pill-green { background:#e6f9ee; color:#1e8449; padding:3px 10px;
                border-radius:12px; font-size:12px; margin:2px;
                display:inline-block; }
  .pill-red   { background:#fdecea; color:#c0392b; padding:3px 10px;
                border-radius:12px; font-size:12px; margin:2px;
                display:inline-block; }
  .pill-grey  { background:#f0f0f0; color:#555; padding:3px 10px;
                border-radius:12px; font-size:12px; margin:2px;
                display:inline-block; }
  .pill-blue  { background:#e8f0fe; color:#1a73e8; padding:3px 10px;
                border-radius:12px; font-size:12px; margin:2px;
                display:inline-block; }

  /* section cards */
  .card {
    background: #f5f7fa;
    border: 1px solid #dde1e7;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
  }
  .card h4 { color: #1a73e8; margin-bottom: 10px; font-size: 14px;
             text-transform: uppercase; letter-spacing: 0.4px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def _pills(items: list, css_class: str) -> str:
    return "".join(f'<span class="{css_class}">{i}</span>' for i in items)


def _read_resume(uploaded) -> tuple[str, list]:
    """Return (text, warnings) from an uploaded file."""
    warnings = []
    ext = os.path.splitext(uploaded.name)[1].lower()
    data = uploaded.read()

    if ext == ".pdf":
        result = PDFReader().read(file_bytes=data)
    elif ext in (".docx", ".doc"):
        result = DOCXReader().read(file_bytes=data)
    else:
        return "", [f"Unsupported file type: {ext}"]

    if not result.get("success"):
        return "", [result.get("error", "Unknown read error")]

    warnings = result.get("warnings", [])
    return result.get("text", ""), warnings


def _run_analysis(resume_text: str, jd_text: str, res: dict) -> dict:
    """Run the full pipeline and return a results dict."""
    cleaner   = res["cleaner"]
    r_parser  = res["r_parser"]
    jd_parser = res["jd_parser"]
    extractor = res["extractor"]
    matcher   = res["matcher"]
    scorer    = res["scorer"]

    clean_resume = cleaner.clean_text(resume_text)
    clean_jd     = cleaner.clean_text(jd_text)

    resume_data  = r_parser.parse(resume_text)
    jd_data      = jd_parser.parse(jd_text)

    skill_cmp    = extractor.compare(resume_text, jd_text)
    match_result = matcher.match(resume_data, jd_data)

    # merge skill comparison into match_result
    match_result.setdefault("matched_skills", skill_cmp.get("matched", []))
    match_result.setdefault("missing_skills", skill_cmp.get("missing", []))
    match_result["extra_skills"] = match_result.get("additional_skills", [])

    ats_result = scorer.score(resume_data, jd_data, match_result)

    return dict(
        resume_data  = resume_data,
        jd_data      = jd_data,
        skill_cmp    = skill_cmp,
        match_result = match_result,
        ats_result   = ats_result,
        clean_resume = clean_resume,
        clean_jd     = clean_jd,
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
def _sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## AI Resume Analyzer")
        st.markdown("---")

        st.markdown("### Upload Resume")
        uploaded = st.file_uploader(
            "PDF or DOCX", type=["pdf", "docx"],
            label_visibility="collapsed"
        )

        st.markdown("### Job Description")
        jd_source = st.radio("Source", ["Paste text", "Upload file"],
                             horizontal=True, label_visibility="collapsed")

        jd_text = ""
        if jd_source == "Paste text":
            jd_text = st.text_area("Paste JD here", height=220,
                                   placeholder="Paste the job description…",
                                   label_visibility="collapsed")
        else:
            jd_file = st.file_uploader("JD file", type=["txt", "pdf", "docx"],
                                       label_visibility="collapsed",
                                       key="jd_upload")
            if jd_file:
                ext = os.path.splitext(jd_file.name)[1].lower()
                if ext == ".txt":
                    jd_text = jd_file.read().decode("utf-8", errors="ignore")
                elif ext == ".pdf":
                    r = PDFReader().read(file_bytes=jd_file.read())
                    jd_text = r.get("text", "")
                elif ext in (".docx", ".doc"):
                    r = DOCXReader().read(file_bytes=jd_file.read())
                    jd_text = r.get("text", "")

        st.markdown("---")
        analyze_btn = st.button("Analyze", use_container_width=True,
                                type="primary")

        st.markdown("---")
        st.caption("Built with Streamlit · spaCy · scikit-learn")

    return uploaded, jd_text.strip(), analyze_btn


# ── tab renderers ─────────────────────────────────────────────────────────────
def _tab_dashboard(results: dict):
    ats   = results["ats_result"]
    score = ats.get("overall_score", 0)
    label = ats.get("label", "")
    bd    = ats.get("breakdown", {})

    # top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ATS Score",       f"{score:.1f}/100")
    m2.metric("Rating",          label)
    m3.metric("Skills Matched",  len(results["match_result"].get("matched_skills", [])))
    m4.metric("Skills Missing",  len(results["match_result"].get("missing_skills", [])))

    st.markdown("---")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(gauge_chart(score), use_container_width=True)
    with c2:
        st.plotly_chart(radar_chart(bd),    use_container_width=True)

    st.plotly_chart(category_bar_chart(bd), use_container_width=True)

    # strengths / weaknesses
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="card"><h4>Strengths</h4>', unsafe_allow_html=True)
        for s in ats.get("strengths", []):
            st.markdown(f'<span class="pill-green">{s}</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="card"><h4>Areas to Improve</h4>',
                    unsafe_allow_html=True)
        for w in ats.get("weaknesses", []):
            st.markdown(f'<span class="pill-red">{w}</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _tab_skills(results: dict):
    match = results["match_result"]

    matched = match.get("matched_skills", [])
    missing = match.get("missing_skills", [])
    extra   = match.get("extra_skills",   [])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(
            skill_match_donut(matched, missing, extra),
            use_container_width=True
        )
    with c2:
        skill_score = match.get("skill_match_pct", 0)
        st.metric("Skill Match Score", f"{skill_score:.1f}%")
        st.progress(min(int(skill_score), 100))
        st.markdown(f"**Matched:** {len(matched)}  |  "
                    f"**Missing:** {len(missing)}  |  "
                    f"**Extra:** {len(extra)}")

        suggestions = match.get("suggestions", [])
        if suggestions:
            st.markdown("**Suggestions**")
            for s in suggestions:
                st.markdown(f"- {s}")

    st.markdown("---")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Matched Skills**")
        st.markdown(_pills(matched, "pill-green") or "_None_", unsafe_allow_html=True)
    with t2:
        st.markdown("**Missing Skills**")
        st.markdown(_pills(missing, "pill-red") or "_None_", unsafe_allow_html=True)
    with t3:
        st.markdown("**Extra Skills**")
        st.markdown(_pills(extra, "pill-grey") or "_None_", unsafe_allow_html=True)

    # skills by category
    extractor = _load_resources()["extractor"]
    resume_by_cat = extractor.get_skills_by_category(
        results["resume_data"].get("skills", [])
    )
    if resume_by_cat:
        st.markdown("---")
        st.markdown("**Resume Skills by Category**")
        cols = st.columns(3)
        for i, (cat, skills) in enumerate(resume_by_cat.items()):
            with cols[i % 3]:
                st.markdown(f"*{cat}*")
                st.markdown(
                    _pills(skills, "pill-blue"),
                    unsafe_allow_html=True
                )


def _tab_keywords(results: dict):
    match = results["match_result"]
    r_kws = match.get("resume_keywords", {}) or {}
    j_kws = match.get("jd_keywords",     {}) or {}

    kw_score  = match.get("keyword_match_pct",       0)
    sem_score = match.get("semantic_similarity_pct",  0)
    overall   = match.get("overall_compatibility_pct", 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Keyword Match",   f"{kw_score:.1f}%")
    m2.metric("Semantic Match",  f"{sem_score:.1f}%")
    m3.metric("Overall Match",   f"{overall:.1f}%")

    if r_kws and j_kws:
        st.plotly_chart(
            keyword_bar_chart(r_kws, j_kws),
            use_container_width=True
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Resume Keywords**")
        for kw, freq in list(r_kws.items())[:15]:
            st.markdown(f"`{kw}` — {freq}")
    with c2:
        st.markdown("**Top JD Keywords**")
        for kw, freq in list(j_kws.items())[:15]:
            st.markdown(f"`{kw}` — {freq}")


def _tab_resume(results: dict):
    rd = results["resume_data"]
    jd = results["jd_data"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Candidate")
        fields = [
            ("Name",       rd.get("name")),
            ("Email",      rd.get("email")),
            ("Phone",      rd.get("phone")),
            ("LinkedIn",   rd.get("linkedin")),
            ("GitHub",     rd.get("github")),
            ("Experience", f"{rd.get('experience_years', 0)} yrs"),
            ("Words",      rd.get("word_count")),
        ]
        for k, v in fields:
            if v:
                st.markdown(f"**{k}:** {v}")

        if rd.get("summary"):
            st.markdown("**Summary:**")
            st.info(rd["summary"])

        exp = rd.get("experience", [])
        if exp:
            st.markdown("**Experience Timeline**")
            st.plotly_chart(experience_timeline(exp), use_container_width=True)

    with c2:
        st.markdown("### Job Description")
        jd_fields = [
            ("Title",      jd.get("job_title")),
            ("Company",    jd.get("company")),
            ("Location",   jd.get("location")),
            ("Type",       jd.get("employment_type")),
            ("Level",      jd.get("experience_level")),
            ("Remote",     "Yes" if jd.get("is_remote") else "No"),
            ("Exp Needed", f"{jd.get('experience_years', 0)} yrs"),
        ]
        for k, v in jd_fields:
            if v:
                st.markdown(f"**{k}:** {v}")

        req_skills = jd.get("required_skills", [])
        if req_skills:
            st.markdown("**Required Skills:**")
            st.markdown(_pills(req_skills, "pill-blue"), unsafe_allow_html=True)

        pref_skills = jd.get("preferred_skills", [])
        if pref_skills:
            st.markdown("**Preferred Skills:**")
            st.markdown(_pills(pref_skills, "pill-grey"), unsafe_allow_html=True)

    # education
    edu = rd.get("education", [])
    if edu:
        st.markdown("---")
        st.markdown("**Education**")
        for e in edu:
            degree  = e.get("degree",      "")
            field   = e.get("field",        "")
            school  = e.get("institution",  "")
            year    = e.get("year",         "")
            parts   = filter(None, [degree, field, school, year])
            st.markdown(f"- {' · '.join(parts)}")

    # certifications
    certs = rd.get("certifications", [])
    if certs:
        st.markdown("**Certifications**")
        for c in certs:
            st.markdown(f"- {c}")


def _tab_report(results: dict, reporter: ReportGenerator):
    ats   = results["ats_result"]
    match = results["match_result"]
    rd    = results["resume_data"]
    jd    = results["jd_data"]

    score = ats.get("overall_score", 0)
    label = ats.get("label", "")

    st.markdown(f"### Report Ready — **{score:.1f}/100** ({label})")
    st.markdown("Download a self-contained HTML report with all analysis details.")

    html  = reporter.generate(ats, match, rd, jd)
    raw   = reporter.to_bytes(html)

    name  = rd.get("name", "candidate").replace(" ", "_").lower()
    fname = f"resume_analysis_{name}.html"

    st.download_button(
        label="Download HTML Report",
        data=raw,
        file_name=fname,
        mime="text/html",
        use_container_width=True,
        type="primary",
    )

    with st.expander("Preview report HTML"):
        st.components.v1.html(html, height=600, scrolling=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    res = _load_resources()
    uploaded, jd_text, analyze_btn = _sidebar()

    st.title("AI Resume Analyzer")
    st.caption("Upload a resume and paste a job description to get an ATS score, "
               "skill gap analysis, and improvement suggestions.")

    # ── session state ─────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = None
    if "warnings" not in st.session_state:
        st.session_state.warnings = []

    # ── trigger analysis ──────────────────────────────────────────────────────
    if analyze_btn:
        if not uploaded:
            st.error("Please upload a resume (PDF or DOCX).")
        elif not jd_text:
            st.error("Please provide a job description.")
        else:
            with st.spinner("Analyzing…"):
                resume_text, warnings = _read_resume(uploaded)
                if not resume_text:
                    st.error("Could not extract text from the resume.")
                else:
                    st.session_state.warnings = warnings
                    st.session_state.results  = _run_analysis(
                        resume_text, jd_text, res
                    )

    # ── display warnings ──────────────────────────────────────────────────────
    for w in st.session_state.warnings:
        st.warning(w)

    # ── render tabs ───────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results
        tabs = st.tabs([
            "Dashboard",
            "Skill Analysis",
            "Keywords",
            "Resume Details",
            "Report",
        ])
        with tabs[0]: _tab_dashboard(results)
        with tabs[1]: _tab_skills(results)
        with tabs[2]: _tab_keywords(results)
        with tabs[3]: _tab_resume(results)
        with tabs[4]: _tab_report(results, res["reporter"])

    else:
        # landing state
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown("""
<div class="card">
<h4>Upload Resume</h4>
<p style="color:#555;font-size:13px">
  Supports PDF and DOCX formats.<br>
  Text is extracted automatically.
</p>
</div>""", unsafe_allow_html=True)
        c2.markdown("""
<div class="card">
<h4>Paste Job Description</h4>
<p style="color:#555;font-size:13px">
  Copy the full JD from any job board.<br>
  Or upload a TXT / PDF / DOCX file.
</p>
</div>""", unsafe_allow_html=True)
        c3.markdown("""
<div class="card">
<h4>Get Your ATS Score</h4>
<p style="color:#555;font-size:13px">
  9-category scoring, skill gap analysis,<br>
  keyword comparison, and a PDF report.
</p>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

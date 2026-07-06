"""
charts.py — Plotly chart builders for the AI Resume Analyzer dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from utils.logger import get_logger

log = get_logger(__name__)

# ── colour palette ────────────────────────────────────────────────────────────
_SCORE_COLORS = {
    "Excellent": "#2ecc71",
    "Good":      "#3498db",
    "Fair":      "#f39c12",
    "Poor":      "#e74c3c",
}
_CAT_COLOR   = "#4A90D9"
_MATCH_COLORS = ["#2ecc71", "#e74c3c", "#95a5a6"]   # matched / missing / extra


# ── helpers ───────────────────────────────────────────────────────────────────
def _score_color(score: float) -> str:
    if score >= 80: return _SCORE_COLORS["Excellent"]
    if score >= 60: return _SCORE_COLORS["Good"]
    if score >= 40: return _SCORE_COLORS["Fair"]
    return _SCORE_COLORS["Poor"]


def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    base.update(kwargs)
    return base


# ── public chart functions ────────────────────────────────────────────────────

def gauge_chart(score: float, label: str = "ATS Score") -> go.Figure:
    """Gauge / speedometer for the overall ATS score."""
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 60, "increasing": {"color": "#2ecc71"},
               "decreasing": {"color": "#e74c3c"}},
        title={"text": label, "font": {"size": 18, "color": "#e0e0e0"}},
        number={"suffix": "/100", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#555", "tickfont": {"color": "#aaa"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e1e2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#2d1b1b"},
                {"range": [40, 60], "color": "#2d2510"},
                {"range": [60, 80], "color": "#0f2233"},
                {"range": [80, 100], "color": "#0f2d1a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(**_base_layout(height=280))
    return fig


def category_bar_chart(breakdown: dict) -> go.Figure:
    """Horizontal bar chart for ATS category scores."""
    if not breakdown:
        return go.Figure()

    cats   = list(breakdown.keys())
    scores = [breakdown[c].get("score", 0) for c in cats]
    maxes  = [breakdown[c].get("max",   10) for c in cats]
    pcts   = [round(s / m * 100, 1) if m else 0 for s, m in zip(scores, maxes)]
    labels = [c.replace("_", " ").title() for c in cats]
    colors = [_score_color(p) for p in pcts]

    fig = go.Figure(go.Bar(
        x=pcts, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s}/{m}" for s, m in zip(scores, maxes)],
        textposition="outside",
        textfont=dict(color="#ccc", size=11),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Category Breakdown", font=dict(size=15)),
            xaxis=dict(range=[0, 115], showgrid=False, zeroline=False,
                       ticksuffix="%", tickfont=dict(color="#aaa")),
            yaxis=dict(showgrid=False, tickfont=dict(color="#ddd")),
            height=320,
            bargap=0.35,
        )
    )
    return fig


def skill_match_donut(matched: list, missing: list,
                      extra: Optional[list] = None) -> go.Figure:
    """Donut chart: matched / missing / extra skills."""
    extra = extra or []
    counts = [len(matched), len(missing), len(extra)]
    labels = ["Matched", "Missing", "Extra"]
    # drop zero slices
    data = [(l, c, col) for l, c, col in zip(labels, counts, _MATCH_COLORS) if c > 0]
    if not data:
        return go.Figure()
    labels, counts, colors = zip(*data)

    fig = go.Figure(go.Pie(
        labels=list(labels),
        values=list(counts),
        hole=0.55,
        marker=dict(colors=list(colors), line=dict(color="#1e1e2e", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#e0e0e0"),
        hovertemplate="%{label}: %{value} skills<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Skill Match Overview", font=dict(size=15)),
            showlegend=False,
            height=300,
        )
    )
    return fig


def keyword_bar_chart(resume_kws: dict, jd_kws: dict,
                      top_n: int = 15) -> go.Figure:
    """Side-by-side bars comparing top keyword frequencies."""
    all_kws = sorted(
        set(list(resume_kws)[:top_n] + list(jd_kws)[:top_n]),
        key=lambda k: jd_kws.get(k, 0), reverse=True
    )[:top_n]

    r_vals = [resume_kws.get(k, 0) for k in all_kws]
    j_vals = [jd_kws.get(k, 0)    for k in all_kws]

    fig = go.Figure([
        go.Bar(name="Resume", x=all_kws, y=r_vals,
               marker_color="#4A90D9",
               hovertemplate="%{x}: %{y}<extra>Resume</extra>"),
        go.Bar(name="Job Description", x=all_kws, y=j_vals,
               marker_color="#E8A838",
               hovertemplate="%{x}: %{y}<extra>JD</extra>"),
    ])
    fig.update_layout(
        **_base_layout(
            title=dict(text="Keyword Frequency Comparison", font=dict(size=15)),
            barmode="group",
            xaxis=dict(tickangle=-35, tickfont=dict(color="#ccc", size=10),
                       showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#333",
                       tickfont=dict(color="#aaa")),
            legend=dict(font=dict(color="#ccc")),
            height=340,
        )
    )
    return fig


def radar_chart(breakdown: dict) -> go.Figure:
    """Radar / spider chart for category scores (% of max)."""
    if not breakdown:
        return go.Figure()

    cats   = list(breakdown.keys())
    labels = [c.replace("_", " ").title() for c in cats]
    pcts   = [
        round(breakdown[c].get("score", 0) / breakdown[c].get("max", 10) * 100, 1)
        if breakdown[c].get("max") else 0
        for c in cats
    ]
    # close the polygon
    labels_closed = labels + [labels[0]]
    pcts_closed   = pcts   + [pcts[0]]

    fig = go.Figure(go.Scatterpolar(
        r=pcts_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(74,144,217,0.25)",
        line=dict(color="#4A90D9", width=2),
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            polar=dict(
                bgcolor="#1e1e2e",
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(color="#888", size=9),
                                gridcolor="#333"),
                angularaxis=dict(tickfont=dict(color="#ccc", size=11),
                                 gridcolor="#333"),
            ),
            title=dict(text="Skill Radar", font=dict(size=15)),
            height=360,
        )
    )
    return fig


def experience_timeline(experience: list) -> go.Figure:
    """Horizontal Gantt-style bars for work experience entries."""
    if not experience:
        return go.Figure()

    import re
    rows = []
    for i, exp in enumerate(experience):
        title   = exp.get("title",   f"Role {i+1}")
        company = exp.get("company", "")
        dates   = exp.get("dates",   "")
        label   = f"{title} @ {company}" if company else title

        # try to parse years from date string
        years = re.findall(r"\b(19|20)\d{2}\b", dates)
        start = int(years[0]) if len(years) >= 1 else 2020 - i
        end   = int(years[1]) if len(years) >= 2 else start + 1

        rows.append(dict(label=label, start=start, end=end, duration=end - start))

    rows.sort(key=lambda r: r["start"])
    labels   = [r["label"]    for r in rows]
    starts   = [r["start"]    for r in rows]
    durations = [max(r["duration"], 0.5) for r in rows]

    fig = go.Figure(go.Bar(
        x=durations, y=labels,
        base=starts,
        orientation="h",
        marker_color=_CAT_COLOR,
        hovertemplate="%{y}<br>%{base} – %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Experience Timeline", font=dict(size=15)),
            xaxis=dict(title="Year", tickfont=dict(color="#aaa"),
                       showgrid=True, gridcolor="#333"),
            yaxis=dict(tickfont=dict(color="#ddd"), autorange="reversed"),
            height=max(200, 60 * len(rows)),
        )
    )
    return fig

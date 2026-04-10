import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_dataframe, apply_dark_theme
from utils.logger import get_logger
from utils.styles import inject_global_css

logger = get_logger(__name__)

st.set_page_config(page_title="Data Quality", page_icon="🛡️", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.quality-shell {
    position: relative;
}

.quality-shell::before {
    content: "";
    position: absolute;
    inset: -30px 0 auto 0;
    height: 340px;
    background:
        radial-gradient(circle at 50% 12%, rgba(72, 177, 255, 0.16), transparent 38%),
        radial-gradient(circle at 72% 18%, rgba(93, 255, 162, 0.12), transparent 28%);
    pointer-events: none;
}

.quality-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 0.66fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.5rem 0 1rem 0;
}

.quality-copy,
.quality-art {
    position: relative;
    z-index: 1;
}

.quality-kicker {
    color: #91c7ff;
    font: 600 0.8rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.quality-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.55rem;
}

.quality-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(20, 68, 45, 0.98), rgba(14, 39, 35, 0.98));
    border: 1px solid rgba(98, 255, 162, 0.24);
    box-shadow: 0 0 24px rgba(98, 255, 162, 0.16);
    font-size: 2rem;
}

.quality-title h1 {
    margin: 0;
    color: #f8fbff;
    font: 800 clamp(2rem, 4.8vw, 3.25rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.quality-sub {
    margin: 0.65rem 0 0 5.2rem;
    color: #a8bbd8;
    font: 400 1.02rem/1.7 "IBM Plex Sans", sans-serif;
}

.quality-art {
    min-height: 190px;
}

.clipboard-glow {
    position: absolute;
    right: 10%;
    top: 22px;
    width: 170px;
    height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(116, 255, 173, 0.18), transparent 68%);
    filter: blur(14px);
}

.clipboard {
    position: absolute;
    right: 38px;
    top: 14px;
    width: 150px;
    height: 180px;
    border-radius: 18px;
    background: linear-gradient(180deg, #d8e4ef, #93a8be);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 34px rgba(4, 11, 22, 0.38);
    transform: rotate(6deg);
}

.clipboard::before {
    content: "";
    position: absolute;
    left: 46px;
    top: -12px;
    width: 56px;
    height: 22px;
    border-radius: 10px 10px 6px 6px;
    background: linear-gradient(180deg, #eef5fb, #a5bbcd);
}

.clipboard::after {
    content: "";
    position: absolute;
    inset: 16px 14px 14px 14px;
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(248, 251, 255, 0.96), rgba(219, 232, 242, 0.96));
}

.clipboard-line {
    position: absolute;
    left: 58px;
    width: 62px;
    height: 4px;
    border-radius: 999px;
    background: rgba(42, 70, 102, 0.55);
    z-index: 2;
}

.clipboard-line.l1 { top: 52px; }
.clipboard-line.l2 { top: 78px; }
.clipboard-line.l3 { top: 104px; }
.clipboard-line.l4 { top: 130px; width: 50px; }

.check-box {
    position: absolute;
    left: 28px;
    width: 18px;
    height: 18px;
    border-radius: 4px;
    z-index: 2;
}

.check-box.c1 { top: 45px; background: #31bb7c; }
.check-box.c2 { top: 71px; background: #31bb7c; }
.check-box.c3 { top: 97px; background: #ef4d4d; }
.check-box.c4 { top: 123px; background: transparent; border: 2px solid rgba(42, 70, 102, 0.45); box-sizing: border-box; }

.shield {
    position: absolute;
    right: -10px;
    bottom: -10px;
    width: 110px;
    height: 128px;
    background: linear-gradient(180deg, #79ff82, #1d8f38);
    clip-path: polygon(50% 0, 88% 14%, 88% 56%, 50% 100%, 12% 56%, 12% 14%);
    box-shadow: 0 0 24px rgba(118, 255, 130, 0.24);
    z-index: 3;
}

.shield::before {
    content: "";
    position: absolute;
    inset: 10px;
    clip-path: polygon(50% 0, 88% 14%, 88% 56%, 50% 100%, 12% 56%, 12% 14%);
    background: linear-gradient(180deg, #52d35d, #166d2b);
}

.shield::after {
    content: "✓";
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -54%);
    color: #eefeff;
    font: 800 3rem "Inter", sans-serif;
}

.hero-rule {
    height: 2px;
    margin: 0.9rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(95, 255, 157, 0.88), transparent);
    box-shadow: 0 0 24px rgba(95, 255, 157, 0.22);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.2rem 0;
}

.metric-card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(82, 136, 215, 0.58);
    background: linear-gradient(180deg, rgba(20, 37, 67, 0.96), rgba(11, 23, 43, 0.98));
    text-align: center;
}

.metric-card.good {
    border-color: rgba(96, 255, 146, 0.5);
    box-shadow: inset 0 -12px 24px rgba(81, 255, 111, 0.08);
}

.metric-card.bad {
    border-color: rgba(255, 112, 112, 0.44);
    box-shadow: inset 0 -12px 24px rgba(255, 83, 83, 0.06);
}

.metric-title {
    color: #dfeaff;
    font: 600 0.98rem "IBM Plex Sans", sans-serif;
}

.metric-value {
    margin-top: 0.35rem;
    color: #eef7ff;
    font: 800 1.95rem "Inter", sans-serif;
}

.metric-value.green {
    color: #8eff92;
}

.metric-value.red {
    color: #ff646e;
}

.preview-wrap,
.table-wrap {
    padding: 0.8rem;
}

.checks-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.check-card {
    min-height: 132px;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid rgba(88, 132, 211, 0.58);
    background: linear-gradient(180deg, rgba(18, 33, 60, 0.96), rgba(10, 23, 42, 0.98));
}

.check-card.warn {
    border-color: rgba(255, 186, 82, 0.48);
}

.check-card.bad {
    border-color: rgba(255, 104, 104, 0.48);
}

.check-card.good {
    border-color: rgba(98, 255, 162, 0.42);
}

.check-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px;
    height: 54px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    font-size: 1.7rem;
}

.check-title {
    margin-top: 0.7rem;
    color: #edf6ff;
    font: 700 1.05rem "Inter", sans-serif;
}

.check-desc {
    margin-top: 0.4rem;
    color: #9cb7d8;
    font: 400 0.84rem/1.6 "IBM Plex Sans", sans-serif;
}

.warn-bar {
    margin: 1.4rem 0 1.2rem 0;
    padding: 0.95rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 96, 96, 0.34);
    background: linear-gradient(90deg, rgba(61, 18, 28, 0.92), rgba(34, 23, 42, 0.95));
    color: #ffd4d6;
    font: 500 0.95rem "IBM Plex Sans", sans-serif;
}

.score-panel {
    position: relative;
    min-height: 320px;
    overflow: hidden;
    margin: 0.8rem 0 1.4rem 0;
}

.score-grid {
    position: absolute;
    left: 8%;
    right: 8%;
    bottom: 22px;
    height: 92px;
    background:
        linear-gradient(rgba(68, 169, 255, 0.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(68, 169, 255, 0.13) 1px, transparent 1px);
    background-size: 28px 28px;
    transform: perspective(380px) rotateX(68deg);
    opacity: 0.7;
}

.score-dots {
    position: absolute;
    left: 12%;
    right: 12%;
    bottom: 84px;
    height: 110px;
}

.score-dots span {
    position: absolute;
    border-radius: 50%;
    box-shadow: 0 0 12px currentColor;
}

.score-dots .b1 { left: 6%; bottom: 14px; width: 8px; height: 8px; color: #76c5ff; background: currentColor; }
.score-dots .b2 { left: 12%; bottom: 36px; width: 12px; height: 12px; color: #76c5ff; background: currentColor; }
.score-dots .b3 { left: 20%; bottom: 24px; width: 10px; height: 10px; color: #76c5ff; background: currentColor; }
.score-dots .b4 { left: 72%; bottom: 18px; width: 10px; height: 10px; color: #ff5b5b; background: currentColor; }
.score-dots .b5 { left: 79%; bottom: 44px; width: 12px; height: 12px; color: #ff5b5b; background: currentColor; }
.score-dots .b6 { left: 86%; bottom: 28px; width: 8px; height: 8px; color: #76c5ff; background: currentColor; }
.score-dots .b7 { left: 32%; bottom: 56px; width: 6px; height: 6px; color: #76c5ff; background: currentColor; }
.score-dots .b8 { left: 64%; bottom: 60px; width: 6px; height: 6px; color: #76c5ff; background: currentColor; }

.gauge {
    position: absolute;
    left: 50%;
    bottom: 46px;
    width: 250px;
    height: 170px;
    transform: translateX(-50%);
    border-radius: 180px 180px 24px 24px;
    background: linear-gradient(180deg, rgba(22, 40, 69, 0.98), rgba(11, 22, 39, 0.98));
    box-shadow: 0 18px 30px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(181, 226, 255, 0.08);
}

.gauge-arc {
    position: absolute;
    left: 50%;
    top: 16px;
    width: 188px;
    height: 94px;
    transform: translateX(-50%);
    border-radius: 188px 188px 0 0;
    background: conic-gradient(from 180deg, #ff4040 0deg 42deg, #ffcd3c 42deg 90deg, #8eff45 90deg 156deg);
    clip-path: inset(0 0 50% 0);
}

.gauge-arc::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 14px;
    width: 146px;
    height: 73px;
    transform: translateX(-50%);
    border-radius: 146px 146px 0 0;
    background: linear-gradient(180deg, rgba(20, 37, 67, 0.98), rgba(13, 24, 43, 0.98));
    clip-path: inset(0 0 50% 0);
}

.gauge-value {
    position: absolute;
    left: 50%;
    top: 78px;
    transform: translateX(-50%);
    color: #eef8ff;
    font: 800 2.25rem "Inter", sans-serif;
}

.action-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}

@media (max-width: 1100px) {
    .quality-hero {
        grid-template-columns: 1fr;
    }
    .metric-grid,
    .checks-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .metric-grid,
    .checks-grid,
    .action-row {
        grid-template-columns: 1fr;
    }
    .quality-sub {
        margin-left: 0;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

df = get_dataframe()
if df is None:
    st.warning("⚠️ Please upload a file on Home page first.")
    st.stop()

outlier_log = st.session_state.get("outlier_log")
total_cells = df.shape[0] * df.shape[1]
missing = int(df.isnull().sum().sum())
duplicates = int(df.duplicated().sum())

completeness = (1 - missing / total_cells) * 100 if total_cells else 0
uniqueness = (1 - duplicates / len(df)) * 100 if len(df) else 0

quality_data = []
high_missing_cols = 0
id_like_cols = 0
mixed_type_cols = 0
for col in df.columns:
    missing_pct = round((df[col].isnull().sum() / len(df)) * 100, 2) if len(df) else 0
    unique_vals = df[col].nunique()

    issues = []
    if missing_pct > 30:
        issues.append("High Missing")
        high_missing_cols += 1
    if unique_vals == len(df):
        issues.append("ID Column")
        id_like_cols += 1
    if unique_vals > 50:
        issues.append("Too Many Unique")
    if str(df[col].dtype) == "object":
        mixed_type_cols += 1
    if not issues:
        issues.append("Good")

    quality_data.append([col, str(df[col].dtype), missing_pct, unique_vals, ", ".join(issues)])

quality_df = pd.DataFrame(
    quality_data, columns=["Column", "Type", "Missing %", "Unique", "Issues"]
)

integrity_issues = missing + duplicates
outlier_bonus = 0.0
if outlier_log and outlier_log.get("outlier_count", 0) > 0:
    handled_ratio = outlier_log["outlier_count"] / max(outlier_log.get("rows_before", len(df)), 1)
    outlier_bonus = min(5.0, handled_ratio * 100)
consistency_score = max(0, min(100, round((completeness * 0.55) + (uniqueness * 0.45) + outlier_bonus, 2)))

st.markdown('<div class="quality-shell">', unsafe_allow_html=True)
st.markdown(
    """
<div class="quality-hero">
    <div class="quality-copy">
        <div class="quality-kicker">Reliability review workflow</div>
        <div class="quality-title">
            <div class="quality-icon">🛡️</div>
            <h1>Data Quality Assessment</h1>
        </div>
        <div class="quality-sub">Ensure your data is analysis-ready before moving into deeper insights and modeling.</div>
    </div>
    <div class="quality-art">
        <div class="clipboard-glow"></div>
        <div class="clipboard"></div>
        <div class="check-box c1"></div>
        <div class="check-box c2"></div>
        <div class="check-box c3"></div>
        <div class="check-box c4"></div>
        <div class="clipboard-line l1"></div>
        <div class="clipboard-line l2"></div>
        <div class="clipboard-line l3"></div>
        <div class="clipboard-line l4"></div>
        <div class="shield"></div>
    </div>
</div>
<div class="hero-rule"></div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-head">
    <h2>Data Overview</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

if outlier_log:
    st.markdown(
        f"""
<div style="margin:0.3rem 0 1rem 0;padding:0.95rem 1rem;border-radius:16px;border:1px solid rgba(98,255,162,0.34);background:linear-gradient(90deg, rgba(17,54,37,0.94), rgba(14,33,42,0.96));color:#d9ffea;font:500 0.95rem 'IBM Plex Sans',sans-serif;">
    <b style="color:#ffffff">{outlier_log['action']}</b> was applied on column
    <b style="color:#ffffff">{outlier_log['column']}</b> using
    <b style="color:#ffffff">{outlier_log['method']}</b> detection.
    &nbsp;|&nbsp; {outlier_log['outlier_count']} outliers detected
    &nbsp;|&nbsp; Rows: {outlier_log['rows_before']} → {outlier_log['rows_after']}
    ({'-' if outlier_log['rows_removed'] >= 0 else '+'}{abs(outlier_log['rows_removed'])} removed)
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
<div class="metric-grid">
    <div class="metric-card good">
        <div class="metric-title">Completeness Score</div>
        <div class="metric-value green">{round(completeness, 2)}%</div>
    </div>
    <div class="metric-card bad">
        <div class="metric-title">Integrity Issues</div>
        <div class="metric-value red">{integrity_issues}</div>
    </div>
    <div class="metric-card good">
        <div class="metric-title">Consistency Score</div>
        <div class="metric-value green">{consistency_score}%</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if completeness > 95:
    st.success("✅ Excellent data quality!")
elif completeness > 80:
    st.warning("⚠️ Some cleaning needed.")
else:
    st.error("❌ Poor data quality.")

st.markdown(
    """
<div class="section-head">
    <h2>Preview Dataset</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="panel preview-wrap">', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-head">
    <h2>Data Quality Checks</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="checks-grid">
    <div class="check-card warn">
        <div class="check-icon">⚠️</div>
        <div class="check-title">Missing Values</div>
        <div class="check-desc">Identified {missing} null entries across the current dataset.</div>
    </div>
    <div class="check-card bad">
        <div class="check-icon">🧱</div>
        <div class="check-title">Duplicate Records</div>
        <div class="check-desc">Found {duplicates} duplicate rows that may affect analysis quality.</div>
    </div>
    <div class="check-card">
        <div class="check-icon">⚙️</div>
        <div class="check-title">Inconsistent Data</div>
        <div class="check-desc">{mixed_type_cols} columns may need closer review for dtype consistency.</div>
    </div>
    <div class="check-card good">
        <div class="check-icon">✅</div>
        <div class="check-title">Validity Checks</div>
        <div class="check-desc">{high_missing_cols + id_like_cols} fields were flagged by the current rule-based checks.</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="warn-bar">
    Important: Review quality issues carefully before moving to visualization, predictions, or exports.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-head">
    <h2>Column Analysis</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="panel table-wrap">', unsafe_allow_html=True)
st.dataframe(quality_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="section-head">
    <h2>Overall Quality Score</h2>
    <div class="line"></div>
</div>
<div class="panel score-panel">
    <div class="score-grid"></div>
    <div class="score-dots">
        <span class="b1"></span><span class="b2"></span><span class="b3"></span><span class="b4"></span>
        <span class="b5"></span><span class="b6"></span><span class="b7"></span><span class="b8"></span>
    </div>
    <div class="gauge">
        <div class="gauge-arc"></div>
        <div class="gauge-value">{consistency_score}%</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### 📊 Missing Values Chart")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing"]
missing_df = missing_df[missing_df["Missing"] > 0]

if len(missing_df) > 0:
    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing",
        color="Missing",
        color_continuous_scale=["#0c5f3a", "#22a15d", "#83ff92"],
    )
    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✅ No missing values!")


action_left, action_right = st.columns(2)
with action_left:
    st.button("Generate Quality Report", use_container_width=True, disabled=True)
with action_right:
    st.button("Improve Data", use_container_width=True, disabled=True)

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-26
</div>
</div>
""",
    unsafe_allow_html=True,
)

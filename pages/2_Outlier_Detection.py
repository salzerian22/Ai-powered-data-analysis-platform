import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import (
    get_dataframe,
    save_dataframe,
    push_undo,
    pop_undo,
    get_undo_count,
    apply_dark_theme,
)
from utils.logger import get_logger
from utils.styles import inject_global_css

logger = get_logger(__name__)

st.set_page_config(page_title="Outlier Detection", page_icon="📉", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.outlier-shell {
    position: relative;
}

.outlier-shell::before {
    content: "";
    position: absolute;
    inset: -30px 0 auto 0;
    height: 340px;
    background:
        radial-gradient(circle at 52% 12%, rgba(57, 138, 255, 0.16), transparent 38%),
        radial-gradient(circle at 72% 18%, rgba(255, 91, 91, 0.12), transparent 28%);
    pointer-events: none;
}

.outlier-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 0.62fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.5rem 0 1rem 0;
}

.outlier-copy,
.outlier-art {
    position: relative;
    z-index: 1;
}

.outlier-kicker {
    color: #8fb6ff;
    font: 600 0.8rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.outlier-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.55rem;
}

.outlier-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(55, 24, 45, 0.98), rgba(18, 17, 43, 0.98));
    border: 1px solid rgba(255, 91, 91, 0.24);
    box-shadow: 0 0 24px rgba(255, 91, 91, 0.18);
    font-size: 2rem;
}

.outlier-title h1 {
    margin: 0;
    color: #f7fbff;
    font: 800 clamp(2rem, 4.8vw, 3.2rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.outlier-title .accent {
    color: #ff6f7d;
}

.outlier-sub {
    margin: 0.65rem 0 0 5.2rem;
    color: #a8bbd8;
    font: 400 1.02rem/1.7 "IBM Plex Sans", sans-serif;
}

.outlier-art {
    min-height: 180px;
}

.curve-panel {
    position: absolute;
    inset: 10px 0 0 0;
    margin: auto;
    width: min(100%, 420px);
    height: 180px;
}

.curve-panel svg {
    width: 100%;
    height: 100%;
}

.curve-glow {
    position: absolute;
    right: 18%;
    top: 18px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 70, 70, 0.18), transparent 68%);
    filter: blur(12px);
}

.glow-rule-red {
    height: 2px;
    margin: 0.9rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(255, 94, 94, 0.92), transparent);
    box-shadow: 0 0 26px rgba(255, 94, 94, 0.28);
}

.upload-title b {
    color: #ff7b87;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 1.3rem 0;
}

.summary-card {
    padding: 1rem 1.05rem;
    border-radius: 16px;
    border: 1px solid rgba(80, 127, 205, 0.58);
    background: linear-gradient(180deg, rgba(20, 37, 67, 0.96), rgba(11, 23, 43, 0.98));
}

.summary-label {
    color: #a8bfdc;
    font: 600 0.8rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.summary-value {
    margin-top: 0.28rem;
    color: #edf5ff;
    font: 800 1.68rem "Inter", sans-serif;
}

.summary-value.outlier {
    color: #ff656f;
}

.method-card {
    padding: 1rem 1rem 1.1rem 1rem;
}

.method-card .radio-wrap {
    margin-top: 0.35rem;
}

.viz-panel {
    padding: 0.8rem 0.8rem 0.4rem 0.8rem;
}

.warning-bar {
    margin: 1rem 0 1.2rem 0;
    padding: 0.95rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 96, 96, 0.34);
    background: linear-gradient(90deg, rgba(61, 18, 28, 0.92), rgba(34, 23, 42, 0.95));
    color: #ffd4d6;
    font: 500 0.95rem "IBM Plex Sans", sans-serif;
}

.final-wrap {
    padding: 0.8rem;
}

@media (max-width: 1100px) {
    .outlier-hero {
        grid-template-columns: 1fr;
    }

    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .summary-grid {
        grid-template-columns: 1fr;
    }

    .outlier-sub {
        margin-left: 0;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

df = get_dataframe()
if df is None:
    st.warning("⚠️ No data found! Please upload a file on the Home page first.")
    st.stop()

undo_count = get_undo_count()
if undo_count > 0:
    if st.button(f"↩️ Undo Last Change ({undo_count} available)", use_container_width=True):
        if pop_undo():
            st.success("✅ Reverted to previous state!")
            st.rerun()

st.markdown('<div class="outlier-shell">', unsafe_allow_html=True)
st.markdown(
    """
<div class="outlier-hero">
    <div class="outlier-copy">
        <div class="outlier-kicker">Anomaly review workflow</div>
        <div class="outlier-title">
            <div class="outlier-icon">📍</div>
            <h1><span class="accent">Outlier</span> Detection & Handling</h1>
        </div>
        <div class="outlier-sub">Detect and manage anomalies in your data before visualization and modeling.</div>
    </div>
    <div class="outlier-art">
        <div class="curve-glow"></div>
        <div class="curve-panel">
            <svg viewBox="0 0 420 180" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="blueLine" x1="24" y1="152" x2="230" y2="48" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#69D7FF"/>
                        <stop offset="1" stop-color="#DDF6FF"/>
                    </linearGradient>
                    <linearGradient id="redLine" x1="230" y1="48" x2="390" y2="152" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#FFD8DE"/>
                        <stop offset="1" stop-color="#FF4C5E"/>
                    </linearGradient>
                    <linearGradient id="redFill" x1="250" y1="48" x2="370" y2="152" gradientUnits="userSpaceOnUse">
                        <stop stop-color="rgba(255,82,99,0.38)" stop-opacity="0.42"/>
                        <stop offset="1" stop-color="rgba(255,82,99,0.02)" stop-opacity="0.02"/>
                    </linearGradient>
                </defs>
                <line x1="18" y1="154" x2="398" y2="154" stroke="rgba(111,159,224,0.32)" stroke-width="2"/>
                <line x1="232" y1="26" x2="232" y2="154" stroke="rgba(220,236,255,0.55)" stroke-width="2"/>
                <path d="M24 154C62 154 72 130 92 100C116 64 150 34 196 34C216 34 226 36 232 38"
                      stroke="url(#blueLine)" stroke-width="4" stroke-linecap="round"/>
                <path d="M232 38C254 48 286 76 312 108C330 130 346 146 396 154"
                      stroke="url(#redLine)" stroke-width="4" stroke-linecap="round"/>
                <path d="M232 38C254 48 286 76 312 108C330 130 346 146 396 154L396 154L232 154Z"
                      fill="url(#redFill)"/>
                <path d="M70 154V136M86 154V122M102 154V108M118 154V93M134 154V78M150 154V63M166 154V52M182 154V44M198 154V40M214 154V39"
                      stroke="rgba(86, 173, 255, 0.55)" stroke-width="2"/>
                <path d="M248 154V54M264 154V68M280 154V82M296 154V97M312 154V113M328 154V126M344 154V136M360 154V144"
                      stroke="rgba(255, 92, 106, 0.55)" stroke-width="2"/>
                <circle cx="232" cy="38" r="5" fill="#F4FBFF"/>
                <circle cx="232" cy="38" r="11" stroke="rgba(255,255,255,0.18)" stroke-width="2"/>
                <line x1="242" y1="28" x2="274" y2="10" stroke="#FF5F69" stroke-width="2"/>
                <text x="278" y="16" fill="#FF7984" font-size="18" font-family="IBM Plex Sans" font-weight="700">Outlier</text>
                <text x="285" y="118" fill="#FFD9DE" font-size="26" font-family="Inter" font-weight="800">Z &gt; 3</text>
            </svg>
        </div>
    </div>
</div>
<div class="glow-rule-red"></div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="panel upload-panel">
    <div class="upload-title">Detect outliers on the dataset already loaded from <b>Home</b>.</div>
    <div class="upload-note">This page does not upload a new file. It works on your current dataset and preserves the existing detection logic.</div>
</div>
""",
    unsafe_allow_html=True,
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) == 0:
    st.warning("⚠️ No numeric columns found in your dataset. Outlier detection requires numeric data.")
    st.stop()

selected_col = st.selectbox("Select a numeric column", numeric_cols)
method = st.radio("Detection method", ["IQR", "Z-Score"], horizontal=True)

data = df[selected_col].dropna()
if method == "IQR":
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]
else:
    mean = data.mean()
    std = data.std()
    z_scores = (data - mean) / std
    outliers = data[np.abs(z_scores) > 3]
    lower = mean - 3 * std
    upper = mean + 3 * std

total_values = len(data)
outlier_count = len(outliers)
outlier_pct = (outlier_count / total_values * 100) if total_values > 0 else 0

st.markdown(
    f"""
<div class="section-head">
    <h2>Data Summary</h2>
    <div class="line"></div>
</div>
<div class="summary-grid">
    <div class="summary-card">
        <div class="summary-label">Selected Column</div>
        <div class="summary-value">{selected_col}</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Detection Method</div>
        <div class="summary-value">{method}</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Total Records</div>
        <div class="summary-value">{total_values}</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Potential Outliers</div>
        <div class="summary-value outlier">{outlier_count}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="panel method-card">', unsafe_allow_html=True)
st.markdown(
    f"""
<div style="color:#e9f3ff;font:700 1.1rem Inter,sans-serif;">Detection Results</div>
<div style="color:#95b1d6;font:400 0.92rem/1.7 'IBM Plex Sans',sans-serif;margin-top:0.35rem;">
Lower bound: <b style="color:#ffffff">{lower:.4f}</b> &nbsp;|&nbsp;
Upper bound: <b style="color:#ffffff">{upper:.4f}</b> &nbsp;|&nbsp;
Outlier share: <b style="color:#ff7d86">{outlier_pct:.2f}%</b>
</div>
""",
    unsafe_allow_html=True,
)
if outlier_count > 0:
    st.markdown("#### 🔻 Outlier Data Points")
    st.dataframe(outliers.head(20).reset_index(), use_container_width=True)
else:
    st.success("✅ No outliers detected in this column!")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-head">
    <h2>Visualize Outliers</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
apply_dark_theme(fig)
st.markdown('<div class="panel viz-panel">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="warning-bar">
    Important: Review identified outliers before applying removal or capping.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-head">
    <h2>Handle Outliers</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

action = st.radio(
    "Choose an action",
    ["Do Nothing", "Cap outliers (safe)", "Remove outliers (loses rows)"],
    horizontal=True,
)

new_df = df.copy()
if action == "Cap outliers (safe)":
    new_df[selected_col] = np.clip(df[selected_col], lower, upper)
elif action == "Remove outliers (loses rows)":
    new_df = df[(df[selected_col] >= lower) & (df[selected_col] <= upper)].reset_index(drop=True)

if action != "Do Nothing":
    before_col, after_col = st.columns(2)
    before_col.metric("📝 Rows Before", df.shape[0])
    after_col.metric("📝 Rows After", new_df.shape[0])

    apply_col, keep_col = st.columns(2)
    with apply_col:
        if st.button("🚨 Apply Outlier Handling", use_container_width=True):
            rows_before = int(df.shape[0])
            rows_after = int(new_df.shape[0])
            st.session_state["outlier_log"] = {
                "action": action,
                "column": selected_col,
                "method": method,
                "outlier_count": int(outlier_count),
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
            }
            push_undo()
            save_dataframe(new_df)
            st.success(f"✅ Applied: {action}")
            st.rerun()
    with keep_col:
        st.button("Keep as Is", disabled=True, use_container_width=True)

st.markdown(
    """
<div class="section-head">
    <h2>Final Dataset Status</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

df = get_dataframe()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📝 Total Rows", df.shape[0])
col2.metric("📌 Total Columns", df.shape[1])
col3.metric("❌ Missing Values", df.isnull().sum().sum())
col4.metric("🔁 Duplicate Rows", df.duplicated().sum())

st.markdown('<div class="panel final-wrap">', unsafe_allow_html=True)
st.markdown("### 👀 Dataset Preview")
st.dataframe(df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="outlier_handled_data.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-26
</div>
</div>
""",
    unsafe_allow_html=True,
)

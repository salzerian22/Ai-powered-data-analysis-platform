import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils.helpers import (page_banner, divider,
                           save_dataframe, get_dataframe, push_undo, pop_undo,
                           apply_smart_missing_value_treatment)
from utils.column_classifier import get_column_roles, get_columns_by_role
from utils.styles import inject_global_css

st.set_page_config(
    page_title="AI Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_global_css()
st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; }
.landing-hero {
    position: relative; display: grid; grid-template-columns: minmax(0,0.95fr) minmax(320px,0.8fr);
    gap: 2.2rem; align-items: center; padding: 3rem 0 1.8rem 0;
}
.landing-hero::before {
    content: ""; position: absolute; inset: -30px 0 auto 0; height: 340px;
    background:
        radial-gradient(circle at 50% 12%, rgba(31,121,255,0.18), transparent 36%),
        radial-gradient(circle at 50% 0%, rgba(0,197,255,0.08), transparent 42%);
    pointer-events: none;
}
.hero-copy, .hero-scene { position: relative; z-index: 1; }
.hero-copy { text-align: center; }
.hero-badge {
    display: inline-flex; padding: 0.45rem 1rem; border-radius: 999px;
    border: 1px solid rgba(70,151,255,0.42); background: rgba(12,29,55,0.8);
    color: #80c7ff; font: 600 0.72rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em; text-transform: uppercase;
}
.hero-title {
    margin: 1.2rem auto 1rem auto; max-width: 760px; color: #f8fbff;
    font-family: "Inter", sans-serif; font-size: clamp(2.4rem, 5vw, 4.3rem);
    line-height: 0.96; letter-spacing: -0.04em;
}
.hero-title span {
    display: block; background: linear-gradient(90deg, #ffffff 0%, #8dd2ff 52%, #47a8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { max-width: 650px; margin: 0 auto; color: #7d9ac2; font: 400 1rem/1.8 "IBM Plex Sans", sans-serif; }
.hero-scene { position: relative; height: 320px; }
.hero-orb {
    position: absolute; right: 12%; top: 7%; width: 150px; height: 150px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, rgba(120,225,255,0.9), rgba(24,111,255,0.18) 55%, transparent 72%);
    filter: blur(4px); animation: floatOrb 6s ease-in-out infinite;
}
.hero-dashboard {
    position: absolute; inset: 1.1rem 0 0 0; margin: auto; width: min(100%, 460px); height: 270px; padding: 1rem;
    border-radius: 24px; border: 1px solid rgba(73,141,255,0.42);
    background: linear-gradient(180deg, rgba(11,24,48,0.96), rgba(7,17,32,0.98));
    box-shadow: 0 30px 60px rgba(1,9,20,0.5), inset 0 1px 0 rgba(133,194,255,0.14);
    transform: rotateY(-18deg) rotateX(14deg); animation: slowSpin 16s ease-in-out infinite;
}
.dashboard-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.dashboard-chip {
    padding: 0.35rem 0.75rem; border-radius: 999px; background: rgba(58,154,255,0.1);
    border: 1px solid rgba(58,154,255,0.18); color: #9ed4ff; font: 600 0.68rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.dashboard-dots { display: flex; gap: 0.35rem; }
.dashboard-dots span { width: 8px; height: 8px; border-radius: 50%; background: rgba(142,204,255,0.45); }
.dashboard-main { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 0.9rem; height: calc(100% - 3rem); }
.dashboard-graph, .dashboard-side {
    border-radius: 18px; border: 1px solid rgba(69,122,208,0.35); background: rgba(12,25,49,0.86);
}
.dashboard-graph { position: relative; overflow: hidden; padding: 1rem; }
.dashboard-graph::before {
    content: ""; position: absolute; inset: auto 0 0 0; height: 58%;
    background:
        linear-gradient(rgba(73,164,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(73,164,255,0.08) 1px, transparent 1px);
    background-size: 34px 34px; transform: perspective(320px) rotateX(67deg); opacity: 0.72;
}
.graph-line { position: absolute; left: 8%; right: 8%; bottom: 30%; height: 90px; }
.graph-line svg { width: 100%; height: 100%; }
.dashboard-side { display: grid; grid-template-rows: 1fr 1fr; gap: 0.8rem; padding: 0.8rem; }
.mini-panel { border-radius: 14px; background: rgba(7,18,34,0.95); border: 1px solid rgba(70,136,224,0.28); padding: 0.8rem; }
.mini-title { color: #cfe6ff; font: 600 0.72rem "IBM Plex Sans", sans-serif; letter-spacing: 0.1em; text-transform: uppercase; }
.mini-bars { display: flex; align-items: end; gap: 0.35rem; height: 72px; margin-top: 0.75rem; }
.mini-bars span { width: 12px; border-radius: 4px 4px 0 0; background: linear-gradient(180deg, #60d6ff, #216fff); }
.mini-bars span:nth-child(1) { height: 24px; }
.mini-bars span:nth-child(2) { height: 54px; }
.mini-bars span:nth-child(3) { height: 38px; }
.mini-bars span:nth-child(4) { height: 62px; }
.mini-bars span:nth-child(5) { height: 30px; }
.mini-signal {
    position: relative; margin-top: 0.85rem; height: 74px; border-radius: 12px; overflow: hidden;
    background: linear-gradient(180deg, rgba(10,28,54,0.95), rgba(6,16,32,0.95));
}
.mini-signal::before, .mini-signal::after {
    content: ""; position: absolute; border-radius: 999px; border: 1px solid rgba(103,183,255,0.4);
    inset: 18px 34px; animation: ping 2.5s ease-out infinite;
}
.mini-signal::after { animation-delay: 1.2s; }
.mini-signal span {
    position: absolute; left: 50%; top: 50%; width: 12px; height: 12px;
    transform: translate(-50%, -50%); border-radius: 50%; background: #71d4ff;
    box-shadow: 0 0 20px rgba(113,212,255,0.8);
}
.stats-strip {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1.2rem; align-items: center;
    margin: 1.8rem 0 1.4rem 0; padding: 1.2rem 0;
    border-top: 1px solid rgba(39,96,178,0.5); border-bottom: 1px solid rgba(39,96,178,0.5);
}
.stat-item { text-align: center; }
.stat-num { color: #38a3ff; font-family: "Inter", sans-serif; font-size: 1.65rem; font-weight: 800; }
.stat-label { margin-top: 0.3rem; color: #6588b8; font: 500 0.68rem "IBM Plex Sans", sans-serif; letter-spacing: 0.16em; text-transform: uppercase; }
.section-rule { position: relative; border-top: 1px solid rgba(39,96,178,0.42); margin: 1.7rem 0 1rem 0; }
.section-rule::after {
    content: ""; position: absolute; left: 22%; right: 22%; top: -1px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(78,177,255,0.85), transparent);
}
.section-label { color: #5d91d6; font: 600 0.72rem "IBM Plex Sans", sans-serif; letter-spacing: 0.16em; text-transform: uppercase; }
.section-heading { margin-top: 0.35rem; color: #e9f3ff; font-family: "Inter", sans-serif; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em; }
.section-sub { margin: 0.35rem 0 1.35rem 0; color: #7f9ec8; font: 400 0.96rem/1.7 "IBM Plex Sans", sans-serif; }
.feat-grid, .why-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
.feat-card, .why-card {
    position: relative; padding: 1.15rem 1.15rem 1rem 1.15rem; border-radius: 14px;
    border: 1px solid rgba(31,84,156,0.75);
    background: linear-gradient(180deg, rgba(15,33,60,0.92), rgba(8,19,36,0.98));
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.feat-card::before, .why-card::before {
    content: ""; position: absolute; inset: 0; border-radius: inherit;
    background: radial-gradient(circle at var(--mx, 50%) var(--my, 50%), rgba(111,196,255,0.14), transparent 36%);
    opacity: 0; transition: opacity 0.2s ease; pointer-events: none;
}
.feat-card:hover, .why-card:hover { border-color: rgba(88,171,255,0.9); box-shadow: 0 18px 34px rgba(0,0,0,0.28); }
.feat-card:hover::before, .why-card:hover::before { opacity: 1; }
.feat-icon, .why-icon {
    display: inline-flex; align-items: center; justify-content: center; width: 1.55rem; height: 1.55rem;
    margin-bottom: 0.8rem; border-radius: 0.35rem; background: rgba(56,163,255,0.14); color: #a7d8ff;
    font: 700 0.78rem "IBM Plex Sans", sans-serif;
}
.feat-title, .why-title { color: #e9f3ff; font: 700 0.98rem "Inter", sans-serif; margin-bottom: 0.38rem; }
.feat-desc, .why-desc { color: #7f9ec8; font: 400 0.82rem/1.65 "IBM Plex Sans", sans-serif; }
.why-showcase { position: relative; padding-top: 1rem; }
.why-grid { gap: 1.35rem; }
.why-card {
    min-height: 208px; padding: 1.25rem; border-radius: 18px;
    background:
        radial-gradient(circle at 78% 75%, rgba(79,160,255,0.2), transparent 24%),
        linear-gradient(180deg, rgba(30,48,84,0.92), rgba(11,24,46,0.98));
}
.why-card-header { display: flex; align-items: center; gap: 0.9rem; }
.why-copy { max-width: 58%; }
.why-card-media { position: absolute; right: 1rem; bottom: 0.9rem; width: 120px; height: 92px; border-radius: 18px; }
.why-icon { width: 2.7rem; height: 2.7rem; margin-bottom: 0; border-radius: 0.9rem; font-size: 1rem; }
.why-card.is-trusted .why-card-media { display: flex; align-items: end; justify-content: center; }
.avatar-panel {
    position: relative; width: 100px; height: 78px; border-radius: 16px;
    background: linear-gradient(180deg, rgba(105,170,255,0.34), rgba(22,53,112,0.8));
    border: 1px solid rgba(134,195,255,0.22);
}
.avatar-panel::before {
    content: ""; position: absolute; left: 50%; top: 14px; width: 28px; height: 28px; transform: translateX(-50%);
    border-radius: 50%; background: linear-gradient(180deg, #f8d0b7, #d89a7a);
}
.avatar-panel::after {
    content: ""; position: absolute; left: 50%; bottom: 10px; width: 54px; height: 34px; transform: translateX(-50%);
    border-radius: 18px 18px 10px 10px; background: linear-gradient(180deg, #55a9ff, #2b5fd9);
}
.why-card.is-fast .why-card-media { display: flex; align-items: end; padding: 0 0.15rem; }
.growth-bars { position: relative; display: flex; align-items: end; gap: 0.45rem; width: 100%; height: 100%; }
.growth-bars span { width: 18px; border-radius: 5px 5px 0 0; background: linear-gradient(180deg, #84dbff, #4b8eff); }
.growth-bars span:nth-child(1) { height: 32px; }
.growth-bars span:nth-child(2) { height: 48px; }
.growth-bars span:nth-child(3) { height: 66px; }
.growth-bars span:nth-child(4) { height: 82px; }
.growth-arrow {
    position: absolute; right: 0; top: 2px; width: 86px; height: 74px; border-right: 3px solid rgba(236,246,255,0.88);
    border-top: 3px solid rgba(236,246,255,0.88); border-radius: 0 14px 0 0; transform: skewY(-20deg) rotate(14deg);
}
.growth-arrow::after {
    content: ""; position: absolute; right: -8px; top: -7px; width: 0; height: 0;
    border-left: 10px solid rgba(236,246,255,0.96); border-top: 7px solid transparent; border-bottom: 7px solid transparent;
}
.why-card.is-secure .why-card-media { display: flex; align-items: end; justify-content: flex-end; }
.server-stack { display: flex; flex-direction: column; gap: 0.45rem; }
.server-stack span {
    width: 70px; height: 19px; border-radius: 999px; background: linear-gradient(180deg, #79d7ff, #3d86ff);
}
.upload-zone {
    border: 1.5px dashed rgba(61,143,255,0.52); border-radius: 16px; padding: 1.8rem 1.2rem; text-align: center;
    background: linear-gradient(180deg, rgba(11,25,48,0.92), rgba(8,17,33,0.98));
}
.upload-title { color: #d9ecff; font: 700 1rem "IBM Plex Sans", sans-serif; }
.upload-hint { margin-top: 0.35rem; color: #7595bf; font: 400 0.83rem "IBM Plex Sans", sans-serif; }
@media (max-width: 1100px) {
    .landing-hero { grid-template-columns: 1fr; }
    .hero-scene { max-width: 520px; width: 100%; margin: 0 auto; }
    .feat-grid, .why-grid, .stats-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 720px) {
    .feat-grid, .why-grid, .stats-strip { grid-template-columns: 1fr; }
    .hero-dashboard { transform: none; animation: none; }
    .why-copy { max-width: 100%; }
    .why-card-media { position: relative; right: auto; bottom: auto; margin-top: 1rem; }
}
@keyframes ping { 0% { transform: scale(0.65); opacity: 0.65; } 100% { transform: scale(1.32); opacity: 0; } }
@keyframes floatOrb { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(12px); } }
@keyframes slowSpin { 0%,100% { transform: rotateY(-18deg) rotateX(14deg); } 50% { transform: rotateY(-8deg) rotateX(10deg) translateY(-8px); } }
</style>
""", unsafe_allow_html=True)

UPLOADER_KEY = "main_dataset_uploader"

def reset_dataset_state():
    keys_to_clear = ["df", "column_roles", "mode", "auto_done", "auto_actions", "undo_stack", "chart_memory", "active_processed_name"]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

previous_upload_name = st.session_state.get("active_upload_name")
current_upload = st.session_state.get(UPLOADER_KEY)
if previous_upload_name and current_upload is None:
    reset_dataset_state()
    st.session_state["active_upload_name"] = None
    st.rerun()

if "df" not in st.session_state:
    st.markdown("""
    <div class="landing-hero">
        <div class="hero-copy">
            <div class="hero-badge">Intelligent analytics workspace</div>
            <div class="hero-title">AI-Powered Data <span>Analysis</span></div>
            <div class="hero-sub">
                Upload your CSV or Excel file and move from raw tables to cleaned data,
                visuals, quality checks, predictions, and AI summaries in one streamlined workspace.
            </div>
        </div>
        <div class="hero-scene">
            <div class="hero-orb"></div>
            <div class="hero-dashboard">
                <div class="dashboard-top">
                    <div class="dashboard-chip">Live dashboard</div>
                    <div class="dashboard-dots"><span></span><span></span><span></span></div>
                </div>
                <div class="dashboard-main">
                    <div class="dashboard-graph">
                        <div class="graph-line">
                            <svg viewBox="0 0 320 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 80C42 77 53 36 88 38C123 40 138 68 171 60C205 52 216 15 254 20C281 24 297 46 315 32" stroke="url(#lineGlow)" stroke-width="4" stroke-linecap="round"/>
                                <defs>
                                    <linearGradient id="lineGlow" x1="5" y1="80" x2="315" y2="32" gradientUnits="userSpaceOnUse">
                                        <stop stop-color="#70E0FF"/>
                                        <stop offset="1" stop-color="#2A71FF"/>
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                    </div>
                    <div class="dashboard-side">
                        <div class="mini-panel">
                            <div class="mini-title">Cleaning Flow</div>
                            <div class="mini-bars"><span></span><span></span><span></span><span></span><span></span></div>
                        </div>
                        <div class="mini-panel">
                            <div class="mini-title">Anomaly Pulse</div>
                            <div class="mini-signal"><span></span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="stats-strip">
        <div class="stat-item"><div class="stat-num">7</div><div class="stat-label">Analysis modules</div></div>
        <div class="stat-item"><div class="stat-num">3</div><div class="stat-label">ML models</div></div>
        <div class="stat-item"><div class="stat-num">50MB</div><div class="stat-label">Upload limit</div></div>
        <div class="stat-item"><div class="stat-num">AI</div><div class="stat-label">Insights engine</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What it does</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Everything you need in one platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Designed to feel like a focused analytics workspace instead of a basic upload screen.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card" data-tilt-card="true"><span class="feat-icon">CL</span><div class="feat-title">Data Cleaning</div><div class="feat-desc">Handle duplicates, missing values, and data types with automated cleanup or manual control.</div></div>
        <div class="feat-card" data-tilt-card="true"><span class="feat-icon">OD</span><div class="feat-title">Outlier Detection</div><div class="feat-desc">Review anomalies with IQR-based logic, visual checks, and reversible actions.</div></div>
        <div class="feat-card" data-tilt-card="true"><span class="feat-icon">DQ</span><div class="feat-title">Data Quality</div><div class="feat-desc">Surface completeness, uniqueness, and consistency signals before deeper analysis.</div></div>
        <div class="feat-card" data-tilt-card="true"><span class="feat-icon">VZ</span><div class="feat-title">Visualization</div><div class="feat-desc">Generate polished interactive charts that help patterns stand out immediately.</div></div>
        <div class="feat-card" data-tilt-card="true"><span class="feat-icon">AI</span><div class="feat-title">AI Insights</div><div class="feat-desc">Ask questions in natural language and get contextual summaries of your dataset.</div></div>
        <div class="feat-card" data-tilt-card="true"><span class="feat-icon">PR</span><div class="feat-title">Predictions</div><div class="feat-desc">Train regression models with metrics, diagnostics, and feature-level interpretation.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Why us</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Why Choose Our Platform?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A polished first page with a more visual product feel, while keeping your existing workflow intact.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="why-showcase">
        <div class="why-grid">
            <div class="why-card is-trusted" data-tilt-card="true">
                <div class="why-card-header"><div class="why-icon">TT</div><div class="why-copy"><div class="why-title">Trusted by Teams</div><div class="why-desc">Built for collaborative data work, from first upload to polished reports and decision-ready insights.</div></div></div>
                <div class="why-card-media"><div class="avatar-panel"></div></div>
            </div>
            <div class="why-card is-fast" data-tilt-card="true">
                <div class="why-card-header"><div class="why-icon">FR</div><div class="why-copy"><div class="why-title">Fast and Reliable</div><div class="why-desc">Move through cleaning, quality checks, charts, and prediction flows without breaking momentum.</div></div></div>
                <div class="why-card-media"><div class="growth-bars"><span></span><span></span><span></span><span></span><div class="growth-arrow"></div></div></div>
            </div>
            <div class="why-card is-secure" data-tilt-card="true">
                <div class="why-card-header"><div class="why-icon">SS</div><div class="why-copy"><div class="why-title">Secure and Scalable</div><div class="why-desc">Safer query handling and dependable local processing keep the workflow production-minded.</div></div></div>
                <div class="why-card-media"><div class="server-stack"><span></span><span></span><span></span></div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Get started</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Upload your dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-title">Drop your file here or use the uploader below</div>
        <div class="upload-hint">Supported formats: CSV and Excel (.xlsx) | Maximum size: 50 MB</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed", key=UPLOADER_KEY)
    st.markdown("""
    <div style="text-align:center;padding:42px 0 10px 0;font-size:0.8em;color:#315783;letter-spacing:0.05em;">
        Copyright 2025-26 | Shri Ramdeobaba College of Engineering and Management | Department of Data Science<br>
        <span style="color:#24496f;">Shubham Tiwari | Shrajal Raghuwanshi | Parth Thakur | Prathamesh Rathod</span>
    </div>
    """, unsafe_allow_html=True)
else:
    page_banner("📊", "AI-Powered Data Analysis Platform", "Upload your data and let AI do the heavy lifting")
    uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed", key=UPLOADER_KEY)

if uploaded_file is not None:
    st.session_state["active_upload_name"] = uploaded_file.name

if uploaded_file is not None:
    # ── SIZE VALIDATION ───────────────────────────────────────
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 50:
        st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum allowed is 50 MB.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding="windows-1252")
                except UnicodeDecodeError:
                  uploaded_file.seek(0)
                  df = pd.read_csv(uploaded_file, encoding="latin-1")
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    if df.empty:
        st.error("❌ The uploaded file is empty!")
        st.stop()

    # Save on first upload or when the uploaded file changes.
    if st.session_state.get('active_processed_name') != uploaded_file.name:
        save_dataframe(df)
        st.session_state['column_roles'] = get_column_roles(df)
        st.session_state['active_processed_name'] = uploaded_file.name

    st.success("✅ File uploaded successfully!")
    divider()

    # ══════════════════════════════════════════════════════════
    #  MODE SELECTOR — shown when no mode is chosen yet
    # ══════════════════════════════════════════════════════════
    if 'mode' not in st.session_state:
        st.markdown("## 🎯 How would you like to work with your data?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="custom-card" style="min-height: 220px;">
                <h3>⚡ Do it for me</h3>
                <p style="color:#aad4ff">
                The system cleans, analyzes, and visualizes your data
                <b style="color:white">automatically</b>.<br><br>
                Best for: Quick overviews, presentations, and when you
                want results fast.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Start Automated Analysis", use_container_width=True):
                st.session_state['mode'] = 'auto'
                st.rerun()

        with col2:
            st.markdown("""
            <div class="custom-card" style="min-height: 220px;">
                <h3>🛠️ I will do it myself</h3>
                <p style="color:#aad4ff">
                You control every step — cleaning, feature selection,
                chart types, and model configuration.<br><br>
                Best for: Detailed exploration and full control over
                the analysis pipeline.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📋 Go to Manual Mode", use_container_width=True):
                st.session_state['mode'] = 'manual'
                st.rerun()

        st.stop()

    # ══════════════════════════════════════════════════════════
    #  AUTOMATED MODE — clean + show transparency panel
    # ══════════════════════════════════════════════════════════
    if st.session_state.get('mode') == 'auto':
        df = get_dataframe()

        # Run automated cleaning only once
        if 'auto_done' not in st.session_state:
            push_undo()  # save pre-cleaning snapshot
            actions = []
            roles = st.session_state.get('column_roles', {})

            # 1. Remove duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                actions.append(f"🗑️ Removed {dup_count} duplicate rows")

            # 2. Fill missing values per column using smart strategy
            df, missing_actions, _ = apply_smart_missing_value_treatment(df, roles)
            actions.extend(missing_actions)

            # 3. Note identifier columns excluded from analysis
            id_cols = get_columns_by_role(roles, 'identifier')
            for col in id_cols:
                actions.append(f"🔒 Column '{col}' excluded from analysis — detected as identifier")

            st.session_state['auto_actions'] = actions
            st.session_state['auto_done'] = True
            save_dataframe(df)

        # ── Transparency Panel ────────────────────────────────
        st.markdown("## ⚡ Automated Analysis Complete")

        with st.expander("📋 What the system did to your data", expanded=True):
            actions = st.session_state.get('auto_actions', [])
            if actions:
                for action in actions:
                    st.write(f"  {action}")
            else:
                st.write("  ✅ No cleaning was needed — your data was already clean!")

            st.markdown("---")
            if st.button("↩️ Undo all automated changes"):
                pop_undo()
                del st.session_state['auto_done']
                del st.session_state['auto_actions']
                del st.session_state['mode']
                st.rerun()

        divider()

        # Show cleaned data overview
        st.markdown("## 📋 Cleaned Dataset Overview")
        df = get_dataframe()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Total Rows", df.shape[0])
        col2.metric("📌 Total Columns", df.shape[1])
        col3.metric("❌ Missing Values", df.isnull().sum().sum())
        col4.metric("🔁 Duplicate Rows", df.duplicated().sum())
        divider()

        st.markdown("## 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("""
        <div class="custom-card">
            <h3>🚀 Your data is ready!</h3>
            <p style="color:#aad4ff">
            Automated cleaning is complete. Use the sidebar to explore:
            <b style="color:white">Outlier Detection, Visualization, AI Insights, Predictions, or Export Report</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MANUAL MODE — show overview + navigate via sidebar
    # ══════════════════════════════════════════════════════════
    elif st.session_state.get('mode') == 'manual':
        df = get_dataframe()
        st.markdown("## 📋 Quick Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Total Rows", df.shape[0])
        col2.metric("📌 Total Columns", df.shape[1])
        col3.metric("❌ Missing Values", df.isnull().sum().sum())
        col4.metric("🔁 Duplicate Rows", df.duplicated().sum())
        divider()

        st.markdown("## 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("""
        <div class="custom-card">
            <h3>🛠️ Manual mode active</h3>
            <p style="color:#aad4ff">
            Navigate using the <b style="color:white">sidebar on the left</b> to:
            <b style="color:white">Data Cleaning → Outlier Detection → Data Quality → Visualization →
            AI Insights → Predictions → Export Report</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <h2 style="color:#4da6ff">📊 AI Platform</h2>
        <p style="color:#aad4ff; font-size:0.9em">
        Shri Ramdeobaba College<br>
        Department of Data Science<br>
        Session 2025-26
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show current mode in sidebar
    if 'mode' in st.session_state:
        mode_label = "⚡ Automated" if st.session_state['mode'] == 'auto' else "🛠️ Manual"
        st.info(f"Mode: {mode_label}")
        if st.button("🔄 Switch Mode"):
            if 'auto_done' in st.session_state:
                del st.session_state['auto_done']
            if 'auto_actions' in st.session_state:
                del st.session_state['auto_actions']
            del st.session_state['mode']
            st.rerun()

    divider()
    st.markdown("""
    <p style="color:#aad4ff; font-size:0.85em">
    <b style="color:white">👨‍💻 Team Members</b><br>
    • Shubham Tiwari<br>
    • Shrajal Raghuwanshi<br>
    • Parth Thakur<br>
    • Prathamesh Rathod<br><br>
    <b style="color:white">👩‍🏫 Guide</b><br>
    Prof. Shruti Kolte Mam
    </p>
    """, unsafe_allow_html=True)

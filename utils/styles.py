from __future__ import annotations

import streamlit as st


def inject_global_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1627 50%, #0a0e1a 100%);
    color: #ffffff;
    font-family: "IBM Plex Sans", sans-serif;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    font-family: "Inter", sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050810 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}

[data-testid="stSidebar"] * {
    font-family: "IBM Plex Sans", sans-serif;
}

/* Hide Streamlit's generated multipage nav header, which can leak raw icon text. */
[data-testid="stSidebarNav"] > div:first-child {
    display: none;
}

[data-testid="stSidebarNav"] ul {
    padding-top: 0.35rem;
}

.custom-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0, 120, 255, 0.1);
}

.custom-card h1,
.custom-card h2,
.custom-card h3,
.custom-card h4 {
    color: #f1f7ff;
    font-family: "Inter", sans-serif;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 18px rgba(0, 120, 255, 0.08);
}

[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #bfd4ee;
    font-family: "IBM Plex Sans", sans-serif;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff;
    font-family: "Inter", sans-serif;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #0066cc, #0044aa);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font: 600 0.95rem "IBM Plex Sans", sans-serif;
    transition: all 0.3s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0077ee, #0055bb);
    transform: translateY(-2px);
}

.page-banner {
    background: linear-gradient(135deg, #0044aa, #0066cc, #0088ff);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 30px;
    text-align: center;
}

.page-banner h1 {
    color: #ffffff !important;
    font-size: 2.5em;
    margin: 0;
}

.page-banner p {
    color: #aad4ff;
    font-size: 1.1em;
    margin: 8px 0 0 0;
}

.blue-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #0066cc, transparent);
    margin: 20px 0;
    border: none;
}

.panel {
    border-radius: 20px;
    border: 1px solid rgba(88, 132, 211, 0.72);
    background: linear-gradient(180deg, rgba(15, 28, 52, 0.96), rgba(9, 20, 37, 0.98));
    box-shadow: inset 0 1px 0 rgba(132, 181, 255, 0.08), 0 0 28px rgba(42, 105, 187, 0.12);
}

.upload-panel {
    padding: 1.55rem 1.4rem;
    text-align: center;
}

.upload-title {
    color: #eaf6ff;
    font: 500 1rem/1.7 "IBM Plex Sans", sans-serif;
}

.upload-note {
    margin-top: 0.55rem;
    color: #8aa8cd;
    font: 400 0.88rem "IBM Plex Sans", sans-serif;
}

.section-head {
    margin: 1.8rem 0 0.7rem 0;
}

.section-head h2 {
    margin: 0;
    color: #f1f7ff;
    font: 800 1.8rem "Inter", sans-serif;
    letter-spacing: -0.03em;
}

.section-head .line {
    height: 1px;
    margin-top: 0.45rem;
    background: linear-gradient(90deg, rgba(113, 255, 176, 0.5), rgba(118, 170, 255, 0.4), transparent);
}

.footer-note {
    text-align: center;
    color: #688ab1;
    font: 400 0.82rem "IBM Plex Sans", sans-serif;
    margin-top: 2rem;
}

h1 {
    color: #4da6ff !important;
}

h2 {
    color: #3d9af5 !important;
}

h3 {
    color: #2d8ae8 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

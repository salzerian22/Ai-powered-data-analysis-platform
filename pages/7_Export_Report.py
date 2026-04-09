import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import tempfile
import io
import sys, os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import divider, get_dataframe, apply_dark_theme, GROQ_API_KEY
from utils.column_classifier import get_column_roles, get_columns_by_role
from utils.logger import get_logger
from utils.styles import inject_global_css

logger = get_logger(__name__)

st.set_page_config(page_title="Export Report", page_icon="📄", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.report-shell {
    position: relative;
}

.report-shell::before {
    content: "";
    position: absolute;
    inset: -30px 0 auto 0;
    height: 380px;
    background:
        radial-gradient(circle at 50% 8%, rgba(106, 189, 255, 0.16), transparent 34%),
        radial-gradient(circle at 76% 18%, rgba(255, 210, 117, 0.12), transparent 26%);
    pointer-events: none;
}

.report-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(320px, 0.86fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.35rem 0 1rem 0;
}

.report-copy,
.report-art {
    position: relative;
    z-index: 1;
}

.report-kicker {
    color: #ffd88c;
    font: 600 0.82rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.report-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.6rem;
}

.report-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(58, 43, 12, 0.96), rgba(19, 16, 10, 0.98));
    border: 1px solid rgba(255, 213, 108, 0.3);
    box-shadow: 0 0 24px rgba(255, 196, 77, 0.14);
    font-size: 2rem;
}

.report-title h1 {
    margin: 0;
    color: #f8fbff;
    font: 800 clamp(2rem, 4.8vw, 3.15rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.report-title .accent {
    color: #ffd685;
}

.report-sub {
    margin: 0.7rem 0 0 5.2rem;
    color: #cfdbeb;
    font: 400 1.02rem/1.8 "IBM Plex Sans", sans-serif;
}

.report-art {
    min-height: 250px;
}

.doc-glow {
    position: absolute;
    right: 46px;
    top: 28px;
    width: 240px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(95, 190, 255, 0.18), transparent 68%);
    filter: blur(12px);
}

.doc-trail {
    position: absolute;
    right: 172px;
    top: 92px;
    width: 180px;
    height: 86px;
}

.doc-trail svg {
    width: 100%;
    height: 100%;
}

.doc-stack {
    position: absolute;
    right: 16px;
    top: 10px;
    width: 270px;
    height: 240px;
}

.doc-stack svg {
    width: 100%;
    height: 100%;
}

.hero-rule {
    height: 2px;
    margin: 0.95rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(110, 214, 255, 0.88), transparent);
    box-shadow: 0 0 22px rgba(110, 214, 255, 0.16);
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 0.95rem;
}

.summary-card {
    min-height: 104px;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(96, 136, 212, 0.62);
    background: linear-gradient(180deg, rgba(18, 30, 53, 0.97), rgba(10, 20, 38, 1));
}

.summary-card.gold {
    border-color: rgba(255, 210, 117, 0.34);
    box-shadow: 0 0 20px rgba(255, 196, 77, 0.08);
}

.summary-label {
    color: #d7e4f2;
    font: 600 0.96rem "IBM Plex Sans", sans-serif;
}

.summary-value {
    margin-top: 0.35rem;
    color: #ffffff;
    font: 800 1.95rem "Inter", sans-serif;
}

.customize-shell {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 0.7fr);
    gap: 1rem;
    align-items: center;
}

.report-preview-art {
    min-height: 260px;
}

.report-preview-art svg {
    width: 100%;
    height: 100%;
}

.preview-metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1rem;
}

.preview-pill {
    padding: 0.85rem 0.95rem;
    border-radius: 14px;
    border: 1px solid rgba(255, 210, 117, 0.3);
    background: linear-gradient(180deg, rgba(28, 31, 40, 0.98), rgba(15, 18, 28, 1));
    color: #f0f6ff;
    font: 600 0.94rem "IBM Plex Sans", sans-serif;
    text-align: center;
}

.preview-pill.blue {
    border-color: rgba(119, 192, 255, 0.3);
}

.preview-note {
    margin-top: 1rem;
    padding: 1rem 1.08rem;
    border-radius: 16px;
    border: 1px solid rgba(92, 132, 205, 0.56);
    background: linear-gradient(180deg, rgba(16, 28, 48, 0.98), rgba(9, 18, 34, 1));
    color: #deebfb;
    font: 400 0.95rem/1.72 "IBM Plex Sans", sans-serif;
}

@media (max-width: 1100px) {
    .report-hero,
    .summary-grid,
    .customize-shell,
    .preview-metric-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .report-sub {
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

record_count = df.shape[0]
top_findings = min(6, max(3, len(df.select_dtypes(include=["number"]).columns) + len(df.select_dtypes(include=["object"]).columns)))

st.markdown('<div class="report-shell">', unsafe_allow_html=True)
st.markdown(
    """
<div class="report-hero">
    <div class="report-copy">
        <div class="report-kicker">Final export and delivery</div>
        <div class="report-title">
            <div class="report-icon">📄</div>
            <h1><span class="accent">Export</span> Report</h1>
        </div>
        <div class="report-sub">Generate your analysis summary and package your dataset insights into a meeting-ready PDF report.</div>
    </div>
    <div class="report-art">
        <div class="doc-glow"></div>
        <div class="doc-trail">
            <svg viewBox="0 0 210 100" xmlns="http://www.w3.org/2000/svg">
                <path d="M6 76L38 58L68 66L96 38L126 46L158 26L204 34" fill="none" stroke="rgba(111,215,255,0.82)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="38" cy="58" r="4" fill="#7cdfff"/>
                <circle cx="96" cy="38" r="5" fill="#7cdfff"/>
                <circle cx="158" cy="26" r="5" fill="#ffd97c"/>
            </svg>
        </div>
        <div class="doc-stack">
            <svg viewBox="0 0 300 260" xmlns="http://www.w3.org/2000/svg">
                <rect x="152" y="24" width="104" height="156" rx="8" fill="#d9e3ee"/>
                <rect x="138" y="34" width="108" height="166" rx="8" fill="#e7eef6"/>
                <rect x="126" y="46" width="112" height="176" rx="8" fill="#f2f6fb"/>
                <rect x="98" y="56" width="118" height="182" rx="8" fill="#f8fbff" stroke="#d9e7f4" stroke-width="2"/>
                <rect x="88" y="42" width="70" height="34" rx="6" fill="#cc483d"/>
                <text x="123" y="64" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700" fill="#ffffff">PDF</text>
                <rect x="118" y="92" width="78" height="68" rx="4" fill="url(#chartGrad)"/>
                <defs>
                    <linearGradient id="chartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#112f5a"/>
                        <stop offset="100%" stop-color="#2560a8"/>
                    </linearGradient>
                </defs>
                <rect x="126" y="132" width="10" height="20" rx="2" fill="#79bdff"/>
                <rect x="142" y="120" width="10" height="32" rx="2" fill="#79bdff"/>
                <rect x="158" y="108" width="10" height="44" rx="2" fill="#79bdff"/>
                <rect x="174" y="94" width="10" height="58" rx="2" fill="#79bdff"/>
                <path d="M122 146L142 132L158 136L174 114L192 120" fill="none" stroke="#ffd67d" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M110 168H196" stroke="#cbd8e6" stroke-width="2"/>
                <path d="M110 182H190" stroke="#d3dfe9" stroke-width="2"/>
                <path d="M110 196H186" stroke="#d3dfe9" stroke-width="2"/>
            </svg>
        </div>
    </div>
</div>
<div class="hero-rule"></div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="section-head">
    <h2>Select a Report Summary</h2>
    <div class="line"></div>
</div>
<div class="summary-grid">
    <div class="summary-card">
        <div class="summary-label">Dataset Records</div>
        <div class="summary-value">{record_count:,}</div>
    </div>
    <div class="summary-card gold">
        <div class="summary-label">Top Findings</div>
        <div class="summary-value">{top_findings} key insights</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

divider()

st.markdown(
    """
<div class="section-head">
    <h2>Customize Your Report</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

customize_left, customize_right = st.columns([1.15, 0.85], gap="large")
with customize_left:
    overview_include = st.checkbox("Overview & Summary", value=True)
    cleaning_include = st.checkbox("Data Cleaning Results", value=True)
    outlier_include = st.checkbox("Outlier Detection", value=True)
    quality_include = st.checkbox("Data Quality Assessment", value=True)
    viz_include = st.checkbox("Visualization Insights", value=True)
    ai_include = st.checkbox("AI Insights", value=True)
    predictive_include = st.checkbox("Predictive Modeling", value=True)

with customize_right:
    st.markdown(
        """
<div class="report-preview-art">
    <svg viewBox="0 0 320 300" xmlns="http://www.w3.org/2000/svg">
        <rect x="138" y="34" width="112" height="208" rx="8" fill="#dfe7f0"/>
        <rect x="126" y="46" width="116" height="218" rx="8" fill="#edf2f8"/>
        <rect x="114" y="58" width="120" height="228" rx="8" fill="#f8fbff" stroke="#dde8f5" stroke-width="2"/>
        <text x="174" y="96" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700" fill="#17345f">Data Analysis Report</text>
        <path d="M136 108H212" stroke="#7d92ab" stroke-width="2"/>
        <rect x="142" y="126" width="66" height="62" rx="4" fill="#1e4f8d"/>
        <rect x="150" y="164" width="10" height="16" rx="2" fill="#79bdff"/>
        <rect x="166" y="152" width="10" height="28" rx="2" fill="#79bdff"/>
        <rect x="182" y="140" width="10" height="40" rx="2" fill="#79bdff"/>
        <path d="M148 174L166 160L182 164L200 140" fill="none" stroke="#ffd67d" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="138" y="202" width="12" height="12" rx="2" fill="#101f33"/>
        <rect x="138" y="228" width="12" height="12" rx="2" fill="#101f33"/>
        <rect x="138" y="254" width="12" height="12" rx="2" fill="#101f33"/>
        <path d="M142 208L145 211L150 204" fill="none" stroke="#ffffff" stroke-width="2"/>
        <path d="M142 234L145 237L150 230" fill="none" stroke="#ffffff" stroke-width="2"/>
        <path d="M142 260L145 263L150 256" fill="none" stroke="#ffffff" stroke-width="2"/>
        <path d="M160 208H220" stroke="#9aacc0" stroke-width="2"/>
        <path d="M160 234H214" stroke="#9aacc0" stroke-width="2"/>
        <path d="M160 260H218" stroke="#9aacc0" stroke-width="2"/>
    </svg>
</div>
""",
        unsafe_allow_html=True,
    )

divider()

st.markdown(
    """
<div class="section-head">
    <h2>Report Settings</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    report_title = st.text_input("Report Title", value="AI Data Analysis Report")
    author_name = st.text_input(
        "Author / Team Name",
        value="Shubham Tiwari, Shrajal Raghuwanshi, Parth Thakur, Prathamesh Rathod",
    )
with col2:
    dataset_name = st.text_input("Dataset Name", value=st.session_state.get("uploaded_filename", "Dataset"))
    include_charts = st.checkbox("Include auto-generated charts", value=True)

st.markdown(
    """
<div class="section-head">
    <h2>Report Preview</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

preview_labels = []
if overview_include:
    preview_labels.append(("Overview", "gold"))
if cleaning_include:
    preview_labels.append(("Cleaning", "gold"))
if outlier_include:
    preview_labels.append(("Outliers", "blue"))
if quality_include:
    preview_labels.append(("Quality", "gold"))
if viz_include:
    preview_labels.append(("Visualization", "blue"))
if ai_include:
    preview_labels.append(("AI Insights", "gold"))
if predictive_include:
    preview_labels.append(("Predictions", "blue"))

preview_cols = st.columns(4)
for i, (label, style) in enumerate(preview_labels[:4]):
    with preview_cols[i]:
        st.markdown(f'<div class="preview-pill {style}">{label}</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="preview-note">
    The exported PDF will include your report title, dataset overview, executive summary, selected analysis sections, and optional charts in a presentation-ready format.
</div>
""",
    unsafe_allow_html=True,
)

generate_pdf = st.button("📥 Generate & Download Dashboard PDF", use_container_width=True)



if generate_pdf:
    from fpdf import FPDF

    with st.spinner("Building your dashboard report... ⏳"):

        def generate_executive_summary():
            try:
                from groq import Groq

                client = Groq(api_key=GROQ_API_KEY)
                stats = df.describe().to_string()
                missing = df.isnull().sum()[df.isnull().sum() > 0].to_string()
                if not missing:
                    missing = "No missing values"

                prompt = (
                    f"Data stats:\n{stats}\n\n"
                    f"Missing values:\n{missing}\n\n"
                    f"Write exactly 3 key business insights as bullet points. "
                    f"Each bullet should be 1 sentence. Be specific with numbers."
                )
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=300,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                return (
                    "- Dataset contains {} rows and {} columns.\n"
                    "- {} missing values detected across all columns.\n"
                    "- {} duplicate rows found in the dataset.".format(
                        df.shape[0],
                        df.shape[1],
                        df.isnull().sum().sum(),
                        df.duplicated().sum(),
                    )
                )

        def safe_text(text):
            if not isinstance(text, str):
                text = str(text)
            replacements = {
                "\u2022": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2026": "...",
                "\u200b": "",
                "\ufeff": "",
                "\u00a0": " ",
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            return text.encode("latin-1", "replace").decode("latin-1")

        def chart_to_bytes(fig, width=700, height=350):
            try:
                return pio.to_image(fig, format="png", width=width, height=height)
            except Exception:
                return None

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(0, 100, 200)
        pdf.ln(40)
        pdf.cell(0, 15, safe_text(report_title), ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 10, safe_text(f"Dataset: {dataset_name}"), ln=True, align="C")
        pdf.cell(0, 10, safe_text(f"Date: {datetime.now().strftime('%B %d, %Y')}"), ln=True, align="C")
        pdf.cell(0, 10, safe_text(f"Team: {author_name}"), ln=True, align="C")
        pdf.ln(10)

        pdf.set_draw_color(0, 100, 200)
        pdf.set_line_width(0.8)
        pdf.line(30, pdf.get_y(), 180, pdf.get_y())
        pdf.ln(10)

        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, "Generated by AI-Powered Data Analysis Platform", ln=True, align="C")
        pdf.cell(0, 8, "Shri Ramdeobaba College - Dept. of Data Science", ln=True, align="C")

        pdf.add_page()

        def section_header(title):
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(0, 100, 200)
            pdf.cell(0, 12, safe_text(title), ln=True)
            pdf.set_draw_color(0, 100, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            pdf.set_x(pdf.l_margin)

        def body_text():
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(pdf.l_margin)

        section_header("Executive Summary")
        body_text()

        exec_summary = generate_executive_summary()
        for line in exec_summary.split("\n"):
            line = line.strip()
            if line:
                pdf.set_x(pdf.l_margin)
                try:
                    pdf.multi_cell(0, 7, safe_text(line))
                except Exception:
                    pdf.set_x(pdf.l_margin)
                    pdf.cell(0, 7, safe_text(line[:120]), ln=True)
        pdf.ln(8)

        section_header("Dataset Overview")
        body_text()

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0

        metrics = [
            ("Total Rows", str(df.shape[0])),
            ("Total Columns", str(df.shape[1])),
            ("Missing Values", f"{df.isnull().sum().sum()} ({missing_pct:.1f}%)"),
            ("Duplicate Rows", str(df.duplicated().sum())),
            ("Memory Usage", f"{memory_mb:.2f} MB"),
            ("Numeric Columns", str(len(df.select_dtypes(include=['number']).columns))),
        ]

        for i, (label, value) in enumerate(metrics):
            pdf.cell(90, 8, safe_text(f"  {label}: {value}"), ln=False)
            if i % 2 == 1:
                pdf.ln()
        if len(metrics) % 2 == 1:
            pdf.ln()
        pdf.ln(10)

        if include_charts:
            roles = st.session_state.get("column_roles", get_column_roles(df))
            num_cols = get_columns_by_role(roles, "numeric", "ordinal")
            cat_cols = get_columns_by_role(roles, "categorical")

            charts_generated = []

            for col in num_cols[:3]:
                fig = px.histogram(df, x=col, color_discrete_sequence=["#4da6ff"], title=f"Distribution of {col}")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8f9fa", font_color="#333", title_font_color="#0064c8")
                charts_generated.append((f"Distribution of {col}", fig))

            for cat in cat_cols[:2]:
                if num_cols:
                    num = num_cols[0]
                    grouped = df.groupby(cat)[num].mean().reset_index()
                    grouped = grouped.sort_values(by=num, ascending=False).head(15)
                    fig = px.bar(
                        grouped,
                        x=cat,
                        y=num,
                        title=f"Average {num} by {cat}",
                        color=num,
                        color_continuous_scale=["#003399", "#0066cc", "#00aaff"],
                    )
                    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8f9fa", font_color="#333", title_font_color="#0064c8")
                    charts_generated.append((f"Average {num} by {cat}", fig))

            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
                fig.update_layout(paper_bgcolor="white", font_color="#333", title_font_color="#0064c8")
                charts_generated.append(("Correlation Heatmap", fig))

            if charts_generated:
                pdf.add_page()
                section_header("Visualizations")
                chart_count_on_page = 0

                for title, fig in charts_generated:
                    img_bytes = chart_to_bytes(fig)
                    if img_bytes is None:
                        continue

                    if chart_count_on_page >= 2:
                        pdf.add_page()
                        chart_count_on_page = 0

                    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp_img.write(img_bytes)
                    tmp_img.close()

                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(0, 80, 160)
                    pdf.cell(0, 8, safe_text(title), ln=True)
                    pdf.image(tmp_img.name, x=15, w=180)
                    pdf.ln(5)
                    chart_count_on_page += 1

                    try:
                        os.unlink(tmp_img.name)
                    except Exception:
                        pass

        pdf.add_page()
        section_header("Data Quality Summary")
        body_text()

        completeness = (1 - df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 100
        pdf.cell(0, 8, safe_text(f"  Completeness: {completeness:.1f}%"), ln=True)
        pdf.cell(0, 8, safe_text(f"  Duplicate Rows: {df.duplicated().sum()}"), ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 80, 160)
        pdf.cell(0, 8, safe_text("Outlier Detection (IQR Method)"), ln=True)
        body_text()

        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_count = len(numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)])
                pdf.cell(0, 7, safe_text(f"  {col}: {outlier_count} outliers  |  Bounds: [{lower:.2f}, {upper:.2f}]"), ln=True)
        pdf.ln(5)

        section_header("Statistical Summary")

        if not numeric_df.empty:
            stats_df = numeric_df.describe()
            cols_to_show = list(stats_df.columns)[:6]
            usable_w = pdf.w - pdf.l_margin - pdf.r_margin
            col_width = usable_w / (len(cols_to_show) + 1)

            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(0, 100, 200)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(col_width, 7, "Stat", border=1, fill=True, align="C")
            for col in cols_to_show:
                pdf.cell(col_width, 7, safe_text(str(col)[:12]), border=1, fill=True, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            for stat in stats_df.index:
                pdf.set_x(pdf.l_margin)
                pdf.set_fill_color(245, 248, 255)
                pdf.cell(col_width, 6, str(stat), border=1, fill=True, align="C")
                for col in cols_to_show:
                    val = stats_df[col][stat]
                    pdf.cell(col_width, 6, f"{val:.2f}", border=1, align="C")
                pdf.ln()
        pdf.ln(5)

        section_header("Methodology")
        body_text()

        auto_actions = st.session_state.get("auto_actions", [])
        if auto_actions:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Automated Cleaning Applied:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for action in auto_actions:
                pdf.set_x(pdf.l_margin)
                try:
                    pdf.multi_cell(0, 6, safe_text(f"  - {action}"))
                except Exception:
                    pdf.set_x(pdf.l_margin)
                    pdf.cell(0, 6, safe_text(f"  - {action}"[:80]), ln=True)
            pdf.ln(3)

        roles = st.session_state.get("column_roles", {})
        if roles:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Column Role Classifications:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for col, role in roles.items():
                pdf.cell(0, 6, safe_text(f"  - {col}: {role}"), ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 150, 150)
        pdf.ln(10)
        pdf.cell(0, 8, "Shri Ramdeobaba College - Dept. of Data Science - Session 2025-26", ln=True, align="C")

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp_file.name)

        with open(tmp_file.name, "rb") as f:
            pdf_bytes = f.read()

        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass

    st.success("✅ Dashboard PDF Generated Successfully!")

    st.download_button(
        label="📥 Click Here to Download PDF",
        data=pdf_bytes,
        file_name="data_analysis_dashboard.pdf",
        mime="application/pdf",
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

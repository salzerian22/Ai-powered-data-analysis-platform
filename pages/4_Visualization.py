import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import (
    divider,
    get_dataframe,
    save_dataframe,
    apply_dark_theme,
    GROQ_API_KEY,
    get_chart_memory,
    record_chart_view,
    render_plotly_chart,
)
from utils.column_classifier import get_column_roles, get_columns_by_role
from utils.visualization import CHART_TYPES, recommend_chart, render_chart, prepare_x_axis
from utils.heatmap import should_show_heatmap, describe_correlation
from utils.chart_summary import show_chart_summary
from utils.logger import get_logger
from utils.styles import inject_global_css
from groq import Groq

logger = get_logger(__name__)

st.set_page_config(page_title="Visualization", page_icon="📊", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.viz-shell {
    position: relative;
}

.viz-shell::before {
    content: "";
    position: absolute;
    inset: -30px 0 auto 0;
    height: 360px;
    background:
        radial-gradient(circle at 50% 12%, rgba(77, 177, 255, 0.16), transparent 38%),
        radial-gradient(circle at 76% 18%, rgba(92, 255, 165, 0.1), transparent 28%);
    pointer-events: none;
}

.viz-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 0.68fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.5rem 0 1rem 0;
}

.viz-copy,
.viz-art {
    position: relative;
    z-index: 1;
}

.viz-kicker {
    color: #8dc6ff;
    font: 600 0.8rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.viz-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.55rem;
}

.viz-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(16, 60, 112, 0.94), rgba(9, 28, 55, 0.98));
    border: 1px solid rgba(93, 255, 164, 0.22);
    box-shadow: 0 0 24px rgba(52, 210, 255, 0.18);
    font-size: 2rem;
}

.viz-title h1 {
    margin: 0;
    color: #f7fbff;
    font: 800 clamp(2rem, 4.8vw, 3.25rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.viz-title .accent {
    color: #ff7d89;
}

.viz-sub {
    margin: 0.65rem 0 0 5.2rem;
    color: #a8bbd8;
    font: 400 1.02rem/1.7 "IBM Plex Sans", sans-serif;
}

.viz-art {
    min-height: 200px;
}

.globe-glow {
    position: absolute;
    right: 18%;
    top: 20px;
    width: 170px;
    height: 170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(107, 214, 255, 0.18), transparent 70%);
    filter: blur(14px);
}

.globe-base {
    position: absolute;
    right: 30px;
    bottom: 0;
    width: 182px;
    height: 34px;
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(64, 82, 108, 0.96), rgba(24, 36, 56, 0.98));
    transform: perspective(240px) rotateX(62deg);
    box-shadow: 0 0 18px rgba(93, 171, 255, 0.16);
}

.globe-sphere {
    position: absolute;
    right: 54px;
    top: 4px;
    width: 176px;
    height: 176px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 40% 35%, rgba(194, 244, 255, 0.95), rgba(59, 133, 255, 0.36) 45%, rgba(16, 43, 85, 0.68) 72%),
        linear-gradient(180deg, rgba(75, 173, 255, 0.48), rgba(12, 51, 110, 0.34));
    border: 1px solid rgba(160, 225, 255, 0.26);
    box-shadow: 0 0 28px rgba(82, 176, 255, 0.22);
    overflow: hidden;
}

.globe-sphere::before,
.globe-sphere::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
}

.globe-sphere::before {
    background:
        repeating-linear-gradient(90deg, transparent 0 18px, rgba(192, 239, 255, 0.2) 18px 19px),
        repeating-linear-gradient(0deg, transparent 0 18px, rgba(192, 239, 255, 0.14) 18px 19px);
    opacity: 0.7;
}

.globe-sphere::after {
    background: radial-gradient(circle at 30% 24%, rgba(255,255,255,0.34), transparent 20%);
}

.viz-bars {
    position: absolute;
    left: 10px;
    bottom: 36px;
    display: flex;
    align-items: end;
    gap: 0.5rem;
}

.viz-bars span {
    width: 14px;
    border-radius: 5px 5px 0 0;
    background: linear-gradient(180deg, #75f3ff, #18b8ff);
    box-shadow: 0 0 14px rgba(93, 214, 255, 0.22);
}

.viz-bars span:nth-child(1) { height: 22px; }
.viz-bars span:nth-child(2) { height: 34px; }
.viz-bars span:nth-child(3) { height: 50px; }
.viz-bars span:nth-child(4) { height: 64px; }

.viz-line {
    position: absolute;
    right: 16px;
    top: 54px;
    width: 160px;
    height: 86px;
}

.viz-line svg {
    width: 100%;
    height: 100%;
}

.hero-rule {
    height: 2px;
    margin: 0.9rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(108, 238, 255, 0.88), transparent);
    box-shadow: 0 0 24px rgba(108, 238, 255, 0.2);
}

.upload-title b {
    color: #7aff8f;
}

.type-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.type-card {
    min-height: 122px;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid rgba(88, 132, 211, 0.58);
    background: linear-gradient(180deg, rgba(18, 33, 60, 0.96), rgba(10, 23, 42, 0.98));
}

.type-card.good {
    border-color: rgba(98, 255, 162, 0.42);
    box-shadow: 0 0 18px rgba(98, 255, 162, 0.12);
}

.type-card.blue {
    border-color: rgba(125, 197, 255, 0.45);
}

.type-card.hot {
    border-color: rgba(255, 140, 140, 0.4);
}

.type-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px;
    height: 54px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    font-size: 1.7rem;
}

.type-title {
    margin-top: 0.65rem;
    color: #edf6ff;
    font: 700 1.02rem "Inter", sans-serif;
}

.type-desc {
    margin-top: 0.35rem;
    color: #9cb7d8;
    font: 400 0.82rem/1.55 "IBM Plex Sans", sans-serif;
}

.dash-grid {
    display: grid;
    grid-template-columns: 1.2fr 0.9fr;
    gap: 1rem;
    margin-top: 1rem;
}

.dash-card {
    min-height: 220px;
    padding: 0.8rem;
}

.dash-card h3 {
    margin: 0 0 0.6rem 0;
    color: #eef7ff;
    font: 700 1.04rem "Inter", sans-serif;
}

.warn-bar {
    margin: 1.1rem 0 1.2rem 0;
    padding: 0.95rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 96, 96, 0.34);
    background: linear-gradient(90deg, rgba(61, 18, 28, 0.92), rgba(34, 23, 42, 0.95));
    color: #ffd4d6;
    font: 500 0.95rem "IBM Plex Sans", sans-serif;
}

.builder-wrap {
    padding: 0.9rem 1rem;
}

@media (max-width: 1100px) {
    .viz-hero,
    .dash-grid {
        grid-template-columns: 1fr;
    }

    .type-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .type-grid {
        grid-template-columns: 1fr;
    }

    .viz-sub {
        margin-left: 0;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

df = get_dataframe()
if df is None:
    st.warning("⚠️ Please upload a dataset from the Home page first.")
    st.stop()


def is_meaningful_column(col, df):
    col_lower = col.lower()
    if any(k in col_lower for k in ["id", "unnamed", "index"]):
        if df[col].nunique() > 0.9 * len(df):
            return False
    if df[col].nunique() <= 1:
        return False
    return True


df = df[[c for c in df.columns if is_meaningful_column(c, df)]]

st.session_state["column_roles"] = get_column_roles(df)
column_roles = st.session_state["column_roles"]

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception:
    groq_client = None


def score_graph(x, y, df, corr=None, is_time=False):
    score = 50
    if is_time:
        score += 40
    if corr is not None:
        score += abs(corr) * 30
    if df[x].nunique() < 10:
        score += 10

    mem = get_chart_memory()
    usage = mem.get(f"{x}|{y}", 0) + mem.get(f"{y}|{x}", 0)
    score += min(usage * 5, 20)
    return score


st.markdown('<div class="viz-shell">', unsafe_allow_html=True)
st.markdown(
    """
<div class="viz-hero">
    <div class="viz-copy">
        <div class="viz-kicker">Charting and dashboard workflow</div>
        <div class="viz-title">
            <div class="viz-icon">📊</div>
            <h1><span class="accent">Data</span> Visualization</h1>
        </div>
        <div class="viz-sub">Transform your data into insights with auto-generated and manual charts.</div>
    </div>
    <div class="viz-art">
        <div class="globe-glow"></div>
        <div class="globe-base"></div>
        <div class="globe-sphere"></div>
        <div class="viz-bars"><span></span><span></span><span></span><span></span></div>
        <div class="viz-line">
            <svg viewBox="0 0 180 90" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M8 72L46 58L78 66L108 40L140 48L172 18" stroke="#ff5a5a" stroke-width="3" stroke-linecap="round"/>
                <path d="M8 76L46 60L78 66L108 38L140 50L172 22" stroke="rgba(121,255,143,0.84)" stroke-width="3" stroke-linecap="round"/>
                <circle cx="46" cy="58" r="5" fill="#ff6b6b"/>
                <circle cx="108" cy="40" r="6" fill="#ff6b6b"/>
                <circle cx="172" cy="18" r="5" fill="#8cff8b"/>
            </svg>
        </div>
    </div>
</div>
<div class="hero-rule"></div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-head">
    <h2>Upload Data</h2>
    <div class="line"></div>
</div>
<div class="panel upload-panel">
    <div class="upload-title">Visualize the dataset already loaded from <b>Home</b> and explore charts below.</div>
    <div class="upload-note">This page keeps all original visualization functionality and works on your current in-app dataset only.</div>
</div>
""",
    unsafe_allow_html=True,
)

divider()
st.markdown("### 🔎 Filter Data")
base_df = df.copy()
col_filter = st.selectbox("Filter by column", [None] + df.columns.tolist(), key="global_filter_col")
if col_filter:
    vals = df[col_filter].dropna().unique().tolist()
    current_vals = st.session_state.get("global_filter_vals")
    if not isinstance(current_vals, list):
        current_vals = vals
    selected_vals = [val for val in current_vals if val in vals]
    if not selected_vals:
        selected_vals = vals
    selected_vals = st.multiselect("Select values", vals, default=selected_vals, key="global_filter_vals")
    filtered_df = df[df[col_filter].isin(selected_vals)]
    if filtered_df.empty:
        st.warning("No rows match the current filter selection. Showing the full dataset for visualization instead.")
        df = base_df
    else:
        df = filtered_df
        st.caption(f"Active filter: {col_filter} = {selected_vals}")

st.markdown(
    """
<div class="section-head">
    <h2>Choose Visualization</h2>
    <div class="line"></div>
</div>
<div class="type-grid">
    <div class="type-card good">
        <div class="type-icon">📶</div>
        <div class="type-title">Bar Chart</div>
        <div class="type-desc">Compare categories and values.</div>
    </div>
    <div class="type-card blue">
        <div class="type-icon">📈</div>
        <div class="type-title">Line Chart</div>
        <div class="type-desc">View trends and patterns.</div>
    </div>
    <div class="type-card hot">
        <div class="type-icon">🥧</div>
        <div class="type-title">Pie Chart</div>
        <div class="type-desc">Visualize distributions.</div>
    </div>
    <div class="type-card good">
        <div class="type-icon">🟩</div>
        <div class="type-title">Heatmap</div>
        <div class="type-desc">Show correlations in a matrix.</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Chart recommendations and builders below are fully functional and use your actual dataset.")

num_cols = [col for col in get_columns_by_role(column_roles, "numeric", "ordinal") if col in df.columns]
cat_cols = [col for col in get_columns_by_role(column_roles, "categorical") if col in df.columns]
time_cols = [col for col in get_columns_by_role(column_roles, "datetime") if col in df.columns]
id_cols = [col for col in get_columns_by_role(column_roles, "identifier") if col in df.columns]
year_like_cols = [col for col in df.columns if "year" in col.lower()]
pseudo_cat = [col for col in num_cols if df[col].nunique() <= 10]
all_cat = cat_cols + pseudo_cat

detected_time_col = None
year_cols = [
    col for col in num_cols
    if "year" in col.lower() and df[col].dropna().between(1900, 2100).all()
]

df_parse = df.copy()
for col in time_cols:
    if col in id_cols:
        continue
    try:
        parsed = pd.to_datetime(df_parse[col], errors="coerce")
        if parsed.notna().sum() > 0:
            df_parse[col] = parsed
            detected_time_col = col
            break
    except Exception:
        continue

if detected_time_col:
    df = df_parse
elif year_cols:
    detected_time_col = year_cols[0]
else:
    sequence_col = "__row_order__"
    if sequence_col not in df.columns:
        df = df.copy()
        df[sequence_col] = range(1, len(df) + 1)

graph_options = []
for col in num_cols:
    if col in year_like_cols:
        continue
    graph_options.append({"type": "Distribution", "x": col, "y": col, "score": score_graph(col, col, df)})

for cat in all_cat:
    for num in num_cols:
        if cat == num:
            continue
        if num in year_like_cols:
            continue
        graph_options.append({"type": "Category vs Value", "x": cat, "y": num, "score": score_graph(cat, num, df)})

used_titles = set()
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if abs(corr.iloc[i, j]) > 0.5:
                if num_cols[i] in year_like_cols or num_cols[j] in year_like_cols:
                    continue
                title = f"{num_cols[i]} vs {num_cols[j]}"
                if title not in used_titles:
                    used_titles.add(title)
                    graph_options.append(
                        {
                            "type": "Relationship",
                            "x": num_cols[i],
                            "y": num_cols[j],
                            "score": score_graph(num_cols[i], num_cols[j], df, corr=corr.iloc[i, j]),
                        }
                    )

if detected_time_col:
    for num in num_cols:
        if num == detected_time_col:
            continue
        graph_options.append(
            {
                "type": "Trend",
                "x": detected_time_col,
                "y": num,
                "score": score_graph(detected_time_col, num, df, is_time=True),
            }
        )
elif num_cols:
    for num in num_cols:
        if num in year_like_cols:
            continue
        graph_options.append(
            {
                "type": "Trend",
                "x": "__row_order__",
                "y": num,
                "score": score_graph("__row_order__", num, df, is_time=True) - 5,
            }
        )

graph_options = sorted(graph_options, key=lambda g: g["score"], reverse=True)[:20]

names = []
for g in graph_options:
    if g["type"] == "Distribution":
        names.append(f"📦 Distribution of {g['x']}")
    elif g["type"] == "Category vs Value":
        names.append(f"📊 {g['y']} by {g['x']}")
    elif g["type"] == "Relationship":
        names.append(f"🔗 {g['x']} vs {g['y']}")
    elif g["type"] == "Trend":
        if g["x"] == "__row_order__":
            names.append(f"📈 {g['y']} trend by row order")
        else:
            names.append(f"📈 {g['y']} over {g['x']}")

st.markdown(
    """
<div class="section-head">
    <h2>Interactive Dashboard</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

if len(names) == 0:
    st.info("ℹ️ No smart charts detected for this dataset — use the manual selector below.")
else:
    choice = st.selectbox("Select a smart chart", names, key="smart_chart")
    g = graph_options[names.index(choice)]
    record_chart_view(g["x"], g.get("y", g["x"]))

    if g["type"] == "Distribution":
        fig = px.histogram(df, x=g["x"], title=f"Distribution of {g['x']}", color_discrete_sequence=["#4da6ff"])
        apply_dark_theme(fig)
        smart_info = f"This shows how values of '{g['x']}' are distributed."
        summary_cols = [g["x"]]
        summary_type = "Histogram"
    elif g["type"] == "Category vs Value":
        grouped = df.groupby(g["x"])[g["y"]].mean().reset_index()
        grouped = grouped.sort_values(by=g["y"], ascending=False)
        fig = px.bar(
            grouped,
            x=g["x"],
            y=g["y"],
            title=f"{g['y']} by {g['x']}",
            color=g["y"],
            color_continuous_scale=["#003399", "#0066cc", "#00aaff"],
        )
        apply_dark_theme(fig)
        smart_info = f"This shows average '{g['y']}' across different '{g['x']}' categories."
        summary_cols = [g["x"], g["y"]]
        summary_type = "Bar chart"
    elif g["type"] == "Relationship":
        fig = px.scatter(df, x=g["x"], y=g["y"], title=f"{g['y']} over {g['x']}", opacity=0.6, color_discrete_sequence=["#4da6ff"])
        apply_dark_theme(fig)
        smart_info = f"This shows the relationship between '{g['x']}' and '{g['y']}'."
        summary_cols = [g["x"], g["y"]]
        summary_type = "Scatter plot"
    else:
        grouped = df.groupby(g["x"])[g["y"]].mean().reset_index()
        grouped = grouped.sort_values(by=g["x"])
        grouped = grouped.copy()
        grouped[g["x"]] = prepare_x_axis(grouped[g["x"]])
        fig = px.line(grouped, x=g["x"], y=g["y"], markers=True, color_discrete_sequence=["#4da6ff"])
        fig.update_layout(title=f"{g['y']} over {g['x']}")
        apply_dark_theme(fig)
        if g["x"] == "__row_order__":
            smart_info = f"This shows how '{g['y']}' changes across dataset row order when no time column is available."
            summary_cols = [g["y"]]
        else:
            smart_info = f"This shows how '{g['y']}' changes over time."
            summary_cols = [g["x"], g["y"]]
        summary_type = "Line chart"

    dash_left, dash_right = st.columns([1.2, 0.9], gap="large")
    with dash_left:
        render_plotly_chart(fig, use_container_width=True)
    with dash_right:
        st.markdown("### Chart Insight")
        st.info(smart_info)
        if groq_client:
            show_chart_summary(df, summary_cols, summary_type, groq_client)

st.markdown(
    """
<div class="warn-bar">
    Important: Review filtered data and chart intent before exporting or sharing dashboard results.
</div>
""",
    unsafe_allow_html=True,
)

divider()
st.markdown("### 🛠️ Manual Chart Builder")
st.markdown('<div class="panel builder-wrap">', unsafe_allow_html=True)
st.markdown("<p style='color:#aad4ff'>Pick your own columns and chart type</p>", unsafe_allow_html=True)

selectable_cols = [c for c in df.columns if column_roles.get(c) != "identifier"]

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.multiselect("Select columns to visualize", selectable_cols, key="manual_cols")
with col2:
    default_chart = recommend_chart(selected, column_roles) if selected else "Bar chart"
    default_idx = CHART_TYPES.index(default_chart) if default_chart in CHART_TYPES else 0
    chart_type = st.selectbox("Chart type", CHART_TYPES, index=default_idx, key="manual_chart_type")

if len(selected) == 0:
    st.info("ℹ️ Select one or more columns above to generate a chart.")
else:
    if chart_type in ("Scatter plot", "Line chart") and len(selected) < 2:
        st.warning(f"⚠️ {chart_type} needs at least 2 columns. Please select one more.")
    else:
        fig = render_chart(df, selected, chart_type, column_roles=column_roles)
        if fig is not None:
            apply_dark_theme(fig)
            render_plotly_chart(fig, use_container_width=True)
            if groq_client:
                show_chart_summary(df, selected, chart_type, groq_client)
        else:
            st.error(f"❌ Could not render {chart_type} with the selected columns. Try a different combination.")
st.markdown("</div>", unsafe_allow_html=True)

if len(num_cols) >= 2:
    divider()
    st.markdown("### 📊 Correlation Heatmap")
    show, reason, pairs = should_show_heatmap(df, num_cols)

    if not show:
        st.info(f"ℹ️ Heatmap skipped — {reason}")
    else:
        st.caption(f"✅ {reason}")
        corr_heatmap = df[num_cols].corr()
        fig = px.imshow(corr_heatmap, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        apply_dark_theme(fig)
        render_plotly_chart(fig, use_container_width=True)

        st.markdown("**Statistically significant correlations:**")
        for col_a, col_b, r, p_val in pairs:
            strength, direction = describe_correlation(r)
            st.write(f"• **{col_a}** vs **{col_b}**: r={r} ({strength} {direction}, p={p_val})")
        st.caption("Only pairs with p < 0.05 and |r| > 0.2 are shown.")

action_left, action_right = st.columns(2)
with action_left:
    st.button("Generate Report", use_container_width=True, disabled=True)
with action_right:
    st.button("Download Dashboard", use_container_width=True, disabled=True)

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-26
</div>
</div>
""",
    unsafe_allow_html=True,
)

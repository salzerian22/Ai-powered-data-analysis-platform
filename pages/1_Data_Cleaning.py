import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import (
    divider,
    get_dataframe,
    save_dataframe,
    push_undo,
    pop_undo,
    get_undo_count,
    apply_smart_missing_value_treatment,
)
from utils.column_classifier import get_column_roles
from utils.logger import get_logger
from utils.styles import inject_global_css

logger = get_logger(__name__)

st.set_page_config(page_title="Data Cleaning", page_icon="🧹", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.clean-shell {
    position: relative;
}

.clean-shell::before {
    content: "";
    position: absolute;
    inset: -30px 0 auto 0;
    height: 320px;
    background:
        radial-gradient(circle at 50% 15%, rgba(46, 176, 255, 0.18), transparent 38%),
        radial-gradient(circle at 70% 18%, rgba(70, 255, 240, 0.08), transparent 26%);
    pointer-events: none;
}

.clean-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 0.95fr) minmax(240px, 0.55fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.5rem 0 1.2rem 0;
}

.clean-hero-copy,
.clean-hero-art {
    position: relative;
    z-index: 1;
}

.clean-hero-copy {
    padding-left: 0.4rem;
}

.clean-kicker {
    color: #77dfff;
    font: 600 0.82rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.clean-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.55rem;
}

.clean-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(16, 60, 112, 0.94), rgba(9, 28, 55, 0.98));
    border: 1px solid rgba(81, 223, 255, 0.26);
    box-shadow: 0 0 24px rgba(44, 190, 255, 0.22);
    font-size: 2rem;
}

.clean-title h1 {
    margin: 0;
    color: #f6fbff;
    font: 800 clamp(2.1rem, 5vw, 3.5rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.clean-sub {
    margin: 0.6rem 0 0 5.2rem;
    color: #9db7d8;
    font: 400 1.1rem/1.7 "IBM Plex Sans", sans-serif;
}

.clean-hero-art {
    min-height: 180px;
}

.broom-orb {
    position: absolute;
    inset: 12px 0 0 0;
    margin: auto;
    width: 210px;
    height: 150px;
    border-radius: 32px;
    background: radial-gradient(circle at 40% 50%, rgba(92, 231, 255, 0.34), transparent 48%);
    filter: blur(10px);
}

.broom-stick {
    position: absolute;
    right: 22px;
    top: -2px;
    width: 11px;
    height: 148px;
    border-radius: 999px;
    background: linear-gradient(180deg, #7b4a39, #341c15);
    transform: rotate(24deg);
    transform-origin: top center;
    box-shadow: inset -2px 0 0 rgba(255,255,255,0.08);
}

.broom-head {
    position: absolute;
    right: 46px;
    top: 82px;
    width: 96px;
    height: 34px;
    border-radius: 18px 18px 12px 12px;
    background: linear-gradient(180deg, #58e9f7, #2280c5);
    transform: rotate(21deg);
    box-shadow: 0 0 24px rgba(53, 194, 255, 0.22), inset 0 -4px 0 rgba(8, 45, 88, 0.2);
}

.broom-head::before {
    content: "";
    position: absolute;
    left: -6px;
    right: -12px;
    bottom: -28px;
    height: 44px;
    border-radius: 0 0 48px 48px;
    background: linear-gradient(180deg, #f7e1bf, #cfa366);
    clip-path: polygon(4% 0, 96% 0, 100% 18%, 92% 100%, 8% 100%, 0 18%);
}

.broom-head::after {
    content: "";
    position: absolute;
    left: 22px;
    right: 22px;
    top: -7px;
    height: 10px;
    border-radius: 999px;
    background: rgba(232, 248, 255, 0.18);
}

.spark {
    position: absolute;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #9ef9ff;
    box-shadow: 0 0 12px rgba(158, 249, 255, 0.95);
}

.spark.s1 { right: 152px; top: 34px; }
.spark.s2 { right: 124px; top: 16px; width: 6px; height: 6px; }
.spark.s3 { right: 108px; top: 48px; width: 5px; height: 5px; }
.spark.s4 { right: 58px; top: 40px; width: 10px; height: 10px; }

.glow-rule {
    height: 2px;
    margin: 0.9rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(92, 242, 255, 0.92), transparent);
    box-shadow: 0 0 24px rgba(71, 224, 255, 0.42);
}

.upload-title b {
    color: #61e5ff;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 1.4rem 0;
}

.status-card {
    padding: 1rem 1.05rem;
    border-radius: 16px;
    border: 1px solid rgba(74, 154, 231, 0.6);
    background: linear-gradient(180deg, rgba(21, 39, 70, 0.96), rgba(11, 23, 43, 0.98));
}

.status-label {
    color: #a9c0dd;
    font: 600 0.8rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.status-value {
    margin-top: 0.28rem;
    color: #65e5ff;
    font: 800 1.7rem "Inter", sans-serif;
}

.preview-wrap {
    padding: 0.85rem;
}

.action-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.6rem 0;
}

.action-card {
    min-height: 176px;
    padding: 1.2rem 1rem;
    border-radius: 18px;
    border: 1px solid rgba(67, 147, 223, 0.68);
    background: linear-gradient(180deg, rgba(19, 39, 72, 0.96), rgba(10, 23, 42, 0.98));
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
}

.action-card.active {
    border-color: rgba(98, 240, 255, 0.92);
    box-shadow: 0 0 28px rgba(54, 219, 255, 0.18), 0 12px 26px rgba(0, 0, 0, 0.24);
    background:
        radial-gradient(circle at 20% 0%, rgba(86, 236, 255, 0.12), transparent 40%),
        linear-gradient(180deg, rgba(20, 46, 82, 0.98), rgba(10, 23, 42, 0.98));
}

.action-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    background: rgba(90, 227, 255, 0.1);
    border: 1px solid rgba(90, 227, 255, 0.18);
    color: #6feeff;
    font: 700 0.7rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.action-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: rgba(80, 222, 255, 0.12);
    font-size: 1.7rem;
    margin-bottom: 0.9rem;
}

.action-title {
    color: #eef7ff;
    font: 700 1.12rem "Inter", sans-serif;
}

.action-desc {
    margin-top: 0.45rem;
    color: #8eafd3;
    font: 400 0.84rem/1.6 "IBM Plex Sans", sans-serif;
}

.step-panel {
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    margin-top: 0.6rem;
}

.step-panel-title {
    color: #eef7ff;
    font: 800 1.25rem "Inter", sans-serif;
    margin-bottom: 0.25rem;
}

.step-panel-sub {
    color: #8fb1d8;
    font: 400 0.9rem/1.6 "IBM Plex Sans", sans-serif;
    margin-bottom: 1rem;
}

.chart-panel {
    position: relative;
    min-height: 260px;
    overflow: hidden;
    margin: 0.8rem 0 1.5rem 0;
}

.chart-grid {
    position: absolute;
    left: 8%;
    right: 8%;
    bottom: 26px;
    height: 84px;
    background:
        linear-gradient(rgba(68, 169, 255, 0.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(68, 169, 255, 0.13) 1px, transparent 1px);
    background-size: 28px 28px;
    transform: perspective(380px) rotateX(68deg);
    opacity: 0.7;
}

.chart-bars {
    position: absolute;
    left: 50%;
    bottom: 62px;
    display: flex;
    align-items: end;
    gap: 1rem;
    transform: translateX(-50%);
}

.chart-bars span {
    width: 30px;
    border-radius: 8px 8px 0 0;
    background: linear-gradient(180deg, #79f6ff, #16b7ff);
    box-shadow: 0 0 18px rgba(72, 229, 255, 0.36);
}

.chart-bars span:nth-child(1) { height: 70px; }
.chart-bars span:nth-child(2) { height: 118px; }
.chart-bars span:nth-child(3) { height: 92px; }
.chart-bars span:nth-child(4) { height: 146px; }
.chart-bars span:nth-child(5) { height: 58px; }
.chart-bars span:nth-child(6) { height: 108px; }

.bottom-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}

@media (max-width: 1100px) {
    .clean-hero {
        grid-template-columns: 1fr;
    }
    .action-grid,
    .status-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .action-grid,
    .status-grid,
    .bottom-actions {
        grid-template-columns: 1fr;
    }
    .clean-sub {
        margin-left: 0;
    }
    .clean-title {
        align-items: flex-start;
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
missing_total = int(df.isnull().sum().sum())
duplicate_total = int(df.duplicated().sum())
dtype_issue_count = int(sum(str(dtype) == "object" for dtype in df.dtypes))

if "cleaning_active_step" not in st.session_state:
    st.session_state["cleaning_active_step"] = "duplicates"

st.markdown('<div class="clean-shell">', unsafe_allow_html=True)
st.markdown(
    """
<div class="clean-hero">
    <div class="clean-hero-copy">
        <div class="clean-kicker">Dataset repair workflow</div>
        <div class="clean-title">
            <div class="clean-icon">🧹</div>
            <h1>Data Cleaning</h1>
        </div>
        <div class="clean-sub">Fix your dataset before analysis without changing your existing workflow.</div>
    </div>
    <div class="clean-hero-art">
        <div class="broom-orb"></div>
        <div class="broom-stick"></div>
        <div class="broom-head"></div>
        <div class="spark s1"></div>
        <div class="spark s2"></div>
        <div class="spark s3"></div>
        <div class="spark s4"></div>
    </div>
</div>
<div class="glow-rule"></div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="panel upload-panel">
    <div class="upload-title">Clean the dataset already loaded from <b>Home</b> and apply fixes below.</div>
    <div class="upload-note">No new upload happens on this page. All actions work on your current in-app dataset only.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="status-grid">
    <div class="status-card">
        <div class="status-label">Total Rows</div>
        <div class="status-value">{df.shape[0]}</div>
    </div>
    <div class="status-card">
        <div class="status-label">Missing Values</div>
        <div class="status-value">{missing_total}</div>
    </div>
    <div class="status-card">
        <div class="status-label">Duplicate Rows</div>
        <div class="status-value">{duplicate_total}</div>
    </div>
    <div class="status-card">
        <div class="status-label">Potential Type Issues</div>
        <div class="status-value">{dtype_issue_count}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

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
st.dataframe(df.head(10), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-head">
    <h2>Clean Your Data</h2>
    <div class="line"></div>
</div>
""",
    unsafe_allow_html=True,
)

step_cards = [
    ("duplicates", "Step 1", "🗑️", "Remove Duplicates", "Eliminate redundant rows while keeping the rest of your dataset untouched."),
    ("missing", "Step 2", "💧", "Handle Missing Values", "Fill or drop null values using the same cleaning options your page already supports."),
    ("columns", "Step 3", "🗂️", "Remove Columns", "Drop unwanted fields and keep only the columns that matter for later analysis."),
    ("types", "Step 4", "🔧", "Fix Data Types", "Convert columns into consistent numeric, string, or datetime formats safely."),
]

card_cols = st.columns(4)
for col, (step_key, step_label, icon, title, desc) in zip(card_cols, step_cards):
    active_class = " active" if st.session_state["cleaning_active_step"] == step_key else ""
    with col:
        st.markdown(
            f"""
        <div class="action-card{active_class}">
            <div class="action-step">{step_label}</div>
            <div class="action-icon">{icon}</div>
            <div class="action-title">{title}</div>
            <div class="action-desc">{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button(f"Open {title}", key=f"open_clean_step_{step_key}", use_container_width=True):
            st.session_state["cleaning_active_step"] = step_key
            st.rerun()

if undo_count > 0:
    if st.button(f"↩️ Undo Last Change ({undo_count} available)", use_container_width=True):
        if pop_undo():
            st.success("✅ Reverted to previous state!")
            st.rerun()

active_step = st.session_state["cleaning_active_step"]

divider()

if active_step == "duplicates":
    st.markdown(
        """
    <div class="panel step-panel">
        <div class="step-panel-title">Remove Duplicate Rows</div>
        <div class="step-panel-sub">Review duplicate records and remove them without changing the rest of the page workflow.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    duplicates = df.duplicated().sum()
    if duplicates == 0:
        st.success("✅ No duplicate rows found!")
    else:
        st.warning(f"⚠️ Found {duplicates} duplicate rows!")
        if st.button("🗑️ Remove Duplicate Rows"):
            push_undo()
            df = df.drop_duplicates().reset_index(drop=True)
            save_dataframe(df)
            st.success(f"✅ Removed {duplicates} duplicate rows!")
            st.metric("Rows remaining", df.shape[0])

elif active_step == "missing":
    st.markdown(
        """
    <div class="panel step-panel">
        <div class="step-panel-title">Handle Missing Values</div>
        <div class="step-panel-sub">Choose how null values should be treated and apply the same cleaning logic already used by the page.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        st.success("✅ No missing values found!")
    else:
        st.warning(f"⚠️ Found missing values in {len(missing)} columns!")

        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(df) * 100).round(2),
            }
        )
        st.dataframe(missing_df, use_container_width=True)

        if st.session_state.get("mode") == "auto":
            st.info("âš¡ Automated mode applies the best missing-value treatment automatically on this step.")

            roles = st.session_state.get("column_roles")
            if not roles:
                roles = get_column_roles(df)
                st.session_state["column_roles"] = roles

            cleaned_df, auto_actions, changed = apply_smart_missing_value_treatment(df, roles)

            if changed:
                push_undo()
                save_dataframe(cleaned_df)
                existing_actions = st.session_state.get("auto_actions", [])
                st.session_state["auto_actions"] = existing_actions + auto_actions
                st.success("âœ… Missing values were handled automatically for automated mode.")
                st.rerun()

            st.markdown("#### Automated handling summary:")
            if auto_actions:
                for action in auto_actions:
                    st.write(action)
            else:
                st.write("âœ… No automated changes were needed on this step.")

        else:
            st.markdown("#### Choose how to handle missing values:")

            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox(
                    "Select Method",
                    [
                        "Drop rows with missing values",
                        "Fill with Mean",
                        "Fill with Median",
                        "Fill with Mode",
                        "Fill with Custom Value",
                    ],
                )
            with col2:
                if method == "Fill with Custom Value":
                    custom_value = st.text_input("Enter custom value", "0")

        if st.session_state.get("mode") != "auto" and st.button("✅ Apply Missing Value Treatment"):
            push_undo()
            if method == "Drop rows with missing values":
                before = len(df)
                df = df.dropna().reset_index(drop=True)
                after = len(df)
                save_dataframe(df)
                st.success(f"✅ Dropped {before - after} rows!")

            elif method == "Fill with Mean":
                numeric_cols = df.select_dtypes(include=["number"]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                save_dataframe(df)
                st.success("✅ Filled missing values with Mean!")

            elif method == "Fill with Median":
                numeric_cols = df.select_dtypes(include=["number"]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                save_dataframe(df)
                st.success("✅ Filled missing values with Median!")

            elif method == "Fill with Mode":
                skipped = []
                for col in df.columns:
                    if df[col].isnull().any():
                        mode_vals = df[col].mode()
                        if len(mode_vals) > 0:
                            df[col] = df[col].fillna(mode_vals[0])
                        else:
                            skipped.append(col)
                save_dataframe(df)
                st.success("✅ Filled missing values with Mode!")
                if skipped:
                    st.warning(f"⚠️ Skipped entirely empty columns: {', '.join(skipped)}")

            elif method == "Fill with Custom Value":
                filled_cols = []
                failed_cols = []
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            try:
                                df[col] = df[col].fillna(float(custom_value))
                                filled_cols.append(col)
                            except ValueError:
                                failed_cols.append(col)
                        else:
                            df[col] = df[col].fillna(custom_value)
                            filled_cols.append(col)
                save_dataframe(df)
                st.success(f"✅ Filled {len(filled_cols)} column(s) with '{custom_value}'!")
                if failed_cols:
                    st.warning(
                        f"⚠️ Could not fill numeric columns with '{custom_value}' (not a number): {', '.join(failed_cols)}"
                    )

elif active_step == "columns":
    st.markdown(
        """
    <div class="panel step-panel">
        <div class="step-panel-title">Remove Unwanted Columns</div>
        <div class="step-panel-sub">Select the fields you do not want to keep in the cleaned dataset.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols_to_remove = st.multiselect("Select columns to remove", df.columns.tolist())

    if len(cols_to_remove) > 0:
        if st.button("🗑️ Remove Selected Columns"):
            push_undo()
            df = df.drop(columns=cols_to_remove)
            save_dataframe(df)
            st.success(f"✅ Removed columns: {cols_to_remove}")

elif active_step == "types":
    st.markdown(
        """
    <div class="panel step-panel">
        <div class="step-panel-title">Fix Data Types</div>
        <div class="step-panel-sub">Convert a selected column into a more appropriate data type without changing the rest of the page functionality.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_col = st.selectbox("Select Column", df.columns)
    with col2:
        new_type = st.selectbox("Convert to Type", ["int", "float", "string", "datetime"])
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Convert Type"):
            push_undo()
            try:
                if new_type == "int":
                    df[selected_col] = pd.to_numeric(df[selected_col], errors="coerce").astype("Int64")
                elif new_type == "float":
                    df[selected_col] = pd.to_numeric(df[selected_col], errors="coerce")
                elif new_type == "string":
                    df[selected_col] = df[selected_col].astype(str)
                elif new_type == "datetime":
                    df[selected_col] = pd.to_datetime(df[selected_col], errors="coerce")
                save_dataframe(df)
                st.success(f"✅ Converted {selected_col} to {new_type}!")
            except Exception as e:
                logger.exception("Failed to convert data type")
                st.error(f"❌ Could not convert: {e}")

st.markdown(
    """
<div class="section-head">
    <h2>Missing Data Overview</h2>
    <div class="line"></div>
</div>
<div class="panel chart-panel">
    <div class="chart-grid"></div>
    <div class="chart-bars"><span></span><span></span><span></span><span></span><span></span><span></span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### ✅ Cleaned Dataset Status")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📝 Total Rows", df.shape[0])
col2.metric("📌 Total Columns", df.shape[1])
col3.metric("❌ Missing Values", df.isnull().sum().sum())
col4.metric("🔁 Duplicate Rows", df.duplicated().sum())

st.markdown(
    """
<div class="custom-card">
    <h3>🚀 Your cleaned data is ready!</h3>
    <p style="color:#aad4ff">
    All changes are <b style="color:white">automatically saved</b>.
    Navigate to <b style="color:white">Data Quality, Visualization, AI Insights, Predictions,
    or Export Report</b> from the sidebar — they will all use your cleaned dataset.
    <br><br>No need to download and re-upload!
    </p>
</div>
""",
    unsafe_allow_html=True,
)

divider()
st.markdown("### 👀 Cleaned Dataset Preview")
st.dataframe(df, use_container_width=True)

divider()
st.markdown("### 📥 Download Cleaned Dataset (Optional)")
st.markdown(
    "<p style='color:#aad4ff'>Save a local copy of your cleaned data for your records.</p>",
    unsafe_allow_html=True,
)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Cleaned CSV",
    data=csv,
    file_name="cleaned_data.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-28
</div>
</div>
""",
    unsafe_allow_html=True,
)

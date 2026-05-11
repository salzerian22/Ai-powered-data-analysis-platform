import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import io
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import divider, get_dataframe, apply_dark_theme, render_plotly_chart
from utils.logger import get_logger
from utils.styles import inject_global_css

logger = get_logger(__name__)

st.set_page_config(page_title="Predictions", page_icon="📈", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.pred-shell { position: relative; }
.pred-shell::before {
    content: "";
    position: absolute;
    inset: -28px 0 auto 0;
    height: 380px;
    background:
        radial-gradient(circle at 50% 8%, rgba(255, 212, 106, 0.16), transparent 34%),
        radial-gradient(circle at 78% 18%, rgba(255, 196, 77, 0.12), transparent 26%);
    pointer-events: none;
}
.pred-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(310px, 0.86fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.35rem 0 1rem 0;
}
.pred-copy, .pred-art { position: relative; z-index: 1; }
.pred-kicker { color: #ffd978; font: 600 0.82rem "IBM Plex Sans", sans-serif; letter-spacing: 0.18em; text-transform: uppercase; }
.pred-title { display: flex; align-items: center; gap: 1rem; margin-top: 0.6rem; }
.pred-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 70px; height: 70px; border-radius: 18px;
    background: linear-gradient(180deg, rgba(74, 58, 12, 0.96), rgba(28, 22, 8, 0.98));
    border: 1px solid rgba(255, 213, 108, 0.34);
    box-shadow: 0 0 24px rgba(255, 196, 77, 0.16);
    font-size: 2rem;
}
.pred-title h1 { margin: 0; color: #f9fbff; font: 800 clamp(2rem, 4.7vw, 3.15rem) "Inter", sans-serif; letter-spacing: -0.04em; }
.pred-title .accent { color: #ffcf63; }
.pred-sub { margin: 0.7rem 0 0 5.2rem; color: #ced8ea; font: 400 1.02rem/1.8 "IBM Plex Sans", sans-serif; }
.pred-art { min-height: 240px; }
.orb-glow { position: absolute; right: 56px; top: 18px; width: 230px; height: 230px; border-radius: 50%; background: radial-gradient(circle, rgba(255, 201, 83, 0.22), transparent 68%); filter: blur(12px); }
.orb-base { position: absolute; right: 52px; bottom: 6px; width: 200px; height: 38px; border-radius: 18px; background: linear-gradient(180deg, rgba(81, 56, 16, 0.98), rgba(24, 18, 10, 1)); transform: perspective(240px) rotateX(66deg); box-shadow: 0 0 20px rgba(255, 182, 65, 0.18); }
.orb-sphere { position: absolute; right: 72px; top: 2px; width: 190px; height: 190px; border-radius: 50%; background: radial-gradient(circle at 34% 28%, rgba(255,255,255,0.92), rgba(255, 236, 169, 0.48) 24%, rgba(255, 194, 68, 0.2) 44%, rgba(44, 29, 12, 0.58) 76%), linear-gradient(180deg, rgba(255, 213, 111, 0.36), rgba(120, 82, 16, 0.2)); border: 1px solid rgba(255, 220, 138, 0.32); box-shadow: 0 0 34px rgba(255, 196, 77, 0.2); overflow: hidden; }
.orb-sphere::before { content: ""; position: absolute; inset: 18px; border-radius: 50%; border: 1px solid rgba(255, 227, 163, 0.18); }
.orb-chart { position: absolute; left: 26px; top: 56px; width: 138px; height: 78px; }
.orb-chart svg { width: 100%; height: 100%; }
.chart-trail { position: absolute; right: 204px; top: 74px; width: 186px; height: 90px; }
.chart-trail svg { width: 100%; height: 100%; }
.hero-rule { height: 2px; margin: 0.95rem 0 1.8rem 0; background: linear-gradient(90deg, transparent, rgba(255, 214, 121, 0.88), transparent); box-shadow: 0 0 22px rgba(255, 214, 121, 0.18); }
.model-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 0.95rem; }
.model-card { min-height: 108px; padding: 1rem; border-radius: 18px; border: 1px solid rgba(255, 211, 104, 0.38); background: linear-gradient(180deg, rgba(28, 31, 40, 0.98), rgba(15, 18, 28, 1)); box-shadow: 0 0 20px rgba(255, 194, 77, 0.08); }
.model-icon { font-size: 1.5rem; }
.model-title { margin-top: 0.5rem; color: #fff4d4; font: 700 1.04rem "Inter", sans-serif; }
.model-desc { margin-top: 0.34rem; color: #c8d2e5; font: 400 0.86rem/1.55 "IBM Plex Sans", sans-serif; }
.results-box { margin-top: 1rem; padding: 1rem 1.1rem; border-radius: 18px; border: 1px solid rgba(255, 211, 104, 0.34); background: linear-gradient(180deg, rgba(17, 27, 44, 0.98), rgba(9, 18, 32, 1)); color: #e7edf8; font: 400 0.94rem/1.75 "IBM Plex Sans", sans-serif; }
.results-box strong { color: #ffd875; }
.target-badge { display: inline-block; padding: 0.18rem 0.7rem; border-radius: 999px; font: 600 0.78rem "IBM Plex Sans", sans-serif; margin-left: 0.5rem; }
.badge-reg { background: rgba(255,196,77,0.18); color: #ffd875; border: 1px solid rgba(255,196,77,0.38); }
.badge-cls { background: rgba(98,255,162,0.15); color: #7fffc4; border: 1px solid rgba(98,255,162,0.32); }
.badge-skip { background: rgba(255,104,104,0.14); color: #ff8a8a; border: 1px solid rgba(255,104,104,0.28); }
.multi-pred-card { margin: 0.6rem 0; padding: 1rem 1.1rem; border-radius: 14px; border: 1px solid rgba(255,211,104,0.28); background: linear-gradient(180deg, rgba(22,30,50,0.98), rgba(12,18,32,1)); }
.multi-pred-label { color: #a8c0e0; font: 600 0.82rem "IBM Plex Sans", sans-serif; text-transform: uppercase; letter-spacing: 0.1em; }
.multi-pred-value { color: #ffd875; font: 800 1.6rem "Inter", sans-serif; margin-top: 0.2rem; }
.multi-pred-meta { color: #7a95b8; font: 400 0.82rem "IBM Plex Sans", sans-serif; margin-top: 0.18rem; }
@media (max-width: 1100px) { .pred-hero, .model-grid { grid-template-columns: 1fr; } }
@media (max-width: 720px) { .pred-sub { margin-left: 0; } }
</style>
""",
    unsafe_allow_html=True,
)

# ── Data ─────────────────────────────────────────────────────
df = get_dataframe()
if df is None:
    st.warning("⚠️ No data found! Please upload a file on the Home page first.")
    st.stop()


# ── Smart target analysis ─────────────────────────────────────
def analyse_targets(df):
    n = len(df)
    result = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            result[col] = {"type": "skip", "reason": "all nulls"}
            continue
        n_unique = series.nunique()
        dtype = df[col].dtype
        if n_unique == n:
            result[col] = {"type": "skip", "reason": "all unique — likely an ID"}
            continue
        if n_unique == 1:
            result[col] = {"type": "skip", "reason": "constant — no variance"}
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            if n_unique <= 20:
                result[col] = {"type": "classification", "reason": f"{n_unique} unique numeric values"}
            else:
                std = series.std()
                if std == 0:
                    result[col] = {"type": "skip", "reason": "zero variance"}
                else:
                    result[col] = {"type": "regression", "reason": f"continuous numeric ({n_unique} unique)"}
        elif pd.api.types.is_object_dtype(dtype) or str(dtype) == "category":
            if n_unique <= 20:
                result[col] = {"type": "classification", "reason": f"{n_unique} unique categories"}
            else:
                result[col] = {"type": "skip", "reason": f"too many categories ({n_unique})"}
        else:
            result[col] = {"type": "skip", "reason": f"unsupported dtype ({dtype})"}
    return result


target_analysis = analyse_targets(df)
predictable_cols = {c: v for c, v in target_analysis.items() if v["type"] != "skip"}


def analyse_features(df, excluded_cols=None):
    excluded_cols = set(excluded_cols or [])
    result = {}
    for col in df.columns:
        if col in excluded_cols:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        n_unique = series.nunique()
        dtype = df[col].dtype
        if n_unique <= 1:
            continue
        if n_unique == len(series) and ("id" in col.lower() or "code" in col.lower() or "index" in col.lower()):
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            result[col] = {"type": "numeric", "reason": "usable numeric predictor"}
        elif pd.api.types.is_object_dtype(dtype) or str(dtype) == "category":
            if n_unique <= 20:
                result[col] = {"type": "categorical", "reason": f"categorical predictor ({n_unique} values)"}
        elif pd.api.types.is_bool_dtype(dtype):
            result[col] = {"type": "categorical", "reason": "boolean predictor"}
    return result


def profile_dataset(df: pd.DataFrame, target_col: str | None,
                    feature_cols: list[str]) -> dict:
    """
    Profiles the dataset and returns a dict of characteristics used
    for model recommendation and suitability warnings.
    """
    profile = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "task_type": None,
        "has_text_cols": False,
        "has_date_cols": False,
        "has_categorical_features": False,
        "class_imbalance": False,
        "high_cardinality_cols": [],
        "missing_pct": 0.0,
        "is_small_dataset": len(df) < 200,
        "is_very_small_dataset": len(df) < 50,
        "warnings": [],
        "special_modes": [],
    }

    # Task type detection
    if target_col and target_col in df.columns:
        target_series = df[target_col].dropna()
        n_unique_target = target_series.nunique()
        if pd.api.types.is_numeric_dtype(df[target_col]) and n_unique_target > 20:
            profile["task_type"] = "regression"
        elif n_unique_target <= 20:
            profile["task_type"] = "classification"
        elif pd.api.types.is_object_dtype(df[target_col]) and n_unique_target > 20:
            profile["task_type"] = "skip"
            profile["warnings"].append(
                f"⚠️ Target '{target_col}' has {n_unique_target} unique text "
                f"values — too many for classification."
            )
    else:
        profile["task_type"] = "clustering"
        profile["special_modes"].append("clustering")

    # Feature characteristics
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]):
            n_uniq = df[col].nunique()
            if n_uniq > 50:
                profile["high_cardinality_cols"].append(col)
                profile["warnings"].append(
                    f"⚠️ Feature '{col}' has {n_uniq} unique categories "
                    f"(high cardinality) — may slow training."
                )
            elif n_uniq > 1:
                profile["has_categorical_features"] = True
        # Date detection
        col_lower = col.lower()
        if any(h in col_lower for h in ["date", "time", "month", "day", "year", "timestamp"]):
            profile["has_date_cols"] = True
            profile["special_modes"].append("time_series")
            profile["warnings"].append(
                f"⚠️ Column '{col}' looks like a date — "
                f"use chronological split instead of random split."
            )
        # Text detection
        if pd.api.types.is_object_dtype(df[col]) and df[col].dropna().str.split().str.len().mean() > 4:
            profile["has_text_cols"] = True
            profile["special_modes"].append("nlp")

    # Missing values
    selected = [c for c in feature_cols + ([target_col] if target_col else []) if c in df.columns]
    if selected:
        total_cells = len(df) * len(selected)
        missing_cells = df[selected].isnull().sum().sum()
        profile["missing_pct"] = round((missing_cells / total_cells) * 100, 1) if total_cells > 0 else 0
        if profile["missing_pct"] > 0:
            rows_lost = len(df) - len(df[selected].dropna())
            profile["warnings"].append(
                f"⚠️ {profile['missing_pct']}% missing data — "
                f"~{rows_lost} rows will be dropped during training."
            )

    # Class imbalance check (classification only)
    if profile["task_type"] == "classification" and target_col and target_col in df.columns:
        counts = df[target_col].value_counts(normalize=True)
        if counts.max() > 0.80:
            profile["class_imbalance"] = True
            profile["warnings"].append(
                f"⚠️ Class imbalance detected — '{counts.idxmax()}' dominates "
                f"({counts.max()*100:.0f}%). Accuracy alone may be misleading."
            )

    # Size warnings
    if profile["is_very_small_dataset"]:
        profile["warnings"].append(
            f"⚠️ Only {len(df)} rows — results may be unreliable. "
            f"Consider adding more data."
        )
    elif profile["is_small_dataset"]:
        profile["warnings"].append(
            f"⚠️ Small dataset ({len(df)} rows) — avoid complex models "
            f"like Gradient Boosting or SVM."
        )

    return profile


def get_model_catalogue(task_type: str, profile: dict) -> list[dict]:
    """
    Returns an ordered list of model dicts for the given task.
    Each dict: {name, model_obj, tag, reason, recommended, badge}
    tag values: "✅ Recommended" | "⚠️ Use with caution" | "🔵 Baseline"
    """
    n = profile["n_rows"]
    has_cat = profile["has_categorical_features"]
    is_small = profile["is_small_dataset"]

    if task_type == "regression":
        return [
            {
                "name": "Random Forest Regressor",
                "model_obj": "RandomForestRegressor(n_estimators=100, random_state=42)",
                "tag": "✅ Recommended",
                "reason": "Best for mixed numeric/categorical tabular data. Handles non-linearity well.",
                "recommended": True,
                "badge": "badge-reg",
            },
            {
                "name": "Gradient Boosting Regressor",
                "model_obj": "GradientBoostingRegressor(n_estimators=100, random_state=42)",
                "tag": "✅ Recommended",
                "reason": "Strong for structured prediction. Often top performer on tabular data.",
                "recommended": True,
                "badge": "badge-reg",
            },
            {
                "name": "Decision Tree Regressor",
                "model_obj": "DecisionTreeRegressor(max_depth=6, random_state=42)",
                "tag": "⚠️ Use with caution",
                "reason": "Interpretable but prone to overfitting without depth limit.",
                "recommended": False,
                "badge": "badge-skip",
            },
            {
                "name": "KNN Regressor",
                "model_obj": "KNeighborsRegressor(n_neighbors=5)",
                "tag": "⚠️ Use with caution" if not is_small else "✅ Recommended",
                "reason": "Good for small datasets. Slow on large data.",
                "recommended": is_small,
                "badge": "badge-reg" if is_small else "badge-skip",
            },
            {
                "name": "Ridge Regression",
                "model_obj": "Ridge()",
                "tag": "🔵 Baseline",
                "reason": "Regularized linear model. Fast and stable baseline.",
                "recommended": False,
                "badge": "badge-cls",
            },
            {
                "name": "Lasso Regression",
                "model_obj": "Lasso()",
                "tag": "🔵 Baseline",
                "reason": "Forces sparse features. Good when many features are irrelevant.",
                "recommended": False,
                "badge": "badge-cls",
            },
            {
                "name": "Linear Regression",
                "model_obj": "LinearRegression()",
                "tag": "🔵 Baseline",
                "reason": "Simplest baseline. May underfit non-linear relationships.",
                "recommended": False,
                "badge": "badge-cls",
            },
            {
                "name": "SVR",
                "model_obj": "SVR()",
                "tag": "⚠️ Use with caution",
                "reason": "Works well on small datasets. Too slow for large data.",
                "recommended": False,
                "badge": "badge-skip",
            },
        ]

    elif task_type == "classification":
        return [
            {
                "name": "Random Forest Classifier",
                "model_obj": "RandomForestClassifier(n_estimators=100, random_state=42)",
                "tag": "✅ Recommended",
                "reason": "Robust to noise. Handles mixed feature types well.",
                "recommended": True,
                "badge": "badge-cls",
            },
            {
                "name": "Gradient Boosting Classifier",
                "model_obj": "GradientBoostingClassifier(n_estimators=100, random_state=42)",
                "tag": "✅ Recommended",
                "reason": "High accuracy on structured tabular classification tasks.",
                "recommended": True,
                "badge": "badge-cls",
            },
            {
                "name": "Decision Tree Classifier",
                "model_obj": "DecisionTreeClassifier(max_depth=6, random_state=42)",
                "tag": "⚠️ Use with caution",
                "reason": "Highly interpretable but overfits easily.",
                "recommended": False,
                "badge": "badge-skip",
            },
            {
                "name": "KNN Classifier",
                "model_obj": "KNeighborsClassifier(n_neighbors=5)",
                "tag": "⚠️ Use with caution",
                "reason": "Good for small, balanced datasets. Slow at scale.",
                "recommended": False,
                "badge": "badge-skip",
            },
            {
                "name": "SVM Classifier",
                "model_obj": "SVC(probability=True, random_state=42)",
                "tag": "⚠️ Use with caution",
                "reason": "Strong for binary classification. Slow on large datasets.",
                "recommended": False,
                "badge": "badge-skip",
            },
            {
                "name": "Naive Bayes",
                "model_obj": "GaussianNB()",
                "tag": "🔵 Baseline",
                "reason": "Very fast. Works well for simple or text classification.",
                "recommended": False,
                "badge": "badge-cls",
            },
            {
                "name": "Logistic Regression",
                "model_obj": "LogisticRegression(max_iter=500, random_state=42)",
                "tag": "🔵 Baseline",
                "reason": "Linear baseline for classification. Fast and interpretable.",
                "recommended": False,
                "badge": "badge-cls",
            },
        ]

    elif task_type == "clustering":
        return [
            {
                "name": "KMeans",
                "model_obj": "KMeans(n_clusters=3, random_state=42, n_init='auto')",
                "tag": "✅ Recommended",
                "reason": "Best general-purpose clustering. Fast and interpretable.",
                "recommended": True,
                "badge": "badge-reg",
            },
            {
                "name": "Agglomerative Clustering",
                "model_obj": "AgglomerativeClustering(n_clusters=3)",
                "tag": "✅ Recommended",
                "reason": "Hierarchical clustering — no need to specify cluster count upfront.",
                "recommended": True,
                "badge": "badge-reg",
            },
            {
                "name": "DBSCAN",
                "model_obj": "DBSCAN(eps=0.5, min_samples=5)",
                "tag": "⚠️ Use with caution",
                "reason": "Finds arbitrary shapes. Sensitive to eps parameter.",
                "recommended": False,
                "badge": "badge-skip",
            },
        ]

    return []


def build_preprocessor(X: pd.DataFrame) -> tuple:
    """
    Build a ColumnTransformer that handles numeric and categorical
    columns automatically. Returns (preprocessor, numeric_cols,
    categorical_cols).
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        from sklearn.preprocessing import StandardScaler
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ))

    if not transformers:
        from sklearn.preprocessing import FunctionTransformer
        preprocessor = FunctionTransformer()
    else:
        preprocessor = ColumnTransformer(transformers, remainder="drop")

    return preprocessor, numeric_cols, categorical_cols


def extract_date_features(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """
    Extract year, month, day, weekday from detected date columns.
    Returns a new DataFrame with date columns replaced by numeric features.
    """
    df = df.copy()
    for col in date_cols:
        if col not in df.columns:
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                df[f"{col}_year"]    = parsed.dt.year
                df[f"{col}_month"]   = parsed.dt.month
                df[f"{col}_day"]     = parsed.dt.day
                df[f"{col}_weekday"] = parsed.dt.weekday
                df.drop(columns=[col], inplace=True)
        except Exception:
            pass
    return df


feature_info = analyse_features(df, [])
numeric_features = [c for c, v in feature_info.items() if v["type"] == "numeric"]
categorical_features = [c for c, v in feature_info.items() if v["type"] == "categorical"]
feature_pool = numeric_features + categorical_features
numeric_feature_pool = numeric_features

# ── Hero ──────────────────────────────────────────────────────
st.markdown('<div class="pred-shell">', unsafe_allow_html=True)
st.markdown(
    """
<div class="pred-hero">
    <div class="pred-copy">
        <div class="pred-kicker">Forecast and model workspace</div>
        <div class="pred-title">
            <div class="pred-icon">📈</div>
            <h1><span class="accent">Predictive</span> Modeling</h1>
        </div>
        <div class="pred-sub">Forecast future trends, compare model performance, and generate predictions from your dataset.</div>
    </div>
    <div class="pred-art">
        <div class="chart-trail">
            <svg viewBox="0 0 210 100" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 70L40 52L68 64L98 42L126 50L154 26L206 34" fill="none" stroke="rgba(255,215,120,0.85)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="40" cy="52" r="4" fill="#ffd875"/>
                <circle cx="98" cy="42" r="5" fill="#ffd875"/>
                <circle cx="154" cy="26" r="5" fill="#ffd875"/>
                <circle cx="206" cy="34" r="4" fill="#fff2b5"/>
            </svg>
        </div>
        <div class="orb-glow"></div>
        <div class="orb-base"></div>
        <div class="orb-sphere">
            <div class="orb-chart">
                <svg viewBox="0 0 150 90" xmlns="http://www.w3.org/2000/svg">
                    <rect x="16" y="50" width="12" height="24" rx="2" fill="#ffca57"/>
                    <rect x="38" y="38" width="12" height="36" rx="2" fill="#ffca57"/>
                    <rect x="60" y="24" width="12" height="50" rx="2" fill="#ffca57"/>
                    <rect x="82" y="44" width="12" height="30" rx="2" fill="#ffca57"/>
                    <rect x="104" y="12" width="12" height="62" rx="2" fill="#ffca57"/>
                    <path d="M8 66L32 48L54 54L78 28L102 36L142 12" fill="none" stroke="#fff4b5" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M132 12L142 12L138 22" fill="none" stroke="#fff4b5" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
    </div>
</div>
<div class="hero-rule"></div>
""",
    unsafe_allow_html=True,
)

if len(predictable_cols) == 0:
    st.error("No predictable columns found. Need at least one numeric or low-cardinality column.")
    st.stop()

# ── Target analysis expander ──────────────────────────────────
st.markdown('<div class="section-head"><h2>Dataset Target Analysis</h2><div class="line"></div></div>', unsafe_allow_html=True)
analysis_rows = []
for col, info in target_analysis.items():
    badge = {"regression": "badge-reg", "classification": "badge-cls", "skip": "badge-skip"}[info["type"]]
    label = {"regression": "Regression", "classification": "Classification", "skip": "Skip"}[info["type"]]
    analysis_rows.append(
        f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="color:#c8d8f0;font:400 0.9rem IBM Plex Sans,sans-serif;flex:1;">{col}</span>'
        f'<span style="color:#7a95b8;font:400 0.82rem IBM Plex Sans,sans-serif;flex:2;">{info["reason"]}</span>'
        f'<span class="target-badge {badge}">{label}</span></div>'
    )
with st.expander("📊 View column suitability analysis", expanded=False):
    st.markdown(f'<div style="padding:0.5rem 0.2rem;">{"".join(analysis_rows)}</div>', unsafe_allow_html=True)

# ── Model cards ───────────────────────────────────────────────
st.markdown(
    """
<div class="section-head"><h2>Select a Model</h2><div class="line"></div></div>
<div class="model-grid">
    <div class="model-card"><div class="model-icon">📈</div><div class="model-title">Linear / Logistic Regression</div><div class="model-desc">Fast baseline. Linear for continuous targets, Logistic for categories.</div></div>
    <div class="model-card"><div class="model-icon">🌿</div><div class="model-title">Ridge</div><div class="model-desc">Regularized regression — more stable when features overlap.</div></div>
    <div class="model-card"><div class="model-icon">🌲</div><div class="model-title">Random Forest</div><div class="model-desc">Nonlinear ensemble for both regression and classification.</div></div>
</div>
""",
    unsafe_allow_html=True,
)

divider()

# ── Configuration ─────────────────────────────────────────────
st.markdown("### ⚙️ Configure Your Model")
st.markdown(
    "<p style='color:#aad4ff'>Select target and features — "
    "model recommendations update automatically.</p>",
    unsafe_allow_html=True,
)

all_predictable = list(predictable_cols.keys())
cfg_left, cfg_right = st.columns(2)

with cfg_left:
    target_cols = st.multiselect(
        "🎯 Target Columns (what to predict)",
        options=all_predictable,
        default=[all_predictable[0]] if all_predictable else [],
        help="Regression = continuous numeric. Classification = categorical/low-cardinality.",
    )

with cfg_right:
    excluded = set(target_cols)
    feature_info = analyse_features(df, list(excluded))
    all_feature_candidates = list(feature_info.keys())
    feature_cols = st.multiselect(
        "📊 Feature Columns (predictors — numeric AND categorical)",
        all_feature_candidates,
        default=all_feature_candidates[:min(5, len(all_feature_candidates))],
        help="Both numeric and categorical columns are supported.",
    )

if not target_cols:
    st.warning("⚠️ Please select at least one target column.")
    st.stop()
if not feature_cols:
    st.warning("⚠️ Please select at least one feature column.")
    st.stop()

# ── Dataset profile + warnings ─────────────────────────────────
first_target = target_cols[0] if target_cols else None
profile = profile_dataset(df, first_target, feature_cols)

if profile["warnings"]:
    with st.expander("⚠️ Dataset Suitability Warnings", expanded=True):
        for w in profile["warnings"]:
            st.warning(w)

# ── Auto Model Recommendation panel ───────────────────────────
task_type_for_reco = profile["task_type"] or "regression"
catalogue = get_model_catalogue(task_type_for_reco, profile)
model_names = [m["name"] for m in catalogue]

if "nlp" in profile["special_modes"]:
    st.info(
        "📝 Text columns detected. Consider using TF-IDF + Logistic Regression "
        "or Naive Bayes for text classification."
    )
if "time_series" in profile["special_modes"]:
    st.info(
        "📅 Date columns detected. Features extracted automatically. "
        "Consider chronological train/test split."
    )

st.markdown("#### 🤖 Recommended Models for This Dataset")
rec_cols = st.columns(min(len(catalogue), 3))
for i, m in enumerate(catalogue):
    col_idx = i % min(len(catalogue), 3)
    border_color = (
        "rgba(98,255,162,0.5)"  if m["tag"].startswith("✅") else
        "rgba(255,213,104,0.5)" if m["tag"].startswith("⚠️") else
        "rgba(100,160,255,0.4)"
    )
    with rec_cols[col_idx]:
        st.markdown(
            f"""
            <div style="padding:0.8rem;border-radius:12px;margin-bottom:0.6rem;
                        border:1px solid {border_color};
                        background:linear-gradient(180deg,rgba(18,28,50,0.98),
                        rgba(10,18,35,1));">
              <div style="font-weight:700;color:#f0f6ff;font-size:0.9rem;">
                {m['tag']} &nbsp;{m['name']}
              </div>
              <div style="color:#a0b8d8;font-size:0.8rem;margin-top:0.35rem;">
                {m['reason']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

model_choice = st.selectbox(
    "🤖 Select Model to Train",
    model_names,
    index=0,
    help="Recommended models are listed first.",
    key="model_choice",
)

# Show "Why recommended?" for selected model
selected_meta = next((m for m in catalogue if m["name"] == model_choice), None)
if selected_meta:
    tag_color = (
        "#7fffc4" if selected_meta["tag"].startswith("✅") else
        "#ffd875" if selected_meta["tag"].startswith("⚠️") else
        "#80c7ff"
    )
    st.markdown(
        f"<p style='color:{tag_color};font-size:0.88rem;margin-top:0.2rem;'>"
        f"{selected_meta['tag']} — {selected_meta['reason']}</p>",
        unsafe_allow_html=True,
    )

# Run-all toggle
run_all_models = st.checkbox(
    "🏁 Run All Recommended Models and compare",
    value=False,
    help="Trains all ✅ Recommended models and shows a comparison table.",
)

# ── Train ─────────────────────────────────────────────────────
if st.button("🚀 Train & Evaluate Model", use_container_width=True):
    st.session_state["pred_models"]       = {}
    st.session_state["pred_feature_cols"] = feature_cols
    st.session_state["pred_target_cols"]  = target_cols
    st.session_state["pred_model_choice"] = model_choice
    st.session_state["pred_profile"]      = profile
    st.session_state["pred_catalogue"]    = catalogue
    st.session_state["pred_run_all"]      = run_all_models
    st.session_state["pred_comparison"]   = []

    # Build list of models to train
    if run_all_models:
        models_to_run = [m for m in catalogue if m["recommended"]]
        if not any(m["name"] == model_choice for m in models_to_run):
            models_to_run.insert(0, next(m for m in catalogue if m["name"] == model_choice))
    else:
        models_to_run = [m for m in catalogue if m["name"] == model_choice]

    with st.spinner("Training models... ⏳"):
        for target_col in target_cols:
            task_type = predictable_cols[target_col]["type"]
            selected_cols = feature_cols + [target_col]

            # ── Preprocessing ──────────────────────────────
            work_df = df[selected_cols].copy()

            # Date feature extraction
            date_hint_cols = [
                c for c in feature_cols
                if any(h in c.lower() for h in
                       ["date","time","month","day","year","timestamp"])
                and c in work_df.columns
            ]
            if date_hint_cols:
                work_df = extract_date_features(work_df, date_hint_cols)

            # Drop rows with missing values
            clean_df = work_df.dropna()
            rows_dropped = len(df) - len(clean_df)
            if rows_dropped > 0:
                st.warning(
                    f"⚠️ '{target_col}': dropped {rows_dropped} rows "
                    f"with missing values."
                )
            if len(clean_df) < 10:
                st.error(
                    f"❌ '{target_col}': only {len(clean_df)} rows after "
                    f"cleaning — skipping."
                )
                continue

            # Separate X and y
            actual_feature_cols = [c for c in clean_df.columns if c != target_col]
            X = clean_df[actual_feature_cols]
            y_raw = clean_df[target_col]

            le = None
            if task_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
            else:
                y = y_raw.values

            # Guard: classification needs ≥2 classes with ≥2 rows each
            if task_type == "classification":
                class_counts = pd.Series(y).value_counts()
                if len(class_counts) < 2 or class_counts.min() < 2:
                    st.error(
                        f"❌ '{target_col}': each class needs ≥2 rows — skipping."
                    )
                    continue
                stratify_y = y
            else:
                class_counts = None
                stratify_y = None

            # Guard: multiclass split size (Fix 1.1)
            n_classes = len(np.unique(y)) if task_type == "classification" else 1
            test_count = max(n_classes, math.ceil(0.2 * len(clean_df)))
            test_size = min(test_count / len(clean_df), 0.4)

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    stratify=stratify_y,
                )
            except ValueError as e:
                st.error(
                    f"Cannot split '{target_col}': {e}. "
                    f"Try adding more rows."
                )
                continue

            # Build preprocessor from actual feature types
            preprocessor, num_cols, cat_cols = build_preprocessor(X)

            # Train each model in the run list
            comparison_rows = []
            for model_meta in models_to_run:
                try:
                    # Instantiate model from string repr
                    model_obj = eval(model_meta["model_obj"], {
                        "LinearRegression": LinearRegression,
                        "Ridge": Ridge, "Lasso": Lasso,
                        "DecisionTreeRegressor": DecisionTreeRegressor,
                        "RandomForestRegressor": RandomForestRegressor,
                        "GradientBoostingRegressor": GradientBoostingRegressor,
                        "KNeighborsRegressor": KNeighborsRegressor,
                        "SVR": SVR,
                        "LogisticRegression": LogisticRegression,
                        "DecisionTreeClassifier": DecisionTreeClassifier,
                        "RandomForestClassifier": RandomForestClassifier,
                        "GradientBoostingClassifier": GradientBoostingClassifier,
                        "KNeighborsClassifier": KNeighborsClassifier,
                        "SVC": SVC,
                        "GaussianNB": GaussianNB,
                        "KMeans": KMeans,
                        "DBSCAN": DBSCAN,
                        "AgglomerativeClustering": AgglomerativeClustering,
                    })

                    pipeline = Pipeline([
                        ("prep", preprocessor),
                        ("model", model_obj),
                    ])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    train_pred = pipeline.predict(X_train)

                    # CV setup
                    n_cv = min(5, max(2, len(clean_df) // 5))
                    if task_type == "classification":
                        n_cv = min(n_cv, int(class_counts.min()))
                        cv_strategy = StratifiedKFold(
                            n_splits=n_cv, shuffle=True, random_state=42
                        )
                        scoring = "f1_weighted"
                    else:
                        cv_strategy = KFold(
                            n_splits=n_cv, shuffle=True, random_state=42
                        )
                        scoring = "r2"

                    cv_scores = cross_val_score(
                        pipeline, X, y,
                        cv=cv_strategy, scoring=scoring
                    )

                    # Metrics
                    if task_type == "regression":
                        r2        = r2_score(y_test, y_pred)
                        train_r2  = r2_score(y_train, train_pred)
                        mse       = mean_squared_error(y_test, y_pred)
                        rmse      = np.sqrt(mse)
                        mae       = mean_absolute_error(y_test, y_pred)
                        dummy     = DummyRegressor(strategy="mean")
                        dummy.fit(X_train, y_train)
                        dummy_r2  = r2_score(y_test, dummy.predict(X_test))
                        overfit   = (train_r2 - r2) > 0.15
                        metrics   = dict(
                            r2=r2, train_r2=train_r2, mse=mse,
                            rmse=rmse, mae=mae,
                            dummy_r2=dummy_r2, cv_scores=cv_scores,
                            overfit=overfit,
                        )
                        comparison_rows.append({
                            "Model": model_meta["name"],
                            "R²": round(r2, 4),
                            "RMSE": round(rmse, 2),
                            "MAE": round(mae, 2),
                            "CV Mean": round(cv_scores.mean(), 4),
                            "Tag": model_meta["tag"],
                        })
                    else:
                        acc       = accuracy_score(y_test, y_pred)
                        train_acc = accuracy_score(y_train, train_pred)
                        f1        = f1_score(
                            y_test, y_pred,
                            average="weighted", zero_division=0
                        )
                        dummy     = DummyClassifier(strategy="most_frequent")
                        dummy.fit(X_train, y_train)
                        dummy_acc = accuracy_score(
                            y_test, dummy.predict(X_test)
                        )
                        overfit   = (train_acc - acc) > 0.15
                        report_labels = np.unique(
                            np.concatenate([y_test, y_pred])
                        )
                        report    = classification_report(
                            y_test, y_pred,
                            labels=report_labels,
                            target_names=[
                                str(le.inverse_transform([lb])[0])
                                for lb in report_labels
                            ],
                            zero_division=0,
                        )
                        cm        = confusion_matrix(
                            y_test, y_pred, labels=report_labels
                        )
                        metrics   = dict(
                            acc=acc, train_acc=train_acc,
                            f1=f1, dummy_acc=dummy_acc,
                            cv_scores=cv_scores, report=report,
                            cm=cm, cm_labels=report_labels,
                            overfit=overfit,
                            le=le,
                        )
                        comparison_rows.append({
                            "Model": model_meta["name"],
                            "Accuracy": round(acc, 4),
                            "F1": round(f1, 4),
                            "CV Mean": round(cv_scores.mean(), 4),
                            "Tag": model_meta["tag"],
                        })

                    # Feature importance
                    inner_model = pipeline.named_steps["model"]
                    all_feat_names = (
                        num_cols +
                        (
                            list(pipeline.named_steps["prep"]
                                 .named_transformers_["cat"]
                                 .get_feature_names_out(cat_cols))
                            if cat_cols and "cat" in pipeline.named_steps["prep"].named_transformers_
                            else []
                        )
                    )
                    if hasattr(inner_model, "feature_importances_"):
                        importances = inner_model.feature_importances_
                    elif hasattr(inner_model, "coef_"):
                        coef = inner_model.coef_
                        importances = (
                            np.abs(coef).mean(axis=0)
                            if coef.ndim > 1 else np.abs(coef)
                        )
                    else:
                        importances = np.zeros(len(all_feat_names))

                    imp_len = min(len(importances), len(all_feat_names))
                    imp_df  = (
                        pd.DataFrame({
                            "Feature": all_feat_names[:imp_len],
                            "Importance": importances[:imp_len],
                        })
                        .sort_values("Importance", ascending=False)
                        .head(15)
                    )

                    # Only store first model's full results for display
                    if model_meta["name"] == model_choice or \
                       target_col not in st.session_state["pred_models"]:
                        st.session_state["pred_models"][target_col] = {
                            "pipeline":      pipeline,
                            "task_type":     task_type,
                            "metrics":       metrics,
                            "importance_df": imp_df,
                            "result_df": pd.DataFrame({
                                "Actual": y_test,
                                "Predicted": y_pred,
                            }),
                            "y_test":   y_test,
                            "y_pred":   y_pred,
                            "le":       le,
                            "model_name": model_meta["name"],
                            "actual_feature_cols": actual_feature_cols,
                            "X_sample": X.head(1),
                        }
                        st.session_state["pred_clean_means"] = {
                            target_col: X.mean(numeric_only=True).to_dict()
                        }

                except Exception as exc:
                    st.warning(
                        f"⚠️ '{model_meta['name']}' failed on "
                        f"'{target_col}': {exc}"
                    )

            if comparison_rows:
                st.session_state["pred_comparison"].extend(comparison_rows)

    st.success("✅ All models trained!")

# ── Results ───────────────────────────────────────────────────
if st.session_state.get("pred_models"):
    stored_features = st.session_state["pred_feature_cols"]
    stored_targets  = st.session_state["pred_target_cols"]
    stored_model    = st.session_state.get("pred_model_choice", model_choice)
    comparison_data = st.session_state.get("pred_comparison", [])

    # ── Model Comparison Table (run-all mode) ──────────────
    if comparison_data and len(comparison_data) > 1:
        divider()
        st.markdown("### 📊 Model Comparison Table")
        comp_df = pd.DataFrame(comparison_data)
        # Highlight best model
        score_col = "R²" if "R²" in comp_df.columns else "Accuracy"
        if score_col in comp_df.columns:
            best_idx = comp_df[score_col].idxmax()
            comp_df["🏆"] = comp_df.index.map(
                lambda i: "✅ Best" if i == best_idx else ""
            )
        st.dataframe(comp_df, use_container_width=True)

        # Export comparison CSV
        comp_csv = comp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Model Comparison CSV",
            data=comp_csv,
            file_name="model_comparison.csv",
            mime="text/csv",
        )

    # ── Per-target detailed results ────────────────────────
    for target_col in stored_targets:
        if target_col not in st.session_state["pred_models"]:
            continue

        data      = st.session_state["pred_models"][target_col]
        task_type = data["task_type"]
        metrics   = data["metrics"]
        imp_df    = data["importance_df"]
        result_df = data["result_df"]
        y_test    = data["y_test"]
        y_pred    = data["y_pred"]
        le        = data.get("le")
        model_name = data.get("model_name", stored_model)

        badge_cls = "badge-reg" if task_type == "regression" else "badge-cls"
        badge_lbl = "Regression" if task_type == "regression" else "Classification"

        st.markdown(
            f'<div class="section-head"><h2>Results — {target_col} '
            f'<span class="target-badge {badge_cls}">{badge_lbl}</span>'
            f'&nbsp;<span style="color:#a0b8d8;font-size:0.9rem;">'
            f'({model_name})</span></h2>'
            f'<div class="line"></div></div>',
            unsafe_allow_html=True,
        )

        if task_type == "regression":
            r2       = metrics["r2"]
            train_r2 = metrics["train_r2"]
            mse      = metrics["mse"]
            rmse     = metrics["rmse"]
            mae      = metrics["mae"]
            dummy_r2 = metrics["dummy_r2"]

            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("R² (Test)",  f"{r2:.3f}",
                      delta=f"{r2 - dummy_r2:+.3f} vs baseline")
            p2.metric("R² (Train)", f"{train_r2:.3f}")
            p3.metric("Baseline R²", f"{dummy_r2:.3f}")
            p4.metric("RMSE",  f"{rmse:.2f}")
            p5.metric("MAE",   f"{mae:.2f}")

            if metrics.get("overfit"):
                st.warning(
                    f"⚠️ Possible overfitting: train R² ({train_r2:.3f}) "
                    f"is much higher than test R² ({r2:.3f})."
                )
            if r2 <= dummy_r2:
                st.error(f"❌ Model doesn't beat the baseline (R²={r2:.3f}).")
            elif r2 >= 0.8:
                st.success(f"✅ Excellent! R² = {r2:.3f}")
            elif r2 >= 0.6:
                st.info(f"ℹ️ Good model. R² = {r2:.3f}")
            else:
                st.warning(f"⚠️ Weak model. R² = {r2:.3f} — try more features.")

        else:
            acc       = metrics["acc"]
            train_acc = metrics["train_acc"]
            f1        = metrics["f1"]
            dummy_acc = metrics["dummy_acc"]

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Accuracy (Test)",  f"{acc:.3f}",
                      delta=f"{acc - dummy_acc:+.3f} vs baseline")
            p2.metric("Accuracy (Train)", f"{train_acc:.3f}")
            p3.metric("F1 (weighted)",    f"{f1:.3f}")
            p4.metric("Baseline Acc",     f"{dummy_acc:.3f}")

            if metrics.get("overfit"):
                st.warning(
                    f"⚠️ Possible overfitting: train accuracy "
                    f"({train_acc:.3f}) >> test accuracy ({acc:.3f})."
                )
            if acc <= dummy_acc:
                st.error(f"❌ Model doesn't beat the baseline.")
            elif acc >= 0.8:
                st.success(f"✅ Excellent! Accuracy = {acc:.3f}")
            elif acc >= 0.6:
                st.info(f"ℹ️ Good. Accuracy = {acc:.3f}")
            else:
                st.warning(f"⚠️ Weak model. Accuracy = {acc:.3f}")

        # Cross-validation
        cv_scores = metrics["cv_scores"]
        st.markdown("#### 🔄 Cross-Validation")
        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("CV Mean",    f"{cv_scores.mean():.3f}")
        cv2.metric("CV Std",     f"{cv_scores.std():.3f}")
        cv3.metric("Min / Max",  f"{cv_scores.min():.3f} / {cv_scores.max():.3f}")
        if cv_scores.std() > 0.1:
            st.warning("⚠️ High variance across folds — results may be unstable.")
        else:
            st.success("✅ Stable across folds.")

        # Classification extras: confusion matrix + report
        if task_type == "classification":
            with st.expander("📋 Classification Report", expanded=False):
                st.text(metrics["report"])

            cm         = metrics.get("cm")
            cm_labels  = metrics.get("cm_labels", [])
            if cm is not None and le is not None:
                try:
                    label_names = [
                        str(le.inverse_transform([lb])[0])
                        for lb in cm_labels
                    ]
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                    fig_cm.patch.set_facecolor("#0b1220")
                    ax_cm.set_facecolor("#0d1f3c")
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=label_names,
                    )
                    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
                    ax_cm.tick_params(colors="white")
                    ax_cm.xaxis.label.set_color("white")
                    ax_cm.yaxis.label.set_color("white")
                    ax_cm.title.set_color("#4da6ff")
                    plt.title(f"Confusion Matrix — {target_col}",
                              color="#4da6ff")
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
                except Exception:
                    pass

        # Regression charts
        if task_type == "regression":
            fig1 = px.scatter(
                result_df, x="Actual", y="Predicted",
                title=f"Actual vs Predicted — {target_col}",
                color_discrete_sequence=["#ffd36d"],
            )
            fig1.add_shape(
                type="line",
                x0=result_df["Actual"].min(), y0=result_df["Actual"].min(),
                x1=result_df["Actual"].max(), y1=result_df["Actual"].max(),
                line=dict(color="#7fc0ff", dash="dash"),
            )
            apply_dark_theme(fig1)
            render_plotly_chart(fig1, use_container_width=True)

            residuals = y_test - y_pred
            fig_r = px.scatter(
                x=y_pred, y=residuals,
                labels={"x": "Fitted values", "y": "Residuals"},
                title=f"Residuals vs Fitted — {target_col}",
                color_discrete_sequence=["#ffd36d"], opacity=0.6,
            )
            fig_r.add_hline(y=0, line_dash="dash", line_color="red")
            apply_dark_theme(fig_r)
            render_plotly_chart(fig_r, use_container_width=True)

        # Feature importance chart
        if not imp_df.empty:
            fig2 = px.bar(
                imp_df, x="Feature", y="Importance",
                title=f"Feature Importance — {target_col} ({model_name})",
                color="Importance",
                color_continuous_scale=["#4b3a11", "#c7902f", "#ffd36d"],
            )
            apply_dark_theme(fig2)
            render_plotly_chart(fig2, use_container_width=True)

            # Export feature importance
            fi_csv = imp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download Feature Importance CSV ({target_col})",
                data=fi_csv,
                file_name=f"feature_importance_{target_col}.csv",
                mime="text/csv",
                key=f"fi_dl_{target_col}",
            )

        # Export prediction results
        pred_csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download Prediction Results CSV ({target_col})",
            data=pred_csv,
            file_name=f"predictions_{target_col}.csv",
            mime="text/csv",
            key=f"pred_dl_{target_col}",
        )
        divider()

# ── Predict for New Data ──────────────────────────────────────
st.markdown(
    '<div class="section-head"><h2>🎯 Predict for New Data</h2>'
    '<div class="line"></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#aad4ff'>Inputs are auto-generated from selected "
    "features. Numeric → number input. Categorical → dropdown. "
    "Date → text input (YYYY-MM-DD).</p>",
    unsafe_allow_html=True,
)

pred_cfg_left, pred_cfg_right = st.columns(2)
with pred_cfg_left:
    pred_target_col = st.selectbox(
        "🎯 Target Variable",
        options=list(predictable_cols.keys()),
        key="new_pred_target_col",
    )
with pred_cfg_right:
    pred_feature_info = analyse_features(df, [pred_target_col])
    pred_feature_candidates = list(pred_feature_info.keys())
    pred_feature_cols = st.multiselect(
        "📊 Feature Columns",
        pred_feature_candidates,
        default=pred_feature_candidates[:min(5, len(pred_feature_candidates))],
        key="new_pred_feature_cols",
    )

if pred_feature_cols:
    pred_clean_df = df[pred_feature_cols + [pred_target_col]].dropna()
    pred_task = predictable_cols[pred_target_col]["type"]

    # ── Auto-generate input widgets by feature type ────────
    input_values = {}
    input_cols = st.columns(min(len(pred_feature_cols), 3))
    for i, feat in enumerate(pred_feature_cols):
        col_widget = input_cols[i % min(len(pred_feature_cols), 3)]
        with col_widget:
            feat_lower = feat.lower()
            # Date column
            if any(h in feat_lower for h in
                   ["date","time","month","day","timestamp"]):
                input_values[feat] = st.text_input(
                    feat, value="2024-01-01",
                    key=f"new_pred_{feat}",
                )
            # Categorical column
            elif pd.api.types.is_object_dtype(df[feat]):
                unique_vals = df[feat].dropna().unique().tolist()
                input_values[feat] = st.selectbox(
                    feat, unique_vals,
                    key=f"new_pred_{feat}",
                )
            # Numeric column
            else:
                mean_val = float(
                    pred_clean_df[feat].mean()
                    if feat in pred_clean_df.columns else 0.0
                )
                input_values[feat] = st.number_input(
                    feat, value=mean_val,
                    key=f"new_pred_{feat}",
                )

    pred_model_choice_new = st.selectbox(
        "🤖 Model",
        model_names if "model_names" in dir() else
        ["Random Forest Regressor", "Linear Regression"],
        key="new_pred_model_choice",
    )

    if st.button("🔮 Predict", use_container_width=True):
        if len(pred_clean_df) < 10:
            st.error("❌ Not enough rows for prediction.")
        else:
            try:
                # Build input row
                input_row = {}
                for feat in pred_feature_cols:
                    feat_lower = feat.lower()
                    val = input_values[feat]
                    if any(h in feat_lower for h in
                           ["date","time","month","day","timestamp"]):
                        try:
                            parsed_date = pd.to_datetime(val)
                            input_row[f"{feat}_year"]    = parsed_date.year
                            input_row[f"{feat}_month"]   = parsed_date.month
                            input_row[f"{feat}_day"]     = parsed_date.day
                            input_row[f"{feat}_weekday"] = parsed_date.weekday()
                        except Exception:
                            input_row[feat] = val
                    else:
                        input_row[feat] = val

                X_input = pd.DataFrame([input_row])

                # Re-train on full data for prediction
                work = pred_clean_df.copy()
                X_full = work[pred_feature_cols]
                y_full_raw = work[pred_target_col]

                pred_le = None
                if pred_task == "classification":
                    pred_le = LabelEncoder()
                    y_full  = pred_le.fit_transform(y_full_raw.astype(str))
                else:
                    y_full = y_full_raw.values

                pred_preprocessor, _, _ = build_preprocessor(X_full)

                # Find model obj from catalogue
                catalogue_now = st.session_state.get(
                    "pred_catalogue",
                    get_model_catalogue(
                        predictable_cols[pred_target_col]["type"],
                        profile_dataset(df, pred_target_col, pred_feature_cols),
                    )
                )
                pred_meta = next(
                    (m for m in catalogue_now
                     if m["name"] == pred_model_choice_new),
                    catalogue_now[0],
                )
                pred_model_obj = eval(pred_meta["model_obj"], {
                    "LinearRegression": LinearRegression,
                    "Ridge": Ridge, "Lasso": Lasso,
                    "DecisionTreeRegressor": DecisionTreeRegressor,
                    "RandomForestRegressor": RandomForestRegressor,
                    "GradientBoostingRegressor": GradientBoostingRegressor,
                    "KNeighborsRegressor": KNeighborsRegressor,
                    "SVR": SVR,
                    "LogisticRegression": LogisticRegression,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "RandomForestClassifier": RandomForestClassifier,
                    "GradientBoostingClassifier": GradientBoostingClassifier,
                    "KNeighborsClassifier": KNeighborsClassifier,
                    "SVC": SVC,
                    "GaussianNB": GaussianNB,
                })
                pred_pipeline_new = Pipeline([
                    ("prep", pred_preprocessor),
                    ("model", pred_model_obj),
                ])
                pred_pipeline_new.fit(X_full, y_full)

                raw_pred = pred_pipeline_new.predict(X_input)[0]

                if pred_task == "classification" and pred_le is not None:
                    try:
                        pred_display = str(
                            pred_le.inverse_transform([int(raw_pred)])[0]
                        )
                    except Exception:
                        pred_display = str(raw_pred)
                else:
                    pred_display = f"{float(raw_pred):.4f}"

                st.markdown(
                    f'<div class="multi-pred-card">'
                    f'<div class="multi-pred-label">'
                    f'Predicted {pred_target_col}</div>'
                    f'<div class="multi-pred-value">{pred_display}</div>'
                    f'<div class="multi-pred-meta">'
                    f'{pred_task} · {pred_model_choice_new}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")
else:
    st.info("Select at least one feature column to enable prediction.")

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-26
</div>
</div>
""",
    unsafe_allow_html=True,
)

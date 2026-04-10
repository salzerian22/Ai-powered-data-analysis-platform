import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
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


numeric_feature_pool = df.select_dtypes(include=["number"]).columns.tolist()

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
st.markdown("<p style='color:#aad4ff'>Select one or more target columns to predict</p>", unsafe_allow_html=True)

all_predictable = list(predictable_cols.keys())
cfg_left, cfg_right = st.columns(2)
with cfg_left:
    target_cols = st.multiselect(
        "🎯 Target Columns (what to predict)",
        options=all_predictable,
        default=[all_predictable[0]] if all_predictable else [],
        help="Only columns suitable for prediction are shown. Regression = continuous. Classification = categorical / low-cardinality.",
    )
with cfg_right:
    excluded = set(target_cols)
    available_features = [c for c in numeric_feature_pool if c not in excluded]
    feature_cols = st.multiselect(
        "📊 Feature Columns (predictors)",
        available_features,
        default=available_features[:min(5, len(available_features))],
        help="Feature options are auto-filtered from the dataset analysis to keep only usable predictor columns.",
    )

model_choice = st.selectbox(
    "🤖 Select Model",
    ["Linear / Logistic Regression", "Ridge", "Random Forest"],
    key="model_choice",
)

if not target_cols:
    st.warning("⚠️ Please select at least one target column.")
    st.stop()
if not feature_cols:
    st.warning("⚠️ Please select at least one feature column.")
    st.stop()

# ── Train ─────────────────────────────────────────────────────
if st.button("🚀 Train & Run Prediction Model", use_container_width=True):
    st.session_state["pred_models"]        = {}
    st.session_state["pred_feature_cols"]  = feature_cols
    st.session_state["pred_target_cols"]   = target_cols
    st.session_state["pred_clean_means"]   = {}
    st.session_state["pred_model_choice"]  = model_choice

    with st.spinner("Training models... ⏳"):
        for target_col in target_cols:
            task_type = predictable_cols[target_col]["type"]
            selected_cols = feature_cols + [target_col]
            clean_df = df[selected_cols].dropna()
            rows_dropped = len(df) - len(clean_df)
            if rows_dropped > 0:
                st.warning(f"⚠️ '{target_col}': dropped {rows_dropped} rows with missing values.")
            if len(clean_df) < 10:
                st.error(f"❌ '{target_col}': only {len(clean_df)} rows after NaN removal — skipping.")
                continue

            X = clean_df[feature_cols]
            y_raw = clean_df[target_col]

            le = None
            if task_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
            else:
                y = y_raw.values

            if task_type == "classification":
                class_counts = pd.Series(y).value_counts()
                if len(class_counts) < 2 or class_counts.min() < 2:
                    st.error(f"❌ '{target_col}': each class needs at least 2 rows for training and testing — skipping.")
                    continue
                stratify_y = y
            else:
                class_counts = None
                stratify_y = None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )

            if task_type == "regression":
                if model_choice == "Linear / Logistic Regression":
                    base_model = LinearRegression()
                elif model_choice == "Ridge":
                    base_model = Ridge()
                else:
                    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = "r2"
            else:
                if model_choice in ("Linear / Logistic Regression", "Ridge"):
                    base_model = LogisticRegression(max_iter=500, random_state=42)
                else:
                    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = "f1_weighted"

            pipeline = Pipeline([("scaler", StandardScaler()), ("model", base_model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            n_cv = min(5, max(2, len(clean_df) // 5))
            if task_type == "classification":
                n_cv = min(n_cv, int(class_counts.min()))
                cv_strategy = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
            else:
                cv_strategy = KFold(n_splits=n_cv, shuffle=True, random_state=42)

            if task_type == "regression":
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                dummy = DummyRegressor(strategy="mean")
                dummy.fit(X_train, y_train)
                dummy_r2 = r2_score(y_test, dummy.predict(X_test))
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring="r2")
                metrics = dict(r2=r2, mse=mse, rmse=rmse, dummy_r2=dummy_r2, cv_scores=cv_scores)
            else:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                dummy = DummyClassifier(strategy="most_frequent")
                dummy.fit(X_train, y_train)
                dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring="f1_weighted")
                report_labels = np.unique(np.concatenate([y_test, y_pred]))
                report = classification_report(
                    y_test, y_pred,
                    labels=report_labels,
                    target_names=[str(le.inverse_transform([label])[0]) for label in report_labels],
                    zero_division=0,
                )
                metrics = dict(acc=acc, f1=f1, dummy_acc=dummy_acc, cv_scores=cv_scores, report=report)

            if model_choice == "Random Forest":
                importances = pipeline.named_steps["model"].feature_importances_
            elif task_type == "regression":
                importances = np.abs(pipeline.named_steps["model"].coef_)
            else:
                coef = pipeline.named_steps["model"].coef_
                importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

            imp_df = (
                pd.DataFrame({"Feature": feature_cols, "Importance": importances})
                .sort_values("Importance", ascending=False)
            )

            st.session_state["pred_models"][target_col] = {
                "pipeline":      pipeline,
                "task_type":     task_type,
                "metrics":       metrics,
                "importance_df": imp_df,
                "result_df":     pd.DataFrame({"Actual": y_test, "Predicted": y_pred}),
                "y_test":        y_test,
                "y_pred":        y_pred,
                "le":            le,
            }
            st.session_state["pred_clean_means"][target_col] = X.mean().to_dict()

    st.success("✅ All models trained!")

# ── Results — outside button block so they survive reruns ─────
if st.session_state.get("pred_models"):
    stored_features = st.session_state["pred_feature_cols"]
    stored_targets  = st.session_state["pred_target_cols"]
    stored_model    = st.session_state.get("pred_model_choice", model_choice)

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
        le        = data["le"]
        badge_cls = "badge-reg" if task_type == "regression" else "badge-cls"
        badge_lbl = "Regression" if task_type == "regression" else "Classification"

        st.markdown(
            f'<div class="section-head"><h2>Results — {target_col} '
            f'<span class="target-badge {badge_cls}">{badge_lbl}</span></h2><div class="line"></div></div>',
            unsafe_allow_html=True,
        )

        if task_type == "regression":
            r2, mse, rmse, dummy_r2 = metrics["r2"], metrics["mse"], metrics["rmse"], metrics["dummy_r2"]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("R² Score", f"{r2:.3f}", delta=f"{r2 - dummy_r2:+.3f} vs baseline")
            p2.metric("Baseline R²", f"{dummy_r2:.3f}")
            p3.metric("MSE", f"{mse:.2f}")
            p4.metric("RMSE", f"{rmse:.2f}")
            if r2 <= dummy_r2:
                st.error(f"❌ '{target_col}' model doesn't beat the baseline.")
            elif r2 >= 0.8:
                st.success(f"✅ Excellent! R² = {r2:.3f}")
            elif r2 >= 0.6:
                st.info(f"ℹ️ Good model. R² = {r2:.3f}")
            else:
                st.warning(f"⚠️ Weak model. R² = {r2:.3f} — consider adding more features.")
        else:
            acc, f1, dummy_acc = metrics["acc"], metrics["f1"], metrics["dummy_acc"]
            p1, p2, p3 = st.columns(3)
            p1.metric("Accuracy", f"{acc:.3f}", delta=f"{acc - dummy_acc:+.3f} vs baseline")
            p2.metric("F1 (weighted)", f"{f1:.3f}")
            p3.metric("Baseline Accuracy", f"{dummy_acc:.3f}")
            if acc <= dummy_acc:
                st.error(f"❌ '{target_col}' model doesn't beat the baseline.")
            elif acc >= 0.8:
                st.success(f"✅ Excellent! Accuracy = {acc:.3f}")
            elif acc >= 0.6:
                st.info(f"ℹ️ Good model. Accuracy = {acc:.3f}")
            else:
                st.warning(f"⚠️ Weak model. Accuracy = {acc:.3f}")

        cv_scores = metrics["cv_scores"]
        st.markdown("#### 🔄 Cross-Validation")
        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("CV Mean", f"{cv_scores.mean():.3f}")
        cv2.metric("CV Std",  f"{cv_scores.std():.3f}")
        cv3.metric("Min / Max", f"{cv_scores.min():.3f} / {cv_scores.max():.3f}")
        if cv_scores.std() > 0.1:
            st.warning("⚠️ High variance across folds.")
        else:
            st.success("✅ Stable across folds.")

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
        else:
            with st.expander("📋 Classification Report"):
                st.text(metrics["report"])

        fig2 = px.bar(
            imp_df, x="Feature", y="Importance",
            title=f"Feature Importance — {target_col}",
            color="Importance",
            color_continuous_scale=["#4b3a11", "#c7902f", "#ffd36d"],
        )
        apply_dark_theme(fig2)
        render_plotly_chart(fig2, use_container_width=True)
        divider()

    # ── Predict for New Data ──────────────────────────────────
    st.markdown(
        '<div class="section-head"><h2>🎯 Predict for New Data</h2><div class="line"></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#aad4ff'>Choose the target variable, valid feature columns, and the model here. This section uses dataset analysis to show only usable options.</p>",
        unsafe_allow_html=True,
    )

    pred_cfg_left, pred_cfg_mid, pred_cfg_right = st.columns(3)
    with pred_cfg_left:
        pred_target_col = st.selectbox(
            "🎯 Target Variable",
            options=list(predictable_cols.keys()),
            key="new_pred_target_col",
        )
    with pred_cfg_mid:
        pred_feature_candidates = [c for c in analyse_features(df, [pred_target_col]).keys() if c in numeric_feature_pool]
        pred_feature_cols = st.multiselect(
            "📊 Feature Columns",
            pred_feature_candidates,
            default=pred_feature_candidates[:min(5, len(pred_feature_candidates))],
            key="new_pred_feature_cols",
        )
    with pred_cfg_right:
        pred_model_choice = st.selectbox(
            "🤖 Model to Use for Prediction",
            ["Linear / Logistic Regression", "Ridge", "Random Forest"],
            key="new_pred_model_choice",
        )

    if pred_feature_cols:
        pred_clean_df = df[pred_feature_cols + [pred_target_col]].dropna()
        pred_means = pred_clean_df[pred_feature_cols].mean().to_dict() if not pred_clean_df.empty else {col: 0.0 for col in pred_feature_cols}

        input_values = {}
        n_feat = len(pred_feature_cols)
        input_cols = st.columns(min(n_feat, 4))
        for i, feat in enumerate(pred_feature_cols):
            with input_cols[i % min(n_feat, 4)]:
                input_values[feat] = st.number_input(
                    feat,
                    value=float(pred_means.get(feat, 0.0)),
                    key=f"new_pred_{feat}",
                )

        if st.button("🔮 Predict Target", use_container_width=True):
            if len(pred_clean_df) < 10:
                st.error("❌ Not enough valid rows to make a prediction with the selected target and features.")
            else:
                pred_task = predictable_cols[pred_target_col]["type"]
                X_pred_train = pred_clean_df[pred_feature_cols]
                y_pred_raw = pred_clean_df[pred_target_col]

                pred_le = None
                if pred_task == "classification":
                    pred_le = LabelEncoder()
                    y_pred_train = pred_le.fit_transform(y_pred_raw.astype(str))
                else:
                    y_pred_train = y_pred_raw.values

                if pred_task == "regression":
                    if pred_model_choice == "Linear / Logistic Regression":
                        pred_model = LinearRegression()
                    elif pred_model_choice == "Ridge":
                        pred_model = Ridge()
                    else:
                        pred_model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    if pred_model_choice in ("Linear / Logistic Regression", "Ridge"):
                        pred_model = LogisticRegression(max_iter=500, random_state=42)
                    else:
                        pred_model = RandomForestClassifier(n_estimators=100, random_state=42)

                pred_pipeline = Pipeline([("scaler", StandardScaler()), ("model", pred_model)])
                pred_pipeline.fit(X_pred_train, y_pred_train)

                input_array = [input_values[f] for f in pred_feature_cols]
                raw_pred = pred_pipeline.predict([input_array])[0]

                if pred_task == "classification" and pred_le is not None:
                    try:
                        pred_display = str(pred_le.inverse_transform([int(raw_pred)])[0])
                    except Exception:
                        pred_display = str(raw_pred)
                else:
                    pred_display = f"{float(raw_pred):.4f}"

                st.markdown(f"#### Prediction using {pred_model_choice}")
                st.markdown(
                    f'<div class="multi-pred-card">'
                    f'<div class="multi-pred-label">{pred_target_col}</div>'
                    f'<div class="multi-pred-value">{pred_display}</div>'
                    f'<div class="multi-pred-meta">{pred_task} · {pred_model_choice}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Select at least one valid feature column in this section to enable prediction.")

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-26
</div>
</div>
""",
    unsafe_allow_html=True,
)

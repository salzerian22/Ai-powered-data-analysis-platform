import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import divider, get_dataframe, apply_dark_theme
from utils.logger import get_logger
from utils.styles import inject_global_css

logger = get_logger(__name__)


st.set_page_config(page_title="Predictions", page_icon="📈", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.pred-shell {
    position: relative;
}

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

.pred-copy,
.pred-art {
    position: relative;
    z-index: 1;
}

.pred-kicker {
    color: #ffd978;
    font: 600 0.82rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.pred-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.6rem;
}

.pred-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(74, 58, 12, 0.96), rgba(28, 22, 8, 0.98));
    border: 1px solid rgba(255, 213, 108, 0.34);
    box-shadow: 0 0 24px rgba(255, 196, 77, 0.16);
    font-size: 2rem;
}

.pred-title h1 {
    margin: 0;
    color: #f9fbff;
    font: 800 clamp(2rem, 4.7vw, 3.15rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.pred-title .accent {
    color: #ffcf63;
}

.pred-sub {
    margin: 0.7rem 0 0 5.2rem;
    color: #ced8ea;
    font: 400 1.02rem/1.8 "IBM Plex Sans", sans-serif;
}

.pred-art {
    min-height: 240px;
}

.orb-glow {
    position: absolute;
    right: 56px;
    top: 18px;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 201, 83, 0.22), transparent 68%);
    filter: blur(12px);
}

.orb-base {
    position: absolute;
    right: 52px;
    bottom: 6px;
    width: 200px;
    height: 38px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(81, 56, 16, 0.98), rgba(24, 18, 10, 1));
    transform: perspective(240px) rotateX(66deg);
    box-shadow: 0 0 20px rgba(255, 182, 65, 0.18);
}

.orb-sphere {
    position: absolute;
    right: 72px;
    top: 2px;
    width: 190px;
    height: 190px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 34% 28%, rgba(255,255,255,0.92), rgba(255, 236, 169, 0.48) 24%, rgba(255, 194, 68, 0.2) 44%, rgba(44, 29, 12, 0.58) 76%),
        linear-gradient(180deg, rgba(255, 213, 111, 0.36), rgba(120, 82, 16, 0.2));
    border: 1px solid rgba(255, 220, 138, 0.32);
    box-shadow: 0 0 34px rgba(255, 196, 77, 0.2);
    overflow: hidden;
}

.orb-sphere::before {
    content: "";
    position: absolute;
    inset: 18px;
    border-radius: 50%;
    border: 1px solid rgba(255, 227, 163, 0.18);
}

.orb-chart {
    position: absolute;
    left: 26px;
    top: 56px;
    width: 138px;
    height: 78px;
}

.orb-chart svg {
    width: 100%;
    height: 100%;
}

.chart-trail {
    position: absolute;
    right: 204px;
    top: 74px;
    width: 186px;
    height: 90px;
}

.chart-trail svg {
    width: 100%;
    height: 100%;
}

.hero-rule {
    height: 2px;
    margin: 0.95rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(255, 214, 121, 0.88), transparent);
    box-shadow: 0 0 22px rgba(255, 214, 121, 0.18);
}

.model-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 0.95rem;
}

.model-card {
    min-height: 108px;
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid rgba(255, 211, 104, 0.38);
    background: linear-gradient(180deg, rgba(28, 31, 40, 0.98), rgba(15, 18, 28, 1));
    box-shadow: 0 0 20px rgba(255, 194, 77, 0.08);
}

.model-icon {
    font-size: 1.5rem;
}

.model-title {
    margin-top: 0.5rem;
    color: #fff4d4;
    font: 700 1.04rem "Inter", sans-serif;
}

.model-desc {
    margin-top: 0.34rem;
    color: #c8d2e5;
    font: 400 0.86rem/1.55 "IBM Plex Sans", sans-serif;
}

.prompt-note {
    margin-top: 1rem;
    padding: 1rem 1.08rem;
    border-radius: 18px;
    border: 1px solid rgba(103, 129, 170, 0.52);
    background: linear-gradient(180deg, rgba(14, 25, 44, 0.98), rgba(8, 18, 33, 1));
    color: #d4deef;
    font: 400 0.95rem/1.75 "IBM Plex Sans", sans-serif;
}

.results-box {
    margin-top: 1rem;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(255, 211, 104, 0.34);
    background: linear-gradient(180deg, rgba(17, 27, 44, 0.98), rgba(9, 18, 32, 1));
    color: #e7edf8;
    font: 400 0.94rem/1.75 "IBM Plex Sans", sans-serif;
}

.results-box strong {
    color: #ffd875;
}

@media (max-width: 1100px) {
    .pred-hero,
    .model-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .pred-sub {
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

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

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

if len(numeric_cols) < 2:
    st.info("ℹ️ Need at least 2 numeric columns for prediction.")
    st.stop()

st.markdown(
    """
<div class="section-head">
    <h2>Select a Model</h2>
    <div class="line"></div>
</div>
<div class="model-grid">
    <div class="model-card">
        <div class="model-icon">📈</div>
        <div class="model-title">Linear Regression</div>
        <div class="model-desc">Fast linear baseline for continuous targets and easy interpretability.</div>
    </div>
    <div class="model-card">
        <div class="model-icon">🌿</div>
        <div class="model-title">Ridge</div>
        <div class="model-desc">Regularized regression that can stay more stable when features overlap.</div>
    </div>
    <div class="model-card">
        <div class="model-icon">🌲</div>
        <div class="model-title">Random Forest</div>
        <div class="model-desc">Nonlinear ensemble model that can capture more complex relationships.</div>
    </div>
</div>
<div class="prompt-note">Configure your target, choose predictor columns, select a model, and then train it using the exact same prediction pipeline already built into this page.</div>
""",
    unsafe_allow_html=True,
)

divider()
st.markdown("### ⚙️ Configure Your Model")
st.markdown("<p style='color:#aad4ff'>Select your target and feature columns</p>", unsafe_allow_html=True)

cfg_left, cfg_right = st.columns(2)
with cfg_left:
    target_col = st.selectbox("🎯 Target Column (what to predict)", numeric_cols)
with cfg_right:
    feature_cols = st.multiselect("📊 Feature Columns (predictors)", [c for c in numeric_cols if c != target_col])

if len(feature_cols) == 0:
    st.warning("⚠️ Please select at least one feature column!")
    st.stop()

model_choice = st.selectbox(
    "🤖 Select Model",
    ["Linear Regression", "Ridge", "Random Forest"],
    key="model_choice",
)

if st.button("🚀 Train & Run Prediction Model", use_container_width=True):
    with st.spinner("Training model... ⏳"):
        selected_cols = feature_cols + [target_col]
        clean_df = df[selected_cols].dropna()
        rows_dropped = len(df) - len(clean_df)

        if rows_dropped > 0:
            st.warning(f"⚠️ Dropped {rows_dropped} rows with missing values in selected columns.")

        if len(clean_df) < 10:
            st.error(
                "❌ Not enough data after removing missing values (need at least 10 rows). "
                "Please clean your data first on the Data Cleaning page."
            )
            st.stop()

        X = clean_df[feature_cols]
        y = clean_df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge":
            model = Ridge()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.markdown(
            """
<div class="section-head">
    <h2>Model Performance</h2>
    <div class="line"></div>
</div>
""",
            unsafe_allow_html=True,
        )

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        dummy_r2 = r2_score(y_test, dummy.predict(X_test))

        perf1, perf2, perf3, perf4 = st.columns(4)
        perf1.metric("R² Score", f"{r2:.3f}", delta=f"{r2 - dummy_r2:+.3f} vs baseline")
        perf2.metric("Baseline R² (predict mean)", f"{dummy_r2:.3f}")
        perf3.metric("MSE", f"{mse:.2f}")
        perf4.metric("RMSE", f"{rmse:.2f}")

        if r2 <= dummy_r2:
            st.error("❌ Your model does NOT beat the baseline (predicting the mean). It is not learning anything useful from the features.")
        elif r2 >= 0.8:
            st.success(f"✅ Excellent Model! R² = {r2:.3f} (beats baseline by {r2 - dummy_r2:+.3f})")
        elif r2 >= 0.6:
            st.info(f"ℹ️ Good Model! R² = {r2:.3f} (beats baseline by {r2 - dummy_r2:+.3f})")
        else:
            st.warning(f"⚠️ Weak Model. R² = {r2:.3f} — consider adding more features.")

        divider()

        st.markdown("### 🔄 Cross-Validation (5-Fold)")
        st.markdown("<p style='color:#aad4ff'>Evaluates model stability across different data splits</p>", unsafe_allow_html=True)

        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("R² (5-fold CV mean)", f"{cv_scores.mean():.3f}")
        cv2.metric("R² (std across folds)", f"{cv_scores.std():.3f}")
        cv3.metric("Min / Max fold R²", f"{cv_scores.min():.3f} / {cv_scores.max():.3f}")

        if cv_scores.std() > 0.1:
            st.warning("⚠️ High variance across folds — model performance varies significantly with different data splits. Interpret R² with caution.")
        else:
            st.success("✅ Low variance across folds — model performance is stable.")

        with st.expander("📋 Individual fold scores"):
            fold_df = pd.DataFrame(
                {
                    "Fold": [f"Fold {i+1}" for i in range(5)],
                    "R² Score": cv_scores.round(4),
                }
            )
            st.dataframe(fold_df, use_container_width=True)
            st.caption("Each fold uses 80% of data for training and 20% for testing, rotating which portion is held out.")

        divider()

        st.markdown(
            """
<div class="section-head">
    <h2>Prediction Results</h2>
    <div class="line"></div>
</div>
""",
            unsafe_allow_html=True,
        )

        result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig1 = px.scatter(
            result_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Values",
            color_discrete_sequence=["#ffd36d"],
        )
        fig1.add_shape(
            type="line",
            x0=result_df["Actual"].min(),
            y0=result_df["Actual"].min(),
            x1=result_df["Actual"].max(),
            y1=result_df["Actual"].max(),
            line=dict(color="#7fc0ff", dash="dash"),
        )
        apply_dark_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)

        divider()

        st.markdown("### 🔬 Residual Diagnostics")
        st.markdown("<p style='color:#aad4ff'>Checks whether linear regression assumptions hold</p>", unsafe_allow_html=True)

        residuals = y_test.values - y_pred
        fig_resid = px.scatter(
            x=y_pred,
            y=residuals,
            labels={"x": "Fitted values", "y": "Residuals"},
            title="Residuals vs Fitted Values",
            color_discrete_sequence=["#ffd36d"],
            opacity=0.6,
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        apply_dark_theme(fig_resid)
        st.plotly_chart(fig_resid, use_container_width=True)
        st.caption("ℹ️ Residuals should scatter randomly around 0. Patterns (funnel shape, curves) indicate model assumptions are violated.")

        res1, res2, res3 = st.columns(3)
        res1.metric("Mean Residual", f"{residuals.mean():.4f}")
        res2.metric("Std of Residuals", f"{residuals.std():.4f}")
        res3.metric("Max |Residual|", f"{np.max(np.abs(residuals)):.4f}")

        divider()

        st.markdown("### 🏆 Feature Importance")
        if model_choice != "Random Forest":
            std_coefs = np.abs(pipeline.named_steps["model"].coef_)
            importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": std_coefs}).sort_values("Importance", ascending=False)
            st.caption("ℹ️ Importance = |standardized coefficient|. Features are scaled before regression so coefficients are directly comparable.")
        else:
            importances = pipeline.named_steps["model"].feature_importances_
            importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False)
            st.caption("ℹ️ Importance = feature importances from Random Forest.")

        fig2 = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            title="Feature Importance",
            color="Importance",
            color_continuous_scale=["#4b3a11", "#c7902f", "#ffd36d"],
        )
        apply_dark_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        divider()

        st.markdown("### 📋 Prediction Results Table")
        result_df["Difference"] = abs(result_df["Actual"] - result_df["Predicted"]).round(2)
        st.dataframe(result_df.round(2), use_container_width=True)

        divider()

        st.markdown("### 🎯 Predict for New Data")
        st.markdown("<p style='color:#aad4ff'>Enter values below to get a prediction</p>", unsafe_allow_html=True)

        input_values = []
        cols = st.columns(min(len(feature_cols), 4))
        for i, col in enumerate(feature_cols):
            with cols[i % len(cols)]:
                val = st.number_input(col, value=float(clean_df[col].mean()), key=f"pred_input_{col}")
                input_values.append(val)

        if st.button("🔮 Predict", use_container_width=True):
            prediction = pipeline.predict([input_values])[0]
            st.metric(label=f"Predicted {target_col}", value=f"{prediction:.2f}")
            st.markdown(
                f"""
<div class="results-box">
    <strong>Prediction Result:</strong> Based on the values you entered, the model estimates <strong>{target_col}</strong> at <strong>{prediction:.2f}</strong>.
</div>
""",
                unsafe_allow_html=True,
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

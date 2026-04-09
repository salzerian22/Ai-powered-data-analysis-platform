import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import (
    divider,
    get_dataframe,
    GROQ_API_KEY,
    sanitize_query,
    clean_llm_query,
)
from utils.logger import get_logger
from utils.styles import inject_global_css
from groq import Groq

logger = get_logger(__name__)


try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception:
    groq_client = None

st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
inject_global_css()

st.markdown(
    """
<style>
.ai-shell {
    position: relative;
}

.ai-shell::before {
    content: "";
    position: absolute;
    inset: -30px 0 auto 0;
    height: 380px;
    background:
        radial-gradient(circle at 50% 10%, rgba(89, 182, 255, 0.18), transparent 38%),
        radial-gradient(circle at 80% 18%, rgba(114, 236, 255, 0.1), transparent 28%);
    pointer-events: none;
}

.ai-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(300px, 0.82fr);
    gap: 1.8rem;
    align-items: center;
    padding: 1.4rem 0 1rem 0;
}

.ai-copy,
.ai-art {
    position: relative;
    z-index: 1;
}

.ai-kicker {
    color: #92d0ff;
    font: 600 0.82rem "IBM Plex Sans", sans-serif;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.ai-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.6rem;
}

.ai-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(18, 57, 106, 0.96), rgba(8, 25, 50, 0.98));
    border: 1px solid rgba(103, 255, 211, 0.26);
    box-shadow: 0 0 24px rgba(82, 175, 255, 0.18);
    font-size: 2rem;
}

.ai-title h1 {
    margin: 0;
    color: #f6fbff;
    font: 800 clamp(2rem, 4.8vw, 3.2rem) "Inter", sans-serif;
    letter-spacing: -0.04em;
}

.ai-title .accent {
    color: #a8d2ff;
}

.ai-sub {
    margin: 0.65rem 0 0.4rem 5.25rem;
    color: #b7c7df;
    font: 400 1.02rem/1.8 "IBM Plex Sans", sans-serif;
}

.ai-powered {
    margin-left: 5.25rem;
    color: #d5ebff;
    font: 600 1.02rem "IBM Plex Sans", sans-serif;
}

.ai-powered span {
    color: #84beff;
}

.ai-art {
    min-height: 240px;
}

.hero-chart {
    position: absolute;
    right: 146px;
    top: 48px;
    width: 200px;
    height: 120px;
    opacity: 0.95;
}

.hero-chart svg {
    width: 100%;
    height: 100%;
}

.brain-glow {
    position: absolute;
    right: 64px;
    top: 22px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(98, 207, 255, 0.16), transparent 68%);
    filter: blur(12px);
}

.hero-orbit {
    position: absolute;
    right: 8px;
    top: 30px;
    width: 320px;
    height: 190px;
    border-radius: 50%;
    border: 1px solid rgba(117, 208, 255, 0.14);
    transform: perspective(420px) rotateY(-18deg);
}

.ai-node {
    position: absolute;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #8be1ff;
    box-shadow: 0 0 12px rgba(139, 225, 255, 0.8);
}

.node-a { right: 246px; top: 56px; }
.node-b { right: 218px; top: 100px; }
.node-c { right: 188px; top: 70px; }
.node-d { right: 56px; top: 126px; background: #ff6f84; }

.ai-face {
    position: absolute;
    right: 40px;
    top: 0;
    width: 212px;
    height: 256px;
}

.hero-rule {
    height: 2px;
    margin: 0.95rem 0 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(102, 224, 255, 0.88), transparent);
    box-shadow: 0 0 22px rgba(102, 224, 255, 0.18);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 0.9rem;
}

.metric-card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(94, 132, 211, 0.7);
    background: linear-gradient(180deg, rgba(16, 28, 51, 0.96), rgba(8, 20, 36, 0.98));
    box-shadow: inset 0 1px 0 rgba(132, 181, 255, 0.08), 0 0 24px rgba(42, 105, 187, 0.1);
}

.metric-label {
    color: #bfd4ee;
    font: 500 0.92rem "IBM Plex Sans", sans-serif;
}

.metric-value {
    margin-top: 0.3rem;
    color: #ffffff;
    font: 800 2rem "Inter", sans-serif;
}

.metric-value.hot {
    color: #ff7272;
}

.panel-box {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(89, 131, 204, 0.62);
    background: linear-gradient(180deg, rgba(17, 29, 49, 0.96), rgba(9, 18, 35, 0.98));
    box-shadow: inset 0 1px 0 rgba(132, 181, 255, 0.08), 0 0 20px rgba(32, 93, 170, 0.08);
}

.prompt-strip {
    margin-top: 1rem;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(89, 131, 204, 0.62);
    background: linear-gradient(180deg, rgba(13, 25, 46, 0.98), rgba(8, 19, 36, 1));
}

.insight-box {
    margin-top: 1rem;
    padding: 1.05rem 1.15rem;
    border-radius: 18px;
    border: 1px solid rgba(93, 131, 204, 0.62);
    background: linear-gradient(180deg, rgba(14, 26, 45, 0.98), rgba(7, 16, 31, 1));
    color: #d7e6f7;
    font: 400 0.95rem/1.8 "IBM Plex Sans", sans-serif;
}

.insight-box .highlight {
    color: #ff8e8e;
    font-weight: 700;
}

@media (max-width: 1100px) {
    .ai-hero,
    .metric-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .ai-sub,
    .ai-powered {
        margin-left: 0;
    }

    .ai-title {
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

numeric_cols = df.select_dtypes(include="number").columns.tolist()
string_cols = df.select_dtypes(include="object").columns.tolist()
top_findings = len(numeric_cols) + len(string_cols)

st.markdown('<div class="ai-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="ai-hero">
    <div class="ai-copy">
        <div class="ai-kicker">AI-driven analysis workspace</div>
        <div class="ai-title">
            <div class="ai-icon">🤖</div>
            <h1><span class="accent">AI</span> Insights</h1>
        </div>
        <div class="ai-sub">Let the model analyze your dataset, answer plain-English questions, and support deeper data exploration.</div>
        <div class="ai-powered">Powered by <span>LLaMA 3.3</span></div>
    </div>
    <div class="ai-art">
        <div class="hero-chart">
            <svg viewBox="0 0 220 120" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="lineBlue" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#69d6ff"/>
                        <stop offset="100%" stop-color="#8ef3ff"/>
                    </linearGradient>
                </defs>
                <path d="M4 92H216" stroke="rgba(137,200,255,0.16)" stroke-width="1"/>
                <path d="M22 88L56 78L84 84L118 52L146 62L178 34L212 46" stroke="url(#lineBlue)" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M22 90L56 96L84 86L118 72L146 82L178 68L212 74" stroke="rgba(112,177,255,0.4)" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="56" cy="78" r="4" fill="#84e8ff"/>
                <circle cx="118" cy="52" r="5" fill="#84e8ff"/>
                <circle cx="178" cy="34" r="5" fill="#84e8ff"/>
                <circle cx="212" cy="46" r="4" fill="#ff6f84"/>
            </svg>
        </div>
        <div class="brain-glow"></div>
        <div class="hero-orbit"></div>
        <div class="ai-node node-a"></div>
        <div class="ai-node node-b"></div>
        <div class="ai-node node-c"></div>
        <div class="ai-node node-d"></div>
        <div class="ai-face">
            <svg viewBox="0 0 230 270" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="headGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#f4fbff"/>
                        <stop offset="38%" stop-color="#bddcf7"/>
                        <stop offset="100%" stop-color="#17365f"/>
                    </linearGradient>
                    <linearGradient id="neckGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#122b52"/>
                        <stop offset="100%" stop-color="#07192f"/>
                    </linearGradient>
                </defs>
                <path d="M152 16C187 23 205 54 206 89C208 122 199 149 182 168C170 181 164 196 164 215V236H119L108 210C93 201 78 190 68 171C52 145 46 120 47 91C49 52 72 25 111 17C126 14 141 13 152 16Z" fill="url(#headGrad)" stroke="rgba(162,220,255,0.45)" stroke-width="2"/>
                <path d="M126 58C140 58 152 70 152 85C152 100 141 112 126 112C111 112 100 100 100 85C100 70 112 58 126 58Z" fill="rgba(18,41,74,0.72)"/>
                <circle cx="124" cy="86" r="10" fill="#7de0ff"/>
                <circle cx="124" cy="86" r="20" fill="none" stroke="rgba(125,224,255,0.34)" stroke-width="5"/>
                <path d="M85 90C97 85 106 83 119 84" stroke="#0d2344" stroke-width="3" stroke-linecap="round"/>
                <path d="M82 126C98 131 111 132 129 130" stroke="#16335c" stroke-width="3" stroke-linecap="round"/>
                <path d="M74 161C92 164 111 163 131 156" stroke="#0f2749" stroke-width="3" stroke-linecap="round"/>
                <path d="M122 175L150 188L161 230H109L101 193Z" fill="url(#neckGrad)" stroke="rgba(112,178,255,0.24)" stroke-width="2"/>
                <path d="M150 48C173 54 186 69 191 93C194 114 188 135 171 150" fill="none" stroke="#0c2244" stroke-width="8" stroke-linecap="round"/>
                <circle cx="184" cy="106" r="31" fill="rgba(17,36,68,0.96)" stroke="rgba(128,217,255,0.4)" stroke-width="2"/>
                <circle cx="184" cy="106" r="15" fill="#7de0ff"/>
                <circle cx="184" cy="106" r="23" fill="none" stroke="rgba(125,224,255,0.34)" stroke-width="6"/>
                <path d="M43 214L84 188L108 230H28Z" fill="#09192f" stroke="rgba(112,178,255,0.18)" stroke-width="2"/>
                <path d="M162 230L200 204L220 247H164Z" fill="#09192f" stroke="rgba(112,178,255,0.18)" stroke-width="2"/>
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
    <h2>Analysis Overview</h2>
    <div class="line"></div>
</div>
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Records Analyzed</div>
        <div class="metric-value">{df.shape[0]:,}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Top Findings</div>
        <div class="metric-value hot">{top_findings}</div>
    </div>
</div>
<div class="prompt-strip">Use the controls below to generate insights, query the dataset, or chat with the assistant.</div>
""",
    unsafe_allow_html=True,
)

divider()
st.markdown(
    """
<div class="section-head">
    <h2>Generated Insights</h2>
    <div class="line"></div>
</div>
<div class="insight-box">
    Use <span class="highlight">Generate AI Insights</span> for a full summary, ask a plain-English data question below, or chat with the assistant for iterative exploration.
</div>
""",
    unsafe_allow_html=True,
)

def render_ai_generated_insights():
    st.markdown("### 🔍 AI-Generated Insights")
    st.markdown("<p style='color:#aad4ff'>Click the button to get AI analysis</p>", unsafe_allow_html=True)

    if st.button("✨ Generate AI Insights", use_container_width=True):
        if groq_client is None:
            st.error("❌ AI client is not available right now. Please check your GROQ API key.")
        else:
            with st.spinner("AI is analyzing your data... ⏳"):
                summary = f"""
                Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
                Columns: {list(df.columns)}
                Statistical Summary: {df.describe().to_string()}
                Missing Values: {df.isnull().sum().to_string()}
                Duplicate Rows: {df.duplicated().sum()}
                """
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst."},
                        {
                            "role": "user",
                            "content": f"""
                            Analyze this dataset and provide:
                            1. Key observations
                            2. Important patterns
                            3. Data quality issues
                            4. Business recommendations
                            Dataset Info: {summary}
                            Give clear, simple, human-readable insights.
                        """,
                        },
                    ],
                )
                st.success("✅ AI Analysis Complete!")
                st.write(response.choices[0].message.content)


def render_natural_language_query():
    divider()
    st.markdown("### 💬 Natural Language Query")
    st.markdown("<p style='color:#aad4ff'>Ask questions about your data in plain English!</p>", unsafe_allow_html=True)

    dynamic_examples = []
    if numeric_cols:
        dynamic_examples.append(f"Show rows where {numeric_cols[0]} > {int(df[numeric_cols[0]].median())}")
    if string_cols:
        top_val = df[string_cols[0]].mode().iloc[0] if not df[string_cols[0]].mode().empty else "X"
        dynamic_examples.append(f'Find rows where {string_cols[0]} == "{top_val}"')
    if len(numeric_cols) >= 2:
        dynamic_examples.append(
            f"Show rows where {numeric_cols[0]} > {int(df[numeric_cols[0]].quantile(0.25))} and {numeric_cols[1]} < {int(df[numeric_cols[1]].quantile(0.75))}"
        )

    if not dynamic_examples:
        dynamic_examples = ["Show the first 10 rows", "Find rows where column > value"]

    st.info(f"💡 Examples: {' | '.join(dynamic_examples)}")

    query_left, query_right = st.columns([5, 1])
    with query_left:
        user_question = st.text_input(
            "Ask about your data:",
            placeholder="e.g. Show rows where Salary > 70000",
            key="ai_query_input",
        )
    with query_right:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Analyze", use_container_width=True)

    if search_btn and user_question:
        if groq_client is None:
            st.error("❌ AI client is not available right now. Please check your GROQ API key.")
        else:
            with st.spinner("Thinking... ⏳"):
                prompt = f"""You are a Python pandas expert. Your ONLY job is to convert a
user question into a pandas `df.query()` expression string.

DataFrame columns and dtypes:
{df.dtypes.to_string()}

Sample rows:
{df.head(3).to_string()}

RULES (follow strictly):
1. Return ONLY the query string — no explanation, no code fences, no quotes around the whole thing.
2. For numeric comparisons: use operators directly, e.g.  Salary > 70000
3. For string comparisons: use ==  with double quotes around the value, e.g.  Department == "IT"
4. If a column name contains spaces or special characters, wrap it in backticks, e.g.  `First Name` == "Alice"
5. For multiple conditions use 'and' / 'or', e.g.  Age > 30 and Department == "HR"
6. Do NOT use .str, .isin(), .apply(), or any method calls.
7. Do NOT wrap output in df.query(...).
8. Do NOT use parentheses for function calls.

User question: {user_question}
"""
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Return only the pandas df.query() expression string. Nothing else."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                raw_query = response.choices[0].message.content.strip()
                query = clean_llm_query(raw_query)

                is_safe, reason = sanitize_query(query)
                if not is_safe:
                    st.error(f"❌ Unsafe query blocked: {reason}. Try rephrasing!")
                    st.caption(f"Raw LLM output: `{raw_query}`")
                else:
                    try:
                        result = df.query(query)
                        if len(result) == 0:
                            st.warning("⚠️ No results found!")
                            st.caption(f"Query used: `{query}`")
                        else:
                            st.success(f"✅ Found {len(result)} matching rows!")
                            st.write(f"**Query:** `{query}`")
                            st.dataframe(result, use_container_width=True)
                    except Exception as e:
                        logger.exception("Failed to process natural language query")
                        st.error("❌ Could not process. Try rephrasing your question!")
                        st.caption(f"Query attempted: `{query}` — Error: {e}")


def render_chat_section():
    divider()
    st.markdown("### 🤖 Chat with AI about your Data")
    st.markdown("<p style='color:#aad4ff'>Ask me anything about your dataset!</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    chat_left, chat_right = st.columns([5, 1])
    with chat_left:
        user_chat = st.text_input(
            "💬 Type your message...",
            key="chat_input",
            placeholder="e.g. What is the average salary?",
        )
    with chat_right:
        st.markdown("<br>", unsafe_allow_html=True)
        send_btn = st.button("Send", use_container_width=True)

    if send_btn and user_chat:
        st.session_state.messages.append({"role": "user", "content": user_chat})

        with st.chat_message("user"):
            st.write(user_chat)

        with st.chat_message("assistant"):
            if groq_client is None:
                st.error("❌ AI client is not available right now. Please check your GROQ API key.")
            else:
                with st.spinner("Thinking... ⏳"):
                    data_context = f"""
                    You are a helpful data analysis assistant.
                    Dataset: {df.shape[0]} rows, {df.shape[1]} columns
                    Columns: {list(df.columns)}
                    Stats: {df.describe().to_string()}
                    Missing: {df.isnull().sum().to_string()}
                    Duplicates: {df.duplicated().sum()}
                    Sample: {df.head(5).to_string()}
                    Answer clearly and friendly.
                    """
                    messages = [{"role": "system", "content": data_context}]
                    for msg in st.session_state.messages[-10:]:
                        messages.append(msg)

                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                    )
                    reply = response.choices[0].message.content
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

render_ai_generated_insights()
render_natural_language_query()
render_chat_section()

clear_left, clear_right = st.columns([4, 1])
with clear_right:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """
<div class="footer-note">
    2025 Shri Ramdeobaba College Department of Data Science | Session 2025-26
</div>
</div>
""",
    unsafe_allow_html=True,
)

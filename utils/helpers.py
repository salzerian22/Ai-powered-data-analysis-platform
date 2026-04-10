from __future__ import annotations

import streamlit as st
import os
import re
import copy
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils.logger import get_logger

logger = get_logger(__name__)

PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0b1220",
        "plot_bgcolor": "#0d1f3c",
        "font": {"color": "#ffffff"},
        "title": {"font": {"color": "#4da6ff"}},
        "xaxis": {
            "gridcolor": "rgba(160,190,230,0.18)",
            "linecolor": "rgba(160,190,230,0.30)",
            "zerolinecolor": "rgba(160,190,230,0.22)",
        },
        "yaxis": {
            "gridcolor": "rgba(160,190,230,0.18)",
            "linecolor": "rgba(160,190,230,0.30)",
            "zerolinecolor": "rgba(160,190,230,0.22)",
        },
    }
}

pio.templates["codex_dark"] = PLOTLY_DARK_TEMPLATE
pio.templates.default = "codex_dark"
px.defaults.template = "codex_dark"

# ── API Key Loading (secure) ─────────────────────────────────
def get_groq_api_key(required: bool = False) -> str | None:
    """Load GROQ API key from Streamlit Secrets (production)
    or .env file (local development). Never hardcode keys."""
    # 1. Try Streamlit Secrets (used on Streamlit Cloud)
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # 2. Fallback to .env file (local development)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if key:
            return key
    except ImportError:
        pass

    # 3. Last resort: environment variable
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key

    logger.error("Missing GROQ_API_KEY")
    if required:
        st.error("Missing GROQ_API_KEY! Set it in .streamlit/secrets.toml or .env file.")
        st.stop()
    return None


GROQ_API_KEY = get_groq_api_key(required=False)

# ── Page Styling ─────────────────────────────────────────────
def set_page_style() -> None:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1627 50%, #0a0e1a 100%);
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050810 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f;
    }
    .custom-card {
        background: linear-gradient(135deg, #0d1f3c, #0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0,120,255,0.1);
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1f3c, #0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0066cc, #0044aa);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
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
    .page-banner h1 { color: white !important; font-size: 2.5em; margin: 0; }
    .page-banner p { color: #aad4ff; font-size: 1.1em; margin: 8px 0 0 0; }
    .blue-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #0066cc, transparent);
        margin: 20px 0;
        border: none;
    }
    h1 { color: #4da6ff !important; }
    h2 { color: #3d9af5 !important; }
    h3 { color: #2d8ae8 !important; }
    </style>
    """, unsafe_allow_html=True)

def page_banner(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="page-banner">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def divider() -> None:
    st.markdown('<hr class="blue-divider">', unsafe_allow_html=True)

# ── Plotly Dark Theme Helper ─────────────────────────────────
def apply_dark_theme(fig: object) -> object:
    """Apply consistent dark theme to any Plotly figure."""
    fig.layout.template = None
    fig.update_layout(
        template=None,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0d1f3c",
        font=dict(color="white"),
        title_font=dict(color="#4da6ff"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(160,190,230,0.18)",
        zerolinecolor="rgba(160,190,230,0.22)",
        linecolor="rgba(160,190,230,0.30)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(160,190,230,0.18)",
        zerolinecolor="rgba(160,190,230,0.22)",
        linecolor="rgba(160,190,230,0.30)",
    )
    return fig


def render_plotly_chart(fig: object, **kwargs) -> None:
    """Render Plotly without Streamlit's theme overriding figure colors."""
    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, theme=None, **kwargs)

# ── DataFrame State Management ───────────────────────────────
MAX_UNDO_STACK = 5

def get_dataframe() -> pd.DataFrame | None:
    if "df" not in st.session_state:
        return None
    return st.session_state.df

def save_dataframe(df: pd.DataFrame) -> None:
    st.session_state.df = df

def push_undo() -> None:
    """Save a snapshot of the current dataframe before a destructive op."""
    if "df" not in st.session_state:
        return
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    st.session_state.undo_stack.append(copy.deepcopy(st.session_state.df))
    if len(st.session_state.undo_stack) > MAX_UNDO_STACK:
        st.session_state.undo_stack.pop(0)

def pop_undo() -> bool:
    """Restore the most recent dataframe snapshot. Returns True on success."""
    if "undo_stack" not in st.session_state or len(st.session_state.undo_stack) == 0:
        return False
    st.session_state.df = st.session_state.undo_stack.pop()
    return True

def get_undo_count() -> int:
    """Return how many undo snapshots are available."""
    if "undo_stack" not in st.session_state:
        return 0
    return len(st.session_state.undo_stack)


def apply_smart_missing_value_treatment(
    df: pd.DataFrame, roles: dict[str, str] | None = None
) -> tuple[pd.DataFrame, list[str], bool]:
    """Apply the automated missing-value strategy used across the app.

    Numeric columns are filled with their median, non-numeric columns with
    their mode, and identifier columns are skipped.
    """
    roles = roles or {}
    cleaned_df = df.copy()
    actions: list[str] = []
    changed = False

    for col in cleaned_df.columns:
        missing = int(cleaned_df[col].isnull().sum())
        if missing == 0:
            continue

        role = roles.get(col, "numeric")

        if role == "identifier":
            actions.append(f"⏭️ Column '{col}' skipped — detected as identifier")
            continue

        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            median_val = cleaned_df[col].median()
            if pd.isna(median_val):
                actions.append(f"⚠️ Column '{col}' is entirely empty — skipped")
                continue
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            actions.append(
                f"📊 Filled {missing} missing values in '{col}' with median ({median_val:.2f})"
            )
            changed = True
            continue

        mode_vals = cleaned_df[col].mode()
        if len(mode_vals) > 0:
            cleaned_df[col] = cleaned_df[col].fillna(mode_vals[0])
            actions.append(
                f"📝 Filled {missing} missing values in '{col}' with mode ({mode_vals[0]})"
            )
            changed = True
        else:
            actions.append(f"⚠️ Column '{col}' is entirely empty — skipped")

    return cleaned_df, actions, changed

# ── Chart Memory (session-based) ─────────────────────────────
def get_chart_memory() -> dict[str, int]:
    """Return the chart view frequency dict from session state."""
    return st.session_state.get('chart_memory', {})

def record_chart_view(x: str, y: str) -> None:
    """Increment view count for a column pair.
    Called every time the user selects a chart in the auto-chart section.
    """
    key = f"{x}|{y}"
    mem = get_chart_memory()
    mem[key] = mem.get(key, 0) + 1
    st.session_state['chart_memory'] = mem

# ── Query Sanitization ───────────────────────────────────────
_DANGEROUS_TOKENS = re.compile(
    r'(__[\w]+__|import|exec|eval|compile|open|globals|locals|getattr|setattr|'
    r'delattr|hasattr|vars|dir|type|super|classmethod|staticmethod|property|'
    r'lambda|os\.|sys\.|subprocess|shutil|pathlib|builtins|breakpoint)',
    re.IGNORECASE
)

# Matches function-call patterns like  func(  or  obj.method(
_FUNC_CALL = re.compile(r'[A-Za-z_]\w*\s*\(')

def sanitize_query(query_str: str) -> tuple[bool, str]:
    """Validate that an LLM-generated pandas query string is safe.
    Returns (is_safe: bool, reason: str)."""
    if not query_str or not query_str.strip():
        return False, "Empty query"

    query_str = query_str.strip()

    match = _DANGEROUS_TOKENS.search(query_str)
    if match:
        return False, f"Blocked dangerous token: '{match.group()}'"

    # Block actual function calls (identifier followed by "(") but allow
    # bare parentheses used for grouping conditions.
    if _FUNC_CALL.search(query_str):
        return False, "Function calls are not allowed in queries"

    if ';' in query_str:
        return False, "Multiple statements are not allowed"

    return True, "OK"


def clean_llm_query(raw: str) -> str:
    """Strip common LLM response noise from a pandas query string."""
    q = raw.strip()
    # Remove markdown code fences
    q = q.replace("```python", "").replace("```", "").strip()
    # Remove leading 'df.query(' wrapper if the LLM added it
    if q.lower().startswith("df.query("):
        q = q[len("df.query("):]
        if q.endswith(")"):
            q = q[:-1]
    # Strip surrounding quotes (single or double)
    if (q.startswith('"') and q.endswith('"')) or \
       (q.startswith("'") and q.endswith("'")):
        q = q[1:-1]
    return q.strip()

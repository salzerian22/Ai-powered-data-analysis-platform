from __future__ import annotations

"""
Chart Summary Module
────────────────────
Generates AI-powered 5-6 word headlines and detailed insights
for every chart. Uses computed statistics — never raw data.
"""

import pandas as pd
import streamlit as st


def generate_chart_summary(
    df: pd.DataFrame,
    selected_cols: list[str],
    chart_type: str,
    groq_client: object,
) -> tuple[str, str]:
    """Generate a headline + detail insight for a chart.

    Step 1: Compute actual statistics from data.
    Step 2: Pass computed values to LLM — never raw data.

    Args:
        df: pandas DataFrame
        selected_cols: list of column names used in chart
        chart_type: string describing the chart type
        groq_client: initialized Groq client

    Returns:
        tuple: (headline: str, detail: str)
    """
    # ── Step 1: Compute statistics ────────────────────────────
    stats_context = {}
    for col in selected_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_context[col] = {
                'mean': round(df[col].mean(), 2),
                'median': round(df[col].median(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2),
                'std': round(df[col].std(), 2)
            }
        else:
            top = df[col].value_counts().head(3).to_dict()
            stats_context[col] = {
                'top_values': top,
                'unique_count': df[col].nunique()
            }

    # ── Step 2: Generate insight via LLM ──────────────────────
    prompt = f"""
    Chart type: {chart_type}
    Columns: {selected_cols}
    Computed statistics: {stats_context}

    Task 1: Write a 5-6 word headline summarizing the KEY finding.
    Task 2: Write 2-3 sentences explaining: what the chart shows,
            what is notable, and what question this raises.
    Format: HEADLINE|||DETAIL
    """

    try:
        response = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        parts = content.split('|||')
        headline = parts[0].strip()
        detail = parts[1].strip() if len(parts) > 1 else ''
        return headline, detail
    except Exception as e:
        return '📊 Chart generated successfully', f'Could not generate AI summary: {e}'


def show_chart_summary(
    df: pd.DataFrame,
    selected_cols: list[str],
    chart_type: str,
    groq_client: object,
) -> None:
    """Display the chart summary below a chart in Streamlit.

    Renders a caption headline + expandable detail section.
    Handles errors gracefully — never crashes the page.
    """
    try:
        headline, detail = generate_chart_summary(
            df, selected_cols, chart_type, groq_client
        )
        st.caption(f"💡 {headline}")
        if detail:
            with st.expander("See more"):
                st.write(detail)
    except Exception:
        pass  # never let summary generation crash the chart page

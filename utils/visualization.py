from __future__ import annotations

"""
Visualization Module
────────────────────
Safe chart renderer and smart chart recommender.
Uses column roles from column_classifier to recommend the best chart type.
All Plotly rendering is wrapped in try/except — never crashes the page.
"""

import pandas as pd
import plotly.express as px


# ── Available chart types ─────────────────────────────────────
CHART_TYPES = [
    'Histogram',
    'Bar chart',
    'Line chart',
    'Scatter plot',
    'Box plot',
    'Violin plot',
    'Pie chart'
]


def chart_title(selected: list[str], chart_type: str) -> str:
    """Build a stable fallback title for charts."""
    if not selected:
        return chart_type
    if len(selected) == 1:
        return f"Distribution of {selected[0]}" if chart_type == 'Histogram' else selected[0]
    if chart_type == 'Bar chart':
        return f"{selected[1]} by {selected[0]}"
    return f"{selected[1]} over {selected[0]}"


def is_year_axis(column_name: str) -> bool:
    return "year" in column_name.lower()


def normalize_selected_axes(selected: list[str]) -> list[str]:
    """Force year-like columns onto the x-axis when possible."""
    if len(selected) >= 2:
        if is_year_axis(selected[1]) and not is_year_axis(selected[0]):
            reordered = selected.copy()
            reordered[0], reordered[1] = reordered[1], reordered[0]
            return reordered
    return selected


def prepare_x_axis(series: pd.Series) -> pd.Series:
    """Keep year-like integer columns categorical so Plotly doesn't treat them as epoch timestamps."""
    non_null = series.dropna()
    if non_null.empty:
        return series

    if pd.api.types.is_integer_dtype(series) and non_null.between(1900, 2100).all():
        return series.astype("Int64").astype(str)

    return series


def recommend_chart(selected_cols: list[str], column_roles: dict[str, str]) -> str:
    """Recommend the best chart type based on column roles.

    Args:
        selected_cols: list of column names the user selected
        column_roles: dict from get_column_roles() {col_name: role}

    Returns:
        str: one of CHART_TYPES
    """
    if not selected_cols:
        return 'Bar chart'

    if len(selected_cols) == 1:
        role = column_roles.get(selected_cols[0], 'numeric')
        if role in ('numeric', 'ordinal'):
            return 'Histogram'
        if role == 'categorical':
            return 'Bar chart'
        if role == 'datetime':
            return 'Line chart'
        if role == 'boolean':
            return 'Pie chart'
        return 'Bar chart'

    elif len(selected_cols) == 2:
        r0 = column_roles.get(selected_cols[0], 'numeric')
        r1 = column_roles.get(selected_cols[1], 'numeric')

        if is_year_axis(selected_cols[0]) and r1 in ('numeric', 'ordinal'):
            return 'Line chart'
        if is_year_axis(selected_cols[1]) and r0 in ('numeric', 'ordinal'):
            return 'Line chart'
        if r0 == 'categorical' and r1 in ('numeric', 'ordinal'):
            return 'Bar chart'
        if r0 in ('numeric', 'ordinal') and r1 == 'categorical':
            return 'Bar chart'
        if r0 in ('numeric', 'ordinal') and r1 in ('numeric', 'ordinal'):
            return 'Scatter plot'
        if r0 == 'datetime' and r1 in ('numeric', 'ordinal'):
            return 'Line chart'
        if r1 == 'datetime' and r0 in ('numeric', 'ordinal'):
            return 'Line chart'
        if r0 == 'categorical' and r1 == 'categorical':
            return 'Bar chart'
        return 'Bar chart'

    else:
        # 3+ columns — scatter with color dimension
        return 'Scatter plot'


def render_chart(
    df: pd.DataFrame,
    selected: list[str],
    chart_type: str,
    column_roles: dict[str, str] | None = None,
) -> object | None:
    """Safely render a Plotly chart. Returns fig or None on error.

    Args:
        df: pandas DataFrame
        selected: list of selected column names
        chart_type: one of CHART_TYPES

    Returns:
        plotly Figure or None if rendering fails
    """
    if not selected:
        return None

    selected = normalize_selected_axes(selected)

    try:
        if chart_type == 'Histogram':
            return px.histogram(df, x=selected[0],
                                title=chart_title(selected, chart_type),
                                color_discrete_sequence=['#4da6ff'])

        elif chart_type == 'Bar chart':
            if len(selected) >= 2:
                # Use column_roles if available, otherwise fall back to dtype
                is_cat = (column_roles.get(selected[0]) == 'categorical') if column_roles else (df[selected[0]].dtype == 'object')
                if is_cat:
                    grouped = df.groupby(selected[0])[selected[1]].mean().reset_index()
                    grouped = grouped.sort_values(by=selected[1], ascending=False)
                    return px.bar(grouped, x=selected[0], y=selected[1],
                                  title=chart_title(selected, chart_type),
                                  color=selected[1],
                                  color_continuous_scale=['#003399', '#0066cc', '#00aaff'])
                else:
                    grouped = df.groupby(selected[1])[selected[0]].mean().reset_index()
                    grouped = grouped.sort_values(by=selected[0], ascending=False)
                    return px.bar(grouped, x=selected[1], y=selected[0],
                                  title=f"{selected[0]} by {selected[1]}",
                                  color=selected[0],
                                  color_continuous_scale=['#003399', '#0066cc', '#00aaff'])
            # Single column — value counts
            counts = df[selected[0]].value_counts().reset_index()
            counts.columns = [selected[0], 'count']
            return px.bar(counts, x=selected[0], y='count',
                          title=chart_title(selected, chart_type),
                          color='count',
                          color_continuous_scale=['#003399', '#0066cc', '#00aaff'])

        elif chart_type == 'Scatter plot':
            if len(selected) < 2:
                return None
            return px.scatter(df, x=selected[0], y=selected[1],
                              title=chart_title(selected, chart_type),
                              color=selected[2] if len(selected) > 2 else None,
                              opacity=0.6,
                              color_discrete_sequence=['#4da6ff'])

        elif chart_type == 'Line chart':
            if len(selected) < 2:
                return None
            grouped = df.groupby(selected[0])[selected[1]].mean().reset_index()
            grouped = grouped.sort_values(by=selected[0])
            grouped = grouped.copy()
            grouped[selected[0]] = prepare_x_axis(grouped[selected[0]])
            return px.line(grouped, x=selected[0], y=selected[1],
                           title=chart_title(selected, chart_type),
                           markers=True,
                           color_discrete_sequence=['#4da6ff'])

        elif chart_type == 'Box plot':
            return px.box(df, y=selected[0],
                          x=selected[1] if len(selected) > 1 else None,
                          title=chart_title(selected, chart_type),
                          color_discrete_sequence=['#4da6ff'])

        elif chart_type == 'Violin plot':
            return px.violin(
                df,
                y=selected[0],
                x=selected[1] if len(selected) > 1 else None,
                title=chart_title(selected, chart_type),
                box=True,
                color_discrete_sequence=['#4da6ff']
            )

        elif chart_type == 'Pie chart':
            if len(selected) >= 2:
                return px.pie(df, names=selected[0], values=selected[1], title=chart_title(selected, chart_type))
            counts = df[selected[0]].value_counts().reset_index()
            counts.columns = [selected[0], 'count']
            return px.pie(counts, names=selected[0], values='count', title=chart_title(selected, chart_type))

    except Exception:
        return None

    return None

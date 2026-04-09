from __future__ import annotations

"""
Heatmap Module
──────────────
Significance-gated correlation heatmap.
Only shows a heatmap when statistically significant correlations exist.
Uses Pearson r + p-value to avoid misleading visualizations.
"""

import pandas as pd
from scipy import stats


def should_show_heatmap(
    df: pd.DataFrame,
    num_cols: list[str],
    min_rows: int = 20,
    alpha: float = 0.05,
) -> tuple[bool, str, list[tuple[str, str, float, float]]]:
    """Determine if a correlation heatmap is statistically meaningful.

    Args:
        df: pandas DataFrame
        num_cols: list of numeric column names
        min_rows: minimum rows required for reliable correlation (default 20)
        alpha: significance level for p-value test (default 0.05)

    Returns:
        tuple: (show: bool, reason: str, significant_pairs: list)
            - show: whether to display the heatmap
            - reason: human-readable explanation
            - significant_pairs: list of (col_a, col_b, r, p_value) tuples
    """
    if len(num_cols) < 2:
        return False, 'Need at least 2 numeric columns for correlations.', []

    if len(df) < min_rows:
        return False, f'Only {len(df)} rows — need {min_rows}+ for reliable correlations.', []

    significant_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col_a = df[num_cols[i]].dropna()
            col_b = df[num_cols[j]].dropna()
            common = col_a.index.intersection(col_b.index)

            if len(common) < min_rows:
                continue

            r, p = stats.pearsonr(col_a[common], col_b[common])

            # Only include if statistically significant AND meaningful strength
            if p < alpha and abs(r) > 0.2:
                significant_pairs.append((
                    num_cols[i],
                    num_cols[j],
                    round(r, 2),
                    round(p, 4)
                ))

    if not significant_pairs:
        return False, 'No statistically significant correlations found (p < 0.05, |r| > 0.2).', []

    return True, f'{len(significant_pairs)} significant pair(s) found.', significant_pairs


def describe_correlation(r: float) -> tuple[str, str]:
    """Return human-readable description of a correlation coefficient.

    Args:
        r: Pearson correlation coefficient

    Returns:
        tuple: (strength: str, direction: str)
    """
    direction = 'positive' if r > 0 else 'negative'
    abs_r = abs(r)
    if abs_r > 0.7:
        strength = 'strong'
    elif abs_r > 0.4:
        strength = 'moderate'
    else:
        strength = 'weak'
    return strength, direction

from __future__ import annotations

"""
Column Role Classifier
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Attaches semantic metadata to columns WITHOUT renaming them.
Used by both Manual and Automated modes to drive smart defaults
for cleaning, visualization, and ML feature selection.

Roles: identifier | datetime | boolean | numeric | ordinal | categorical | text
"""

import pandas as pd

def classify_column(series: pd.Series, col_name: str) -> str:
    """Return semantic role for a single column.

    Roles:
        identifier  â€” ID, key, code, ref columns (excluded from ML)
        datetime    â€” date/time columns
        boolean     â€” true/false or 0/1 columns
        numeric     â€” continuous numeric (>20 unique values)
        ordinal     â€” numeric with few unique values (â‰¤20)
        categorical â€” string/object with â‰¤30 unique values
        text        â€” free-form text with >30 unique values
    """
    name = col_name.lower().strip()

    # â”€â”€ 1. Identifier detection (before numeric check) â”€â”€â”€â”€â”€â”€â”€â”€
    id_patterns = ['id', '_id', 'code', 'key', 'uuid', 'ref', 'no', 'number']
    if any(name == p or name.endswith(p) for p in id_patterns):
        return 'identifier'

    # â”€â”€ 2. Datetime detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    date_hints = ['date', 'time', 'year', 'month', 'day', 'timestamp']
    if any(hint in name for hint in date_hints):
        try:
            pd.to_datetime(series.dropna().head(20))
            return 'datetime'
        except Exception:
            pass

    # â”€â”€ 3. Boolean detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pd.api.types.is_bool_dtype(series):
        return 'boolean'
    unique_vals = set(series.dropna().unique())
    if unique_vals and unique_vals <= {0, 1, True, False, 'yes', 'no', 'Yes', 'No', 'YES', 'NO'}:
        return 'boolean'

    # â”€â”€ 4. Numeric vs ordinal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric' if series.nunique() > 20 else 'ordinal'

    # â”€â”€ 5. Categorical vs free text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    return 'categorical' if series.nunique() <= 30 else 'text'


def get_column_roles(df: pd.DataFrame) -> dict[str, str]:
    """Classify all columns in a DataFrame.

    Returns:
        dict: {column_name: role_string}

    Example:
        >>> roles = get_column_roles(df)
        >>> roles
        {'CustomerID': 'identifier', 'Age': 'numeric', 'City': 'categorical'}
    """
    return {col: classify_column(df[col], col) for col in df.columns}


def get_columns_by_role(roles: dict[str, str], *target_roles: str) -> list[str]:
    """Filter column names by one or more roles.

    Example:
        >>> numeric_cols = get_columns_by_role(roles, 'numeric', 'ordinal')
    """
    return [col for col, role in roles.items() if role in target_roles]

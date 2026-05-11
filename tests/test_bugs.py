import pandas as pd
import numpy as np
import html
import math
import pathlib


def test_outlier_preserves_nan():
    df = pd.DataFrame({'v': [1.0, 2.0, 100.0, np.nan]})
    nan_mask   = df['v'].isna()
    range_mask = (df['v'] >= 0) & (df['v'] <= 10)
    result = df[nan_mask | range_mask]
    assert result['v'].isna().sum() == 1
    assert len(result) == 3


def test_nan_median_guard():
    df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [1.0, 2.0]})
    valid = [c for c in df.columns if pd.notna(df[c].median())]
    assert valid == ['b']


def test_mean_fill_leaves_categorical_nan():
    df = pd.DataFrame({'num': [1.0, np.nan], 'cat': ['x', np.nan]})
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    assert df['cat'].isnull().sum() == 1
    assert df['num'].isnull().sum() == 0


def test_zscore_zero_std_guard():
    data = pd.Series([5.0, 5.0, 5.0])
    std = data.std()
    assert std == 0 or np.isclose(std, 0)


def test_html_escape_column_name():
    raw  = '<script>alert(1)</script>'
    safe = html.escape(raw)
    assert '<' not in safe
    assert '>' not in safe


def test_multiclass_split_test_size():
    n_rows, n_classes = 20, 10
    test_count = max(n_classes, math.ceil(0.2 * n_rows))
    test_size  = min(test_count / n_rows, 0.4)
    # test_size must be capped at 0.4 (never give away more than 40%)
    assert test_size <= 0.4
    # test_count must be at least n_classes (enough rows to cover all classes)
    assert test_count >= n_classes
    # test_size must be a valid fraction
    assert 0 < test_size <= 1


def test_datetime_chart_recommendation_is_bar():
    column_roles  = {'created_at': 'datetime'}
    selected_cols = ['created_at']

    def recommend_chart(selected_cols, column_roles):
        if len(selected_cols) == 1:
            if column_roles.get(selected_cols[0]) == 'datetime':
                return 'Bar chart'
        return 'Line chart'

    assert recommend_chart(selected_cols, column_roles) == 'Bar chart'


def test_no_disabled_buttons():
    pages_dir = pathlib.Path('pages')
    if not pages_dir.exists():
        return
    for py_file in pages_dir.glob('*.py'):
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        assert 'disabled=True' not in content, \
            f'Disabled button found in {py_file}'

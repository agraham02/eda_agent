from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.dataset_cache import get_dataset_cached as get_dataset
from ..utils.errors import (
    COLUMN_NOT_FOUND,
    DATASET_NOT_FOUND,
    TYPE_MISMATCH,
    exception_to_error,
    wrap_success,
)
from ..utils.parsing import parse_columns_csv
from ..utils.schemas import (
    BivariateSummaryResult,
    CorrelationMatrixResult,
    UnivariateSummaryItem,
    UnivariateSummaryResult,
)


# -----------------------------
# UNIVARIATE SUMMARY
# -----------------------------
def build_univariate_summary(
    dataset_id: str,
    columns: Optional[List[str]] = None,
) -> UnivariateSummaryResult:
    """Build univariate summaries returning a schema model."""
    df = get_dataset(dataset_id)

    if columns is None:
        columns = list(df.columns)

    items: List[UnivariateSummaryItem] = []

    for col in columns:
        series = df[col]
        name = col
        dtype_str = str(series.dtype)
        n_total = int(series.shape[0])
        n_missing = int(series.isna().sum())
        missing_pct = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.empty:
                items.append(
                    UnivariateSummaryItem(
                        name=name,
                        dtype=dtype_str,
                        n=n_total,
                        n_missing=n_missing,
                        missing_pct=missing_pct,
                        type="numeric",
                        mean=None,
                        median=None,
                        mode=[],
                        std=None,
                        iqr=None,
                        q1=None,
                        q3=None,
                        min=None,
                        max=None,
                        n_outliers=None,
                    )
                )
                continue
            mean = clean.mean()
            median = clean.median()
            mode_vals = clean.mode().tolist()

            # Spread
            std = clean.std()
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            min_val = clean.min()
            max_val = clean.max()

            # Outlier detection (1.5*IQR rule) — matches your notes
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((clean < lower) | (clean > upper)).sum())
            items.append(
                UnivariateSummaryItem(
                    name=name,
                    dtype=dtype_str,
                    n=n_total,
                    n_missing=n_missing,
                    missing_pct=missing_pct,
                    type="numeric",
                    mean=float(mean) if pd.notna(mean) else None,
                    median=float(median) if pd.notna(median) else None,
                    mode=mode_vals,
                    std=float(std) if pd.notna(std) else None,
                    iqr=float(iqr),
                    q1=float(q1),
                    q3=float(q3),
                    min=float(min_val),
                    max=float(max_val),
                    n_outliers=n_outliers,
                )
            )

        # Categorical variables
        else:
            counts = series.value_counts(dropna=False)
            proportions = (counts / len(series)).to_dict()
            items.append(
                UnivariateSummaryItem(
                    name=name,
                    dtype=dtype_str,
                    n=n_total,
                    n_missing=n_missing,
                    missing_pct=missing_pct,
                    type="categorical",
                    mode=series.mode().astype(str).tolist(),
                    unique_values=list(counts.index.astype(str)),
                    counts={str(k): int(v) for k, v in counts.to_dict().items()},
                    proportions={str(k): float(v) for k, v in proportions.items()},
                )
            )

    return UnivariateSummaryResult(dataset_id=dataset_id, summaries=items)


# -----------------------------
# BIVARIATE SUMMARY
# -----------------------------
def build_bivariate_summary(
    dataset_id: str,
    x: str,
    y: str,
    group_by: Optional[str] = None,
) -> BivariateSummaryResult:
    df = get_dataset(dataset_id)
    if x not in df.columns or y not in df.columns:
        raise ValueError("One or both columns not found in dataset")

    X = df[x]
    Y = df[y]

    # Determine variable types
    x_numeric = pd.api.types.is_numeric_dtype(X)
    y_numeric = pd.api.types.is_numeric_dtype(Y)

    # -------------------------
    # numeric - numeric
    # -------------------------
    if x_numeric and y_numeric:
        clean = df[[x, y]].dropna()
        corr = clean[x].corr(clean[y])
        cov = clean[x].cov(clean[y])
        return BivariateSummaryResult(
            dataset_id=dataset_id,
            type="numeric-numeric",
            payload={
                "x": x,
                "y": y,
                "n_complete": len(clean),
                "correlation": float(corr),
                "covariance": float(cov),
            },
        )

    # -------------------------
    # numeric - categorical
    # -------------------------
    if x_numeric and not y_numeric:
        group_stats = df.groupby(y)[x].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )
        return BivariateSummaryResult(
            dataset_id=dataset_id,
            type="numeric-categorical",
            payload={
                "numeric": x,
                "categorical": y,
                "group_summary": group_stats.reset_index().to_dict(orient="records"),
            },
        )

    if not x_numeric and y_numeric:
        group_stats = df.groupby(x)[y].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )
        return BivariateSummaryResult(
            dataset_id=dataset_id,
            type="numeric-categorical",
            payload={
                "numeric": y,
                "categorical": x,
                "group_summary": group_stats.reset_index().to_dict(orient="records"),
            },
        )

    # -------------------------
    # categorical - categorical
    # -------------------------
    contingency = pd.crosstab(df[x], df[y], dropna=False)
    total = contingency.values.sum()
    proportions = contingency / total

    # expected counts under independence
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / total
    expected_df = pd.DataFrame(
        expected, index=contingency.index, columns=contingency.columns
    )

    return BivariateSummaryResult(
        dataset_id=dataset_id,
        type="categorical-categorical",
        payload={
            "x": x,
            "y": y,
            "contingency_counts": contingency.reset_index().to_dict(orient="records"),
            "proportions": proportions.reset_index().to_dict(orient="records"),
            "expected_counts": expected_df.reset_index().to_dict(orient="records"),
        },
    )


# -----------------------------
# CORRELATION MATRIX
# -----------------------------
def build_correlation_matrix(
    dataset_id: str,
    columns: Optional[List[str]] = None,
) -> CorrelationMatrixResult:
    df = get_dataset(dataset_id)
    if columns is not None:
        df = df[columns]
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    corr_dict: Dict[str, Dict[str, float]] = {
        str(row_key): {str(col_key): float(val) for col_key, val in row_dict.items()}
        for row_key, row_dict in corr.to_dict().items()
    }
    return CorrelationMatrixResult(
        dataset_id=dataset_id,
        columns=list(numeric_df.columns),
        correlation_matrix=corr_dict,
    )


# -----------------------------
# TOOL WRAPPERS
# -----------------------------
def eda_univariate_summary_tool(
    dataset_id: str, columns_csv: str = ""
) -> Dict[str, Any]:
    """Tool wrapper for univariate summary.

    columns_csv: Comma-separated column names. Leave empty to summarize all columns.
    """
    try:
        columns = parse_columns_csv(columns_csv)
        result = build_univariate_summary(dataset_id, columns)
        return wrap_success(result.model_dump())
    except KeyError as e:
        return exception_to_error(
            DATASET_NOT_FOUND,
            e,
            hint="Ingest dataset with ingest_csv_tool first",
        )
    except (ValueError, TypeError) as e:
        return exception_to_error(
            COLUMN_NOT_FOUND,
            e,
            hint="Check that column names exist in dataset",
        )


def eda_bivariate_summary_tool(
    dataset_id: str,
    x: str,
    y: str,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        result = build_bivariate_summary(dataset_id, x, y, group_by)
        return wrap_success(result.model_dump())
    except KeyError as e:
        return exception_to_error(
            DATASET_NOT_FOUND,
            e,
            hint="Ingest dataset with ingest_csv_tool first",
        )
    except ValueError as e:
        return exception_to_error(
            COLUMN_NOT_FOUND,
            e,
            hint="Check that column names exist in dataset",
        )
    except TypeError as e:
        return exception_to_error(
            TYPE_MISMATCH,
            e,
            hint="Verify column data types are appropriate for bivariate analysis",
        )


def eda_correlation_matrix_tool(
    dataset_id: str, columns_csv: str = ""
) -> Dict[str, Any]:
    """Tool wrapper for correlation matrix.

    columns_csv: Comma-separated column names. Leave empty to use all columns.
    """
    try:
        columns = parse_columns_csv(columns_csv)
        result = build_correlation_matrix(dataset_id, columns)
        return wrap_success(result.model_dump())
    except KeyError as e:
        return exception_to_error(
            DATASET_NOT_FOUND,
            e,
            hint="Ingest dataset with ingest_csv_tool first",
        )
    except (ValueError, TypeError) as e:
        return exception_to_error(
            COLUMN_NOT_FOUND,
            e,
            hint="Check that column names exist in dataset. Only numeric columns can be correlated.",
        )

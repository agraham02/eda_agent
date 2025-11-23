from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..tools.ingestion_tools import infer_semantic_type  # reuse your logic
from ..utils.data_store import get_dataset


def _numeric_summary(series: pd.Series) -> Optional[Dict[str, Any]]:
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return None

    desc = series.describe()
    q1 = desc["25%"]
    q3 = desc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()

    # Optional: truncate to avoid huge JSON
    outliers_preview = outliers[:20]  # first 20 only

    return {
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not np.isnan(desc["std"]) else None,
        "min": float(desc["min"]),
        "q1": float(q1),
        "median": float(desc["50%"]),
        "q3": float(q3),
        "max": float(desc["max"]),
        "iqr": float(iqr),
        "outlier_count": len(outliers),
        "outliers": outliers_preview,
        "outliers_truncated": len(outliers) > 20,
    }


def data_quality_tool(dataset_id: str) -> Dict[str, Any]:
    """
    Run basic data quality checks on a dataset that has already been
    ingested and stored in the in-memory data store.
    """
    df = get_dataset(dataset_id)

    n_rows, n_cols = df.shape
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = float(duplicate_count / max(1, n_rows))

    columns: List[Dict[str, Any]] = []
    dataset_issues: List[str] = []

    if duplicate_count > 0:
        dataset_issues.append(
            f"Dataset has {duplicate_count} duplicate rows "
            f"({duplicate_pct:.1%} of all rows)."
        )

    for col in df.columns:
        series = df[col]
        pandas_dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        missing_pct = float(n_missing / max(1, n_rows))
        n_unique = int(series.nunique(dropna=True))

        semantic_type = infer_semantic_type(pandas_dtype, n_unique, n_rows)

        is_constant = n_unique <= 1
        is_all_unique = n_unique == (n_rows - n_missing)

        col_issues: List[str] = []

        if missing_pct > 0.3:
            col_issues.append(
                f"High missingness: {missing_pct:.1%} of values are missing."
            )
        elif 0 < missing_pct <= 0.3:
            col_issues.append(
                f"Some missing values: {missing_pct:.1%} of values are missing."
            )

        if is_constant:
            col_issues.append("Column is constant (only one unique non-null value).")

        if semantic_type in {"numeric", "numeric_categorical"}:
            numeric_stats = _numeric_summary(series.dropna())
        else:
            numeric_stats = None

        columns.append(
            {
                "name": col,
                "pandas_dtype": pandas_dtype,
                "semantic_type": semantic_type,
                "n_missing": n_missing,
                "missing_pct": missing_pct,
                "n_unique": n_unique,
                "is_constant": is_constant,
                "is_all_unique": is_all_unique,
                "issues": col_issues,
                "numeric_summary": numeric_stats,
            }
        )

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "duplicate_rows": {
            "count": duplicate_count,
            "pct": duplicate_pct,
        },
        "columns": columns,
        "dataset_issues": dataset_issues,
    }

    return result

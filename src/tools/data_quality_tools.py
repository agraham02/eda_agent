from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..tools.ingestion_tools import infer_semantic_type  # reuse logic
from ..utils.data_store import get_dataset
from ..utils.errors import DATASET_NOT_FOUND, exception_to_error, wrap_success
from ..utils.schemas import (
    DataQualityColumn,
    DataQualityResult,
    NumericSummary,
    SemanticType,
)


def compute_readiness_score(
    n_rows: int,
    duplicate_pct: float,
    columns: List[DataQualityColumn],
) -> Dict[str, Any]:
    """Compute a simple readiness score (0-100) with component breakdown.

    Heuristic components (each starts at 100 then penalized):
      - missingness: average column missing pct weighted by columns
      - duplicates: penalty proportional to duplicate pct
      - constants: proportion of constant columns
      - high_missing_columns: proportion of columns with >40% missing
      - outliers: aggregate outlier density across numeric columns

    Overall score: mean of component scores (clamped 0-100).
    """
    try:
        if n_rows <= 0:
            return {
                "overall": 0,
                "components": {},
                "notes": ["Empty dataset"],
            }

        total_cols = len(columns) or 1
        missing_avgs: List[float] = []
        constant_flags = 0
        high_missing_flags = 0
        outlier_counts = 0
        numeric_value_counts = 0

        for col in columns:
            missing_avgs.append(col.missing_pct)
            if col.is_constant:
                constant_flags += 1
            if col.missing_pct > 0.4:
                high_missing_flags += 1
            if col.numeric_summary is not None:
                outlier_counts += col.numeric_summary.outlier_count
                numeric_value_counts += (
                    col.numeric_summary.outlier_count + 1
                )  # avoid zero division later

        avg_missing = sum(missing_avgs) / len(missing_avgs) if missing_avgs else 0.0
        missing_score = max(
            0, 100 - (avg_missing * 100 * 1.2)
        )  # 20% extra penalty multiplier

        duplicate_score = max(0, 100 - (duplicate_pct * 100 * 1.5))  # heavier penalty

        constant_ratio = constant_flags / total_cols
        constant_score = max(0, 100 - (constant_ratio * 100 * 2.0))

        high_missing_ratio = high_missing_flags / total_cols
        high_missing_score = max(0, 100 - (high_missing_ratio * 100 * 2.5))

        if numeric_value_counts > 0:
            outlier_density = outlier_counts / numeric_value_counts
        else:
            outlier_density = 0.0
        # treat higher density as worse; mild penalty
        outlier_score = max(0, 100 - (outlier_density * 100 * 0.8))

        components = {
            "missingness": round(missing_score, 2),
            "duplicates": round(duplicate_score, 2),
            "constants": round(constant_score, 2),
            "high_missing_columns": round(high_missing_score, 2),
            "outliers": round(outlier_score, 2),
        }
        overall = max(0, min(100, round(sum(components.values()) / len(components), 2)))

        notes: List[str] = []
        if avg_missing > 0.3:
            notes.append("High average missingness; consider aggressive cleaning.")
        if duplicate_pct > 0.05:
            notes.append("Notable duplicate rows present.")
        if constant_ratio > 0.1:
            notes.append("Several constant columns provide no variance.")
        if high_missing_ratio > 0.2:
            notes.append("Multiple columns with >40% missing values.")
        if outlier_density > 0.15:
            notes.append("High outlier density in numeric columns.")

        readiness = {
            "overall": overall,
            "components": components,
            "notes": notes,
        }
        return readiness
    except (ValueError, TypeError, ZeroDivisionError) as e:
        # Fallback if scoring calculation fails
        return {
            "overall": 0,
            "components": {},
            "notes": [f"Error computing readiness score: {str(e)}"],
        }


def _numeric_summary(
    series: pd.Series, outlier_method: str = "both"
) -> Optional[NumericSummary]:
    try:
        if not pd.api.types.is_numeric_dtype(series.dtype):
            return None

        desc = series.describe()
        q1 = desc["25%"]
        q3 = desc["75%"]
        iqr = q3 - q1
        mean = desc["mean"]
        std = desc["std"]

        # IQR-based outlier detection
        iqr_outliers_mask = pd.Series([False] * len(series), index=series.index)
        if outlier_method in ["iqr", "both"]:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers_mask = (series < lower_bound) | (series > upper_bound)

        # Z-score based outlier detection (|z| > 3)
        zscore_outliers_mask = pd.Series([False] * len(series), index=series.index)
        if outlier_method in ["zscore", "both"]:
            if not np.isnan(std) and std > 0:
                z_scores = np.abs((series - mean) / std)
                zscore_outliers_mask = z_scores > 3

        # Combine outliers based on method
        if outlier_method == "both":
            outliers_mask = iqr_outliers_mask | zscore_outliers_mask
        elif outlier_method == "zscore":
            outliers_mask = zscore_outliers_mask
        else:  # "iqr"
            outliers_mask = iqr_outliers_mask

        outliers = series[outliers_mask].tolist()

        # Optional: truncate to avoid huge JSON
        outliers_preview = outliers[:20]  # first 20 only

        return NumericSummary(
            mean=float(mean),
            std=float(std) if not np.isnan(std) else None,
            min=float(desc["min"]),
            q1=float(q1),
            median=float(desc["50%"]),
            q3=float(q3),
            max=float(desc["max"]),
            iqr=float(iqr),
            outlier_count=len(outliers),
            outliers=outliers_preview,
            outliers_truncated=len(outliers) > 20,
            outlier_method=outlier_method,
        )
    except (ValueError, TypeError, KeyError) as e:
        # If numeric summary fails, return None
        return None


def data_quality_tool(dataset_id: str, outlier_method: str = "both") -> Dict[str, Any]:
    """
    Run basic data quality checks on a dataset that has already been
    ingested and stored in the in-memory data store.

    Args:
        dataset_id: The ID of the dataset to analyze
        outlier_method: Method for outlier detection - "iqr", "zscore", or "both" (default: "both")
    """
    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return exception_to_error(
            DATASET_NOT_FOUND,
            e,
            hint="Ingest dataset with ingest_csv_tool before quality analysis",
        )

    n_rows, n_cols = df.shape
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = float(duplicate_count / max(1, n_rows))

    column_models: List[DataQualityColumn] = []
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
            numeric_stats = _numeric_summary(
                series.dropna(), outlier_method=outlier_method
            )
        else:
            numeric_stats = None

        column_models.append(
            DataQualityColumn(
                name=col,
                pandas_dtype=pandas_dtype,
                semantic_type=SemanticType(semantic_type) if semantic_type in SemanticType.__members__.values() else SemanticType.UNKNOWN,  # type: ignore
                n_missing=n_missing,
                missing_pct=missing_pct,
                n_unique=n_unique,
                is_constant=is_constant,
                is_all_unique=is_all_unique,
                issues=col_issues,
                numeric_summary=numeric_stats,
            )
        )

    readiness_score = compute_readiness_score(
        n_rows=n_rows, duplicate_pct=duplicate_pct, columns=column_models
    )

    result_model = DataQualityResult(
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_columns=n_cols,
        duplicate_rows={"count": duplicate_count, "pct": duplicate_pct},
        columns=column_models,
        dataset_issues=dataset_issues,
        readiness_score=readiness_score,
    )
    return wrap_success(result_model.model_dump())

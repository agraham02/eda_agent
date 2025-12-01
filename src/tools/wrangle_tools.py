# wrangle_tools.py

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.data_store import get_dataset, register_dataset
from ..utils.errors import (
    COLUMN_NOT_FOUND,
    DATASET_NOT_FOUND,
    EXPRESSION_ERROR,
    INVALID_PARAMETER,
    exception_to_error,
    make_error,
    wrap_success,
)
from ..utils.schemas import FilterResult, MutateResult, SelectResult


def apply_row_filter(
    dataset_id: str,
    condition: str,
) -> Dict[str, Any]:
    """
    Internal helper to filter rows based on a condition expression.

    The condition should be a pandas query string, for example:
      "age > 30 and country == 'US'"
      "`Life expectancy` > 70"

    Returns metadata and a new dataset_id for the filtered frame.
    """
    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Ingest dataset before filtering",
            context={"dataset_id": dataset_id},
        )

    # Normalize condition: convert df['column'] or df["column"] to `column`
    # This handles cases where the agent generates Python-style indexing
    normalized_condition = condition

    # Pattern: df['column_name'] or df["column_name"]
    pattern = r"df\[(['\"])([^'\"]+)\1\]"

    def replace_with_backticks(match):
        col_name = match.group(2)
        # Use backticks for query() syntax
        return f"`{col_name}`"

    normalized_condition = re.sub(pattern, replace_with_backticks, normalized_condition)

    try:
        # Use pandas query syntax for safety and familiarity.
        filtered = df.query(normalized_condition)
    except Exception as e:
        raise ValueError(f"Invalid filter condition '{condition}': {e}")

    new_dataset_id = register_dataset(
        filtered,
        filename="filtered",
        parent_dataset_id=dataset_id,
        transformation_note=f"filter: {condition[:100]}",
    )

    result = FilterResult(
        original_dataset_id=dataset_id,
        new_dataset_id=new_dataset_id,
        condition=condition,
        n_rows_before=int(len(df)),
        n_rows_after=int(len(filtered)),
        n_columns=int(df.shape[1]),
        n_rows=int(len(filtered)),
    )
    return wrap_success(result.model_dump())


def select_columns(
    dataset_id: str,
    columns: List[str],
) -> Dict[str, Any]:
    """
    Internal helper to select a subset of columns.

    Returns metadata and a new dataset_id for the selected frame.
    """
    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Ingest dataset before selecting columns",
            context={"dataset_id": dataset_id},
        )

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")

    selected = df[columns].copy()
    new_dataset_id = register_dataset(
        selected,
        filename="selected",
        parent_dataset_id=dataset_id,
        transformation_note=f"select: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}",
    )

    result = SelectResult(
        original_dataset_id=dataset_id,
        new_dataset_id=new_dataset_id,
        selected_columns=columns,
        n_rows=int(len(selected)),
        n_columns_before=int(df.shape[1]),
        n_columns_after=int(selected.shape[1]),
    )
    return wrap_success(result.model_dump())


def mutate_columns(
    dataset_id: str,
    expressions: Dict[str, str],
) -> Dict[str, Any]:
    """
    Internal helper to create or update columns based on expression strings.

    `expressions` is a mapping from new_column_name -> expression string.

    Each expression is evaluated in the context of the DataFrame, for example:
      {
        "bmi": "weight_kg / (height_m ** 2)",
        "income_k": "income / 1000"
      }

    Returns metadata and a new dataset_id for the modified frame.
    """
    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Ingest dataset before mutating columns",
            context={"dataset_id": dataset_id},
        )
    modified = df.copy()

    existing_cols = set(modified.columns)
    new_cols_created: List[str] = []

    for col_name, expr in expressions.items():
        try:
            # Use DataFrame.eval so expressions operate on columns, not Python globals.
            modified[col_name] = modified.eval(expr)
        except Exception as e:
            raise ValueError(f"Failed to compute expression for '{col_name}': {e}")

        if col_name not in existing_cols:
            new_cols_created.append(col_name)

    expr_summary = ", ".join(f"{k}={v[:20]}" for k, v in list(expressions.items())[:3])
    new_dataset_id = register_dataset(
        modified,
        filename="mutated",
        parent_dataset_id=dataset_id,
        transformation_note=f"mutate: {expr_summary}",
    )

    result = MutateResult(
        original_dataset_id=dataset_id,
        new_dataset_id=new_dataset_id,
        expressions=expressions,
        n_rows=int(len(modified)),
        n_columns_before=int(df.shape[1]),
        n_columns_after=int(modified.shape[1]),
        new_columns_created=new_cols_created,
    )
    return wrap_success(result.model_dump())


def wrangle_filter_rows_tool(
    dataset_id: str,
    condition: str,
) -> Dict[str, Any]:
    """
    Tool wrapper to filter rows using a condition string.

    Delegates to apply_row_filter and returns its metadata.
    """
    try:
        return apply_row_filter(dataset_id=dataset_id, condition=condition)
    except ValueError as e:
        return exception_to_error(
            EXPRESSION_ERROR,
            e,
            hint="Check filter condition syntax and column names",
        )
    except Exception as e:
        return exception_to_error(
            INVALID_PARAMETER,
            e,
            hint="Verify dataset_id and condition are valid",
        )


def wrangle_select_columns_tool(
    dataset_id: str,
    columns: List[str],
) -> Dict[str, Any]:
    """
    Tool wrapper to select a subset of columns from the dataset.

    Delegates to select_columns and returns its metadata.
    """
    try:
        return select_columns(dataset_id=dataset_id, columns=columns)
    except ValueError as e:
        return exception_to_error(
            COLUMN_NOT_FOUND,
            e,
            hint="Check that all column names exist in the dataset",
        )
    except Exception as e:
        return exception_to_error(
            INVALID_PARAMETER,
            e,
            hint="Verify dataset_id and columns are valid",
        )


def wrangle_mutate_columns_tool(
    dataset_id: str,
    expressions: Dict[str, str],
) -> Dict[str, Any]:
    """
    Tool wrapper to create or modify columns in the dataset.

    Delegates to mutate_columns and returns its metadata.
    """
    try:
        return mutate_columns(dataset_id=dataset_id, expressions=expressions)
    except ValueError as e:
        return exception_to_error(
            EXPRESSION_ERROR,
            e,
            hint="Check expression syntax and referenced column names",
        )
    except Exception as e:
        return exception_to_error(
            INVALID_PARAMETER,
            e,
            hint="Verify dataset_id and expressions are valid",
        )


def wrangle_remove_outliers_tool(
    dataset_id: str,
    outlier_metadata: Dict[str, Any],
    columns: Optional[List[str]] = None,
    strategy: str = "remove",
) -> Dict[str, Any]:
    """
    Smart outlier removal tool that uses pre-computed outlier metadata from data_quality_tool.

    This tool removes outliers based on the bounds calculated during quality analysis,
    eliminating the need for users to manually specify thresholds.

    Args:
        dataset_id: The ID of the dataset to filter
        outlier_metadata: The outlier_metadata dict from data_quality_tool containing
                         columns_with_outliers with lower_bound and upper_bound for each column
        columns: Optional list of column names to remove outliers from. If None, removes
                outliers from ALL columns in the metadata.
        strategy: "remove" (default) - removes rows with outliers
                 "clip" - clips outlier values to bounds (future feature)

    Returns:
        FilterResult with the new dataset_id and summary of removed rows

    Example usage:
        # The agent can pass outlier_metadata from state:
        wrangle_remove_outliers_tool(
            dataset_id="ds_abc123",
            outlier_metadata=state["outlier_metadata"],
            columns=["Life expectancy"]  # or None for all columns
        )
    """
    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Ingest dataset before removing outliers",
            context={"dataset_id": dataset_id},
        )

    # Validate outlier_metadata structure
    if not outlier_metadata or "columns_with_outliers" not in outlier_metadata:
        return make_error(
            INVALID_PARAMETER,
            "outlier_metadata is missing or invalid",
            hint="Run data_quality_tool first to generate outlier_metadata",
            context={"dataset_id": dataset_id},
        )

    columns_with_outliers = outlier_metadata.get("columns_with_outliers", [])
    if not columns_with_outliers:
        return wrap_success(
            {
                "message": "No outliers found in metadata - dataset unchanged",
                "original_dataset_id": dataset_id,
                "new_dataset_id": dataset_id,
                "n_rows_before": len(df),
                "n_rows_after": len(df),
                "columns_processed": [],
            }
        )

    # Filter to requested columns if specified
    if columns:
        columns_lower = [c.lower() for c in columns]
        columns_with_outliers = [
            c
            for c in columns_with_outliers
            if c.get("column_name", "").lower() in columns_lower
        ]

    if not columns_with_outliers:
        return wrap_success(
            {
                "message": "Specified columns have no outliers in metadata",
                "original_dataset_id": dataset_id,
                "new_dataset_id": dataset_id,
                "n_rows_before": len(df),
                "n_rows_after": len(df),
                "columns_processed": columns or [],
            }
        )

    # Build combined filter condition from all column bounds
    filter_conditions = []
    columns_processed = []
    removal_details = []

    for col_info in columns_with_outliers:
        col_name = col_info.get("column_name")
        lower_bound = col_info.get("lower_bound")
        upper_bound = col_info.get("upper_bound")
        outlier_count = col_info.get("outlier_count", 0)

        if col_name and col_name in df.columns:
            conditions = []
            if lower_bound is not None:
                conditions.append(f"`{col_name}` >= {lower_bound}")
            if upper_bound is not None:
                conditions.append(f"`{col_name}` <= {upper_bound}")

            if conditions:
                filter_conditions.append(" and ".join(conditions))
                columns_processed.append(col_name)
                removal_details.append(
                    f"'{col_name}': {outlier_count} outliers (bounds: [{lower_bound:.4g}, {upper_bound:.4g}])"
                )

    if not filter_conditions:
        return wrap_success(
            {
                "message": "No valid filter conditions could be built from metadata",
                "original_dataset_id": dataset_id,
                "new_dataset_id": dataset_id,
                "n_rows_before": len(df),
                "n_rows_after": len(df),
                "columns_processed": [],
            }
        )

    # Combine all conditions with AND (keep rows that are within bounds for ALL columns)
    combined_condition = " and ".join(f"({c})" for c in filter_conditions)

    try:
        filtered = df.query(combined_condition)
    except Exception as e:
        return exception_to_error(
            EXPRESSION_ERROR,
            e,
            hint=f"Error applying outlier filter: {combined_condition[:100]}",
        )

    n_rows_removed = len(df) - len(filtered)

    new_dataset_id = register_dataset(
        filtered,
        filename="outliers_removed",
        parent_dataset_id=dataset_id,
        transformation_note=f"removed outliers from {', '.join(columns_processed)}",
    )

    result = {
        "operation": "remove_outliers",
        "original_dataset_id": dataset_id,
        "new_dataset_id": new_dataset_id,
        "n_rows_before": len(df),
        "n_rows_after": len(filtered),
        "n_rows_removed": n_rows_removed,
        "n_columns": df.shape[1],
        "columns_processed": columns_processed,
        "removal_details": removal_details,
        "filter_applied": combined_condition,
        "message": f"Removed {n_rows_removed} rows containing outliers from {len(columns_processed)} column(s)",
    }

    return wrap_success(result)


def get_outlier_removal_options(
    outlier_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper function to generate user-friendly outlier removal options from metadata.

    Returns a structured set of options the agent can present to the user
    in a suggest-and-confirm pattern.

    Args:
        outlier_metadata: The outlier_metadata dict from data_quality_tool

    Returns:
        Dict with:
        - options: List of removal options with descriptions
        - total_outliers: Total count across all columns
        - columns_affected: List of column names with outliers
    """
    if not outlier_metadata or "columns_with_outliers" not in outlier_metadata:
        return {
            "options": [],
            "total_outliers": 0,
            "columns_affected": [],
            "message": "No outlier metadata available. Run data quality check first.",
        }

    columns_with_outliers = outlier_metadata.get("columns_with_outliers", [])
    total_outliers = outlier_metadata.get("total_outlier_count", 0)

    if not columns_with_outliers:
        return {
            "options": [],
            "total_outliers": 0,
            "columns_affected": [],
            "message": "No outliers detected in the dataset.",
        }

    options = []
    columns_affected = []

    # Option 1: Remove all outliers
    options.append(
        {
            "id": "all",
            "description": f"Remove ALL outliers ({total_outliers} total across all columns)",
            "columns": [c["column_name"] for c in columns_with_outliers],
            "estimated_rows_removed": "varies based on overlap",
        }
    )

    # Per-column options
    for col_info in columns_with_outliers:
        col_name = col_info.get("column_name", "unknown")
        outlier_count = col_info.get("outlier_count", 0)
        lower_bound = col_info.get("lower_bound")
        upper_bound = col_info.get("upper_bound")
        outlier_pct = col_info.get("outlier_pct", 0)

        columns_affected.append(col_name)

        bounds_desc = ""
        if lower_bound is not None and upper_bound is not None:
            bounds_desc = f"keep values in [{lower_bound:.4g}, {upper_bound:.4g}]"
        elif lower_bound is not None:
            bounds_desc = f"keep values >= {lower_bound:.4g}"
        elif upper_bound is not None:
            bounds_desc = f"keep values <= {upper_bound:.4g}"

        options.append(
            {
                "id": col_name,
                "description": f"Remove {outlier_count} outliers from '{col_name}' ({outlier_pct:.1%} of values) - {bounds_desc}",
                "columns": [col_name],
                "estimated_rows_removed": outlier_count,
                "bounds": {"lower": lower_bound, "upper": upper_bound},
            }
        )

    return {
        "options": options,
        "total_outliers": total_outliers,
        "columns_affected": columns_affected,
        "message": f"Found {total_outliers} outliers across {len(columns_affected)} column(s). Choose an option to remove them.",
    }

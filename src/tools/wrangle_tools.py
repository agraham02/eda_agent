# wrangle_tools.py

from typing import Any, Dict, List

import pandas as pd

from ..utils.data_store import get_dataset, register_dataset


def apply_row_filter(
    dataset_id: str,
    condition: str,
) -> Dict[str, Any]:
    """
    Internal helper to filter rows based on a condition expression.

    The condition should be a pandas query string, for example:
      "age > 30 and country == 'US'"

    Returns metadata and a new dataset_id for the filtered frame.
    """
    df = get_dataset(dataset_id)

    try:
        # Use pandas query syntax for safety and familiarity.
        filtered = df.query(condition)
    except Exception as e:
        raise ValueError(f"Invalid filter condition '{condition}': {e}")

    new_dataset_id = register_dataset(filtered)

    return {
        "operation": "filter_rows",
        "original_dataset_id": dataset_id,
        "new_dataset_id": new_dataset_id,
        "condition": condition,
        "n_rows_before": int(len(df)),
        "n_rows_after": int(len(filtered)),
        "n_columns": int(df.shape[1]),
    }


def select_columns(
    dataset_id: str,
    columns: List[str],
) -> Dict[str, Any]:
    """
    Internal helper to select a subset of columns.

    Returns metadata and a new dataset_id for the selected frame.
    """
    df = get_dataset(dataset_id)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")

    selected = df[columns].copy()
    new_dataset_id = register_dataset(selected)

    return {
        "operation": "select_columns",
        "original_dataset_id": dataset_id,
        "new_dataset_id": new_dataset_id,
        "selected_columns": columns,
        "n_rows": int(len(selected)),
        "n_columns_before": int(df.shape[1]),
        "n_columns_after": int(selected.shape[1]),
    }


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
    df = get_dataset(dataset_id)
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

    new_dataset_id = register_dataset(modified)

    return {
        "operation": "mutate_columns",
        "original_dataset_id": dataset_id,
        "new_dataset_id": new_dataset_id,
        "expressions": expressions,
        "n_rows": int(len(modified)),
        "n_columns_before": int(df.shape[1]),
        "n_columns_after": int(modified.shape[1]),
        "new_columns_created": new_cols_created,
    }


def wrangle_filter_rows_tool(
    dataset_id: str,
    condition: str,
) -> Dict[str, Any]:
    """
    Tool wrapper to filter rows using a condition string.

    Delegates to apply_row_filter and returns its metadata.
    """
    return apply_row_filter(dataset_id=dataset_id, condition=condition)


def wrangle_select_columns_tool(
    dataset_id: str,
    columns: List[str],
) -> Dict[str, Any]:
    """
    Tool wrapper to select a subset of columns from the dataset.

    Delegates to select_columns and returns its metadata.
    """
    return select_columns(dataset_id=dataset_id, columns=columns)


def wrangle_mutate_columns_tool(
    dataset_id: str,
    expressions: Dict[str, str],
) -> Dict[str, Any]:
    """
    Tool wrapper to create or modify columns in the dataset.

    Delegates to mutate_columns and returns its metadata.
    """
    return mutate_columns(dataset_id=dataset_id, expressions=expressions)

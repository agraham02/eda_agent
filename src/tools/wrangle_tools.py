# wrangle_tools.py

import re
from typing import Any, Dict, List

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

    new_dataset_id = register_dataset(filtered)

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
    new_dataset_id = register_dataset(selected)

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

    new_dataset_id = register_dataset(modified)

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

from typing import Any, Dict, List

import pandas as pd

from ..utils.data_store import register_dataset
from ..utils.errors import INGESTION_ERROR, exception_to_error, wrap_success
from ..utils.schemas import ColumnInfo, IngestionResult, SemanticType


def infer_semantic_type(dtype: str, n_unique: int, n_rows: int) -> str:
    """
    Very simple heuristic for semantic type.
    We can improve this later.
    """
    dtype = dtype.lower()

    # Treat explicit datetime separately if you parse it
    if "datetime" in dtype or "date" in dtype:
        return "datetime"

    # Numeric types
    if any(x in dtype for x in ["int", "float", "decimal"]):
        # if too few unique values relative to rows, may be categorical
        if n_unique <= 20 or n_unique <= 0.05 * n_rows:
            return "numeric_categorical"
        return "numeric"

    # Boolean / logical
    if "bool" in dtype:
        return "boolean"

    # Fallbacks: object, string, etc
    if "object" in dtype or "str" in dtype:
        # Again, rough heuristic for categorical vs text
        if n_unique <= 50 and n_unique <= 0.1 * n_rows:
            return "categorical"
        return "text"

    return "unknown"


def ingest_csv(file_path: str, max_sample_rows: int = 20) -> Dict[str, Any]:
    """
    Load a CSV, register it in the in-memory store, and return
    structured artifacts for downstream agents.
    """
    df = pd.read_csv(file_path)

    dataset_id = register_dataset(df)

    n_rows, n_cols = df.shape

    column_models: List[ColumnInfo] = []
    warnings: List[str] = []

    for col in df.columns:
        series = df[col]
        pandas_dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        missing_pct = float(n_missing / max(1, n_rows))
        n_unique = int(series.nunique(dropna=True))

        semantic_type = infer_semantic_type(pandas_dtype, n_unique, n_rows)

        if missing_pct > 0.3:
            warnings.append(f"Column '{col}' has high missingness ({missing_pct:.1%}).")

        # Take a few non-null example values as strings
        non_null = series.dropna().astype(str).head(5).tolist()

        column_models.append(
            ColumnInfo(
                name=col,
                pandas_dtype=pandas_dtype,
                semantic_type=SemanticType(semantic_type) if semantic_type in SemanticType.__members__.values() else SemanticType.UNKNOWN,  # type: ignore
                n_missing=n_missing,
                missing_pct=missing_pct,
                n_unique=n_unique,
                example_values=non_null,
            )
        )

    sample_rows = [
        {str(k): v for k, v in row.items()}
        for row in df.head(max_sample_rows).to_dict(orient="records")
    ]

    result_model = IngestionResult(
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_columns=n_cols,
        columns=column_models,
        sample_rows=sample_rows,
        warnings=warnings,
        source={"file_path": file_path, "format": "csv"},
    )

    payload = result_model.model_dump()
    return wrap_success(payload)


def ingest_csv_tool(file_path: str) -> Dict[str, Any]:
    """
    Ingest a CSV file into the Data Whisperer system.
    """
    try:
        return ingest_csv(file_path)
    except Exception as e:
        return exception_to_error(
            INGESTION_ERROR, e, hint="Ensure file path is accessible and valid CSV"
        )

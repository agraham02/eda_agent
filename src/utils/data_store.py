# data_store.py
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd

from .persistent_store import get_store

_DATASETS: Dict[str, pd.DataFrame] = {}


def register_dataset(
    df: pd.DataFrame,
    filename: str = "unknown",
    parent_dataset_id: Optional[str] = None,
    transformation_note: Optional[str] = None,
    persist: bool = True,
) -> str:
    """
    Store a dataframe in memory and optionally persist to disk.
    Returns a dataset_id handle with 'ds_' prefix.

    Args:
        df: The dataframe to register
        filename: Original filename or source identifier
        parent_dataset_id: ID of parent dataset if this is a transformation
        transformation_note: Short description of transformation applied
        persist: Whether to save to parquet and SQLite (default True)
    """
    dataset_id = f"ds_{uuid4()}"
    _DATASETS[dataset_id] = df

    # Persist to parquet + SQLite if requested
    if persist:
        try:
            store = get_store()
            store.save_dataset(
                df=df,
                dataset_id=dataset_id,
                filename=filename,
                parent_dataset_id=parent_dataset_id,
                transformation_note=transformation_note,
            )
        except Exception:
            # If persistence fails, still keep in memory
            pass

    return dataset_id


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Retrieve a dataframe by dataset_id.
    First checks in-memory store, then tries to load from persistent storage.
    Raise KeyError if not found in either location.
    """
    # Check in-memory first
    if dataset_id in _DATASETS:
        return _DATASETS[dataset_id]

    # Try loading from persistent storage
    try:
        store = get_store()
        df = store.load_dataset(dataset_id)
        if df is not None:
            # Cache in memory for future access
            _DATASETS[dataset_id] = df
            return df
    except Exception:
        pass

    # Not found anywhere
    available_ids = list(_DATASETS.keys())
    raise KeyError(
        f"Dataset ID '{dataset_id}' not found. "
        f"Available dataset IDs: {available_ids if available_ids else 'None (no datasets registered yet)'}. "
        f"Please ingest a dataset first using the ingest_csv_tool."
    )


def has_dataset(dataset_id: str) -> bool:
    """Check if dataset exists in memory or persistent storage."""
    if dataset_id in _DATASETS:
        return True
    try:
        store = get_store()
        return store.get_dataset_metadata(dataset_id) is not None
    except Exception:
        return False


def list_datasets() -> Dict[str, Dict[str, Any]]:
    """
    List all registered datasets with basic info.
    Includes both in-memory and persisted datasets.
    """
    result = {}

    # Add in-memory datasets
    for dataset_id, df in _DATASETS.items():
        result[dataset_id] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "in_memory": True,
        }

    # Add persisted datasets not already in memory
    try:
        store = get_store()
        for metadata in store.list_datasets():
            if metadata.dataset_id not in result:
                result[metadata.dataset_id] = {
                    "shape": (metadata.n_rows, metadata.n_columns),
                    "columns": metadata.columns,
                    "memory_usage_mb": None,
                    "in_memory": False,
                    "filename": metadata.filename,
                    "ingested_at": metadata.ingested_at.isoformat(),
                    "parent_dataset_id": metadata.parent_dataset_id,
                }
    except Exception:
        pass

    return result


def clear_datasets() -> None:
    """
    Clear all datasets from memory.
    """
    _DATASETS.clear()

# data_store.py
from typing import Any, Dict
from uuid import uuid4

import pandas as pd

_DATASETS: Dict[str, pd.DataFrame] = {}


def register_dataset(df: pd.DataFrame) -> str:
    """
    Store a dataframe in memory and return a dataset_id handle.
    The dataset_id will have a 'ds_' prefix for clarity.
    """
    dataset_id = f"ds_{uuid4()}"
    _DATASETS[dataset_id] = df
    return dataset_id


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Retrieve a dataframe by dataset_id.
    Raise KeyError if not found.
    """
    if dataset_id not in _DATASETS:
        available_ids = list(_DATASETS.keys())
        raise KeyError(
            f"Dataset ID '{dataset_id}' not found. "
            f"Available dataset IDs: {available_ids if available_ids else 'None (no datasets registered yet)'}. "
            f"Please ingest a dataset first using the ingest_csv_tool."
        )
    return _DATASETS[dataset_id]


def has_dataset(dataset_id: str) -> bool:
    return dataset_id in _DATASETS


def list_datasets() -> Dict[str, Dict[str, Any]]:
    """
    List all registered datasets with basic info.
    """
    return {
        dataset_id: {
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        for dataset_id, df in _DATASETS.items()
    }


def clear_datasets() -> None:
    """
    Clear all datasets from memory.
    """
    _DATASETS.clear()

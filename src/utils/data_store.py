# data_store.py
from typing import Dict
from uuid import uuid4

import pandas as pd

_DATASETS: Dict[str, pd.DataFrame] = {}


def register_dataset(df: pd.DataFrame) -> str:
    """
    Store a dataframe in memory and return a dataset_id handle.
    """
    dataset_id = str(uuid4())
    _DATASETS[dataset_id] = df
    return dataset_id


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Retrieve a dataframe by dataset_id.
    Raise KeyError if not found.
    """
    return _DATASETS[dataset_id]


def has_dataset(dataset_id: str) -> bool:
    return dataset_id in _DATASETS

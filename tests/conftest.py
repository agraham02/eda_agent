"""
Test fixtures and configuration for Data Whisperer EDA Agent
Hackathon version - minimal fixtures for fast testing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_store import clear_datasets, register_dataset


@pytest.fixture(autouse=True)
def cleanup_datasets():
    """Clear dataset store after each test to prevent interference"""
    yield
    clear_datasets()


@pytest.fixture
def perfect_df():
    """DataFrame with no quality issues - should score 100"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(100),
            "age": np.random.randint(20, 80, 100),
            "income": np.random.randint(30000, 120000, 100),
            "score": np.random.uniform(0, 100, 100),
        }
    )


@pytest.fixture
def high_missing_df():
    """DataFrame with 60% missing values in one column"""
    data = [1, 2, 3, 4, 5] * 8  # 40 values
    missing = [None] * 60
    return pd.DataFrame({"col": data + missing})


@pytest.fixture
def duplicate_df():
    """DataFrame with 50% duplicate rows"""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5] * 10, "b": ["x", "y", "z", "w", "v"] * 10})
    # Add duplicates - repeat first half
    return pd.concat([df, df.iloc[:25]], ignore_index=True)


@pytest.fixture
def registered_perfect_dataset(perfect_df):
    """Pre-registered dataset_id for tests that need it"""
    return register_dataset(perfect_df)

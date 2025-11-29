"""
Smoke tests for wrangle tools
Hackathon version - basic filter and select operations
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.wrangle_tools import (
    wrangle_filter_rows_tool,
    wrangle_select_columns_tool,
)
from src.utils.data_store import get_dataset, register_dataset


@pytest.mark.smoke
def test_filter_rows_basic(perfect_df):
    """Test basic row filtering"""
    dataset_id = register_dataset(perfect_df)

    # Filter for age > 50
    result = wrangle_filter_rows_tool(dataset_id, "age > 50")
    assert result["ok"] is True

    new_dataset_id = result["new_dataset_id"]
    filtered_df = get_dataset(new_dataset_id)
    assert len(filtered_df) < len(perfect_df)
    assert all(filtered_df["age"] > 50)


@pytest.mark.smoke
def test_select_columns(perfect_df):
    """Test column selection"""
    dataset_id = register_dataset(perfect_df)

    # Select only age and income
    result = wrangle_select_columns_tool(dataset_id, ["age", "income"])
    assert result["ok"] is True

    new_dataset_id = result["new_dataset_id"]
    selected_df = get_dataset(new_dataset_id)
    assert list(selected_df.columns) == ["age", "income"]
    assert len(selected_df) == len(perfect_df)  # Same number of rows

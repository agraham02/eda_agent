"""
Smoke tests for data quality tools
Hackathon version - focused on happy paths with real data
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.data_quality_tools import data_quality_tool
from src.tools.ingestion_tools import ingest_csv_tool
from src.utils.data_store import get_dataset


@pytest.mark.smoke
def test_data_quality_on_real_dataset(perfect_df):
    """Test data quality analysis on dataset"""
    from src.utils.data_store import register_dataset

    # Use fixture instead of file to avoid path issues
    dataset_id = register_dataset(perfect_df)

    # Run quality check
    quality_result = data_quality_tool(dataset_id)
    assert quality_result["ok"] is True
    assert "readiness_score" in quality_result
    assert "columns" in quality_result
    assert quality_result["readiness_score"]["overall"] >= 0
    assert quality_result["readiness_score"]["overall"] <= 100


@pytest.mark.smoke
def test_data_quality_with_missing_dataset():
    """Test error handling when dataset doesn't exist"""
    result = data_quality_tool("fake_dataset_id")
    assert result["ok"] is False
    assert "error" in result


@pytest.mark.integration
def test_full_quality_pipeline(perfect_df):
    """Test full pipeline: register -> quality check"""
    from src.utils.data_store import register_dataset

    dataset_id = register_dataset(perfect_df)
    result = data_quality_tool(dataset_id)

    assert result["ok"] is True
    assert result["readiness_score"]["overall"] > 85  # Should be high quality

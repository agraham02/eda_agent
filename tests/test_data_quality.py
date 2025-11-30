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


@pytest.mark.smoke
def test_iqr_outlier_detection(outlier_df):
    """Test IQR outlier detection finds known outliers"""
    from src.utils.data_store import register_dataset

    dataset_id = register_dataset(outlier_df)
    result = data_quality_tool(dataset_id, outlier_method="iqr")

    assert result["ok"] is True
    # Check that outliers were detected in the numeric column
    col_stats = result["columns"][0]
    assert col_stats["numeric_summary"] is not None
    assert col_stats["numeric_summary"]["outlier_count"] > 0
    assert col_stats["numeric_summary"]["outlier_method"] == "iqr"


@pytest.mark.smoke
def test_zscore_outlier_detection(outlier_df):
    """Test Z-score outlier detection"""
    from src.utils.data_store import register_dataset

    dataset_id = register_dataset(outlier_df)
    result = data_quality_tool(dataset_id, outlier_method="zscore")

    assert result["ok"] is True
    col_stats = result["columns"][0]
    assert col_stats["numeric_summary"] is not None
    assert col_stats["numeric_summary"]["outlier_method"] == "zscore"


@pytest.mark.smoke
def test_both_outlier_methods(outlier_df):
    """Test using both methods returns union of outliers"""
    from src.utils.data_store import register_dataset

    dataset_id = register_dataset(outlier_df)
    result = data_quality_tool(dataset_id, outlier_method="both")

    assert result["ok"] is True
    col_stats = result["columns"][0]
    assert col_stats["numeric_summary"] is not None
    assert col_stats["numeric_summary"]["outlier_method"] == "both"
    # With both methods, we should detect at least some outliers
    assert col_stats["numeric_summary"]["outlier_count"] > 0

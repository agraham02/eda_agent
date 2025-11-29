"""
Smoke tests for readiness scoring
Hackathon version - focused on critical paths
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.data_quality_tools import compute_readiness_score, data_quality_tool
from src.utils.data_store import register_dataset
from src.utils.schemas import DataQualityColumn, SemanticType


@pytest.mark.smoke
def test_perfect_dataset_scores_100():
    """Perfect dataset with no issues should score 100"""
    columns = [
        DataQualityColumn(
            name="col1",
            pandas_dtype="int64",
            n_missing=0,
            missing_pct=0.0,
            n_unique=100,
            is_constant=False,
            is_all_unique=True,
            semantic_type=SemanticType.NUMERIC,
            numeric_summary=None,
        )
    ]
    result = compute_readiness_score(n_rows=100, duplicate_pct=0.0, columns=columns)
    assert result["overall"] == 100
    assert result["components"]["missingness"] == 100


@pytest.mark.smoke
def test_empty_dataset_scores_0():
    """Empty dataset should score 0"""
    result = compute_readiness_score(n_rows=0, duplicate_pct=0.0, columns=[])
    assert result["overall"] == 0
    assert "Empty dataset" in result["notes"]


@pytest.mark.smoke
def test_high_missing_penalized():
    """Dataset with 60% missing should score poorly"""
    columns = [
        DataQualityColumn(
            name="col1",
            pandas_dtype="object",
            n_missing=60,
            missing_pct=0.6,
            n_unique=40,
            is_constant=False,
            is_all_unique=False,
            semantic_type=SemanticType.CATEGORICAL,
            numeric_summary=None,
        )
    ]
    result = compute_readiness_score(n_rows=100, duplicate_pct=0.0, columns=columns)
    assert result["overall"] < 75  # 60% missing gives score around 65
    assert result["components"]["missingness"] < 50


@pytest.mark.smoke
def test_data_quality_tool_integration(registered_perfect_dataset):
    """Test data_quality_tool returns success with registered dataset"""
    result = data_quality_tool(registered_perfect_dataset)
    assert result["ok"] is True
    assert "readiness_score" in result
    assert result["readiness_score"]["overall"] >= 90  # Should be very high

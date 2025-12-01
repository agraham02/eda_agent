"""Tests for Long-Running Operation (LRO) tools.

These tests verify the behavior of tools that use ADK's request_confirmation()
pattern to pause agent execution and wait for human input.

Note: We import tools directly from their modules to avoid triggering
the root __init__.py which sets up DatabaseSessionService.
"""

import os
import sys

# Add src to path to allow direct imports without triggering __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# Mock ToolConfirmation to simulate ADK's confirmation responses
@dataclass
class MockToolConfirmation:
    """Mock of ADK's ToolConfirmation object."""

    confirmed: bool
    payload: Optional[Dict[str, Any]] = None


class MockToolContext:
    """Mock of ADK's ToolContext for testing LRO tools."""

    def __init__(self, tool_confirmation: Optional[MockToolConfirmation] = None):
        self.tool_confirmation = tool_confirmation
        self._confirmation_requested = False
        self._confirmation_hint: Optional[str] = None
        self._confirmation_payload: Optional[Dict[str, Any]] = None

        # For async artifact saving
        self.save_artifact = AsyncMock(return_value=1)

    def request_confirmation(self, hint: str, payload: Dict[str, Any]) -> None:
        """Mock the request_confirmation method."""
        self._confirmation_requested = True
        self._confirmation_hint = hint
        self._confirmation_payload = payload


# ============================================================================
# Tests for offer_quality_loop_tool
# ============================================================================


class TestOfferQualityLoopTool:
    """Tests for the quality loop LRO tool."""

    def test_first_call_pauses_for_confirmation(self):
        """First call should pause and request confirmation."""
        from src.tools.quality_loop_tools import offer_quality_loop_tool

        # First call - no confirmation yet
        tool_context = MockToolContext(tool_confirmation=None)

        result = offer_quality_loop_tool(
            tool_context=tool_context,
            dataset_id="test_dataset",
            readiness_score=65.0,
            quality_issues=["High missingness in column A", "Duplicate rows found"],
        )

        # Should request confirmation
        assert tool_context._confirmation_requested is True
        assert "Data Quality Assessment Complete" in tool_context._confirmation_hint
        assert "65" in tool_context._confirmation_hint  # Score shown

        # Should return pending status
        assert result["status"] == "pending"
        assert "Awaiting user decision" in result["message"]

    def test_resume_with_approval_returns_run_loop(self):
        """Resuming with approval should signal to run quality loop."""
        from src.tools.quality_loop_tools import offer_quality_loop_tool

        # Simulate user approved
        confirmation = MockToolConfirmation(confirmed=True)
        tool_context = MockToolContext(tool_confirmation=confirmation)

        result = offer_quality_loop_tool(
            tool_context=tool_context,
            dataset_id="test_dataset",
            readiness_score=65.0,
            quality_issues=["Issue 1"],
        )

        # Should return approved with run_loop action
        assert result["ok"] is True
        assert result["data"]["status"] == "approved"
        assert result["data"]["action"] == "run_loop"
        assert result["data"]["dataset_id"] == "test_dataset"

    def test_resume_with_rejection_returns_continue(self):
        """Resuming with rejection should signal to continue without loop."""
        from src.tools.quality_loop_tools import offer_quality_loop_tool

        # Simulate user rejected
        confirmation = MockToolConfirmation(confirmed=False)
        tool_context = MockToolContext(tool_confirmation=confirmation)

        result = offer_quality_loop_tool(
            tool_context=tool_context,
            dataset_id="test_dataset",
            readiness_score=65.0,
            quality_issues=["Issue 1"],
        )

        # Should return rejected with continue action
        assert result["ok"] is True
        assert result["data"]["status"] == "rejected"
        assert result["data"]["action"] == "continue"

    def test_readiness_bands_in_hint(self):
        """Hint should show correct readiness band."""
        from src.tools.quality_loop_tools import offer_quality_loop_tool

        test_cases = [
            (95.0, "Ready", "✅"),
            (80.0, "Minor fixes", "🔧"),
            (60.0, "Needs work", "⚠️"),
            (40.0, "Not ready", "🚨"),
        ]

        for score, expected_band, expected_emoji in test_cases:
            tool_context = MockToolContext(tool_confirmation=None)
            offer_quality_loop_tool(
                tool_context=tool_context,
                dataset_id="test",
                readiness_score=score,
                quality_issues=[],
            )

            assert (
                expected_band in tool_context._confirmation_hint
            ), f"Expected '{expected_band}' for score {score}"
            assert (
                expected_emoji in tool_context._confirmation_hint
            ), f"Expected emoji '{expected_emoji}' for score {score}"


# ============================================================================
# Tests for check_outlier_comparison_tool
# ============================================================================


class TestCheckOutlierComparisonTool:
    """Tests for the outlier comparison LRO tool."""

    def test_first_call_pauses_for_confirmation(self):
        """First call should pause and request confirmation."""
        from src.tools.eda_viz_tools import check_outlier_comparison_tool

        tool_context = MockToolContext(tool_confirmation=None)

        result = check_outlier_comparison_tool(
            tool_context=tool_context,
            dataset_id="test_dataset",
            outlier_pct=0.15,  # 15%
            columns_with_outliers=["age", "income", "balance"],
        )

        # Should request confirmation
        assert tool_context._confirmation_requested is True
        assert "High Outlier Rate Detected" in tool_context._confirmation_hint
        assert "15.0%" in tool_context._confirmation_hint

        # Should return pending status
        assert result["status"] == "pending"
        assert result["columns_count"] == 3

    def test_resume_with_approval_returns_create_comparison(self):
        """Resuming with approval should signal to create comparison viz."""
        from src.tools.eda_viz_tools import check_outlier_comparison_tool

        confirmation = MockToolConfirmation(confirmed=True)
        tool_context = MockToolContext(tool_confirmation=confirmation)

        result = check_outlier_comparison_tool(
            tool_context=tool_context,
            dataset_id="test_dataset",
            outlier_pct=0.15,
            columns_with_outliers=["age", "income"],
        )

        assert result["ok"] is True
        assert result["data"]["status"] == "approved"
        assert result["data"]["action"] == "create_comparison"
        assert result["data"]["columns"] == ["age", "income"]

    def test_resume_with_rejection_returns_skip(self):
        """Resuming with rejection should signal to skip comparison."""
        from src.tools.eda_viz_tools import check_outlier_comparison_tool

        confirmation = MockToolConfirmation(confirmed=False)
        tool_context = MockToolContext(tool_confirmation=confirmation)

        result = check_outlier_comparison_tool(
            tool_context=tool_context,
            dataset_id="test_dataset",
            outlier_pct=0.15,
            columns_with_outliers=["age"],
        )

        assert result["ok"] is True
        assert result["data"]["status"] == "rejected"
        assert result["data"]["action"] == "skip_comparison"

    def test_columns_truncated_in_hint(self):
        """Many columns should be truncated in the hint."""
        from src.tools.eda_viz_tools import check_outlier_comparison_tool

        tool_context = MockToolContext(tool_confirmation=None)
        many_columns = [f"col_{i}" for i in range(10)]

        check_outlier_comparison_tool(
            tool_context=tool_context,
            dataset_id="test",
            outlier_pct=0.15,
            columns_with_outliers=many_columns,
        )

        # Should show truncation notice
        assert "+4 more" in tool_context._confirmation_hint


# ============================================================================
# Tests for create_comparison_viz_tool
# ============================================================================


class TestCreateComparisonVizTool:
    """Tests for the comparison visualization tool."""

    @pytest.mark.asyncio
    async def test_creates_comparison_for_numeric_column(self):
        """Should create comparison viz for numeric column."""
        import numpy as np
        import pandas as pd

        from src.tools.eda_viz_tools import create_comparison_viz_tool
        from src.utils.data_store import store_dataset

        # Create test dataset with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 100)
        outliers = [200, -50, 300]  # Clear outliers
        data = np.concatenate([normal_data, outliers])
        df = pd.DataFrame({"value": data})

        dataset_id = "test_outlier_viz"
        store_dataset(dataset_id, df)

        tool_context = MockToolContext()

        result = await create_comparison_viz_tool(
            tool_context=tool_context,
            dataset_id=dataset_id,
            column="value",
            chart_type="box",
        )

        assert result["ok"] is True
        data = result["data"]
        assert "artifact_filename" in data
        assert "comparison_stats" in data
        assert data["comparison_stats"]["outliers_removed"] > 0
        assert data["column"] == "value"

    @pytest.mark.asyncio
    async def test_fails_for_non_numeric_column(self):
        """Should fail gracefully for non-numeric columns."""
        import pandas as pd

        from src.tools.eda_viz_tools import create_comparison_viz_tool
        from src.utils.data_store import store_dataset

        df = pd.DataFrame({"category": ["A", "B", "C", "D", "E"]})
        store_dataset("test_categorical", df)

        tool_context = MockToolContext()

        result = await create_comparison_viz_tool(
            tool_context=tool_context,
            dataset_id="test_categorical",
            column="category",
            chart_type="box",
        )

        assert result["ok"] is False
        assert "not numeric" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_fails_for_missing_column(self):
        """Should fail gracefully for missing column."""
        import pandas as pd

        from src.tools.eda_viz_tools import create_comparison_viz_tool
        from src.utils.data_store import store_dataset

        df = pd.DataFrame({"value": [1, 2, 3]})
        store_dataset("test_missing_col", df)

        tool_context = MockToolContext()

        result = await create_comparison_viz_tool(
            tool_context=tool_context,
            dataset_id="test_missing_col",
            column="nonexistent",
            chart_type="box",
        )

        assert result["ok"] is False
        assert "not found" in result["error"]["message"].lower()


# ============================================================================
# Tests for exit_quality_loop (existing tool)
# ============================================================================


class TestExitQualityLoop:
    """Tests for the exit quality loop tool."""

    def test_returns_success_status(self):
        """Should return success status for loop termination."""
        from src.tools.quality_loop_tools import exit_quality_loop

        result = exit_quality_loop()

        assert result["ok"] is True
        assert result["data"]["status"] == "quality_accept"
        assert "Exiting loop" in result["data"]["message"]

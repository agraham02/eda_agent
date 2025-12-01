"""Tests for persistent storage layer."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.utils.persistent_store import (
    AnalysisRun,
    DatasetMetadata,
    PersistentStore,
    PlotDensity,
    RunType,
    StructuredResults,
    UserPreferences,
    WritingStyle,
)


@pytest.fixture
def temp_store():
    """Create a temporary store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        datasets_dir = Path(tmpdir) / "datasets"
        store = PersistentStore(db_path=db_path, datasets_dir=datasets_dir)
        yield store


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [85.5, 90.0, 78.5, 92.0, 88.0],
        }
    )


class TestDatasetPersistence:
    """Tests for dataset save/load functionality."""

    def test_save_dataset(self, temp_store, sample_df):
        """Test saving a dataset to parquet."""
        metadata = temp_store.save_dataset(
            df=sample_df,
            dataset_id="ds_test123",
            filename="test.csv",
        )

        assert metadata.dataset_id == "ds_test123"
        assert metadata.filename == "test.csv"
        assert metadata.n_rows == 5
        assert metadata.n_columns == 4
        assert "id" in metadata.columns
        assert metadata.parquet_path is not None
        assert os.path.exists(metadata.parquet_path)

    def test_load_dataset(self, temp_store, sample_df):
        """Test loading a dataset from parquet."""
        temp_store.save_dataset(
            df=sample_df,
            dataset_id="ds_load_test",
            filename="test.csv",
        )

        loaded = temp_store.load_dataset("ds_load_test")
        assert loaded is not None
        assert len(loaded) == 5
        assert list(loaded.columns) == list(sample_df.columns)

    def test_dataset_metadata(self, temp_store, sample_df):
        """Test retrieving dataset metadata."""
        temp_store.save_dataset(
            df=sample_df,
            dataset_id="ds_meta_test",
            filename="test.csv",
            parent_dataset_id="ds_parent",
            transformation_note="filtered rows",
        )

        metadata = temp_store.get_dataset_metadata("ds_meta_test")
        assert metadata is not None
        assert metadata.parent_dataset_id == "ds_parent"
        assert metadata.transformation_note == "filtered rows"

    def test_list_datasets(self, temp_store, sample_df):
        """Test listing all datasets."""
        temp_store.save_dataset(sample_df, "ds_list1", "file1.csv")
        temp_store.save_dataset(sample_df, "ds_list2", "file2.csv")

        datasets = temp_store.list_datasets()
        assert len(datasets) == 2
        dataset_ids = [d.dataset_id for d in datasets]
        assert "ds_list1" in dataset_ids
        assert "ds_list2" in dataset_ids

    def test_dataset_lineage(self, temp_store, sample_df):
        """Test dataset lineage tracking."""
        temp_store.save_dataset(sample_df, "ds_original", "original.csv")
        temp_store.save_dataset(
            sample_df,
            "ds_child",
            "child.csv",
            parent_dataset_id="ds_original",
            transformation_note="filtered age > 30",
        )
        temp_store.save_dataset(
            sample_df,
            "ds_grandchild",
            "grandchild.csv",
            parent_dataset_id="ds_child",
            transformation_note="selected columns",
        )

        lineage = temp_store.get_dataset_lineage("ds_grandchild")
        assert len(lineage) == 3
        assert lineage[0].dataset_id == "ds_grandchild"
        assert lineage[1].dataset_id == "ds_child"
        assert lineage[2].dataset_id == "ds_original"


class TestAnalysisRuns:
    """Tests for analysis run persistence."""

    def test_save_run(self, temp_store, sample_df):
        """Test saving an analysis run."""
        temp_store.save_dataset(sample_df, "ds_run_test", "test.csv")

        run = AnalysisRun(
            dataset_id="ds_run_test",
            user_question="What is the average age?",
            run_type=RunType.DESCRIPTIVE,
            summary_markdown="## Summary\nAverage age is 35.",
            readiness_score={"overall": 85, "components": {}},
        )

        saved = temp_store.save_run(run)
        assert saved.run_id is not None
        assert saved.run_id.startswith("run_")

    def test_get_run(self, temp_store, sample_df):
        """Test retrieving an analysis run."""
        temp_store.save_dataset(sample_df, "ds_get_run", "test.csv")

        run = AnalysisRun(
            run_id="run_test123",
            dataset_id="ds_get_run",
            user_question="Test question",
            run_type=RunType.FULL,
        )
        temp_store.save_run(run)

        retrieved = temp_store.get_run("run_test123")
        assert retrieved is not None
        assert retrieved.dataset_id == "ds_get_run"
        assert retrieved.run_type == RunType.FULL

    def test_get_runs_for_dataset(self, temp_store, sample_df):
        """Test getting runs for a specific dataset."""
        temp_store.save_dataset(sample_df, "ds_multi_run", "test.csv")

        for i in range(5):
            run = AnalysisRun(
                dataset_id="ds_multi_run",
                user_question=f"Question {i}",
                run_type=RunType.QUALITY_CHECK,
            )
            temp_store.save_run(run)

        runs = temp_store.get_runs_for_dataset("ds_multi_run", limit=3)
        assert len(runs) == 3

    def test_compare_runs(self, temp_store, sample_df):
        """Test comparing two runs."""
        temp_store.save_dataset(sample_df, "ds_compare", "test.csv")

        run_a = AnalysisRun(
            run_id="run_a",
            dataset_id="ds_compare",
            user_question="First analysis",
            run_type=RunType.FULL,
            readiness_score={"overall": 70},
            structured_results=StructuredResults(p_values={"t_test": 0.05}),
        )
        run_b = AnalysisRun(
            run_id="run_b",
            dataset_id="ds_compare",
            user_question="Second analysis",
            run_type=RunType.FULL,
            readiness_score={"overall": 85},
            structured_results=StructuredResults(p_values={"t_test": 0.01}),
        )

        temp_store.save_run(run_a)
        temp_store.save_run(run_b)

        comparison = temp_store.compare_runs("run_a", "run_b")
        assert comparison is not None
        assert comparison["same_dataset"] is True
        assert comparison["readiness_delta"] == 15
        assert "t_test" in comparison["p_value_changes"]


class TestUserPreferences:
    """Tests for user preferences persistence."""

    def test_save_preferences(self, temp_store):
        """Test saving user preferences."""
        prefs = UserPreferences(
            user_id="test_user",
            writing_style=WritingStyle.EXECUTIVE,
            default_alpha=0.01,
            plot_density=PlotDensity.MINIMAL,
            auto_quality_check=False,
        )

        saved = temp_store.save_preferences(prefs)
        assert saved.user_id == "test_user"
        assert saved.writing_style == WritingStyle.EXECUTIVE

    def test_get_preferences(self, temp_store):
        """Test retrieving preferences."""
        prefs = UserPreferences(
            user_id="pref_user",
            writing_style=WritingStyle.TECHNICAL,
            default_alpha=0.05,
        )
        temp_store.save_preferences(prefs)

        retrieved = temp_store.get_preferences("pref_user")
        assert retrieved.writing_style == WritingStyle.TECHNICAL
        assert retrieved.default_alpha == 0.05

    def test_get_default_preferences(self, temp_store):
        """Test getting defaults when no preferences saved."""
        prefs = temp_store.get_preferences("nonexistent_user")
        assert prefs.user_id == "nonexistent_user"
        assert prefs.writing_style == WritingStyle.TECHNICAL
        assert prefs.default_alpha == 0.05

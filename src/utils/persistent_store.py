"""Persistent storage layer for datasets, analysis runs, and user preferences.

Uses SQLite for metadata and parquet files for dataset storage.
Designed to work alongside ADK's session/memory services for long-term persistence.
"""

import json
import os
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_DIR = Path("./data")
DATASETS_DIR = DEFAULT_DATA_DIR / "datasets"
DB_PATH = DEFAULT_DATA_DIR / "eda_store.db"


# ============================================================================
# ENUMS
# ============================================================================


class RunType(str, Enum):
    """Type of analysis run."""

    QUALITY_CHECK = "quality_check"
    DESCRIPTIVE = "descriptive"
    INFERENCE = "inference"
    FULL = "full"


class WritingStyle(str, Enum):
    """User's preferred writing style for reports."""

    EXECUTIVE = "executive"
    TECHNICAL = "technical"


class PlotDensity(str, Enum):
    """User's preferred plot density."""

    MINIMAL = "minimal"
    COMPREHENSIVE = "comprehensive"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class DatasetMetadata(BaseModel):
    """Metadata for a persisted dataset."""

    dataset_id: str = Field(..., description="Unique dataset identifier (ds_...)")
    filename: str = Field(..., description="Original filename or source identifier")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the dataset was ingested"
    )
    n_rows: int = Field(ge=0, description="Number of rows")
    n_columns: int = Field(ge=0, description="Number of columns")
    columns: List[str] = Field(default_factory=list, description="Column names")
    column_types: Dict[str, str] = Field(
        default_factory=dict, description="Column name -> dtype mapping"
    )
    parent_dataset_id: Optional[str] = Field(
        None, description="ID of parent dataset if this is a transformation"
    )
    transformation_note: Optional[str] = Field(
        None, description="Short description of transformation applied"
    )
    parquet_path: Optional[str] = Field(
        None, description="Path to parquet file on disk"
    )


class StructuredResults(BaseModel):
    """Compact structured results from an analysis run."""

    # Inference results
    p_values: Dict[str, float] = Field(
        default_factory=dict, description="Test name -> p-value"
    )
    confidence_intervals: Dict[str, List[float]] = Field(
        default_factory=dict, description="Test name -> [lower, upper]"
    )
    effect_sizes: Dict[str, float] = Field(
        default_factory=dict, description="Test name -> effect size (Cohen's d, etc.)"
    )

    # Descriptive highlights
    descriptive_highlights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key descriptive stats (means, correlations, etc.)",
    )

    # Visualization paths
    plot_paths: List[str] = Field(
        default_factory=list, description="Paths to generated plot files"
    )


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid4().hex[:12]}"


class AnalysisRun(BaseModel):
    """Record of a single analysis run."""

    run_id: str = Field(
        default_factory=_generate_run_id,
        description="Unique run identifier",
    )
    dataset_id: str = Field(..., description="Dataset analyzed")
    user_question: str = Field(..., description="Original user question/request")
    run_type: RunType = Field(..., description="Type of analysis performed")
    summary_markdown: str = Field(
        default="", description="Final summary report in markdown"
    )
    structured_results: StructuredResults = Field(
        default_factory=StructuredResults, description="Structured analysis results"
    )
    readiness_score: Optional[Dict[str, Any]] = Field(
        None, description="Data quality readiness score"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the run was created"
    )
    session_id: Optional[str] = Field(None, description="ADK session ID if available")


class UserPreferences(BaseModel):
    """User preferences for personalized analysis."""

    user_id: str = Field(default="default", description="User identifier")
    writing_style: WritingStyle = Field(
        default=WritingStyle.TECHNICAL, description="Preferred report style"
    )
    default_alpha: float = Field(
        default=0.05, ge=0.001, le=0.5, description="Default significance level"
    )
    plot_density: PlotDensity = Field(
        default=PlotDensity.COMPREHENSIVE, description="Preferred plot density"
    )
    auto_quality_check: bool = Field(
        default=True, description="Run quality check before inference automatically"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update time"
    )


# ============================================================================
# PERSISTENT STORE CLASS
# ============================================================================


class PersistentStore:
    """SQLite-backed persistent storage for EDA agent."""

    def __init__(
        self,
        db_path: Path = DB_PATH,
        datasets_dir: Path = DATASETS_DIR,
    ):
        self.db_path = Path(db_path)
        self.datasets_dir = Path(datasets_dir)

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    n_rows INTEGER NOT NULL,
                    n_columns INTEGER NOT NULL,
                    columns TEXT NOT NULL,
                    column_types TEXT NOT NULL,
                    parent_dataset_id TEXT,
                    transformation_note TEXT,
                    parquet_path TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    summary_markdown TEXT,
                    structured_results TEXT,
                    readiness_score TEXT,
                    created_at TEXT NOT NULL,
                    session_id TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    writing_style TEXT NOT NULL,
                    default_alpha REAL NOT NULL,
                    plot_density TEXT NOT NULL,
                    auto_quality_check INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_dataset ON analysis_runs(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_runs_created ON analysis_runs(created_at);
                CREATE INDEX IF NOT EXISTS idx_datasets_parent ON datasets(parent_dataset_id);
            """
            )

    # -------------------------------------------------------------------------
    # DATASET METHODS
    # -------------------------------------------------------------------------

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        filename: str,
        parent_dataset_id: Optional[str] = None,
        transformation_note: Optional[str] = None,
    ) -> DatasetMetadata:
        """Save a dataset to parquet and record metadata in SQLite."""
        # Save parquet
        parquet_path = self.datasets_dir / f"{dataset_id}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Build metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            filename=filename,
            ingested_at=datetime.utcnow(),
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            column_types={col: str(df[col].dtype) for col in df.columns},
            parent_dataset_id=parent_dataset_id,
            transformation_note=transformation_note,
            parquet_path=str(parquet_path),
        )

        # Insert/replace in SQLite
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets 
                (dataset_id, filename, ingested_at, n_rows, n_columns, columns, 
                 column_types, parent_dataset_id, transformation_note, parquet_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.dataset_id,
                    metadata.filename,
                    metadata.ingested_at.isoformat(),
                    metadata.n_rows,
                    metadata.n_columns,
                    json.dumps(metadata.columns),
                    json.dumps(metadata.column_types),
                    metadata.parent_dataset_id,
                    metadata.transformation_note,
                    metadata.parquet_path,
                ),
            )

        return metadata

    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load a dataset from parquet by ID."""
        metadata = self.get_dataset_metadata(dataset_id)
        if metadata and metadata.parquet_path and os.path.exists(metadata.parquet_path):
            return pd.read_parquet(metadata.parquet_path)
        return None

    def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()

        if row:
            return DatasetMetadata(
                dataset_id=row["dataset_id"],
                filename=row["filename"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
                n_rows=row["n_rows"],
                n_columns=row["n_columns"],
                columns=json.loads(row["columns"]),
                column_types=json.loads(row["column_types"]),
                parent_dataset_id=row["parent_dataset_id"],
                transformation_note=row["transformation_note"],
                parquet_path=row["parquet_path"],
            )
        return None

    def list_datasets(self) -> List[DatasetMetadata]:
        """List all persisted datasets."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY ingested_at DESC"
            ).fetchall()

        return [
            DatasetMetadata(
                dataset_id=row["dataset_id"],
                filename=row["filename"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
                n_rows=row["n_rows"],
                n_columns=row["n_columns"],
                columns=json.loads(row["columns"]),
                column_types=json.loads(row["column_types"]),
                parent_dataset_id=row["parent_dataset_id"],
                transformation_note=row["transformation_note"],
                parquet_path=row["parquet_path"],
            )
            for row in rows
        ]

    def get_dataset_lineage(self, dataset_id: str) -> List[DatasetMetadata]:
        """Get the lineage chain for a dataset (ancestors)."""
        lineage = []
        current_id = dataset_id

        while current_id:
            metadata = self.get_dataset_metadata(current_id)
            if metadata:
                lineage.append(metadata)
                current_id = metadata.parent_dataset_id
            else:
                break

        return lineage

    # -------------------------------------------------------------------------
    # ANALYSIS RUN METHODS
    # -------------------------------------------------------------------------

    def save_run(self, run: AnalysisRun) -> AnalysisRun:
        """Save an analysis run to the database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_runs
                (run_id, dataset_id, user_question, run_type, summary_markdown,
                 structured_results, readiness_score, created_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run.run_id,
                    run.dataset_id,
                    run.user_question,
                    run.run_type.value,
                    run.summary_markdown,
                    run.structured_results.model_dump_json(),
                    json.dumps(run.readiness_score) if run.readiness_score else None,
                    run.created_at.isoformat(),
                    run.session_id,
                ),
            )
        return run

    def get_run(self, run_id: str) -> Optional[AnalysisRun]:
        """Get a specific analysis run."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,)
            ).fetchone()

        if row:
            return self._row_to_run(row)
        return None

    def get_runs_for_dataset(
        self, dataset_id: str, limit: int = 10
    ) -> List[AnalysisRun]:
        """Get recent analysis runs for a dataset."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_runs 
                WHERE dataset_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (dataset_id, limit),
            ).fetchall()

        return [self._row_to_run(row) for row in rows]

    def get_recent_runs(self, limit: int = 20) -> List[AnalysisRun]:
        """Get recent analysis runs across all datasets."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_runs 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (limit,),
            ).fetchall()

        return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row: sqlite3.Row) -> AnalysisRun:
        """Convert a database row to an AnalysisRun."""
        structured_results = StructuredResults()
        if row["structured_results"]:
            structured_results = StructuredResults.model_validate_json(
                row["structured_results"]
            )

        readiness_score = None
        if row["readiness_score"]:
            readiness_score = json.loads(row["readiness_score"])

        return AnalysisRun(
            run_id=row["run_id"],
            dataset_id=row["dataset_id"],
            user_question=row["user_question"],
            run_type=RunType(row["run_type"]),
            summary_markdown=row["summary_markdown"] or "",
            structured_results=structured_results,
            readiness_score=readiness_score,
            created_at=datetime.fromisoformat(row["created_at"]),
            session_id=row["session_id"],
        )

    # -------------------------------------------------------------------------
    # USER PREFERENCES METHODS
    # -------------------------------------------------------------------------

    def save_preferences(self, prefs: UserPreferences) -> UserPreferences:
        """Save or update user preferences."""
        prefs.updated_at = datetime.utcnow()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_preferences
                (user_id, writing_style, default_alpha, plot_density, 
                 auto_quality_check, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    prefs.user_id,
                    prefs.writing_style.value,
                    prefs.default_alpha,
                    prefs.plot_density.value,
                    1 if prefs.auto_quality_check else 0,
                    prefs.updated_at.isoformat(),
                ),
            )
        return prefs

    def get_preferences(self, user_id: str = "default") -> UserPreferences:
        """Get user preferences, returning defaults if not set."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()

        if row:
            return UserPreferences(
                user_id=row["user_id"],
                writing_style=WritingStyle(row["writing_style"]),
                default_alpha=row["default_alpha"],
                plot_density=PlotDensity(row["plot_density"]),
                auto_quality_check=bool(row["auto_quality_check"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

        # Return defaults
        return UserPreferences(user_id=user_id)

    # -------------------------------------------------------------------------
    # COMPARISON HELPERS
    # -------------------------------------------------------------------------

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Optional[Dict[str, Any]]:
        """Compare two analysis runs and return differences."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        if not run_a or not run_b:
            return None

        comparison = {
            "run_a": {
                "run_id": run_a.run_id,
                "created_at": run_a.created_at.isoformat(),
            },
            "run_b": {
                "run_id": run_b.run_id,
                "created_at": run_b.created_at.isoformat(),
            },
            "same_dataset": run_a.dataset_id == run_b.dataset_id,
            "readiness_delta": None,
            "p_value_changes": {},
            "summary_a_preview": (
                run_a.summary_markdown[:500] if run_a.summary_markdown else ""
            ),
            "summary_b_preview": (
                run_b.summary_markdown[:500] if run_b.summary_markdown else ""
            ),
        }

        # Compare readiness scores
        if run_a.readiness_score and run_b.readiness_score:
            score_a = run_a.readiness_score.get("overall", 0)
            score_b = run_b.readiness_score.get("overall", 0)
            comparison["readiness_delta"] = score_b - score_a

        # Compare p-values
        all_tests = set(run_a.structured_results.p_values.keys()) | set(
            run_b.structured_results.p_values.keys()
        )
        for test in all_tests:
            p_a = run_a.structured_results.p_values.get(test)
            p_b = run_b.structured_results.p_values.get(test)
            if p_a is not None or p_b is not None:
                comparison["p_value_changes"][test] = {"run_a": p_a, "run_b": p_b}

        return comparison


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

_store: Optional[PersistentStore] = None


def get_store() -> PersistentStore:
    """Get the singleton PersistentStore instance."""
    global _store
    if _store is None:
        _store = PersistentStore()
    return _store

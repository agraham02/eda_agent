"""
Shared Pydantic schemas for data validation across all agents and tools.

This ensures consistent data formats when passing information between agents,
preventing issues like mismatched column names, chart types, or data structures.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# ENUMS - Define allowed values
# ============================================================================


class ChartType(str, Enum):
    """Allowed chart types for visualization."""

    HISTOGRAM = "histogram"
    BOX = "box"
    BOXPLOT = "boxplot"
    SCATTER = "scatter"
    BAR = "bar"
    LINE = "line"
    PIE = "pie"


class SemanticType(str, Enum):
    """Semantic data types for columns."""

    NUMERIC = "numeric"
    NUMERIC_CATEGORICAL = "numeric_categorical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


class TestType(str, Enum):
    """Statistical test types."""

    T_TEST = "t"
    Z_TEST = "z"


class Alternative(str, Enum):
    """Alternative hypothesis options."""

    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"


# ============================================================================
# COLUMN AND DATASET SCHEMAS
# ============================================================================


class ColumnInfo(BaseModel):
    """Schema for column metadata."""

    name: str = Field(..., description="Column name")
    pandas_dtype: str = Field(..., description="Pandas data type")
    semantic_type: SemanticType = Field(..., description="Inferred semantic type")
    n_missing: int = Field(ge=0, description="Count of missing values")
    missing_pct: float = Field(
        ge=0.0, le=1.0, description="Percentage of missing values"
    )
    n_unique: int = Field(ge=0, description="Count of unique values")
    example_values: List[str] = Field(default_factory=list, description="Sample values")

    @field_validator("name")
    @classmethod
    def normalize_column_name(cls, v: str) -> str:
        """Normalize column names to handle quote character issues."""
        return v.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')


class DatasetReference(BaseModel):
    """Schema for dataset identification."""

    dataset_id: str = Field(
        ..., pattern=r"^ds_[a-f0-9\-]+$", description="Unique dataset identifier"
    )
    n_rows: int = Field(ge=0, description="Number of rows")
    n_columns: int = Field(ge=0, description="Number of columns")


# ============================================================================
# VISUALIZATION SCHEMAS
# ============================================================================


class VizSpec(BaseModel):
    """Schema for visualization specifications."""

    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: ChartType = Field(..., description="Type of chart to create")
    x: str = Field(..., description="Column name for x-axis")
    y: Optional[str] = Field(None, description="Column name for y-axis (optional)")
    hue: Optional[str] = Field(
        None, description="Column name for color grouping (optional)"
    )
    bins: int = Field(
        default=10, ge=1, le=100, description="Number of bins for histograms"
    )

    @field_validator("chart_type", mode="before")
    @classmethod
    def normalize_chart_type(cls, v: str) -> str:
        """Normalize chart type variations to standard names."""
        if isinstance(v, str):
            normalized = v.lower().replace("_", "").replace("-", "")
            mapping = {
                "boxplot": "boxplot",
                "box": "box",
                "histogram": "histogram",
                "hist": "histogram",
                "scatter": "scatter",
                "scatterplot": "scatter",
                "bar": "bar",
                "barchart": "bar",
                "line": "line",
                "linechart": "line",
                "pie": "pie",
                "piechart": "pie",
            }
            result = mapping.get(normalized, v)
            # If not found in normalized mapping, try original with underscores replaced
            if result not in [
                "box",
                "boxplot",
                "histogram",
                "scatter",
                "bar",
                "line",
                "pie",
            ]:
                # Handle cases like "box_plot" -> "boxplot"
                original_normalized = v.lower().replace("_", "").replace("-", "")
                result = mapping.get(original_normalized, result)
            return result
        return v

    @field_validator("x", "y", "hue")
    @classmethod
    def normalize_column_names(cls, v: Optional[str]) -> Optional[str]:
        """Normalize column names to handle quote character issues."""
        if v is None:
            return None
        return v.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')


class VizResult(BaseModel):
    """Schema for visualization results."""

    file_path: str = Field(..., description="Path to generated plot image")
    chart_type: ChartType = Field(..., description="Type of chart created")
    dataset_id: str = Field(..., description="Source dataset identifier")


# ============================================================================
# DATA QUALITY SCHEMAS
# ============================================================================


class NumericSummary(BaseModel):
    """Schema for numeric column summary statistics."""

    mean: Optional[float] = None
    std: Optional[float] = None
    min: float
    q1: float
    median: float
    q3: float
    max: float
    iqr: float
    outlier_count: int = Field(ge=0)
    outliers: List[float] = Field(default_factory=list)
    outliers_truncated: bool = False


class DataQualityColumn(BaseModel):
    """Schema for column-level data quality assessment."""

    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    n_missing: int = Field(ge=0)
    missing_pct: float = Field(ge=0.0, le=1.0)
    n_unique: int = Field(ge=0)
    is_constant: bool
    is_all_unique: bool
    issues: List[str] = Field(default_factory=list)
    numeric_summary: Optional[NumericSummary] = None


# ============================================================================
# STATISTICAL TEST SCHEMAS
# ============================================================================


class OneSampleTestResult(BaseModel):
    """Schema for one-sample test results."""

    test_type: str
    dataset_id: str
    column: str
    n: int = Field(ge=1)
    sample_mean: float
    sample_std: float
    hypothesized_mean: float
    statistic: float
    standard_error: float
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    reject_null: bool
    confidence_level: float
    confidence_interval: List[float] = Field(min_length=2, max_length=2)
    alternative: Alternative


class TwoSampleTestResult(BaseModel):
    """Schema for two-sample test results."""

    test_type: str
    dataset_id: str
    column: str
    group_col: str
    group_a: str
    group_b: str
    n_a: int = Field(ge=1)
    n_b: int = Field(ge=1)
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_diff: float
    statistic: float
    standard_error_diff: float
    cohen_d: float
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    reject_null: bool
    confidence_level: float
    confidence_interval_diff: List[float] = Field(min_length=2, max_length=2)
    alternative: Alternative


class BinomialTestResult(BaseModel):
    """Schema for binomial test results."""

    test_type: str = "binomial"
    successes: int = Field(ge=0)
    n: int = Field(ge=1)
    observed_proportion: float = Field(ge=0.0, le=1.0)
    hypothesized_proportion: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    reject_null: bool
    confidence_level: float
    confidence_interval_proportion: List[float] = Field(min_length=2, max_length=2)
    alternative: Alternative


# ============================================================================
# INGESTION & DATA QUALITY RESULT SCHEMAS
# ============================================================================


class IngestionResult(BaseModel):
    """Schema for dataset ingestion output (backward compatible)."""

    dataset_id: str
    n_rows: int
    n_columns: int
    columns: List[ColumnInfo]
    sample_rows: List[Dict[str, Any]]
    warnings: List[str] = []
    source: Dict[str, Any]

    # Backward compatibility helpers removed; use model_dump() directly.


class DataQualityResult(BaseModel):
    """Schema for data quality tool output."""

    dataset_id: str
    n_rows: int
    n_columns: int
    duplicate_rows: Dict[str, Any]
    columns: List[DataQualityColumn]
    dataset_issues: List[str] = []
    readiness_score: Optional[Dict[str, Any]] = None  # Overall + component breakdown


# ============================================================================
# EDA DESCRIBE RESULT SCHEMAS
# ============================================================================


class UnivariateSummaryItem(BaseModel):
    name: str
    dtype: str
    n: int
    n_missing: int
    missing_pct: float
    type: str
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: List[Any] = []
    std: Optional[float] = None
    iqr: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    n_outliers: Optional[int] = None
    unique_values: Optional[List[str]] = None
    counts: Optional[Dict[str, int]] = None
    proportions: Optional[Dict[str, float]] = None


class UnivariateSummaryResult(BaseModel):
    dataset_id: str
    summaries: List[UnivariateSummaryItem]

    # Use model_dump() directly; previous reshaping removed.


class BivariateSummaryResult(BaseModel):
    dataset_id: str
    type: str
    payload: Dict[str, Any]

    # Use model_dump() directly.


class CorrelationMatrixResult(BaseModel):
    dataset_id: str
    columns: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]

    # Use model_dump() directly.


# ============================================================================
# CLT SAMPLING RESULT SCHEMA
# ============================================================================


class CLTSamplingResult(BaseModel):
    dataset_id: str
    column: str
    population_estimate: Dict[str, Any]
    sampling_parameters: Dict[str, Any]
    sampling_distribution: Dict[str, Any]
    sample_means_preview: List[float]

    # Use model_dump() directly.


# ============================================================================
# TRANSFORMATION SCHEMAS
# ============================================================================


class WrangleResult(BaseModel):
    """Schema for data wrangling operation results."""

    operation: str
    original_dataset_id: str
    new_dataset_id: str
    n_rows: int = Field(ge=0)
    message: Optional[str] = None


class FilterResult(WrangleResult):
    """Schema for row filtering results."""

    operation: str = "filter_rows"
    condition: str
    n_rows_before: int = Field(ge=0)
    n_rows_after: int = Field(ge=0)
    n_columns: int = Field(ge=0)


class SelectResult(WrangleResult):
    """Schema for column selection results."""

    operation: str = "select_columns"
    selected_columns: List[str]
    n_columns_before: int = Field(ge=0)
    n_columns_after: int = Field(ge=0)


class MutateResult(WrangleResult):
    """Schema for column mutation results."""

    operation: str = "mutate_columns"
    expressions: Dict[str, str]
    n_columns_before: int = Field(ge=0)
    n_columns_after: int = Field(ge=0)
    new_columns_created: List[str] = Field(default_factory=list)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_column_exists(column_name: str, available_columns: List[str]) -> str:
    """
    Validate and normalize a column name against available columns.

    Args:
        column_name: The column name to validate
        available_columns: List of available column names in the dataset

    Returns:
        The matched column name from the dataset

    Raises:
        ValueError: If the column name cannot be matched
    """
    # Normalize the input
    normalized = (
        column_name.replace("'", "'")
        .replace("'", "'")
        .replace(""", '"').replace(""", '"')
    )

    # Try exact match first
    if normalized in available_columns:
        return normalized

    # Try case-insensitive match
    for col in available_columns:
        if normalized.lower() == col.lower():
            return col

    # No match found
    raise ValueError(
        f"Column '{column_name}' not found in dataset. "
        f"Available columns: {', '.join(available_columns)}"
    )


def normalize_chart_type(chart_type: str) -> ChartType:
    """
    Normalize chart type string to ChartType enum.

    Args:
        chart_type: Raw chart type string

    Returns:
        Validated ChartType enum value

    Raises:
        ValueError: If chart type cannot be normalized
    """
    try:
        return VizSpec(
            dataset_id="dummy",
            chart_type=cast(ChartType, chart_type),  # type: ignore
            x="dummy",
            y=None,
            hue=None,
            bins=10,
        ).chart_type
    except Exception as e:
        allowed = [ct.value for ct in ChartType]
        raise ValueError(
            f"Invalid chart type '{chart_type}'. "
            f"Allowed types: {', '.join(allowed)}"
        ) from e

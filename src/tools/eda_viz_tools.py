import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ..utils.consts import OUTLIER_COMPARISON_THRESHOLD
from ..utils.dataset_cache import get_dataset_cached as get_dataset
from ..utils.errors import (
    RENDER_ERROR,
    VALIDATION_ERROR,
    exception_to_error,
    wrap_success,
)
from ..utils.paths import get_artifact_path
from ..utils.schemas import ChartType, VizResult, VizSpec, validate_column_exists

PLOTS_DIR = get_artifact_path("data_whisperer_plots", create_dir=True)

# Simple in-memory cache to prevent duplicate plot generation within a session.
# Keyed by (dataset_id, chart_type, x, y, hue, bins)
# Value: file_path string for the previously rendered plot.
_PLOT_CACHE: Dict[
    Tuple[str, str, Optional[str], Optional[str], Optional[str], int], str
] = {}
_PLOT_CACHE_FILE = os.path.join(PLOTS_DIR, "plot_cache.json")


def _load_plot_cache() -> None:
    if os.path.isfile(_PLOT_CACHE_FILE):
        try:
            with open(_PLOT_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Keys were stored as joined strings; rehydrate tuple
            for k, v in raw.items():
                parts = k.split("|||")
                dataset_id, chart_type, x, y, hue, bins_str = parts
                key = (
                    dataset_id,
                    chart_type,
                    x or None,
                    y or None,
                    hue or None,
                    int(bins_str),
                )
                _PLOT_CACHE[key] = v
        except Exception:
            # Corrupt cache; ignore
            pass


def _save_plot_cache() -> None:
    try:
        serializable: Dict[str, str] = {}
        for key, path in _PLOT_CACHE.items():
            dataset_id, chart_type, x, y, hue, bins = key
            serial_key = "|||".join(
                [
                    dataset_id,
                    chart_type,
                    x or "",
                    y or "",
                    hue or "",
                    str(bins),
                ]
            )
            serializable[serial_key] = path
        with open(_PLOT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        # Fail silently; caching is auxiliary
        pass


_load_plot_cache()


def _chart_role(chart_type: str) -> str:
    """
    Map chart types to high level roles:
    composition, distribution, relationship, comparison.
    Based on your visualization notes.
    """
    chart_type = chart_type.lower()

    if chart_type in {"histogram", "box", "boxplot", "violin"}:
        return "distribution"
    if chart_type in {"scatter", "line"}:
        return "relationship"
    if chart_type in {"bar", "grouped_bar", "stacked_bar"}:
        return "comparison"
    if chart_type in {"pie"}:
        return "composition"

    return "unknown"


## NOTE: Previous manual normalization helpers have been removed.
## Column and chart type normalization now handled by Pydantic validators
## and validate_column_exists in `schemas.py`.


def render_plot_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal helper that takes a visualization spec, renders a plot
    using matplotlib / seaborn, and returns a dict with a file path
    to the image (so the web UI can display it).
    """
    dataset_id = spec["dataset_id"]
    chart_type = spec["chart_type"]
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")
    bins = spec.get("bins", 10)

    cache_key = (dataset_id, chart_type, x, y, hue, bins)

    # If we have already rendered this exact specification, reuse the file.
    if cache_key in _PLOT_CACHE:
        existing_path = _PLOT_CACHE[cache_key]
        if os.path.isfile(existing_path):
            return {
                "file_path": existing_path,
                "chart_type": chart_type,
                "dataset_id": dataset_id,
                "x": x,
                "y": y,
                "hue": hue,
                "bins": bins,
                "role": _chart_role(chart_type),
                "reused": True,
                "message": "Duplicate visualization spec detected; reused previously rendered plot.",
            }
        # Stale cache entry (file removed); drop and regenerate
        _PLOT_CACHE.pop(cache_key, None)

    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        raise ValueError(
            f"Dataset ID '{dataset_id}' not found. "
            "Please ingest the dataset first using ingest_csv_tool."
        ) from e

    # Column names already normalized earlier via VizSpec + validate_column_exists.

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Choose chart type
    if chart_type == "histogram":
        # Univariate distribution of a numeric variable
        sns.histplot(data=df, x=x, bins=bins, ax=ax)
        ax.set_title(f"Distribution of {x}")
        ax.set_xlabel(str(x))
        ax.set_ylabel("Count")

    elif chart_type in {"box", "boxplot"}:
        # If y is provided, treat as numeric vs category
        if y is None:
            # Single variable boxplot on x
            sns.boxplot(data=df, x=x, ax=ax)
            ax.set_title(f"Boxplot of {x}")
            ax.set_xlabel(str(x))
        else:
            # Category on x, numeric on y
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
            ax.set_title(f"Boxplot of {y} by {x}")
            ax.set_xlabel(str(x))
            ax.set_ylabel(str(y))

    elif chart_type == "scatter":
        # Relationship between two numeric variables
        if y is None:
            raise ValueError("Scatter plot requires both x and y")
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f"Scatter plot of {y} vs {x}")
        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))

    elif chart_type == "bar":
        # Comparison across categories
        # If y is provided, plot y as a stat; otherwise countplot on x
        if y is None:
            sns.countplot(data=df, x=x, hue=hue, ax=ax)
            ax.set_title(f"Count of {x}")
            ax.set_xlabel(str(x))
            ax.set_ylabel("Count")
        else:
            # Bar of mean y by category x
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, estimator="mean")
            ax.set_title(f"Mean {y} by {x}")
            ax.set_xlabel(str(x))
            ax.set_ylabel(f"Mean {y}")

    elif chart_type == "line":
        # Line chart, typically for time series or ordered x
        if y is None:
            raise ValueError("Line plot requires both x and y")
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f"Line plot of {y} over {x}")
        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))

    elif chart_type == "pie":
        # Composition chart for categorical data
        # Use x as category
        counts = df[x].value_counts(dropna=False)
        labels = counts.index.astype(str).tolist()
        sizes = counts.values.tolist()
        ax.pie(sizes, labels=labels, autopct="%0.1f%%")
        ax.set_title(f"Composition of {x}")
        ax.axis("equal")

    else:
        plt.close(fig)
        raise ValueError(f"Unsupported chart_type '{chart_type}' in renderer")

    # Minimal styling consistent with your notes:
    # clear axes labels, titles, and no chart junk. :contentReference[oaicite:2]{index=2}
    fig.tight_layout()

    # Save plot to disk
    filename = f"{uuid.uuid4().hex}_{chart_type}.png"
    file_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(file_path)
    plt.close(fig)

    # Store in cache
    _PLOT_CACHE[cache_key] = file_path
    _save_plot_cache()

    return {
        "file_path": file_path,
        "chart_type": chart_type,
        "dataset_id": dataset_id,
        "x": x,
        "y": y,
        "hue": hue,
        "bins": bins,
        "role": _chart_role(chart_type),
        "reused": False,
        "message": "Visualization rendered successfully.",
    }


def eda_viz_spec_tool(
    dataset_id: str,
    chart_type: str,
    x: str,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Tool wrapper to construct a visualization specification for the dataset.
    Uses Pydantic validation to ensure all parameters are correct.

    Args:
        dataset_id: Unique identifier for the dataset
        chart_type: Type of chart (histogram, box, boxplot, scatter, bar, line, pie)
        x: Column name for x-axis
        y: Column name for y-axis (optional, depends on chart type)
        hue: Column name for color grouping (optional)
        bins: Number of bins for histograms (default 10)

    Returns:
        Dictionary containing the validated visualization specification
    """
    try:
        # Validate inputs using Pydantic schema
        spec = VizSpec(
            dataset_id=dataset_id,
            chart_type=cast(
                ChartType, chart_type
            ),  # Cast to satisfy static type checker
            x=x,
            y=y,
            hue=hue,
            bins=bins,
        )

        # Additional validation: check columns exist in dataset
        df = get_dataset(dataset_id)
        available_columns = df.columns.tolist()

        spec.x = validate_column_exists(spec.x, available_columns)
        if spec.y:
            spec.y = validate_column_exists(spec.y, available_columns)
        if spec.hue:
            spec.hue = validate_column_exists(spec.hue, available_columns)

        # Add role based on chart type
        spec_dict = spec.model_dump()
        chart_type_value = (
            spec_dict["chart_type"].value
            if hasattr(spec_dict["chart_type"], "value")
            else spec_dict["chart_type"]
        )
        spec_dict["role"] = _chart_role(chart_type_value)

        # Return as dict for tool compatibility
        return wrap_success(spec_dict)

    except Exception as e:
        return exception_to_error(
            VALIDATION_ERROR,
            e,
            hint="Check chart_type and column names exist in dataset",
        )


async def eda_render_plot_tool(tool_context: ToolContext, spec: Dict[str, Any]) -> Any:
    """
    Tool wrapper to render a plot from a visualization specification.
    Validates the spec using Pydantic before rendering and returns an ADK Part
    object that the web UI can display as an artifact.

    Returns:
        Part: ADK media Part object containing the image data for display in the web UI
    """
    try:
        # Validate spec using Pydantic
        validated_spec = VizSpec(**spec)

        # Render the plot (will reuse cached version if spec repeats)
        result = render_plot_from_spec(validated_spec.model_dump())

        # If this spec was already rendered, avoid re-saving the same artifact.
        if result.get("reused"):
            validated_result = VizResult(**result)
            return wrap_success(
                {
                    "artifact_filename": os.path.basename(validated_result.file_path),
                    "artifact_version": None,  # Not re-saved to avoid duplicate chart spam
                    "mime_type": "image/png",
                    "message": result.get(
                        "message",
                        "Reused previously rendered visualization (not re-saved)",
                    ),
                    "dataset_id": validated_result.dataset_id,
                    "chart_type": validated_result.chart_type.value,
                    "x": validated_result.x,
                    "y": validated_result.y,
                    "hue": validated_result.hue,
                    "bins": validated_result.bins,
                    "role": validated_result.role,
                    "reused": True,
                }
            )

        # Validate result (new render)
        validated_result = VizResult(**result)

        # Read the image file and create an ADK Part for artifact display
        file_path = validated_result.file_path
        filename = os.path.basename(file_path)

        # Read the file as binary data
        with open(file_path, "rb") as f:
            image_data = f.read()

        # Create a Part object with the image
        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type="image/png",
        )

        # Persist the artifact so it shows up in the Dev UI Artifacts panel
        try:
            version = await tool_context.save_artifact(
                filename=filename, artifact=image_part
            )
        except Exception:
            # If saving fails, still return the Part for best-effort display
            return image_part

        # Return lightweight metadata for the LLM to reference
        return wrap_success(
            {
                "artifact_filename": filename,
                "artifact_version": version,
                "mime_type": "image/png",
                "message": result.get("message", "Visualization saved as artifact"),
                "dataset_id": validated_result.dataset_id,
                "chart_type": validated_result.chart_type.value,
                "x": validated_result.x,
                "y": validated_result.y,
                "hue": validated_result.hue,
                "bins": validated_result.bins,
                "role": validated_result.role,
                "reused": result.get("reused", False),
            }
        )

    except Exception as e:
        return exception_to_error(
            RENDER_ERROR,
            e,
            hint="Verify columns are numeric/categorical as required for the chosen chart type",
        )


# -----------------------------------------------------------------------------
# Long-Running Operation (LRO) Tools for Outlier Visualization
# These use ADK's request_confirmation() pattern to pause for user input
# -----------------------------------------------------------------------------


def check_outlier_comparison_tool(
    tool_context: ToolContext,
    dataset_id: str,
    outlier_pct: float,
    columns_with_outliers: List[str],
) -> Dict[str, Any]:
    """LRO tool that pauses to ask user about outlier comparison visualization.

    This tool is called when data quality analysis detects a high outlier
    percentage (>10% by default). It pauses execution and presents the user
    with the option to see side-by-side visualizations comparing data with
    and without outliers.

    The tool handles three scenarios:
    1. First call (no confirmation yet): Pauses and asks user for decision
    2. Resume with approval: Returns signal to create comparison visualizations
    3. Resume with rejection: Returns signal to skip comparison

    Args:
        tool_context: ADK-provided context for LRO operations
        dataset_id: ID of the dataset being analyzed
        outlier_pct: Overall outlier percentage across numeric columns
        columns_with_outliers: List of column names with detected outliers

    Returns:
        Dictionary with status and action to take:
        - status: "pending" | "approved" | "rejected"
        - action: "create_comparison" | "skip_comparison"
    """
    # Format columns for display (limit to first 6)
    cols_preview = ", ".join(columns_with_outliers[:6])
    if len(columns_with_outliers) > 6:
        cols_preview += f", ... (+{len(columns_with_outliers) - 6} more)"

    # SCENARIO 1: First call - no confirmation yet, pause and ask user
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=(
                f"📈 **High Outlier Rate Detected**\n\n"
                f"**Outlier Percentage:** {outlier_pct:.1%} of values\n"
                f"**Threshold:** {OUTLIER_COMPARISON_THRESHOLD:.0%}\n\n"
                f"**Columns with outliers:**\n  {cols_preview}\n\n"
                "Would you like to see **side-by-side comparison** visualizations?\n"
                "This will show each affected column with and without outliers.\n\n"
                "• **Approve** → Generate comparison plots\n"
                "• **Reject** → Continue with standard visualization"
            ),
            payload={
                "dataset_id": dataset_id,
                "outlier_pct": outlier_pct,
                "columns": columns_with_outliers[:6],  # Limit payload size
            },
        )
        return {
            "ok": True,
            "status": "pending",
            "message": "Awaiting user decision on outlier comparison visualization",
            "outlier_pct": outlier_pct,
            "columns_count": len(columns_with_outliers),
        }

    # SCENARIO 2 & 3: Resuming after user response
    if tool_context.tool_confirmation.confirmed:
        # User approved - create comparison visualizations
        return wrap_success(
            {
                "status": "approved",
                "action": "create_comparison",
                "dataset_id": dataset_id,
                "columns": columns_with_outliers,
                "message": (
                    f"User approved outlier comparison visualization for "
                    f"{len(columns_with_outliers)} column(s)."
                ),
            }
        )
    else:
        # User rejected - skip comparison
        return wrap_success(
            {
                "status": "rejected",
                "action": "skip_comparison",
                "dataset_id": dataset_id,
                "message": "User chose to skip outlier comparison visualization.",
            }
        )


async def create_comparison_viz_tool(
    tool_context: ToolContext,
    dataset_id: str,
    column: str,
    chart_type: str = "box",
) -> Dict[str, Any]:
    """Create side-by-side visualization comparing data with and without outliers.

    Generates a figure with two subplots:
    - Left: Full data including outliers, labeled "Including outliers (n=X)"
    - Right: Data with outliers excluded, labeled "Outliers excluded (n=X)"

    Args:
        tool_context: ADK-provided context for artifact saving
        dataset_id: ID of the dataset to visualize
        column: Column name to create comparison for
        chart_type: Type of chart ("box" or "histogram")

    Returns:
        Dictionary with artifact information and comparison statistics
    """
    try:
        df = get_dataset(dataset_id)

        if column not in df.columns:
            available = ", ".join(df.columns[:10])
            return exception_to_error(
                VALIDATION_ERROR,
                ValueError(f"Column '{column}' not found"),
                hint=f"Available columns: {available}...",
            )

        series = df[column].dropna()

        if not np.issubdtype(series.dtype, np.number):
            return exception_to_error(
                VALIDATION_ERROR,
                ValueError(f"Column '{column}' is not numeric"),
                hint="Outlier comparison requires numeric columns",
            )

        # Calculate outlier bounds using IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Create filtered data (without outliers)
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        series_no_outliers = series[~outlier_mask]

        n_total = len(series)
        n_outliers = outlier_mask.sum()
        n_clean = len(series_no_outliers)

        # Create side-by-side figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if chart_type.lower() == "histogram":
            # Histogram comparison
            bins = min(30, int(np.sqrt(n_total)))
            sns.histplot(series, bins=bins, ax=axes[0], color="steelblue")
            axes[0].set_title(f"Including outliers (n={n_total:,})")
            axes[0].set_xlabel(column)
            axes[0].set_ylabel("Count")

            sns.histplot(series_no_outliers, bins=bins, ax=axes[1], color="seagreen")
            axes[1].set_title(f"Outliers excluded (n={n_clean:,})")
            axes[1].set_xlabel(column)
            axes[1].set_ylabel("Count")
        else:
            # Box plot comparison (default)
            sns.boxplot(y=series, ax=axes[0], color="steelblue")
            axes[0].set_title(f"Including outliers (n={n_total:,})")
            axes[0].set_ylabel(column)

            sns.boxplot(y=series_no_outliers, ax=axes[1], color="seagreen")
            axes[1].set_title(f"Outliers excluded (n={n_clean:,})")
            axes[1].set_ylabel(column)

        # Add overall title
        fig.suptitle(
            f"Outlier Comparison: {column} ({n_outliers:,} outliers removed)",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()

        # Save plot to disk
        filename = f"{uuid.uuid4().hex}_comparison_{column}.png"
        file_path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(file_path, dpi=100)
        plt.close(fig)

        # Read and save as artifact
        with open(file_path, "rb") as f:
            image_data = f.read()

        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type="image/png",
        )

        try:
            version = await tool_context.save_artifact(
                filename=filename, artifact=image_part
            )
        except Exception:
            version = None

        return wrap_success(
            {
                "artifact_filename": filename,
                "artifact_version": version,
                "mime_type": "image/png",
                "message": f"Comparison visualization created for '{column}'",
                "dataset_id": dataset_id,
                "column": column,
                "chart_type": chart_type,
                "comparison_stats": {
                    "total_values": n_total,
                    "outliers_removed": int(n_outliers),
                    "clean_values": n_clean,
                    "outlier_pct": float(n_outliers / n_total) if n_total > 0 else 0,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                },
            }
        )

    except Exception as e:
        return exception_to_error(
            RENDER_ERROR,
            e,
            hint="Check that the column is numeric and the dataset exists",
        )

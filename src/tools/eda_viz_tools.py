import os
import uuid
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import seaborn as sns
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ..utils.data_store import get_dataset
from ..utils.errors import exception_to_error, wrap_success
from ..utils.schemas import ChartType, VizResult, VizSpec, validate_column_exists

PLOTS_DIR = "/tmp/data_whisperer_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


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

    return {
        "file_path": file_path,
        "chart_type": chart_type,
        "dataset_id": dataset_id,
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

        # Return as dict for tool compatibility
        return wrap_success(spec.model_dump())

    except Exception as e:
        return exception_to_error(
            "validation_error",
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

        # Render the plot
        result = render_plot_from_spec(validated_spec.model_dump())

        # Validate result
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
                "message": "Visualization saved as artifact",
                "chart_type": validated_result.chart_type.value,
                "dataset_id": validated_result.dataset_id,
            }
        )

    except Exception as e:
        return exception_to_error(
            "render_error",
            e,
            hint="Verify columns are numeric/categorical as required for the chosen chart type",
        )

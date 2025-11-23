import os
import uuid
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.data_store import get_dataset

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


def build_viz_spec(
    dataset_id: str,
    chart_type: str,
    x: str,
    y: Optional[str],
    hue: Optional[str],
    bins: int,
) -> Dict[str, Any]:
    """
    Internal helper to construct a visualization specification
    (for example, histogram, boxplot, scatterplot, bar chart).

    This does not render the plot. It just validates inputs and packs a spec
    that eda_render_plot_tool can use.
    """
    df = get_dataset(dataset_id)
    chart_type = chart_type.lower()

    # Basic validation of columns
    cols = set(df.columns)

    if x not in cols:
        raise ValueError(f"x column '{x}' not found in dataset")

    if y is not None and y not in cols:
        raise ValueError(f"y column '{y}' not found in dataset")

    if hue is not None and hue not in cols:
        raise ValueError(f"hue column '{hue}' not found in dataset")

    # Simple validation of chart_type
    allowed_chart_types = {
        "histogram",
        "box",
        "boxplot",
        "scatter",
        "bar",
        "line",
        "pie",
    }
    if chart_type not in allowed_chart_types:
        raise ValueError(
            f"Unsupported chart_type '{chart_type}'. "
            f"Allowed types: {sorted(allowed_chart_types)}"
        )

    role = _chart_role(chart_type)

    spec: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "chart_type": chart_type,
        "role": role,
        "x": x,
        "y": y,
        "hue": hue,
        "bins": bins,
        # Helpful context for the viz agent if it wants to describe the chart
        "columns_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "n_rows": int(len(df)),
    }

    return spec


def render_plot_from_spec(spec: Dict[str, Any]) -> str:
    """
    Internal helper that takes a visualization spec, renders a plot
    using matplotlib / seaborn, and returns a file path to the image.
    """
    dataset_id = spec["dataset_id"]
    chart_type = spec["chart_type"]
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")
    bins = spec.get("bins", 10)

    df = get_dataset(dataset_id)

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

    return file_path


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

    The visualization agent should:
    - Choose chart_type based on the user question and data type
      (distribution, relationship, comparison, composition). :contentReference[oaicite:3]{index=3}
    - Provide x, y, and hue columns as needed.
    """
    return build_viz_spec(
        dataset_id=dataset_id,
        chart_type=chart_type,
        x=x,
        y=y,
        hue=hue,
        bins=bins,
    )


def eda_render_plot_tool(spec: Dict[str, Any]) -> str:
    """
    Tool wrapper to render a plot from a visualization specification
    and return a file path to the generated image.
    """
    return render_plot_from_spec(spec)

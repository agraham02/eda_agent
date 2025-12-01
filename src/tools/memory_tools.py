"""Memory and preference tools for long-term storage and cross-run comparison.

These tools allow the agent to:
- Save and load user preferences
- List past analysis runs for a dataset
- Compare two analysis runs
- Load datasets from persistent storage
"""

from typing import Any, Dict, List, Optional

from google.adk.tools.tool_context import ToolContext

from ..utils.errors import (
    DATASET_NOT_FOUND,
    INVALID_PARAMETER,
    make_error,
    wrap_success,
)
from ..utils.persistent_store import (
    AnalysisRun,
    PlotDensity,
    RunType,
    StructuredResults,
    UserPreferences,
    WritingStyle,
    get_store,
)

# ============================================================================
# USER PREFERENCE TOOLS
# ============================================================================


def save_preferences_tool(
    tool_context: ToolContext,
    writing_style: str = "technical",
    default_alpha: float = 0.05,
    plot_density: str = "comprehensive",
    auto_quality_check: bool = True,
) -> Dict[str, Any]:
    """
    Save user preferences for personalized analysis.

    Args:
        writing_style: "executive" (concise) or "technical" (detailed)
        default_alpha: Significance level for hypothesis tests (0.001-0.5)
        plot_density: "minimal" or "comprehensive"
        auto_quality_check: Whether to run quality check before inference

    Returns:
        Confirmation with saved preferences
    """
    # Validate inputs
    try:
        style = WritingStyle(writing_style.lower())
    except ValueError:
        return make_error(
            INVALID_PARAMETER,
            f"Invalid writing_style: {writing_style}",
            hint="Use 'executive' or 'technical'",
        )

    try:
        density = PlotDensity(plot_density.lower())
    except ValueError:
        return make_error(
            INVALID_PARAMETER,
            f"Invalid plot_density: {plot_density}",
            hint="Use 'minimal' or 'comprehensive'",
        )

    if not 0.001 <= default_alpha <= 0.5:
        return make_error(
            INVALID_PARAMETER,
            f"default_alpha must be between 0.001 and 0.5, got {default_alpha}",
        )

    # Get user_id from session state if available
    user_id = "default"
    if hasattr(tool_context, "state") and tool_context.state:
        user_id = tool_context.state.get("user:id", "default")

    prefs = UserPreferences(
        user_id=user_id,
        writing_style=style,
        default_alpha=default_alpha,
        plot_density=density,
        auto_quality_check=auto_quality_check,
    )

    store = get_store()
    saved = store.save_preferences(prefs)

    # Also store in session state for quick access
    tool_context.state["user:writing_style"] = saved.writing_style.value
    tool_context.state["user:default_alpha"] = saved.default_alpha
    tool_context.state["user:plot_density"] = saved.plot_density.value
    tool_context.state["user:auto_quality_check"] = saved.auto_quality_check

    return wrap_success(
        {
            "message": "Preferences saved successfully",
            "preferences": {
                "writing_style": saved.writing_style.value,
                "default_alpha": saved.default_alpha,
                "plot_density": saved.plot_density.value,
                "auto_quality_check": saved.auto_quality_check,
            },
        }
    )


def load_preferences_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Load user preferences from storage.

    Returns current preferences or defaults if none saved.
    """
    user_id = "default"
    if hasattr(tool_context, "state") and tool_context.state:
        user_id = tool_context.state.get("user:id", "default")

    store = get_store()
    prefs = store.get_preferences(user_id)

    # Store in session state for quick access
    tool_context.state["user:writing_style"] = prefs.writing_style.value
    tool_context.state["user:default_alpha"] = prefs.default_alpha
    tool_context.state["user:plot_density"] = prefs.plot_density.value
    tool_context.state["user:auto_quality_check"] = prefs.auto_quality_check

    return wrap_success(
        {
            "preferences": {
                "writing_style": prefs.writing_style.value,
                "default_alpha": prefs.default_alpha,
                "plot_density": prefs.plot_density.value,
                "auto_quality_check": prefs.auto_quality_check,
            },
            "updated_at": prefs.updated_at.isoformat(),
        }
    )


# ============================================================================
# ANALYSIS RUN TOOLS
# ============================================================================


def list_past_analyses_tool(
    dataset_id: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    List past analysis runs.

    Args:
        dataset_id: Filter to runs for this dataset (optional)
        limit: Maximum number of runs to return (default 10)

    Returns:
        List of past analysis runs with metadata
    """
    store = get_store()

    if dataset_id:
        runs = store.get_runs_for_dataset(dataset_id, limit=limit)
    else:
        runs = store.get_recent_runs(limit=limit)

    run_summaries = [
        {
            "run_id": run.run_id,
            "dataset_id": run.dataset_id,
            "run_type": run.run_type.value,
            "user_question": run.user_question[:100]
            + ("..." if len(run.user_question) > 100 else ""),
            "readiness_score": (
                run.readiness_score.get("overall") if run.readiness_score else None
            ),
            "created_at": run.created_at.isoformat(),
            "has_summary": bool(run.summary_markdown),
        }
        for run in runs
    ]

    return wrap_success(
        {
            "count": len(run_summaries),
            "runs": run_summaries,
        }
    )


def get_analysis_run_tool(run_id: str) -> Dict[str, Any]:
    """
    Get full details of a specific analysis run.

    Args:
        run_id: The run ID to retrieve

    Returns:
        Full analysis run details including summary and results
    """
    store = get_store()
    run = store.get_run(run_id)

    if not run:
        return make_error(
            DATASET_NOT_FOUND,
            f"Run {run_id} not found",
            hint="Use list_past_analyses_tool to find valid run IDs",
        )

    return wrap_success(
        {
            "run_id": run.run_id,
            "dataset_id": run.dataset_id,
            "run_type": run.run_type.value,
            "user_question": run.user_question,
            "summary_markdown": run.summary_markdown,
            "structured_results": run.structured_results.model_dump(),
            "readiness_score": run.readiness_score,
            "created_at": run.created_at.isoformat(),
            "session_id": run.session_id,
        }
    )


def compare_runs_tool(run_id_a: str, run_id_b: str) -> Dict[str, Any]:
    """
    Compare two analysis runs and show differences.

    Args:
        run_id_a: First run ID (typically older)
        run_id_b: Second run ID (typically newer)

    Returns:
        Comparison showing readiness delta, p-value changes, and summary previews
    """
    store = get_store()
    comparison = store.compare_runs(run_id_a, run_id_b)

    if not comparison:
        return make_error(
            DATASET_NOT_FOUND,
            f"One or both runs not found: {run_id_a}, {run_id_b}",
            hint="Use list_past_analyses_tool to find valid run IDs",
        )

    return wrap_success(comparison)


# ============================================================================
# DATASET MANAGEMENT TOOLS
# ============================================================================


def list_persisted_datasets_tool() -> Dict[str, Any]:
    """
    List all datasets in persistent storage.

    Returns datasets saved across sessions, including lineage information.
    """
    store = get_store()
    datasets = store.list_datasets()

    dataset_summaries = [
        {
            "dataset_id": ds.dataset_id,
            "filename": ds.filename,
            "shape": [ds.n_rows, ds.n_columns],
            "ingested_at": ds.ingested_at.isoformat(),
            "parent_dataset_id": ds.parent_dataset_id,
            "transformation_note": ds.transformation_note,
        }
        for ds in datasets
    ]

    return wrap_success(
        {
            "count": len(dataset_summaries),
            "datasets": dataset_summaries,
        }
    )


def get_dataset_lineage_tool(dataset_id: str) -> Dict[str, Any]:
    """
    Get the lineage (transformation history) for a dataset.

    Args:
        dataset_id: The dataset to trace lineage for

    Returns:
        List of datasets in the lineage chain, from current to original
    """
    store = get_store()
    lineage = store.get_dataset_lineage(dataset_id)

    if not lineage:
        return make_error(
            DATASET_NOT_FOUND,
            f"Dataset {dataset_id} not found in persistent storage",
            hint="Dataset may only exist in memory; run an analysis to persist it",
        )

    lineage_chain = [
        {
            "dataset_id": ds.dataset_id,
            "filename": ds.filename,
            "shape": [ds.n_rows, ds.n_columns],
            "transformation_note": ds.transformation_note,
            "parent_dataset_id": ds.parent_dataset_id,
        }
        for ds in lineage
    ]

    return wrap_success(
        {
            "dataset_id": dataset_id,
            "lineage_depth": len(lineage_chain),
            "lineage": lineage_chain,
        }
    )


# ============================================================================
# HELPER: SAVE ANALYSIS RUN (called by callback)
# ============================================================================


def save_analysis_run(
    dataset_id: str,
    user_question: str,
    run_type: str,
    summary_markdown: str = "",
    p_values: Optional[Dict[str, float]] = None,
    confidence_intervals: Optional[Dict[str, List[float]]] = None,
    effect_sizes: Optional[Dict[str, float]] = None,
    descriptive_highlights: Optional[Dict[str, Any]] = None,
    plot_paths: Optional[List[str]] = None,
    readiness_score: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> AnalysisRun:
    """
    Save an analysis run to persistent storage.

    This is called by the after_agent_callback to auto-save runs.
    """
    try:
        rt = RunType(run_type)
    except ValueError:
        rt = RunType.FULL

    structured = StructuredResults(
        p_values=p_values or {},
        confidence_intervals=confidence_intervals or {},
        effect_sizes=effect_sizes or {},
        descriptive_highlights=descriptive_highlights or {},
        plot_paths=plot_paths or [],
    )

    run = AnalysisRun(
        dataset_id=dataset_id,
        user_question=user_question,
        run_type=rt,
        summary_markdown=summary_markdown,
        structured_results=structured,
        readiness_score=readiness_score,
        session_id=session_id,
    )

    store = get_store()
    return store.save_run(run)

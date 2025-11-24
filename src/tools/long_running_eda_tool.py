import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

from ..utils.data_store import get_dataset
from ..utils.errors import make_error, wrap_success

# Keywords that suggest a comprehensive request
COMPREHENSIVE_KEYWORDS = {
    "full",
    "comprehensive",
    "everything",
    "all analyses",
    "end-to-end",
    "complete",
}


async def heavy_eda_long_running(
    tool_context: ToolContext, dataset_id: str, requested: List[str]
) -> Dict[str, Any]:
    """Initiate a long-running EDA workflow.

    This function is wrapped by LongRunningFunctionTool so the initial call
    produces a long-running function event (displayed as background progress in UI).

    It does NOT complete the whole EDA immediately. Instead it:
      - Records a plan in session state (temp:heavy_eda_plan)
      - Returns an operation id and planned steps
    Subsequent progress could be reported by a future enhancement (e.g., a polling
    or resume flow). For now the individual agent tools will still run, but the UI
    has a top-level long-running marker to anchor the workflow.
    """
    operation_id = f"eda-run-{uuid.uuid4().hex[:8]}"

    # Attempt to derive dataset size for auto-trigger heuristics
    row_count = None
    col_count = None
    try:
        df = get_dataset(dataset_id)
        row_count, col_count = df.shape
    except Exception:
        pass

    # Simple heuristic: map requested tokens to steps
    lower_req = [r.lower() for r in requested]
    steps: List[str] = []
    if any(k in lower_req for k in ("describe", "summary", "statistics")):
        steps.append("descriptive_statistics")
    if any(k in lower_req for k in ("test", "hypothesis", "inference")):
        steps.append("inference_tests")
    if any(
        k in lower_req for k in ("plot", "chart", "visual", "visualization", "graph")
    ):
        steps.append("visualizations")
    if any(k in lower_req for k in ("transform", "wrangle", "feature", "clean")):
        steps.append("data_transformation")
    if not steps:
        steps = ["descriptive_statistics", "visualizations"]  # default minimal plan

    # Auto-augment plan for large datasets
    LARGE_ROWS = 5000
    LARGE_COLS = 50
    if (row_count and row_count > LARGE_ROWS) or (col_count and col_count > LARGE_COLS):
        if "correlation_matrix" not in steps:
            steps.append("correlation_matrix")
        if "outlier_profile" not in steps:
            steps.append("outlier_profile")

    comprehensive = (len(steps) >= 3) or ((row_count or 0) > LARGE_ROWS)

    plan = {
        "operation_id": operation_id,
        "dataset_id": dataset_id,
        "planned_steps": steps,
        "raw_requested": requested,
        "row_count": row_count,
        "column_count": col_count,
        "comprehensive": comprehensive,
    }

    # Persist lightweight plan to session state (temporary namespace)
    try:
        tool_context.state[f"temp:heavy_eda_plan:{operation_id}"] = plan
        # Maintain an index of operation ids for later listing
        index_key = "temp:heavy_eda_index"
        try:
            existing_index = tool_context.state.get(index_key, [])
        except Exception:
            existing_index = []
        if operation_id not in existing_index:
            existing_index.append(operation_id)
        tool_context.state[index_key] = existing_index
    except Exception:
        # Non-fatal; return plan anyway
        pass

    return wrap_success(
        {
            "operation_id": operation_id,
            "status": "started",
            "planned_steps": steps,
            "row_count": row_count,
            "column_count": col_count,
            "comprehensive": comprehensive,
            "message": "Long-running EDA workflow initialized; sub-agent steps will follow.",
        }
    )


# Wrap with LongRunningFunctionTool so ADK marks it long running.
heavy_eda_long_running_tool = LongRunningFunctionTool(func=heavy_eda_long_running)


def _get_plan(tool_context: ToolContext, operation_id: str) -> Optional[Dict[str, Any]]:
    key_prefix = f"temp:heavy_eda_plan:{operation_id}"
    return tool_context.state.get(key_prefix)


def _save_plan(
    tool_context: ToolContext, operation_id: str, plan: Dict[str, Any]
) -> None:
    tool_context.state[f"temp:heavy_eda_plan:{operation_id}"] = plan


def heavy_eda_progress_update_tool(
    tool_context: ToolContext,
    operation_id: str,
    status: str,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a progress update to a long-running EDA plan.

    Stores updates under plan['updates'] as a list of entries:
    {timestamp, status, detail}.
    """
    plan = _get_plan(tool_context, operation_id)
    if not plan:
        return make_error(
            "not_found",
            f"Operation '{operation_id}' not found",
            hint="Verify operation_id from initial start response",
        )

    updates = plan.setdefault("updates", [])
    updates.append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": status,
            "detail": detail,
        }
    )
    plan["last_update"] = status
    _save_plan(tool_context, operation_id, plan)
    return wrap_success(
        {
            "operation_id": operation_id,
            "status": status,
            "detail": detail,
            "updates_count": len(updates),
        }
    )


def heavy_eda_cancel_tool(
    tool_context: ToolContext,
    operation_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Cancel a long-running EDA plan (soft cancel)."""
    plan = _get_plan(tool_context, operation_id)
    if not plan:
        return make_error(
            "not_found",
            f"Operation '{operation_id}' not found",
            hint="Check active plans before cancelling",
        )
    if plan.get("status") == "cancelled":
        return wrap_success(
            {"operation_id": operation_id, "status": "already_cancelled"}
        )

    plan["status"] = "cancelled"
    plan["cancelled_at"] = datetime.utcnow().isoformat() + "Z"
    if reason:
        plan["cancel_reason"] = reason
    _save_plan(tool_context, operation_id, plan)
    return wrap_success(
        {"operation_id": operation_id, "status": "cancelled", "reason": reason}
    )


def heavy_eda_find_active_tool(
    tool_context: ToolContext, dataset_id: Optional[str] = None
) -> Dict[str, Any]:
    """List active (non-cancelled) heavy EDA plans, optionally filtered by dataset_id.

    Uses an index stored at temp:heavy_eda_index to avoid iterating over internal state.
    """
    try:
        index = tool_context.state.get("temp:heavy_eda_index", [])
    except Exception:
        index = []
    active = []
    for op_id in index:
        key = f"temp:heavy_eda_plan:{op_id}"
        try:
            plan = tool_context.state.get(key)
        except Exception:
            plan = None
        if not plan:
            continue
        if plan.get("status") == "cancelled":
            continue
        if dataset_id and plan.get("dataset_id") != dataset_id:
            continue
        active.append(
            {
                "operation_id": plan.get("operation_id"),
                "dataset_id": plan.get("dataset_id"),
                "status": plan.get("status"),
                "planned_steps": plan.get("planned_steps"),
                "comprehensive": plan.get("comprehensive"),
                "last_update": plan.get("last_update"),
            }
        )
    return wrap_success({"active_plans": active, "count": len(active)})

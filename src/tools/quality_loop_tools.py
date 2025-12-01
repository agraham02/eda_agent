"""Tools for quality improvement loop control.

Provides tools for:
1. Exit signal when dataset quality thresholds are met (LoopAgent termination)
2. LRO (Long-Running Operation) tool that pauses to ask user about quality loop

The LRO pattern uses ADK's ToolContext.request_confirmation() to pause agent
execution and wait for human input before proceeding.
"""

from typing import Any, Dict, List

from google.adk.tools.tool_context import ToolContext

from ..utils.consts import QUALITY_LOOP_THRESHOLD, UserDecision
from ..utils.errors import wrap_success


def exit_quality_loop() -> Dict[str, str]:
    """Signal that quality thresholds are satisfied and the loop can terminate.

    Returns a small status payload; LoopAgent will stop after this tool run
    because the refinement agent will not request further changes.
    """
    return wrap_success(
        {
            "status": "quality_accept",
            "message": "Quality thresholds satisfied. Exiting loop.",
        }
    )


def offer_quality_loop_tool(
    tool_context: ToolContext,
    dataset_id: str,
    readiness_score: float,
    quality_issues: List[str],
) -> Dict[str, Any]:
    """LRO tool that pauses to ask user about running the quality improvement loop.

    This tool uses ADK's request_confirmation() to pause agent execution and
    present the user with a choice after data quality analysis is complete.

    The tool handles three scenarios:
    1. First call (no confirmation yet): Pauses and asks user for decision
    2. Resume with approval: Returns signal to run quality improvement loop
    3. Resume with rejection: Returns signal to continue without loop

    Args:
        tool_context: ADK-provided context for LRO operations
        dataset_id: ID of the dataset being analyzed
        readiness_score: Current readiness score (0-100)
        quality_issues: List of identified quality issues

    Returns:
        Dictionary with status and action to take:
        - status: "pending" | "approved" | "rejected"
        - action: "run_quality_loop" | "continue_without_loop"
    """
    # Determine readiness band for user-friendly display
    if readiness_score >= 90:
        band = "Ready"
        band_emoji = "✅"
    elif readiness_score >= 75:
        band = "Minor fixes needed"
        band_emoji = "🔧"
    elif readiness_score >= 50:
        band = "Needs work"
        band_emoji = "⚠️"
    else:
        band = "Not ready"
        band_emoji = "🚨"

    # Format issues for display (limit to first 5)
    issues_preview = "\n".join(f"  • {issue}" for issue in quality_issues[:5])
    if len(quality_issues) > 5:
        issues_preview += f"\n  • ... and {len(quality_issues) - 5} more issues"

    # SCENARIO 1: First call - no confirmation yet, pause and ask user
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=(
                f"📊 **Data Quality Assessment Complete**\n\n"
                f"**Readiness Score:** {readiness_score:.0f}/100 {band_emoji} ({band})\n"
                f"**Threshold for auto-continue:** {QUALITY_LOOP_THRESHOLD}/100\n\n"
                f"**Key Issues Found:**\n{issues_preview if quality_issues else '  • No major issues detected'}\n\n"
                "**What would you like to do?**\n"
                "• **Approve** → Run automatic quality improvement loop\n"
                "• **Reject** → Continue analysis without improvements"
            ),
            payload={
                "dataset_id": dataset_id,
                "readiness_score": readiness_score,
                "quality_issues": quality_issues[:5],  # Limit payload size
            },
        )
        return {
            "ok": True,
            "status": "pending",
            "message": "Awaiting user decision on quality improvement loop",
            "readiness_score": readiness_score,
            "band": band,
        }

    # SCENARIO 2 & 3: Resuming after user response
    if tool_context.tool_confirmation.confirmed:
        # User approved - run quality loop
        return wrap_success(
            {
                "status": "approved",
                "action": UserDecision.RUN_LOOP.value,
                "dataset_id": dataset_id,
                "message": (
                    f"User approved quality improvement loop. "
                    f"Current readiness: {readiness_score:.0f}/100."
                ),
            }
        )
    else:
        # User rejected - continue without loop
        return wrap_success(
            {
                "status": "rejected",
                "action": UserDecision.CONTINUE.value,
                "dataset_id": dataset_id,
                "message": (
                    f"User chose to continue without quality loop. "
                    f"Proceeding with readiness: {readiness_score:.0f}/100."
                ),
            }
        )

"""
Tools for the summary agent.

The summary agent needs at least one tool to avoid ADK framework issues
with agents that have no tools returning None for content.parts.
"""

from typing import Any, Dict

from ..utils.errors import VALIDATION_ERROR, make_error, wrap_success


def _validate_summary(summary_text: str) -> Dict[str, Any]:
    stripped = summary_text.strip()
    errors = []
    if not stripped:
        errors.append("Summary text is empty.")
    # Required sections
    required_sections = [
        "## Data Signature",
        "## Key Findings",
        "## Model Readiness Assessment",
        "## 4. Recommendations",
    ]
    for section in required_sections:
        if section not in stripped:
            errors.append(f"Missing required section: {section}")
    if errors:
        return make_error(
            VALIDATION_ERROR,
            "; ".join(errors),
            hint="Ensure all required sections are present in the summary",
            context={"summary": summary_text},
        )
    return wrap_success({})


def finalize_summary_tool(summary_text: str) -> Dict[str, Any]:
    """
    Finalize and return the summary report.

    This tool serves as a structured way for the summary agent to return
    its final report, avoiding framework issues with tool-less agents.

    Args:
        summary_text: The complete markdown-formatted summary report

    Returns:
        Dict containing the finalized summary and success status
    """
    validation = _validate_summary(summary_text)
    if not validation.get("ok", True):
        return validation
    return wrap_success(
        {
            "summary": summary_text,
            "message": "Summary report generated successfully",
        }
    )

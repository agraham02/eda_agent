"""Tools for quality improvement loop control.

Provides a simple exit function tool that the refinement agent can call
when dataset quality thresholds are met to signal the LoopAgent to stop.
"""

from typing import Dict

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

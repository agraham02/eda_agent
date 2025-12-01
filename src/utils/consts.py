from enum import Enum

from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)


class StateKeys(str, Enum):
    """Standardized state key names for agent outputs."""

    INGESTION = "ingestion_output"
    DATA_QUALITY = "data_quality_output"
    OUTLIER_METADATA = "outlier_metadata"  # Stores outlier thresholds for reuse
    WRANGLE = "wrangle_output"
    DESCRIBE = "describe_output"
    INFERENCE = "inference_output"
    VIZ = "viz_output"
    SUMMARY = "final_summary"


# -----------------------------------------------------------------------------
# Long-Running Operation (LRO) Thresholds
# These control when the agent pauses to ask for human input
# -----------------------------------------------------------------------------

# Readiness score threshold below which quality loop is offered
QUALITY_LOOP_THRESHOLD = 85

# Outlier percentage threshold above which comparison viz is offered
OUTLIER_COMPARISON_THRESHOLD = 0.10  # 10%


class UserDecision(str, Enum):
    """Possible user decisions for LRO prompts."""

    RUN_LOOP = "run_loop"  # User wants to run quality improvement loop
    CONTINUE = "continue"  # User wants to continue without changes
    END = "end"  # User wants to end agent execution

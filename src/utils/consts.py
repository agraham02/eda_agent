from enum import Enum

from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)


class StateKeys(str, Enum):
    """Standardized state key names for agent outputs."""

    INGESTION = "ingestion_output"
    DATA_QUALITY = "data_quality_output"
    WRANGLE = "wrangle_output"
    DESCRIBE = "describe_output"
    INFERENCE = "inference_output"
    VIZ = "viz_output"
    SUMMARY = "final_summary"

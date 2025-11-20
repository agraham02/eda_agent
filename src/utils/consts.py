from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)

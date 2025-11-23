from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ....tools.eda_describe_tools import (
    eda_bivariate_summary_tool,
    eda_correlation_matrix_tool,
    eda_univariate_summary_tool,
)
from ....utils.consts import retry_config

eda_describe_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_describe_agent",
    description=(
        "Performs descriptive statistics and feature summarization "
        "for one or more columns in a dataset."
    ),
    instruction=(
        """You are the EDA summary specialist.

Use eda_univariate_summary_tool for single-column summaries,
eda_bivariate_summary_tool for relationships between two columns,
and eda_correlation_matrix_tool for correlation matrices.
Explain results clearly in natural language."""
    ),
    tools=[
        eda_univariate_summary_tool,
        eda_bivariate_summary_tool,
        eda_correlation_matrix_tool,
    ],
)

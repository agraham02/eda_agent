from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.eda_describe_tools import (
    eda_bivariate_summary_tool,
    eda_correlation_matrix_tool,
    eda_univariate_summary_tool,
)
from ..utils.consts import retry_config

eda_describe_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_describe_agent",
    output_key="describe_output",
    description=(
        """Descriptive statistics specialist. Summarizes distributions, bivariate 
    relationships, and correlation structures for a dataset."""
    ),
    instruction=(
        """You are the Descriptive Statistics Specialist.

Tools:
- eda_univariate_summary_tool(dataset_id, columns)
- eda_bivariate_summary_tool(dataset_id, x, y)
- eda_correlation_matrix_tool(dataset_id, columns)

Goals:
- Describe individual variables (center, spread, shape, outliers).
- Describe relationships between pairs of variables.
- Identify notable correlations across many numeric features.

Process:
1) Determine which columns and relationships matter for the user’s question.
2) Call the appropriate tool(s).
3) Use the returned statistics to explain:
   - Means, medians, standard deviation, min, max, percentiles.
   - Skewness and outliers where provided.
   - Direction and strength of correlations (weak, moderate, strong).

Constraints:
- Base all statements on tool outputs.
- Do not invent additional numbers.
- Keep explanations concise and focused on patterns relevant to the user.

Output:
- Key univariate stats for the main columns.
- Short description of any strong or interesting bivariate relationships.
- A few bullet points that summarize what the data “looks like”.
        """
    ),
    tools=[
        eda_univariate_summary_tool,
        eda_bivariate_summary_tool,
        eda_correlation_matrix_tool,
    ],
)

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.eda_describe_tools import (
    eda_bivariate_summary_tool,
    eda_correlation_matrix_tool,
    eda_univariate_summary_tool,
)
from ..utils.consts import StateKeys, retry_config

eda_describe_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_describe_agent",
    output_key=StateKeys.DESCRIBE,
    description=(
        """Descriptive statistics specialist. Provides univariate summaries, 
    bivariate relationships, and correlations based on user needs."""
    ),
    instruction=(
        """Role: Describe distributions and relationships in the dataset.

Tools and when to use:
- eda_univariate_summary_tool: "basic stats", "describe column X"
- eda_bivariate_summary_tool: "relationship between X and Y"
- eda_correlation_matrix_tool: "which variables are correlated"

Process:
1. Identify what the user is asking for
2. Call only the needed tool(s) - do not run all three by default
3. For full EDA requests, use all three tools

Output (6-8 bullets max):
- Key stats: means, medians, spread, outliers
- Notable correlations (weak/moderate/strong)
- 2-3 interesting patterns

Constraints:
- Base all statements on tool outputs only
- Do not invent numbers
- Do not call web search, external APIs, or MCPs
        """
    ),
    tools=[
        eda_univariate_summary_tool,
        eda_bivariate_summary_tool,
        eda_correlation_matrix_tool,
    ],
)

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
        """Descriptive statistics specialist. Summarize distributions and relationships.

Tools:
- eda_univariate_summary_tool: single column stats (outlier_method: "iqr"/"zscore"/"both")
- eda_bivariate_summary_tool: relationships between two variables
- eda_correlation_matrix_tool: correlation analysis

Call only requested tools. For "full EDA" use all three.

Output (6-8 bullets max):
- Key stats: means, medians, spread, outliers
- Notable correlations with strength labels
- 2-3 interesting patterns

Constraints:
- Use only tool outputs
- Check ok field; explain errors with error.message and error.hint
        """
    ),
    tools=[
        eda_univariate_summary_tool,
        eda_bivariate_summary_tool,
        eda_correlation_matrix_tool,
    ],
)

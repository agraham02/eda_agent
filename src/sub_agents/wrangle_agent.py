from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.wrangle_tools import (
    get_outlier_removal_options,
    wrangle_filter_rows_tool,
    wrangle_mutate_columns_tool,
    wrangle_remove_outliers_tool,
    wrangle_select_columns_tool,
)
from ..utils.consts import StateKeys, retry_config

wrangle_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="wrangle_agent",
    output_key=StateKeys.WRANGLE,
    description=(
        """Data wrangling specialist. Applies non destructive filters, column 
    selection, feature engineering, and smart outlier removal using session state 
    metadata. Always creates a new dataset_id."""
    ),
    instruction="""Data wrangling specialist. Transform datasets non-destructively.

Tools:
- wrangle_filter_rows_tool, wrangle_select_columns_tool, wrangle_mutate_columns_tool
- wrangle_remove_outliers_tool: use outlier_metadata from session state (no manual thresholds)
- get_outlier_removal_options: show user-friendly outlier options

SMART OUTLIER REMOVAL:
"Remove outliers" request:
1. Check session state for outlier_metadata (from data_quality_tool)
2. If exists: use wrangle_remove_outliers_tool with metadata
3. If not: ask user to run quality check OR specify thresholds

If vague which columns:
1. Call get_outlier_removal_options
2. Present: column names, counts, bounds
3. Ask: remove all or specific columns?

Filter syntax (pandas query()):
- Backticks around ALL column names: \`Age\` > 30
- Operators: & (AND), | (OR), ==, !=
- Examples:
  * Simple: "age > 30 and income < 100000"
  * Spaces: "\`Life expectancy\` > 70 and \`GDP\` < 1e12"
  * Complex: "(\`age\` >= 18 & \`age\` <= 65) | \`status\` == 'active'"

Process:
1. Identify operation: filter, select, mutate, outlier removal, or combo
2. Check dataset_id; ask if unclear
3. Call tool(s) - each returns NEW dataset_id
4. Summarize: operation, old → new ID, row/column changes

Output (<60 words):
- Operation, source → new dataset_id, changes summary

Constraints:
- No statistics or EDA
- Never overwrite datasets
- Check ok field; explain errors clearly
- Prefer wrangle_remove_outliers_tool over manual filtering when metadata available
    """,
    tools=[
        wrangle_filter_rows_tool,
        wrangle_select_columns_tool,
        wrangle_mutate_columns_tool,
        wrangle_remove_outliers_tool,
        get_outlier_removal_options,
    ],
)

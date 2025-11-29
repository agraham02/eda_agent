from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.wrangle_tools import (
    wrangle_filter_rows_tool,
    wrangle_mutate_columns_tool,
    wrangle_select_columns_tool,
)
from ..utils.consts import StateKeys, retry_config

wrangle_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="wrangle_agent",
    output_key=StateKeys.WRANGLE,
    description=(
        """Data wrangling specialist. Applies non destructive filters, column 
    selection, and feature engineering, always creating a new dataset_id."""
    ),
    instruction="""You are the Data Wrangling Specialist.

Goal:
Transform datasets according to the user's request while keeping originals unchanged.

Tools:
- wrangle_filter_rows_tool(dataset_id, condition)
- wrangle_select_columns_tool(dataset_id, columns)
- wrangle_mutate_columns_tool(dataset_id, expressions)

Process:
1) Identify what the user wants: filter, select, mutate, or a combination.
2) Ensure a source dataset_id is known; if not, ask which dataset to use.
3) Call the appropriate tool(s). Each tool returns a NEW dataset_id.
4) Summarize what changed and highlight the new dataset_id.

Filter Condition Syntax:
- Use pandas query() syntax with backticks for column names with spaces/special chars
- Examples:
  * Simple: "age > 30 and income < 100000"
  * With spaces: "`Life expectancy` > 70 and `GDP` < 1e12"
  * Complex: "(`age` >= 18 & `age` <= 65) | `status` == 'active'"
- Use backticks (`) around ALL column names for safety
- Use & for AND, | for OR, == for equality, != for inequality
- Do NOT use df['column'] syntax - the tool handles column references automatically

Constraints:
- Do not compute statistics or perform EDA.
- Never overwrite datasets; always work with the new dataset_id from tools.
- If an operation is ambiguous or unsafe, ask for clarification.
- Do NOT call web search, external APIs, or MCPs.

Output (<60 words):
- Operation performed.
- Source → new dataset_id.
- Row/column changes summary.
    """,
    tools=[
        wrangle_filter_rows_tool,
        wrangle_select_columns_tool,
        wrangle_mutate_columns_tool,
    ],
)

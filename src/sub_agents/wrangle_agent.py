from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.wrangle_tools import (
    wrangle_filter_rows_tool,
    wrangle_mutate_columns_tool,
    wrangle_select_columns_tool,
)
from ..utils.consts import retry_config

wrangle_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="wrangle_agent",
    description=(
        "Performs data wrangling, filtering, selecting, and feature creation "
        "operations on datasets. All transformations result in new dataset_ids."
    ),
    instruction=(
        """You are the data wrangling specialist.

Use:
- wrangle_filter_rows_tool to subset rows by a condition,
- wrangle_select_columns_tool to keep or drop columns,
- wrangle_mutate_columns_tool to create or modify columns.

When interpreting user instructions, use these patterns:

- Filter rows: wrangle_filter_rows_tool(dataset_id, condition="col > 10")
- Filter with string ops: "city.str.contains('New')"
- Select columns: wrangle_select_columns_tool(dataset_id, columns=[...])
- Mutate columns: wrangle_mutate_columns_tool(dataset_id,
    expressions={"new_col": "colA + colB"})

Always return the new dataset_id from the tool result.

Always:
- Explain the transformation,
- Return or reference the new dataset_id created by the tool.

Do not compute statistics or do analysis."""
    ),
    tools=[
        wrangle_filter_rows_tool,
        wrangle_select_columns_tool,
        wrangle_mutate_columns_tool,
    ],
)

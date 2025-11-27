"""Loop-based quality improvement agents.

Defines:
  - quality_refine_agent: decides whether to apply wrangling operations or exit.
  - quality_improvement_loop_agent: LoopAgent composing data_quality_agent + refine agent.

The refinement logic relies ONLY on outputs from data_quality_agent; it never
computes statistics directly. It asks the user for guidance when ambiguous.
"""

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool

from ..sub_agents.data_quality_agent import data_quality_agent
from ..tools.quality_loop_tools import exit_quality_loop
from ..tools.wrangle_tools import (
    wrangle_filter_rows_tool,
    wrangle_mutate_columns_tool,
    wrangle_select_columns_tool,
)
from ..utils.consts import retry_config

quality_refine_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="quality_refine_agent",
    output_key="wrangle_output",  # Reuse wrangle output key convention
    description="Iteratively improves dataset quality via targeted wrangling or exits when acceptance criteria met.",
    instruction="""
    You are the Data Quality Refinement Specialist in a LoopAgent.

Context:
- Latest data quality report: {data_quality_output}
- Each loop iteration runs:
  1) data_quality_agent → produces data_quality_output
  2) You → either exit the loop or apply ONE wrangling operation to create a new dataset.

Exit criteria (any of these is sufficient):
- readiness_score.overall >= 85
- AND all are true:
  - Duplicate row percentage <= 1%
  - No column with missing_pct > 40%
  - Columns with 30–40% missing are imputed or dropped
  - Columns with >60% missing are dropped or explicitly justified
  - No constant column kept unless clearly an identifier

Decision process:
1) Read {data_quality_output}. Identify:
   - Duplicate rate
   - Columns with high missingness
   - Constant columns
   - readiness_score.overall
2) If exit criteria are met, call exit_quality_loop and do nothing else.
3) Otherwise:
   - Propose a short plan (drops, imputations, simple feature changes).
   - If the user has already given preferences in the prompt, follow them.
   - If not, ask for quick confirmation before acting when the choice is subjective.

Allowed actions (max ONE category per iteration):
- Drop columns: use wrangle_select_columns_tool with a list of columns to KEEP.
- Filter rows: use wrangle_filter_rows_tool with a clear filter_expr (for example "col.notna()").
- Impute or create features: use wrangle_mutate_columns_tool with simple expressions
  (for example {"col": "col.fillna(col.median())"}).

Rules:
- Do not compute new statistics; rely only on values in data_quality_output.
- Prefer simple imputations:
  - Numeric roughly symmetric: mean
  - Numeric skewed: median
  - Categorical: mode
- If dataset_id or target columns are unclear, ask the user to clarify.
- Keep each iteration atomic: one main change, then let data_quality_agent re-evaluate.

When you respond:
1) Briefly state current blockers from the latest report.
2) Specify whether you are exiting or which single operation you applied.
3) Return old dataset_id → new dataset_id and list the operations used.
4) Suggest what the next iteration should focus on, or confirm completion if you exited.
""",
    tools=[
        FunctionTool(exit_quality_loop),
        wrangle_filter_rows_tool,
        wrangle_select_columns_tool,
        wrangle_mutate_columns_tool,
    ],
)


quality_improvement_loop_agent = LoopAgent(
    name="quality_improvement_loop",
    sub_agents=[data_quality_agent, quality_refine_agent],
    max_iterations=4,  # Prevent infinite refinement; user can re-invoke if needed.
)

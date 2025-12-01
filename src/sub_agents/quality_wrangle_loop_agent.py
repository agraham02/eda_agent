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
from ..utils.consts import StateKeys, retry_config

quality_refine_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="quality_refine_agent",
    output_key=StateKeys.WRANGLE,  # Reuse wrangle output key convention
    description="Iteratively improves dataset quality via targeted wrangling or exits when acceptance criteria met.",
    instruction="""Data Quality Refinement Specialist. Improve quality iteratively or exit when ready.

Context: {data_quality_output} from data_quality_agent

Exit criteria (call exit_quality_loop when met):
- readiness_score.overall >= 85 AND
- Duplicates <= 1%
- No column >40% missing (30-40% imputed/dropped, >60% dropped/justified)
- No constant columns kept (unless ID)

Decision process:
1. Review: duplicates, high missingness, constants, overall score
2. Exit if criteria met
3. Otherwise: propose plan, ask if subjective, apply ONE action

Actions (ONE per iteration):
- Drop columns: wrangle_select_columns_tool (list columns to KEEP)
- Filter rows: wrangle_filter_rows_tool (e.g., "col.notna()")
- Impute: wrangle_mutate_columns_tool (e.g., {"col": "col.fillna(col.median())"})

Imputation defaults:
- Numeric symmetric → mean | skewed → median
- Categorical → mode

Output:
1. Current blockers
2. Action taken or exit confirmation
3. Old → new dataset_id
4. Next focus or completion note

Constraints:
- Use only data_quality_output; no new calculations
- Check ok field; explain errors clearly
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

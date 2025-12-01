# src/sub_agents/data_quality_agent.py
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.data_quality_tools import data_quality_tool
from ..tools.quality_loop_tools import offer_quality_loop_tool
from ..utils.consts import QUALITY_LOOP_THRESHOLD, StateKeys, retry_config

data_quality_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="data_quality_agent",
    output_key=StateKeys.DATA_QUALITY,
    description=(
        """Data quality specialist. Evaluates missingness, duplicates, outliers,
    constant or ID like columns, and recommends cleanup steps. Can pause to
    ask user about running automatic quality improvement loop. Stores outlier 
    metadata in session state for later use by wrangle_agent."""
    ),
    instruction=(
        """Data Quality Specialist. Evaluate dataset quality and guide cleanup.

Tools:
- data_quality_tool(dataset_id, outlier_method="both")
  Returns: quality metrics + outlier_metadata (bounds, counts, suggested filters)
- offer_quality_loop_tool(dataset_id, readiness_score, issues)
  LRO: pauses for user approval of auto-cleanup loop

Process:
1. Run data_quality_tool
2. Extract outlier_metadata from results - report specific bounds and filter expressions
   Example: "Age: 10 outliers below 18. Filter: `Age` >= 18"
3. Interpret readiness_score (0-100): 90+ Ready, 75-89 Minor fixes, 50-74 Needs work, <50 Not ready
4. If score < 85: call offer_quality_loop_tool and report user decision
5. Recommend fixes with exact column names and filter conditions

Missingness rules of thumb:
- <5%: low impact
- 5-30%: consider imputation
- 60-90%: likely drop
- >90%: drop unless justified

Output (bullet format):
- Readiness: score + band interpretation
- Per-issue summaries: missing data, duplicates, outliers (with bounds), constants
- Component scores breakdown
- Next steps with specific column names and filter expressions

Constraints:
- Use only tool outputs; never fabricate numbers
- Always include outlier bounds and suggested filters
- Check tool ok field; explain errors with error.message and error.hint
        """
    ),
    tools=[data_quality_tool, offer_quality_loop_tool],
)

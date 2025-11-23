# src/sub_agents/data_quality_agent.py
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.data_quality_tools import data_quality_tool
from ..utils.consts import retry_config

data_quality_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="data_quality_agent",
    description=(
        "Analyzes a dataset for data quality issues such as missing values, "
        "duplicate rows, constant columns, and basic numeric outliers. "
        "Consumes a dataset_id, calls data_quality_tool, and explains the "
        "results clearly for the user."
    ),
    instruction=(
        """You are the data quality specialist.

1. Always call data_quality_tool first.
2. Interpret the results using the following rules:

Missing Data:
- <5% missing = likely MCAR; usually safe to ignore.
- 5–30% missing = may be MAR; suggest imputation or group-specific investigation.
- >60% missing = very high; consider the variable potentially MNAR or structurally missing.
- >90% missing = flag as “possible structurally missing” (values may be absent by design).

Outliers:
- Use IQR-based outlier_count and outliers list from numeric_summary.
- If the user asks for outlier values, return numeric_summary["outliers"].

Constant and Unique Columns:
- Constant columns can be safely dropped.
- All-unique columns may be IDs or keys; warn the user if they look like identifiers.

Duplicates:
- Summarize duplicate rows at the dataset level.
- If duplicates >5%, suggest reviewing the collection pipeline.

Imputation Guidance:
- Small missingness: mean/median/mode imputation.
- Moderate missingness: group-based imputation, conditional imputation.
- High missingness: consider dropping the column or using multiple imputation.
- Time series: mention LOCF, NOCB, BOCF if relevant.

Do not fabricate metrics — rely solely on the output from data_quality_tool.
When explaining issues, reference the missingness categories (MCAR, MAR, MNAR)
from the user’s notes in simple language."""
    ),
    tools=[data_quality_tool],
)

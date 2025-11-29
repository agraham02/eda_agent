from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.data_quality_agent import data_quality_agent
from .sub_agents.eda_describe_agent import eda_describe_agent
from .sub_agents.eda_inference_agent import eda_inference_agent
from .sub_agents.eda_viz_agent import eda_viz_agent
from .sub_agents.ingestion_agent import ingestion_agent
from .sub_agents.quality_wrangle_loop_agent import quality_improvement_loop_agent
from .sub_agents.summary_agent import summary_agent
from .sub_agents.wrangle_agent import wrangle_agent
from .utils.consts import retry_config

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="root_orchestrator",
    description=(
        """Top level orchestrator for Data Whisperer. Interprets user requests,
    routes work to specialist agents (ingestion, quality, wrangling, EDA, 
    inference, visualization, summary), and manages session state."""
    ),
    instruction=(
        """
You are the Root Orchestrator for the Data Whisperer system.

Your job:
- Understand the user's request and intent.
- Choose the right specialist agent.
- Keep track of dataset_ids and prior outputs in session state.
- Coordinate multi step workflows, then surface results back to the user.

Available agents (via tools):
- ingestion_agent  → load files, register datasets.
- data_quality_agent → missingness, duplicates, outliers, column issues.
- wrangle_agent → filter/select/mutate columns, create new dataset versions.
- eda_describe_agent → descriptive stats and correlations.
- eda_inference_agent → hypothesis tests and CLT demos.
- eda_viz_agent → plots for distributions and relationships.
- summary_agent → final report that synthesizes all previous outputs.

Routing guidelines:
- File uploads / “load this CSV” → ingestion_agent.
- “Check quality”, “missing values”, “duplicates”, “outliers” → data_quality_agent.
- “Filter”, “keep only these columns”, “create new feature” → wrangle_agent.
- “Describe”, “distributions”, “correlations” → eda_describe_agent.
- “Is this significant”, “test difference”, “compare groups” → eda_inference_agent.
- “Plot”, “visualize”, “histogram”, “scatter”, “time series” → eda_viz_agent.
- “Full analysis”, “summary”, “write a report” → run describe + inference + viz
  (if not already run) then call summary_agent.

Rules:
- Do NOT compute statistics yourself. Always call tools or sub agents.
- Use existing dataset_ids from state; if unclear, list what you have and ask.
- Prefer the standard flow: ingestion → quality → wrangling/EDA → inference → summary,
  but respect explicit user requests.
- Return sub-agent outputs directly without adding commentary or explanations.
- Do NOT call web search, external APIs, or MCPs.

Your role is coordination and state management, not doing analysis.
"""
    ),
    tools=[
        AgentTool(agent=ingestion_agent),
        AgentTool(agent=data_quality_agent),
        AgentTool(agent=wrangle_agent),
        AgentTool(agent=eda_describe_agent),
        AgentTool(agent=eda_inference_agent),
        AgentTool(agent=eda_viz_agent),
        AgentTool(agent=summary_agent),
        AgentTool(agent=quality_improvement_loop_agent),
    ],
)

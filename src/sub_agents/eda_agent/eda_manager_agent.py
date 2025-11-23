from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool

from ...utils.consts import retry_config
from ..wrangle_agent import wrangle_agent
from .sub_agents.eda_inference_agent import eda_inference_agent
from .sub_agents.eda_describe_agent import eda_describe_agent
from .sub_agents.eda_viz_agent import eda_viz_agent

eda_manager_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_manager_agent",
    description=(
        "High level EDA manager that interprets user requests about "
        "exploratory analysis, routing them to summary, inference, "
        "visualization, or wrangling sub-agents."
    ),
    instruction=(
        """You are the EDA manager.

Given a dataset_id and a user question:
- Route descriptive questions (means, distributions, correlations, group summaries)
  to eda_describe_agent.
- Route inferential questions (hypothesis tests, binomial tests, CLT demos)
  to eda_inference_agent.
- Route visualization requests to eda_viz_agent.
- Route data cleaning or transformation requests (filtering, selecting,
  adding columns) to wrangle_agent.

Combine and summarize the results in a coherent final answer.
Do not compute statistics directly."""
    ),
    tools=[
        AgentTool(agent=eda_describe_agent),
        AgentTool(agent=eda_inference_agent),
        AgentTool(agent=eda_viz_agent),
        AgentTool(agent=wrangle_agent),
    ],
)

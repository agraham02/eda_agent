from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.data_quality_agent import data_quality_agent
from .sub_agents.eda_agent.eda_manager_agent import eda_manager_agent
from .sub_agents.ingestion_agent import ingestion_agent
from .sub_agents.wrangle_agent import wrangle_agent
from .utils.consts import retry_config

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="root_orchestrator",
    description=(
        """Coordinates the full Data Whisperer pipeline.
        Responsible for delegating tasks to ingestion, schema, PII,
        data quality, EDA, insights, and reporting agents.
        Maintains the run-level state and decides next steps
        based on intermediate outputs."""
    ),
    instruction=(
        """You are the orchestrator.

        - When the user uploads a file, call the ingestion_agent to ingest it.
        - When the user asks about missing values, duplicates, data quality,
          or whether the dataset is "clean", call the data_quality_agent
          and pass the relevant dataset_id.
        - Do not analyze the raw data yourself. Always delegate to the
          appropriate specialist agent or tool.
        - If information is missing (for example, which dataset to use),
          ask the user to clarify.
        """
    ),
    tools=[
        AgentTool(agent=ingestion_agent),
        AgentTool(agent=data_quality_agent),
        AgentTool(agent=wrangle_agent),
        AgentTool(agent=eda_manager_agent),
    ],
)

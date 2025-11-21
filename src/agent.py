from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.ingestion_agent import ingestion_agent
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
        Do not analyze data yourself.
        When the user uploads a file or requests ingestion,
        delegate to the ingestion_agent tool directly.
        The ingestion_agent is responsible for saving the file
        and performing ingestion.
        Decide which specialized agent should handle the next step.
        If information is missing, request it from the user.
        """
    ),
    tools=[AgentTool(agent=ingestion_agent)],
)

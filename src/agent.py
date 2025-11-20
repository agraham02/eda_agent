from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import google_search

from .utils.consts import retry_config

root_agent = Agent(
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
        Decide which specialized agent should handle the next step.
        If information is missing, request it.
        Use tools when you need external context."""
    ),
    tools=[google_search],
)

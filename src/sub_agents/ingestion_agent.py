# ingestion_agent.py
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini

from ..tools.ingestion_tools import ingest_csv_tool  # adapt name/import to your toolkit
from ..utils.consts import retry_config

ingestion_agent = Agent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="ingestion_agent",
    description=(
        """Ingests raw tabular data (CSV for now), registers it into the
        shared in-memory store, and returns dataset_id plus schema and
        basic column summaries."""
    ),
    instruction=(
        """You are the ingestion specialist.
        Given a file path or URL to a CSV, call the ingest_csv_tool.
        Do not try to parse CSV content in the prompt.
        Return a concise JSON response with dataset_id, basic schema,
        column summaries, and any obvious ingestion warnings."""
    ),
    tools=[ingest_csv_tool],
)

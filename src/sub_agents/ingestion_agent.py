# ingestion_agent.py
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.ingestion_tools import ingest_csv_tool  # adapt name/import to your toolkit
from ..tools.save_file_tool import save_file_tool
from ..utils.consts import StateKeys, retry_config

ingestion_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="ingestion_agent",
    output_key=StateKeys.INGESTION,
    description=(
        """Ingestion specialist. Saves CSV files, loads into data store, 
        returns dataset_id and basic metadata."""
    ),
    instruction=(
        """Load CSV files into system.

Process:
1. save_file_tool → save uploaded file
2. ingest_csv_tool → load CSV, infer schema
3. Report: dataset_id, shape, columns, dtypes, warnings

Constraints:
- CSV only; reject other formats
- Check ok field; explain errors clearly
        """
    ),
    tools=[save_file_tool, ingest_csv_tool],
)

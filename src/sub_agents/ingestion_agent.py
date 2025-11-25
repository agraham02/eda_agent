# ingestion_agent.py
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.ingestion_tools import ingest_csv_tool  # adapt name/import to your toolkit
from ..tools.save_file_tool import save_file_tool
from ..utils.consts import retry_config

ingestion_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="ingestion_agent",
    output_key="ingestion_output",
    description=(
        """Ingestion specialist. Saves uploaded CSV files, loads them into the data 
        store, infers schema, and returns a dataset_id plus basic metadata."""
    ),
    instruction=(
        """You are the Data Ingestion Specialist.

Goal:
Turn uploaded CSV files into registered datasets that other agents can use.

Tools:
- save_file_tool(file) → local file path.
- ingest_csv_tool(file_path) → dataset_id, schema, and basic stats.

Process:
1) If the user provides a file, call save_file_tool.
2) Call ingest_csv_tool with the saved file path.
3) Use the tool response to extract:
   - dataset_id
   - number of rows and columns
   - column names and inferred dtypes
   - any warnings

Constraints:
- Do not attempt to parse CSV content yourself.
- Only handle CSVs; for other formats, explain that only CSV is supported.
- Do not compute extra statistics.

Output:
- Clear confirmation that ingestion succeeded.
- Dataset_id highlighted for future steps.
- Shape and column overview.
- Any warnings from the tool.
- Suggested next steps (for example: run data_quality_agent, eda_describe_agent).
        """
    ),
    tools=[save_file_tool, ingest_csv_tool],
)

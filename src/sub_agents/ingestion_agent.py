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
        """Role: Load CSV files into the system.

Tools:
- save_file_tool: Save uploaded file to disk
- ingest_csv_tool: Load CSV, infer schema, return dataset_id

Process:
1. Call save_file_tool on uploaded file
2. Call ingest_csv_tool with file path
3. Extract from tool result:
   - dataset_id
   - rows and columns
   - column names and dtypes
   - warnings

Output:
- "Ingestion successful"
- Dataset_id (for next steps)
- Shape and column summary
- Warnings (if any)

Constraints:
- CSV only; reject other formats
- Do not parse CSV yourself
- Do not compute extra statistics
- Do not call web search, external APIs, or MCPs

Error Handling:
- Tools return ok=true on success or ok=false with error details.
- Always check the ok field before using results.
- If ok=false, explain error.message and error.hint clearly to the user.
        """
    ),
    tools=[save_file_tool, ingest_csv_tool],
)

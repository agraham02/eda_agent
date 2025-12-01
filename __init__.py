from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

from .src.agent import root_agent

# Configure Runner with persistent session storage (SQLite) and memory service.
# For Vertex AI deployment, replace with:
#   - VertexAiSessionService for sessions
#   - VertexAiMemoryBankService for memory
session_service = DatabaseSessionService(db_url="sqlite:///eda_sessions.db")
memory_service = InMemoryMemoryService()
artifact_service = InMemoryArtifactService()

runner = Runner(
    agent=root_agent,
    app_name="eda_agent_app",
    session_service=session_service,
    artifact_service=artifact_service,
    memory_service=memory_service,
)

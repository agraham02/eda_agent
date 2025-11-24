from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from .src.agent import root_agent

# Configure Runner with session and artifact services so binary outputs (charts)
# are persisted and visible in the Dev UI Artifacts panel.
runner = Runner(
    agent=root_agent,
    app_name="eda_agent_app",
    session_service=InMemorySessionService(),
    artifact_service=InMemoryArtifactService(),
)

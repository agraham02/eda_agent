from google.adk.runners import InMemoryRunner
from .src.agent import root_agent

runner = InMemoryRunner(agent=root_agent)

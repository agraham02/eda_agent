from google.adk.runners import InMemoryRunner
from .agent import root_agent

runner = InMemoryRunner(agent=root_agent)

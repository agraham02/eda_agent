from google.adk.agents import LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.data_quality_agent import data_quality_agent
from .sub_agents.eda_describe_agent import eda_describe_agent
from .sub_agents.eda_inference_agent import eda_inference_agent
from .sub_agents.eda_viz_agent import eda_viz_agent
from .sub_agents.ingestion_agent import ingestion_agent
from .sub_agents.quality_wrangle_loop_agent import quality_improvement_loop_agent
from .sub_agents.summary_agent import summary_agent
from .sub_agents.wrangle_agent import wrangle_agent
from .tools.memory_tools import (
    compare_runs_tool,
    get_analysis_run_tool,
    get_dataset_lineage_tool,
    list_past_analyses_tool,
    list_persisted_datasets_tool,
    load_preferences_tool,
    save_preferences_tool,
)
from .utils.consts import retry_config

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="root_orchestrator",
    description=(
        """Root orchestrator. Routes requests to specialist agents with minimal 
    necessary steps. Manages state and long-term memory."""
    ),
    instruction=(
        """Workflow manager. Route requests efficiently using minimal necessary agents.

CANONICAL WORKFLOW:
1. ingestion_agent → dataset_id
2. data_quality_agent → quality report + outlier_metadata (stored in state)
3. quality_improvement_loop_agent → auto-cleanup (only when user asks "fix"/"auto-improve")
4. wrangle_agent → transforms (uses stored outlier_metadata)
5. eda_describe_agent → stats, correlations, descriptive relationships
6. eda_viz_agent → plots (single or batch)
7. eda_inference_agent → hypothesis tests, p-values (NOT for simple correlations)
8. summary_agent → final report

MEMORY TOOLS:
- Preferences: save_preferences_tool, load_preferences_tool
- History: list_past_analyses_tool, get_analysis_run_tool, compare_runs_tool
- Datasets: list_persisted_datasets_tool, get_dataset_lineage_tool

THREE ROUTING MODES:

A. Direct (specific requests)
- "Plot X vs Y" → call only viz_agent
- "T-test between A and B" → call only inference_agent
- Reuse existing state when possible

B. Partial (needs prerequisites)
- Request requires missing upstream outputs → run minimal chain
- Example: t-test with no data → ingestion → data_quality → inference

C. Full ("analyze", "report", "model readiness")
- Run: ingestion → data_quality → describe → viz → inference → summary

STATE AWARENESS:
Check state before calling agents:
- ingestion_output missing + analysis requested → ingestion first
- data_quality_output missing + quality/report requested → data_quality
- describe_output missing + stats/plots/inference/summary → describe
- viz_output missing + plots/full analysis → viz ONCE (batch mode)
- inference_output missing + hypothesis tests mentioned → inference
- summary_agent requires ingestion_output + at least one other output

CRITICAL ROUTING RULES:
- "Correlation"/"relationship" WITHOUT "significance"/"hypothesis" → describe/viz, NOT inference
- Inference only when user mentions: significance, p-values, hypothesis tests, CIs, "is this real"

SUGGEST-AND-CONFIRM (after milestones):
- After quality: "Found [issues]. Options: (1) remove outliers, (2) handle missing, (3) proceed as-is?"
- After wrangle: "Done. [N] rows. Next: (1) recheck quality, (2) analyze, (3) more transforms?"
- After describe/viz: "Results shown. Want to: (1) dig deeper, (2) run tests, (3) get report?"

CONTEXT PRESERVATION:
- outlier_metadata from data_quality auto-stored → wrangle_agent uses without asking
- Tell users: "Outlier thresholds already noted; can remove without re-specifying"

Output:
- Briefly state which agent(s) calling and why
- Surface sub-agent results directly
- summary_agent output = final result (don't rewrite it)
- End with 2-3 next step options

Constraints:
- Minimal steps to satisfy request
- Never compute stats yourself
- Trust sub-agent outputs
- Check state before calling agents to avoid redundancy
"""
    ),
    tools=[
        AgentTool(agent=ingestion_agent),
        AgentTool(agent=data_quality_agent),
        AgentTool(agent=wrangle_agent),
        AgentTool(agent=eda_describe_agent),
        AgentTool(agent=eda_inference_agent),
        AgentTool(agent=eda_viz_agent),
        AgentTool(agent=summary_agent),
        AgentTool(agent=quality_improvement_loop_agent),
        # Memory and preference tools
        save_preferences_tool,
        load_preferences_tool,
        list_past_analyses_tool,
        get_analysis_run_tool,
        compare_runs_tool,
        list_persisted_datasets_tool,
        get_dataset_lineage_tool,
    ],
)

# -----------------------------------------------------------------------------
# Resumable App Wrapper for Long-Running Operations (LRO)
# This enables the agent to pause and resume for human-in-the-loop decisions
# -----------------------------------------------------------------------------

root_app = App(
    name="data_whisperer",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)

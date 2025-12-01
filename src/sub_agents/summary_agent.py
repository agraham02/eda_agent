# src/sub_agents/summary_agent.py

from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.google_llm import Gemini

from ..tools.memory_tools import save_analysis_run
from ..tools.summary_tools import finalize_summary_tool
from ..utils.consts import StateKeys, retry_config


async def auto_save_analysis_run(
    callback_context: CallbackContext,
) -> None:
    """
    After-agent callback to automatically save analysis runs to persistent storage.

    Extracts relevant data from session state and saves to SQLite.
    """
    try:
        state = callback_context.state

        # Get dataset_id from ingestion output
        ingestion_output = state.get(StateKeys.INGESTION.value, {})
        if isinstance(ingestion_output, str):
            # Sometimes state stores stringified JSON
            import json

            try:
                ingestion_output = json.loads(ingestion_output)
            except (json.JSONDecodeError, TypeError):
                ingestion_output = {}

        dataset_id = ingestion_output.get("dataset_id", "unknown")

        # Get summary from state
        summary_output = state.get(StateKeys.SUMMARY.value, {})
        if isinstance(summary_output, str):
            import json

            try:
                summary_output = json.loads(summary_output)
            except (json.JSONDecodeError, TypeError):
                summary_output = {"summary": summary_output}

        summary_markdown = summary_output.get("summary", "")

        # Get readiness score from data quality output
        dq_output = state.get(StateKeys.DATA_QUALITY.value, {})
        if isinstance(dq_output, str):
            import json

            try:
                dq_output = json.loads(dq_output)
            except (json.JSONDecodeError, TypeError):
                dq_output = {}

        readiness_score = dq_output.get("readiness_score")

        # Get inference results
        inference_output = state.get(StateKeys.INFERENCE.value, {})
        if isinstance(inference_output, str):
            import json

            try:
                inference_output = json.loads(inference_output)
            except (json.JSONDecodeError, TypeError):
                inference_output = {}

        p_values = {}
        confidence_intervals = {}
        effect_sizes = {}

        # Extract from inference output if present
        if inference_output.get("ok"):
            if "p_value" in inference_output:
                test_name = inference_output.get("test_type", "test")
                p_values[test_name] = inference_output["p_value"]
            if "confidence_interval" in inference_output:
                test_name = inference_output.get("test_type", "test")
                confidence_intervals[test_name] = inference_output[
                    "confidence_interval"
                ]
            if "cohen_d" in inference_output:
                effect_sizes["cohen_d"] = inference_output["cohen_d"]

        # Get viz output for plot paths
        viz_output = state.get(StateKeys.VIZ.value, {})
        if isinstance(viz_output, str):
            import json

            try:
                viz_output = json.loads(viz_output)
            except (json.JSONDecodeError, TypeError):
                viz_output = {}

        plot_paths = []
        if viz_output.get("file_path"):
            plot_paths.append(viz_output["file_path"])
        elif viz_output.get("plots"):
            plot_paths = [
                p.get("file_path", "")
                for p in viz_output["plots"]
                if p.get("file_path")
            ]

        # Get descriptive highlights
        describe_output = state.get(StateKeys.DESCRIBE.value, {})
        if isinstance(describe_output, str):
            import json

            try:
                describe_output = json.loads(describe_output)
            except (json.JSONDecodeError, TypeError):
                describe_output = {}

        descriptive_highlights = {}
        if describe_output.get("ok"):
            # Extract key stats
            if "correlation_matrix" in describe_output:
                descriptive_highlights["has_correlations"] = True
            if "summaries" in describe_output:
                descriptive_highlights["n_columns_described"] = len(
                    describe_output["summaries"]
                )

        # Determine run type based on what outputs exist
        run_type = "full"
        if not inference_output:
            if not describe_output:
                run_type = "quality_check"
            else:
                run_type = "descriptive"
        elif not describe_output and not viz_output:
            run_type = "inference"

        # Get user question from session events if possible
        user_question = "Analysis run"
        session = callback_context._invocation_context.session
        if session and session.events:
            for event in session.events:
                if event.content and event.content.role == "user":
                    if (
                        event.content.parts
                        and len(event.content.parts) > 0
                        and event.content.parts[0].text
                    ):
                        user_question = event.content.parts[0].text[:500]
                        break

        # Get session ID
        session_id = session.id if session else None

        # Save the run
        save_analysis_run(
            dataset_id=dataset_id,
            user_question=user_question,
            run_type=run_type,
            summary_markdown=summary_markdown,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            effect_sizes=effect_sizes,
            descriptive_highlights=descriptive_highlights,
            plot_paths=plot_paths,
            readiness_score=readiness_score,
            session_id=session_id,
        )
    except Exception:
        # Don't fail the agent response if saving fails
        pass


summary_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="summary_agent",
    output_key=StateKeys.SUMMARY,
    after_agent_callback=auto_save_analysis_run,
    description="""Final reporting agent. Uses structured outputs from ingestion, data 
        quality, wrangling, descriptive EDA, inference, and visualization to 
        produce a clear, accurate summary report grounded in the actual dataset.""",
    instruction="""Final reporting specialist. Synthesize all outputs into structured report.

Inputs: {ingestion_output}, {data_quality_output}, {wrangle_output?}, {describe_output}, {inference_output}, {viz_output}

Rules:
- Use only provided metrics; never compute or guess
- Reference only columns in ingestion_output.columns[].name

Required structure:

## 1. Data Signature
- Dataset ID, shape, transformations (if any)
- 2-3 sentence analysis overview

## 2. Key Findings (bullets)
- Distributions, correlations (describe_output)
- Tests, p-values, CIs (inference_output)
- Visual patterns (viz_output, reference filenames)
- Inline caveats (e.g., "15% missing in X")

## 3. Model Readiness
From data_quality_output.readiness_score:
- **Overall:** X/100 (Excellent 90-100 / Good 70-89 / Fair 50-69 / Poor <50)
- **Components:** missingness, duplicates, constants, high_missing_columns, outliers
- **Critical Issues:** >30% missingness, dataset_issues
- **Plots:** quality viz references

Gating:
- ≥70: ready with minor cleaning
- 50-69: requires cleaning
- <50: not ready

List priority actions from readiness_score.notes

## 4. Recommendations
- Cleaning/wrangling needed
- Additional analyses
- Next steps

Final step:
- Call finalize_summary_tool once with full markdown
- Return report to user
- If inputs missing, explain which and why report is incomplete

Check ok field; explain errors clearly.
""",
    tools=[finalize_summary_tool],
)

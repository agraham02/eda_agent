from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ....tools.eda_inference_tools import (
    eda_binomial_test_tool,
    eda_clt_sampling_tool,
    eda_one_sample_test_tool,
    eda_two_sample_test_tool,
)
from ....utils.consts import retry_config

eda_inference_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_inference_agent",
    description=(
        "Performs inferential statistics, hypothesis tests, binomial tests, "
        "and Central Limit Theorem sampling demonstrations."
    ),
    instruction=(
        """You are the inference specialist.

Use:
- eda_one_sample_test_tool for one-sample tests,
- eda_two_sample_test_tool for group comparisons,
- eda_binomial_test_tool for binomial proportion tests,
- eda_clt_sampling_tool to illustrate sampling distributions and the CLT.

Interpret p-values, effect sizes, and confidence intervals in plain language."""
    ),
    tools=[
        eda_one_sample_test_tool,
        eda_two_sample_test_tool,
        eda_binomial_test_tool,
        eda_clt_sampling_tool,
    ],
)

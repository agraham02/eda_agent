from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.eda_inference_tools import (
    eda_binomial_test_tool,
    eda_clt_sampling_tool,
    eda_one_sample_test_tool,
    eda_two_sample_test_tool,
)
from ..utils.consts import StateKeys, retry_config

eda_inference_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_inference_agent",
    output_key=StateKeys.INFERENCE,
    description=(
        """Inferential statistics specialist. Runs hypothesis tests, explains 
    p-values and confidence intervals in 2-3 sentences."""
    ),
    instruction=(
        """Inferential statistics specialist. Test statistical significance only.

CAPABILITIES (state upfront if unclear):
- One-sample t-test: compare column mean to value
- Two-sample t-test: compare means between groups
- Binomial test: test observed proportion vs expected
- CLT sampling demo: visualize sampling distribution

For correlations or descriptive relationships → redirect to describe agent.

Entry criteria (run tests only when user mentions):
- Hypotheses, significance, p-values, confidence intervals
- "Is this difference real", "statistically significant"

Test selection:
- "Is X different from Y" → one_sample (Y=number) or two_sample (Y=group)
- "Compare group A vs B" → two_sample
- "Is proportion significant" → binomial
- "Show sampling distribution" → CLT demo

Output per test (2-3 sentences):
- Test type, direction of effect
- p-value, CI, reject/fail to reject null
- Practical interpretation; distinguish statistical vs practical significance
- Never claim causation

Constraints:
- Default α=0.05 unless specified
- Check ok field; explain errors and suggest alternatives
- If column ambiguous, ask once with options
        """
    ),
    tools=[
        eda_one_sample_test_tool,
        eda_two_sample_test_tool,
        eda_binomial_test_tool,
        eda_clt_sampling_tool,
    ],
)

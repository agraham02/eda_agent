from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ....tools.eda_viz_tools import eda_render_plot_tool, eda_viz_spec_tool
from ....utils.consts import retry_config

eda_viz_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_viz_agent",
    description=(
        "Designs and renders visualizations to explore distributions and "
        "relationships in the dataset."
    ),
    instruction=(
        """You are the visualization specialist.

Use eda_viz_spec_tool to choose appropriate plot types and parameters
(histograms, boxplots, scatterplots, bar charts, etc.).
Use eda_render_plot_tool to render the plot and return a reference
to the generated image.

Describe what the visualization shows, including key patterns and
potential outliers."""
    ),
    tools=[
        eda_viz_spec_tool,
        eda_render_plot_tool,
    ],
)

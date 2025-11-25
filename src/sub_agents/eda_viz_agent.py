from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.eda_viz_tools import eda_render_plot_tool, eda_viz_spec_tool
from ..utils.consts import retry_config

eda_viz_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_viz_agent",
    output_key="viz_output",
    description=(
        """Visualization specialist. Designs and generates EDA plots using validated 
    specs, then explains the main patterns they show."""
    ),
    instruction=(
        """You are the Data Visualization Specialist.

Goal:
Create a small but comprehensive set of EDA plots and explain what they show.

Tools:
- eda_viz_spec_tool → validate chart specification.
- eda_render_plot_tool → render the chart and store it as an artifact.

Process:
1) Identify relevant variables and the user’s question.
2) Choose chart types that match data types:
   - Numeric: histogram or box plot.
   - Categorical: bar chart (optionally pie chart if few categories).
   - Numeric vs numeric: scatter or line.
   - Categorical vs numeric: box or violin.
3) For a “full” EDA request, aim for 4–8 plots that cover:
   - Key univariate distributions.
   - Important relationships between variables.
4) For each plot:
   - Build a VizSpec with dataset_id, chart_type, x, y, and optional hue.
   - Call eda_viz_spec_tool, then eda_render_plot_tool.
   - Describe in words what the visualization reveals.

Constraints:
- Do not fabricate relationships. Base interpretations on the variables plotted.
- Keep explanations concise and focused on trends, outliers, and patterns.
- Let the UI handle showing the image; you only return the artifact info
  and the textual interpretation.

Output per run:
- List of plots created (type and variables).
- Short interpretation for each plot.
- One to two sentence summary tying visual patterns back to the user’s goal.
        """
    ),
    tools=[
        eda_viz_spec_tool,
        eda_render_plot_tool,
    ],
)

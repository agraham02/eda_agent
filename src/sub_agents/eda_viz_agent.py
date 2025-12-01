from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from ..tools.eda_viz_tools import (
    check_outlier_comparison_tool,
    create_comparison_viz_tool,
    eda_render_plot_tool,
    eda_viz_spec_tool,
)
from ..utils.consts import StateKeys, retry_config

eda_viz_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="eda_viz_agent",
    output_key=StateKeys.VIZ,
    description=(
        """Visualization specialist. Generates 1-7 plots (EDA-suite 4-7), 
   explains patterns in <60 words per plot. Can offer outlier comparison 
   visualizations when high outlier rates are detected."""
    ),
    instruction=(
        """Visualization specialist. Create DIVERSE, RELEVANT plots using upstream metadata.

Context (auto-injected):
- {ingestion_output}: columns, dtypes, semantic_types
- {data_quality_output}: outliers, constants, missingness
- {describe_output}: summaries (univariate stats), correlation_matrix

Tools:
- eda_viz_spec_tool, eda_render_plot_tool: validate and render
- check_outlier_comparison_tool: LRO for outlier comparison (>10% outliers)
- create_comparison_viz_tool: side-by-side with/without outliers

MODES:

1. Single-plot: Specific request ("histogram of age") → create only that

2. EDA-suite: "visualize"/"EDA plots"/"full EDA" → Diverse batch with flexibility
   
   STEP-BY-STEP ALGORITHM (flexible):
   
   Step 1: Identify column types from describe_output.summaries:
   - Numeric columns: type="numeric" 
   - Categorical columns: type="categorical"
   
   Step 2: Target composition (total 4-7 plots):
   - Distributions: 1-3 plots across different numeric columns (hist and/or box)
   - Categorical: 0-2 bar plots (3-10 unique categories)
   - Relationships: 1-2 scatter plots for strongest pairs
   - Time/trend: 0-1 line plot if temporal column exists
   - Optional pie: only if ≤6 categories AND part-to-whole is clear; prefer bar otherwise
   
   a) **Histogram** (1-2): choose numeric columns
      - Prioritize highest outlier_count or highest std
      - chart_type="histogram", x=column_name
      - Shows distribution shape
   
   b) **Box plot** (0-2): choose different numeric columns
      - Prioritize widest range (max - min) or IQR
      - chart_type="box", x=column_name
      - Shows quartiles and outliers
   
   c) **Bar chart** (0-2): choose categorical columns
      - Prefer 3-10 unique values; avoid constants/high-cardinality
      - chart_type="bar", x=column_name
      - Shows category counts
   
   d) **Scatter plot** (1-2): choose numeric pairs
      - Use describe_output.correlation_matrix; pick highest |corr| pairs (ideally > 0.3)
      - Distinct pairs/targets; avoid reusing same pair
      - chart_type="scatter", x=col1, y=col2, hue=categorical_if_available
      - Shows relationship
   
   e) **Line plot** (0-1): ONLY if Year/Date/Time column exists
      - chart_type="line", x=temporal_column, y=numeric_column, hue=categorical_if_available
      - Shows trend over time

   f) **Pie chart** (0-1): ONLY if ≤6 categories and part-to-whole is intended
      - chart_type="pie", names=categorical_column, values=optional_numeric_counts
      - Prefer bar over pie unless clearly part-to-whole
   
   Step 3: Additions to reach target count (cap 7):
   - Box by group: numeric vs categorical (different columns)
   - Extra hist/box for other interesting numeric columns
   - Second scatter if multiple strong correlations exist
   
   CRITICAL RULES:
   - Allow multiple of a type when justified; avoid redundant duplicates
   - Avoid reusing the same column pair; spread coverage across columns
   - Prioritize univariate (hist/box) before bivariate (scatter); then trend
   - Skip a type if no suitable column (e.g., no categorical, no temporal)
   - If dataset is very small (≤4 columns), still aim for 4-5 diverse plots

3. Outlier comparison: When data_quality_output shows >10% outliers
   - Call check_outlier_comparison_tool (pauses for user)
   - If approved: create_comparison_viz_tool per column

Output per plot (<60 words):
- Chart type, variables
- Key pattern: trend, spread, outliers (2-3 sentences)

Error handling:
- ALWAYS return plot list OR error explanation
- Never empty response

Constraints:
- Check ok field; explain errors with column suggestions
 - In EDA-suite: Aim for distributions + categorical + relationships; add trend if temporal
        """
    ),
    tools=[
        eda_viz_spec_tool,
        eda_render_plot_tool,
        check_outlier_comparison_tool,
        create_comparison_viz_tool,
    ],
)

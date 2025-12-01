# Data Whisperer

An autonomous multi-agent system that evaluates dataset quality and performs exploratory data analysis (EDA) to assess ML readiness.

## Overview

Data Whisperer automates the time-consuming process of dataset evaluation that data scientists typically perform manually. It provides consistent, standardized assessments of data quality and generates comprehensive readiness reports.

## What It Does

-   **Automatic Data Ingestion**: Reads CSV, Excel, and JSON files
-   **Data Quality Assessment**: Checks for missing values, duplicates, outliers, type inconsistencies, and potential data leakage
-   **Exploratory Data Analysis**: Generates distribution plots, correlation heatmaps, and statistical summaries
-   **Readiness Scoring**: Produces a 0-100 score indicating how suitable the dataset is for ML modeling
-   **Comprehensive Reports**: Creates detailed markdown/HTML reports with findings and recommendations

## Key Features

### Multi-Agent Architecture

-   **Orchestrator Agent**: Coordinates the analysis pipeline
-   **Ingestion & Schema Agent**: Reads data and infers column types
-   **Data Quality Agent**: Evaluates data health and computes readiness scores
-   **EDA & Visualization Agent**: Generates plots and statistical summaries
-   **Insight & Report Agent**: Synthesizes findings into a final report

### Data Quality Checks

-   Missingness analysis per column and row
-   Duplicate detection
-   Outlier identification (Z-score and IQR methods)
-   Type consistency validation
-   Class imbalance detection
-   Potential leakage hints
-   High cardinality warnings

### Readiness Score

Datasets receive a score from 0-100 with clear labels:

-   **90-100**: Ready
-   **75-89**: Ready with Minor Fixes
-   **50-74**: Needs Work
-   **0-49**: Not Ready for Modeling

## Getting Started

### Prerequisites

-   Python 3.10+
-   A Google AI Studio API key in `GOOGLE_API_KEY` (required by ADK/Gemini)

```bash
# Bash (macOS/Linux/Git Bash on Windows)
export GOOGLE_API_KEY="<your_api_key>"
```

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # On Windows Git Bash: source .venv/Scripts/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch the web interface (easy demo path)
adk web --port 8000

# 4) (Optional) Run from CLI using the app name defined in code
adk run data_whisperer
```

Upload a CSV and receive a comprehensive analysis report with actionable recommendations.

### Quick Demo (local file)

Use the included diabetes dataset to try the end‑to‑end flow:

```bash
# Start the ADK Web UI
adk web --port 8000

# In the UI, ask:
# "Analyze test_data/diabetes.csv and give me the readiness report."
```

## Report Contents

Each analysis includes:

-   Dataset overview and schema
-   Data readiness score with component breakdown
-   Key issues identified
-   EDA visualizations (histograms, boxplots, correlation heatmaps)
-   Prioritized recommendations
-   Methods, assumptions, and limitations

## Use Cases

-   Pre-modeling data validation
-   Dataset quality assessment
-   Automated EDA generation
-   Identifying data issues early in the ML pipeline
-   Consistent evaluation across data science teams

## Testing & Evaluation

### Running Tests

```bash
# Run smoke tests
pytest tests/ -v -m smoke

# Run all tests including integration
pytest tests/ -v

# Run ADK evaluations (use the app name from src/agent.py)
adk eval data_whisperer tests/eval/
```

Note: If any `.test.json` in `tests/eval/` specifies an `app_name`, ensure it matches `data_whisperer` (the app name in `src/agent.py`).

### Interactive Testing with ADK Web UI

Use the ADK Web UI for manual testing and demo preparation:

```bash
# Start the web interface
adk web --port 8000
```

The Web UI provides:

-   Visual trace debugging of agent behavior
-   Ability to record sessions and export as test files
-   Side-by-side comparisons of expected vs actual outputs
-   Tool trajectory visualization

**Tip for Demos:** Record your demo flow in the Web UI, then export it as a `.test.json` file for regression testing.

## Architecture

This project uses a modular, multi‑agent architecture:

-   Orchestrator: routes requests, manages minimal pipelines
-   Ingestion & Schema: loads datasets, infers dtypes
-   Data Quality: computes readiness score and quality findings
-   EDA Describe: statistics, correlations, relationships
-   Visualization: distribution and relationship plots
-   Inference: significance and hypothesis tests (when requested)
-   Summary: composes the final report, including readiness

App identifier for ADK: `data_whisperer` (see `src/agent.py`).

## Limitations & Next Steps

-   Tested on medium datasets; processing is in‑memory
-   Sequential orchestration; future work to parallelize steps
-   File types focused on CSV; Excel/JSON ingestion planned
-   Wrangle expressions have partial hardening; further sandboxing planned

See `TODO.md` for the prioritized roadmap.

## Submission

For a concise hackathon write‑up (problem, solution, architecture, demo, and impact), see `SUBMISSION.md`.

## Author

Ahmad Graham | November 2025

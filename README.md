# Data Whisperer

An autonomous multi-agent system that automatically evaluates dataset quality and performs exploratory data analysis (EDA) to assess ML readiness.

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

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
adk run eda_agent

# Or use the web interface
adk web --port 8000
```

Upload your dataset and receive a comprehensive analysis report with actionable recommendations.

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

## Author

Ahmad Graham | November 2025

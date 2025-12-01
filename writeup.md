# Data Whisperer: Autonomous Dataset Quality & EDA Agent System

## 1. Problem Statement

Before any model is trained, before dashboards are built, before insights are published, there is a deceptively simple question every data scientist must answer:

Is this dataset even good enough to analyze?

In practice, evaluating data quality is rarely simple. Analysts must inspect schema, check for missing or duplicate records, assess accuracy, understand variable types, search for outliers, and trace where the data came from and how it was collected. These steps are foundational to responsible data work because the reliability of downstream modeling depends entirely on what goes in. When organizations skip or rush this early stage, they often fall into the classic “garbage in, garbage out” cycle that leads to misleading conclusions.

Yet this process is slow and inconsistent. Even when teams share similar tools, each person has their own habits and methods for checking data readiness. The result is duplicated effort, quality blind spots, and datasets that are analyzed or modeled long before they’ve been vetted.

I built Data Whisperer to tackle this problem. It is meant to offer a fast, standardized, and intelligent way to inspect and evaluate a dataset the moment it arrives — without requiring a human to manually walk through dozens of checks and visualizations.

## 2. Why Multi-Agent AI?

Dataset evaluation isn’t one task. It’s a sequence of interdependent mini-tasks:

-   Understanding the data’s origin and structure
-   Confirming variable types
-   Checking for accuracy and consistency
-   Detecting missingness, duplicates, and outliers
-   Exploring distributions and relationships
-   Generating visual summaries
-   Transforming the data when needed
-   Synthesizing insights into a report

These tasks rely on different forms of reasoning. Some require code execution. Others require pattern recognition, or statistical literacy, or visualization instincts. They also benefit from minimal-step routing — for example, describe/viz follows quality checks without recomputing ingestion.

A single monolithic script would either require constant maintenance or fail to generalize beyond toy examples. A multi-agent design solves that by giving each part of the pipeline its own expert:

Each agent has one job.

Each agent produces a structured output.

The orchestrator keeps everything coordinated, cached, and aware of past steps.

By using Google’s Agent Development Kit, I was able to make agents not just “talk,” but run code, generate plots, store artifacts, and pass structured results to each other. This architecture feels much closer to how real data teams operate, except automated and significantly faster.

## 3. What I Created

Data Whisperer is a multi-agent system that ingests a dataset, evaluates its quality, explores its structure, and generates a clear readiness report. The system includes:

### 3.1 Ingestion & Schema Agent

-   Loads the dataset
-   Extracts schema, variable types, and high-level characteristics
-   Flags variables that violate type expectations (e.g., mixed types) and highlights high-cardinality categoricals

### 3.2 Data Quality Agent

-   Quantifies missing values
-   Identifies duplicate rows and suspiciously repeated patterns
-   Detects outliers via IQR and Z-score methods
-   Checks type consistency and highlights potential leakage/high-cardinality risks

This agent explicitly aligns with core data-literacy concepts such as variable definition, type consistency checks, and understanding the limits of exploratory findings.

### 3.3 Wrangler Agent

-   Able to apply transformations requested by other agents
-   Can drop duplicates, impute missingness, reformat columns, convert types
-   Produces a cleaned dataframe version for downstream analysis

### 3.4 EDA & Visualization Agent

-   Generates descriptive statistics
-   Creates appropriate visualizations (histograms, bar charts, scatterplots, boxplots) based on variable types
-   Highlights relationships between variables, potential clustering, and any unusual patterns in distributions
-   Follows chart-selection best practices and attempts to avoid misleading visuals

### 3.5 Summary Agent (Insight & Report)

-   Synthesizes all upstream findings
-   Organizes them into a clean, readable readiness report
-   Includes charts, tables, key findings, and recommended next steps
-   Encodes both descriptive insights and exploratory observations

All of these are coordinated by the Root Orchestrator, which manages state awareness, minimal step routing, and artifact storage.

The final output is a Dataset Readiness Report that answers:

-   How trustworthy is this dataset?
-   What issues were found?
-   What transformations were applied?
-   What does the data look like?
-   Is it suitable for descriptive, exploratory, or inferential work?
-   What should a data scientist do next?

## 4. System Architecture

At a high level, Data Whisperer operates in the following loop:

-   User uploads a dataset
-   Orchestrator routes to domain-specific agents with the fewest necessary steps
-   Agents run code, analyze the data, and store results
-   Results move into shared context and memory
-   The summary agent compiles a final structured document

Memory is a central design choice. The orchestrator and tools cache:

-   Schema summaries
-   Data quality metrics
-   EDA artifacts
-   Chat summaries
-   Generated plots
-   Final report components

This gives the system statefulness across the entire run, allowing each agent to reference prior findings rather than recompute everything.

## 5. The Build

Data Whisperer was built with:

-   Google Agent Development Kit
-   Python
-   Pandas for wrangling and descriptive statistics
-   Matplotlib for visualizations
-   ADK tool-calling for code execution and artifact generation

The agents communicate using structured messages and store intermediate artifacts (like cleaned CSVs, charts, summaries) which are then composed into the final report.

Key Capabilities

-   Sequential orchestration with minimal redundant steps (parallelization planned)
-   Code execution — enabling reproducible transformations
-   Automatic visualization — saving PNGs to be included in the report
-   Context awareness — so each stage builds on the last
-   Deterministic structure — standardized report across any dataset

From a data-literacy perspective, the system enforces good practice: describe the data first, evaluate its origin and structure, check accuracy, inspect distributions, and acknowledge uncertainty before jumping to conclusions.

## 6. Example Demo

A quick local demo using the ADK Web UI:

```bash
adk web --port 8000
```

In the UI, ask: “Analyze `test_data/diabetes.csv` and give me the readiness report.”

A typical run takes a raw CSV and produces:

-   Schema overview
-   Missingness heatmap
-   Duplicate analysis
-   Outlier tables
-   Distribution plots
-   Correlation or relationship visuals
-   Cleaned dataframe
-   A narrative readiness report summarizing everything

The goal is that a data scientist should be able to read the report and immediately understand whether the dataset is usable and what cleaning steps were taken.

## 7. What Makes This Useful

-   **Consistency:** Every dataset is evaluated the same way. The system never forgets a step, never rushes, and never lets a dataset pass without a complete diagnostic.

-   **Speed:** EDA that normally takes 30 minutes to an hour now completes in seconds. This matters when teams work with many datasets or iterate rapidly.

-   **Explainability:** The agents generate transparent code, plots, and summaries, which helps users understand how the system reached each conclusion.

-   **Educational Value:** Because the system explains its steps, students or junior data scientists can use Data Whisperer as a learning tool. It teaches data-literacy concepts by example: identifying variable types, thinking critically about missingness, understanding distributions, and interpreting descriptive summaries.

-   **Scalability:** Multi-agent orchestration makes it easy to add future components: a PII agent, a sampling-health agent, or even statistical inference checks.

## 8. If I Had More Time

There are several extensions I plan to add.

### 8.1 RAG-Enhanced Long-Term Memory

I want the system to remember:

-   Past datasets and their evaluations
-   Common issues in recurring data sources
-   Recommended transformations for specific schemas
-   Previous EDA summaries
-   User’s preferred cleaning patterns

This would let the system learn organizational data habits over time.

### 8.2 Integration with a Dedicated Data Science Knowledge Base

I would build a small RAG or MCP server containing:

-   Best practices from data literacy, wrangling, visualization, and statistics
-   Chart selection rules
-   Sampling and inference guidelines
-   Common pitfalls when interpreting distributions
-   Notes on reliability, accuracy, and limitations

Agents could reference this knowledge when producing their evaluations.

### 8.3 Enhanced Model-Readiness Scoring

The system already produces a 0–100 readiness score combining multiple quality signals. A next step is to expand the rubric with:

-   Leakage heuristics and stronger high-cardinality detection
-   Sampling representativeness indicators
-   Domain-aware expectations where provided

This would further help ML engineers determine whether a dataset is suitable for training or requires more cleaning.

### 8.4 Interactive Chat Layer

After generating the report, the user should be able to:

-   Ask follow-up questions
-   Request transformations
-   Explore relationships between variables
-   Generate additional visuals
-   Run statistical tests (e.g., t-tests, regressions, correlation analyses)
-   Simulate sampling distributions
-   Ask the system to detect potential sources of bias or data-collection artifacts

The chat layer would extend Data Whisperer from an automated evaluator into a collaborative data partner.

## 9. Closing Thoughts

The heart of Data Whisperer is simple:
Data quality should not be an afterthought.

By automating the earliest and most essential steps of analysis, the system helps teams avoid mistakes, work faster, and produce more reliable conclusions. Good data work begins with understanding the data itself — its structure, its accuracy, its limitations, and its quirks. This system captures that philosophy and turns it into an executable workflow.

Data Whisperer isn’t meant to replace data scientists. It’s meant to give them a head start, reduce repetitive work, and improve the baseline quality of every project that follows.

Note: App name for ADK runs/evals is `data_whisperer` (see `src/agent.py`).

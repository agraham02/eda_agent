# Project TODO – Data Whisperer

Legend: (P1) High impact / core promise, (P2) Important, (P3) Nice-to-have / stretch.

## 🚀 Ready-to-Implement PRs (Prioritized)

### PR #1: Agent Instruction Refinement (P1) - **START HERE**

**Estimated effort:** 2-3 hours  
**Files to modify:** `src/agent.py`, `src/sub_agents/*.py`

-   [x] Reduce root agent verbosity: return delegated output without commentary
-   [x] EDA manager: enforce "route only, no interpretation" discipline
-   [x] Data quality agent: constrain speculation to >20% missingness threshold only
-   [x] Visualization agent: limit narratives to <60 words (plot reasoning + 2-3 sentences)
-   [x] Wrangle agent: return transformation summary in <60 words
-   [x] Summary agent: strict template (signature, summary, findings, caveats, recommendations)
-   [x] Add explicit instruction to 6 core EDA agents: "Do not call web search, external APIs, or MCPs"

**Acceptance criteria:**

-   Agent responses are 30-50% shorter
-   No commentary creep in routing agents
-   Session replay shows cleaner delegation

---

### PR #2: State Key Standardization (P1)

**Estimated effort:** 1-2 hours  
**Files to modify:** `src/agent.py`, `src/sub_agents/*.py`, `src/utils/data_store.py`

-   [x] Audit all state key usage across agents
-   [x] Enforce consistent naming: `ingestion_output`, `data_quality_output`, `wrangle_output`, `describe_output`, `inference_output`, `viz_output`
-   [x] Remove any nested/combined containers (e.g., `eda_output` containing others)
-   [x] Update all agents to use standardized keys
-   [x] Add constants file: `src/utils/consts.py` with `STATE_KEYS` enum

**Acceptance criteria:**

-   All state keys follow convention
-   No duplicate/nested keys
-   Constants file prevents typos

---

### PR #3: Testing Infrastructure (P1) - **HACKATHON VERSION**

**Estimated effort:** 2-3 hours  
**Files to create:** `requirements.txt`, `tests/`, `pytest.ini`, `.github/workflows/test.yml`

-   [x] Create `requirements.txt` with loose versions (no pinning for faster setup)
-   [x] Create `tests/` directory structure:
    -   `tests/conftest.py` - Basic fixtures only
    -   `tests/test_readiness_score.py` - 4 smoke tests
    -   `tests/test_data_quality.py` - 3 smoke tests
    -   `tests/test_wrangle.py` - 2 smoke tests
-   [x] Smoke tests for readiness scoring (perfect, empty, high-missing, integration)
-   [x] Add pytest configuration (async mode, markers)
-   [x] Create ADK evaluation test files (demo workflow, error handling)
-   [x] Create GitHub Actions CI workflow (pytest + ADK eval, no coverage/linting)

**Acceptance criteria (Hackathon Edition):**

-   `pytest` runs successfully with smoke tests
-   ADK eval files validate demo workflow
-   CI workflow catches regressions before demos
-   Use `adk web` for interactive testing and test file generation

---

### PR #4: Cross-Platform Path Handling (P1)

**Estimated effort:** 1 hour  
**Files to modify:** `src/tools/*.py`, `src/utils/*.py`

-   [x] Replace hardcoded `/tmp` with `tempfile.gettempdir()`
-   [x] Create utility wrapper: `src/utils/paths.py` with `get_temp_dir()`, `get_artifact_path()`
-   [x] Update all tools to use path utilities
-   [x] Test on Windows and Unix-like systems

**Acceptance criteria:**

-   No hardcoded paths remain
-   Works on Windows, macOS, Linux

---

### PR #5: Unified Error Handling (P1)

**Estimated effort:** 3-4 hours  
**Files to modify:** All `src/tools/*.py`

-   [x] Create error response schema: `{success: bool, error_code: str, message: str, hint: str}`
-   [x] Wrap all tool functions with try/except returning structured errors
-   [x] Add error codes enum in `src/utils/errors.py`
-   [x] Update agents to handle and display structured errors gracefully
-   [x] Add error recovery hints (e.g., "Column 'X' not found. Available: Y, Z")

**Acceptance criteria:**

-   All tools return structured responses
-   Error messages are actionable
-   No raw exceptions reach user

---

### PR #6: Readiness Score in Summary Report (P1)

**Estimated effort:** 2 hours  
**Files to modify:** `src/sub_agents/summary_agent.py`, `src/tools/summary_tools.py`

-   [x] Extend summary agent instructions to include "Readiness Score" section
-   [x] Template structure:

    ```
    ## Model Readiness Assessment
    **Overall Score:** X/100 (Category)

    ### Component Breakdown
    - Missingness: X/100
    - Duplicates: X/100
    - ...

    ### Gating Recommendations
    - [Priority] Action needed before modeling
    ```

-   [x] Pull readiness data from `data_quality_output` state
-   [ ] Include plot artifact references with brief interpretations

**Acceptance criteria:**

-   Summary includes readiness breakdown
-   Actionable gating recommendations
-   Plot filenames referenced in report

---

### PR #7: Z-Score Outlier Method (P1)

**Estimated effort:** 2-3 hours  
**Files to modify:** `src/tools/data_quality_tools.py`, `src/tools/eda_describe_tools.py`

-   [x] Add Z-score outlier detection alongside IQR
-   [x] Parameter: `outlier_method: Literal["iqr", "zscore", "both"] = "both"`
-   [x] Report both methods in data quality output
-   [x] Update describe agent to show both
-   [x] Add tests for Z-score calculation

**Acceptance criteria:**

-   [x] Both methods available
-   [x] Results include method used
-   [x] Tests verify correctness

---

### PR #8: Excel & JSON Ingestion (P1)

**Estimated effort:** 3-4 hours  
**Files to create:** `src/tools/ingest_excel_tool.py`, `src/tools/ingest_json_tool.py`  
**Files to modify:** `src/sub_agents/ingestion_agent.py`

-   [ ] Create `ingest_excel_tool`: handle `.xlsx`, multi-sheet detection
-   [ ] Create `ingest_json_tool`: handle records/array/nested structures
-   [ ] Update ingestion agent to route by file extension
-   [ ] Add type inference for JSON fields
-   [ ] Add tests with sample Excel/JSON files

**Acceptance criteria:**

-   Excel files ingest successfully
-   JSON records/arrays both work
-   Multi-sheet Excel prompts user for sheet selection

---

### PR #9: Persistent Dataset Storage (P1)

**Estimated effort:** 4-5 hours  
**Files to modify:** `src/utils/data_store.py`, `src/utils/dataset_cache.py`  
**Files to create:** `config.yaml`

-   [x] Add parquet export to `dataset_cache.py`
-   [x] Create storage directory (default: `./data/datasets/`)
-   [x] Add retrieval by dataset_id
-   [x] Implement cleanup policy (time-based or count-based)
-   [x] Add configuration file for storage paths
-   [x] Create `list_datasets` and `load_dataset` tools
-   [x] Add dataset metadata tracking (created_at, shape, columns)

**Acceptance criteria:**

-   Datasets persist across sessions
-   Retrieval by ID works
-   Config controls storage location
-   Metadata queryable

---

### PR #10: Outlier Visualization Workflow (P2)

**Estimated effort:** 3-4 hours  
**Files to modify:** `src/tools/eda_viz_tools.py`, `src/sub_agents/eda_viz_agent.py`, `src/tools/data_quality_tools.py`

-   [ ] Add suggestion to data quality output when outliers >10%: "Found X outliers. Say 'show without outliers' to compare."
-   [ ] Add `compare_with_without_outliers: bool` parameter to viz tools
-   [ ] Add clear labels: "Including outliers (n=X)" vs "Outliers excluded (n=X)"
-   [ ] Create `create_comparison_viz` tool for side-by-side plots
-   [ ] Update viz agent to offer comparison when appropriate

**Acceptance criteria:**

-   Data quality suggests comparison
-   Plots labeled clearly
-   Side-by-side comparison works

---

### PR #11: Class Imbalance Detection (P1)

**Estimated effort:** 3-4 hours  
**Files to modify:** `src/tools/data_quality_tools.py`

-   [ ] Add target column selection (user hint or heuristic: low cardinality + categorical)
-   [ ] Calculate class ratios and entropy
-   [ ] Add imbalance warnings to quality report
-   [ ] Suggest stratified sampling/SMOTE when imbalance >70/30
-   [ ] Add tests with imbalanced datasets

**Acceptance criteria:**

-   Detects imbalanced targets
-   Ratios and entropy reported
-   Actionable suggestions included

---

### PR #12: High Cardinality & Leakage Heuristics (P1)

**Estimated effort:** 3-4 hours  
**Files to modify:** `src/tools/data_quality_tools.py`

-   [ ] Detect ID-like columns: sequential ints, UUIDs, >95% unique
-   [ ] Flag potential leakage features
-   [ ] Add cardinality warnings (>50% unique for categoricals)
-   [ ] Pattern detection: UUID regex, sequential checks
-   [ ] Add to quality report with severity levels

**Acceptance criteria:**

-   IDs flagged correctly
-   Leakage warnings appear
-   High cardinality categoricals identified

---

### PR #13: Notes MCP Server (P1)

**Estimated effort:** 6-8 hours  
**Files to create:** `mcp_servers/notes_server/`, `mcp_servers/notes_server/server.py`, `mcp_servers/notes_server/notes/`

-   [ ] Create MCP server scaffolding
-   [ ] Implement `search_notes(query: string)` tool
-   [ ] Index local Markdown files (Notion exports)
-   [ ] Simple keyword/semantic search (use sentence-transformers or BM25)
-   [ ] Return snippets with source references
-   [ ] Add installation/setup docs

**Acceptance criteria:**

-   Server runs independently
-   `search_notes` returns relevant snippets
-   Documented setup process

---

### PR #14: Concept Agent (P2)

**Estimated effort:** 4-5 hours  
**Files to create:** `src/sub_agents/concept_agent.py`  
**Files to modify:** `src/agent.py`

-   [ ] Create new `concept_agent` for non-dataset questions
-   [ ] Wire `search_notes` and `web_search` tools
-   [ ] Update root agent routing: dataset → EDA pipeline, conceptual → concept_agent
-   [ ] Add instruction: "First search notes, then web if needed"
-   [ ] Add example queries to agent description

**Acceptance criteria:**

-   Routes conceptual questions correctly
-   Uses notes before web
-   Provides helpful teaching responses

---

### PR #15: Summary Agent External Tool Access (P2)

**Estimated effort:** 2-3 hours  
**Files to modify:** `src/sub_agents/summary_agent.py`

-   [ ] Wire `search_notes` and `web_search` tools to summary agent
-   [ ] Add hard-wall instruction: "Use search ONLY for definitions/context. Never override dataset facts."
-   [ ] Add guard checks in tool calls
-   [ ] Test with queries that might tempt override

**Acceptance criteria:**

-   Can call search for context
-   Never changes numerical results
-   Notes preferred over web

---

### PR #16: Pipeline Runner CLI (P1)

**Estimated effort:** 4-5 hours  
**Files to create:** `run_pipeline.py`

-   [ ] Create CLI script: `run_pipeline.py --file data.csv --report output.md`
-   [ ] Sequential execution: ingest → quality → describe → viz → inference → summary
-   [ ] Optional flags: `--skip-inference`, `--target-column`, `--max-plots`
-   [ ] Progress indicator
-   [ ] Save final report to file
-   [ ] Add `--help` documentation

**Acceptance criteria:**

-   Single command runs full pipeline
-   Flags work correctly
-   Report saved successfully

---

### PR #17: Type Consistency Validation (P2)

**Estimated effort:** 3-4 hours  
**Files to modify:** `src/tools/data_quality_tools.py`

-   [ ] For object columns, check date parse success %
-   [ ] Check numeric coercion %
-   [ ] Add warnings for inconsistent types
-   [ ] Suggest appropriate dtype conversions
-   [ ] Add to quality report

**Acceptance criteria:**

-   Detects mixed types in object columns
-   Parse success rates reported
-   Conversion suggestions actionable

---

### PR #18: Security Hardening for Wrangle (P2)

**Estimated effort:** 4-5 hours  
**Files to modify:** `src/tools/wrangle_tools.py`

-   [ ] Implement safe expression parser with AST whitelist
-   [ ] Ban: `__`, function calls, imports, exec/eval
-   [ ] Allow: column references, operators, literals
-   [ ] Add validation before execution
-   [ ] Clear error messages for blocked operations
-   [ ] Add tests for malicious inputs

**Acceptance criteria:**

-   Malicious code blocked
-   Safe expressions work
-   Clear rejection messages

---

### PR #19: Parallel Agent Orchestration (P1)

**Estimated effort:** 6-8 hours  
**Files to modify:** `src/agent.py`

-   [ ] Implement fan-out for independent EDA steps (describe + viz + inference)
-   [ ] Use ADK parallel execution patterns
-   [ ] Gather results before summary
-   [ ] Add timeout handling
-   [ ] Update state management for concurrent writes
-   [ ] Add tests for parallel execution

**Acceptance criteria:**

-   Describe/viz/inference run concurrently
-   Results aggregate correctly
-   Execution time reduced

---

### PR #20: Structured Logging (P2)

**Estimated effort:** 3-4 hours  
**Files to create:** `src/utils/logging.py`  
**Files to modify:** All `src/tools/*.py`, `src/sub_agents/*.py`

-   [ ] Implement JSON logging: `{timestamp, level, operation, dataset_id, duration, rows, warnings}`
-   [ ] Add trace IDs for request tracking
-   [ ] Log at tool entry/exit points
-   [ ] Add performance metrics
-   [ ] Make log level configurable
-   [ ] Add log aggregation utilities

**Acceptance criteria:**

-   All operations logged
-   JSON format parseable
-   Trace IDs track requests

---

### PR #21: Documentation & README (P1)

**Estimated effort:** 4-6 hours  
**Files to modify:** `README.md`  
**Files to create:** `docs/architecture.md`, `docs/setup.md`, `CONTRIBUTING.md`

-   [ ] Update README with:
    -   Project overview & value proposition
    -   Setup instructions
    -   End-to-end workflow example
    -   Readiness formula explanation
    -   Limitations & assumptions
-   [ ] Create architecture diagram (agents, tools, data flows)
-   [ ] Document workflow patterns (sequential/parallel/loop)
-   [ ] Create CONTRIBUTING.md (setup, testing, code style)
-   [ ] Add code comments explaining heuristics

**Acceptance criteria:**

-   New users can set up from README
-   Architecture clear from diagram
-   Contributing guide comprehensive

---

## 📋 Implementation Roadmap

### Sprint 1: Foundation & Quality (Weeks 1-2)

**Goal:** Clean architecture, testing, core stability

1. **PR #1** - Agent Instruction Refinement ⭐ START HERE
2. **PR #2** - State Key Standardization
3. **PR #3** - Testing Infrastructure
4. **PR #4** - Cross-Platform Path Handling
5. **PR #5** - Unified Error Handling

### Sprint 2: Core Features (Weeks 3-4)

**Goal:** Expand capabilities, improve reporting

6. **PR #6** - Readiness Score in Summary
7. **PR #7** - Z-Score Outlier Method
8. **PR #8** - Excel & JSON Ingestion
9. **PR #9** - Persistent Dataset Storage
10. **PR #11** - Class Imbalance Detection
11. **PR #12** - High Cardinality & Leakage

### Sprint 3: UX & Workflows (Week 5)

**Goal:** Better user experience, smart workflows

12. **PR #10** - Outlier Visualization Workflow
13. **PR #16** - Pipeline Runner CLI
14. **PR #17** - Type Consistency Validation
15. **PR #21** - Documentation & README

### Sprint 4: Advanced Features (Week 6)

**Goal:** External tools, performance, security

16. **PR #13** - Notes MCP Server
17. **PR #14** - Concept Agent
18. **PR #15** - Summary Agent External Tools
19. **PR #18** - Security Hardening for Wrangle
20. **PR #19** - Parallel Agent Orchestration
21. **PR #20** - Structured Logging

---

## 🎯 Quick Win vs Long-term

### Quick Wins (Can ship independently)

-   PR #1: Agent Instruction Refinement (immediate value)
-   PR #2: State Key Standardization (clean foundation)
-   PR #4: Cross-Platform Paths (removes blocker)
-   PR #6: Readiness Score in Summary (user-visible improvement)

### Foundational (Enables future work)

-   PR #3: Testing Infrastructure
-   PR #5: Unified Error Handling
-   PR #9: Persistent Storage

### High-Impact Features (User-facing value)

-   PR #8: Excel & JSON Ingestion
-   PR #11: Class Imbalance Detection
-   PR #13: Notes MCP Server
-   PR #16: Pipeline Runner CLI

### Performance & Scaling

-   PR #19: Parallel Agent Orchestration
-   PR #20: Structured Logging

---

## 🔧 Additional Enhancements (Lower Priority)

### PR #22: Session State Manager (P2)

-   Dataset lineage tracking
-   Last tool outputs separate from dataset store
-   Session replay capability

### PR #23: Deterministic RNG (P2)

-   Add `seed` parameter to CLT sampling
-   Expose in output for reproducibility

### PR #24: Configuration System (P2)

-   YAML config for thresholds
-   Toggle checks on/off
-   Environment-specific settings

### PR #25: Dockerfile & Deployment (P2)

-   Container for reproducible environment
-   Cloud Run deployment guide
-   Environment variable configuration

### PR #26: Suggestion Engine (P2)

-   Propose wrangling based on quality flags
-   Feature engineering suggestions
-   Auto-fix recommendations

### PR #27: Investigation Tools (P3)

-   `investigate_outliers`: show outlier characteristics
-   `analyze_missingness`: deep-dive into patterns
-   `profile_column`: detailed single-column analysis

---

## ⚠️ Known Limitations to Document

1. Dataset size: tested up to X rows (add benchmark)
2. Memory constraints: in-memory storage only (until PR #9)
3. Security: wrangle expressions partially restricted (full hardening in PR #18)
4. Concurrency: sequential execution (until PR #19)
5. File types: CSV only (Excel/JSON in PR #8)

---

## 📊 Success Metrics

Track these after each sprint:

-   **Code Quality:** Test coverage %, lint pass rate
-   **Performance:** Pipeline execution time, memory usage
-   **UX:** Average steps to insight, error rate
-   **Adoption:** Sessions created, datasets analyzed
-   **Reliability:** Uptime, error recovery rate

---

## Original Backlog (Archive)

### Core Feature Gaps

-   [x] (P1) Implement readiness scoring: aggregate penalties (missingness, duplicates, outliers, constants, high-missing columns) → 0–100 + components (done; integrated in `data_quality_tools.py` and surfaced via `data_quality_agent`).
-   [ ] (P1) Class imbalance detection → **PR #11**
-   [ ] (P1) High cardinality + leakage heuristics → **PR #12**
-   [ ] (P1) Add Z-score outlier method → **PR #7**
-   [ ] (P1) Extend ingestion to Excel/JSON → **PR #8**
-   [ ] (P1) Persistent dataset storage → **PR #9**
-   [ ] (P2) Type consistency validation → **PR #17**
-   [ ] (P2) Session state manager → **PR #22**

### Robustness & Tooling

→ **Covered by PRs #3-5, #18, #23**

### Reporting & UX

→ **Covered by PRs #6, #10, #16, #20, #21**

### Enhancements / Stretch

→ **Covered by PRs #26-27**

### Documentation / Governance

→ **Covered by PR #21**

### Quality Assurance

→ **Covered by PR #3**

### Refactors

→ **Add as needed during implementation**

### Deployment / Ops

→ **Covered by PRs #24-25**

### External Tools & MCP Integration

→ **Covered by PRs #13-15**

---

## 💡 Implementation Tips

### For PR #1 (Agent Instruction Refinement):

```python
# Before:
"""You are an EDA manager. You will route the user's request to the
appropriate sub-agent. First, analyze what they're asking for, then
delegate to the right agent, and finally return the results."""

# After:
"""Route requests to sub-agents. Return their output directly.
Do not add commentary."""
```

### For PR #2 (State Key Standardization):

```python
# src/utils/consts.py
from enum import Enum

class StateKeys(str, Enum):
    INGESTION = "ingestion_output"
    DATA_QUALITY = "data_quality_output"
    WRANGLE = "wrangle_output"
    DESCRIBE = "describe_output"
    INFERENCE = "inference_output"
    VIZ = "viz_output"
```

### For PR #3 (Testing Infrastructure):

```python
# tests/test_readiness_score.py
def test_perfect_dataset_scores_100():
    df = pd.DataFrame({'a': range(100), 'b': range(100)})
    score = calculate_readiness_score(df)
    assert score['total'] == 100

def test_all_missing_scores_0():
    df = pd.DataFrame({'a': [None]*100})
    score = calculate_readiness_score(df)
    assert score['missingness'] == 0
```

### For PR #5 (Unified Error Handling):

```python
# src/utils/errors.py
class ErrorResponse:
    def __init__(self, code: str, message: str, hint: str):
        self.success = False
        self.error_code = code
        self.message = message
        self.hint = hint

# In tools:
try:
    df = get_dataset(dataset_id)
except KeyError:
    return ErrorResponse(
        code="DATASET_NOT_FOUND",
        message=f"Dataset '{dataset_id}' not found",
        hint="Use ingestion_agent to load a dataset first"
    )
```

---

## 📝 PR Template

Use this template for each PR:

```markdown
## Description

[Brief description of what this PR does]

## Related TODO Item

Closes: PR #X from TODO.md

## Changes Made

-   [ ] Change 1
-   [ ] Change 2
-   [ ] Change 3

## Testing

-   [ ] Unit tests added/updated
-   [ ] Manual testing completed
-   [ ] Session replay verified

## Acceptance Criteria Met

-   [ ] Criterion 1
-   [ ] Criterion 2

## Before/After

[Screenshots or logs showing improvement]

## Migration Notes

[Any breaking changes or required updates]
```

---

## 🎬 Getting Started

**Ready to start implementing? Here's your first task:**

1. Create a new branch: `git checkout -b feat/agent-instruction-refinement`
2. Start with **PR #1** (Agent Instruction Refinement)
3. Focus on one agent at a time
4. Test with the session replay
5. Measure response length reduction
6. Open PR with template above

**Need help?** Check the implementation tips above for code examples.

---

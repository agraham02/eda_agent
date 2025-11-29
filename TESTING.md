# Testing Infrastructure - Quick Start Guide

## What Was Implemented

✅ **Lightweight testing infrastructure** designed for hackathon speed and demo reliability

### Files Created

1. **`requirements.txt`** - Loose dependency versions for fast setup
2. **`pytest.ini`** - Basic pytest configuration with async mode
3. **`tests/conftest.py`** - Essential fixtures (cleanup, test datasets)
4. **`tests/test_readiness_score.py`** - 4 smoke tests for scoring logic
5. **`tests/test_data_quality.py`** - 3 smoke tests for quality checks
6. **`tests/test_wrangle.py`** - 2 smoke tests for data transformations
7. **`tests/eval/demo_workflow.test.json`** - ADK evaluation for demo path
8. **`tests/eval/error_handling.test.json`** - ADK evaluation for error recovery
9. **`.github/workflows/test.yml`** - CI pipeline (smoke + integration + ADK eval)

### Total Test Count

-   **9 smoke/integration tests** covering critical paths
-   **2 ADK evaluation files** for trajectory validation
-   **No coverage requirements** (hackathon pragmatism)
-   **No property tests** (too complex for timeframe)

## Running Tests

```bash
# Install dependencies first
pip install -r requirements.txt

# Run smoke tests only (fast - for quick checks)
pytest tests/ -v -m smoke

# Run all tests (includes integration)
pytest tests/ -v

# Run ADK evaluations (validates agent behavior)
adk eval eda_agent tests/eval/
```

## Using ADK Web UI for Testing

The ADK Web UI is your primary tool for interactive testing:

```bash
# Start web interface
adk web --port 8000
```

**Workflow:**

1. Run your demo scenario in the Web UI
2. Verify agent responses and tool calls
3. Export the session as a `.test.json` file
4. Save to `tests/eval/` for regression testing

## CI/CD Pipeline

GitHub Actions runs automatically on push to `main` or `pre-release`:

-   ✅ Smoke tests
-   ✅ Integration tests
-   ✅ ADK evaluations

**No coverage reporting** - keeps CI fast and simple for hackathon.

## Test Strategy Summary

### What We Test

-   ✅ Readiness scoring edge cases (perfect, empty, high-missing)
-   ✅ Data quality tool with real datasets (diabetes.csv)
-   ✅ Basic wrangle operations (filter, select)
-   ✅ End-to-end demo workflow (ingest → quality → viz)
-   ✅ Error handling (missing datasets)

### What We Skip (Hackathon Scope)

-   ❌ Property-based tests
-   ❌ Coverage threshold enforcement
-   ❌ Mocking LLM calls
-   ❌ Test data generation scripts
-   ❌ Extensive edge case testing
-   ❌ Linting/formatting checks

## Next Steps

1. **Run tests locally** to verify everything works:

    ```bash
    pytest tests/ -v
    ```

2. **Try ADK Web UI** to record a demo session:

    ```bash
    adk web --port 8000
    ```

3. **Push to GitHub** to trigger CI pipeline

4. **Before demos**, run smoke tests to catch regressions:
    ```bash
    pytest tests/ -m smoke
    ```

## Troubleshooting

**Import errors?**

```bash
pip install -r requirements.txt
```

**Test failures?**

-   Check that `test_data/diabetes.csv` exists
-   Verify dataset store is being cleaned between tests
-   Run with `-v` flag for detailed output

**ADK eval failures?**

-   Test files may need adjustment for your agent's exact responses
-   Use `adk web` to record actual sessions and export correct format

## Philosophy

This testing setup prioritizes:

-   🎯 **Demo reliability** over exhaustive coverage
-   ⚡ **Fast feedback** over comprehensive validation
-   🛠️ **Practical regression detection** over theoretical correctness
-   🎪 **Hackathon speed** over production-grade rigor

Perfect for a hackathon, but you'd want to expand this for production use!

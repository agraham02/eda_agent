# Data Validation Architecture

## Overview

This project uses **Pydantic schemas** to ensure consistent data formats when passing information between agents and tools. This prevents common issues like:

-   Mismatched column names (e.g., `'Life expectancy'` vs `'Life expectancy'`)
-   Invalid chart types (e.g., `'box_plot'` vs `'boxplot'`)
-   Missing required fields
-   Type mismatches

## Architecture

```
Agent A → Pydantic Schema → Agent B
         (validates)        (receives clean data)
```

### Key Components

1. **`schemas.py`**: Central schema definitions using Pydantic
2. **Enum Classes**: Define allowed values (ChartType, SemanticType, etc.)
3. **Validators**: Automatic normalization (quotes, case, underscores)
4. **Helper Functions**: Shared validation logic

## Usage Examples

### 1. Visualization Tools

**Before (bandaid approach)**:

```python
# Manual normalization scattered across files
chart_type = chart_type.lower().replace("_", "")
column_name = column_name.replace("'", "'")
```

**After (schema-based)**:

```python
from ..utils.schemas import VizSpec

# Automatic validation and normalization
spec = VizSpec(
    dataset_id=dataset_id,
    chart_type="box_plot",  # ✅ Auto-normalized to "boxplot"
    x="Life expectancy'",    # ✅ Auto-normalized quotes
    y=None,
    bins=10
)
```

### 2. Data Quality Assessment

```python
from ..utils.schemas import DataQualityColumn, NumericSummary

# Ensures all required fields are present
column_quality = DataQualityColumn(
    name="age",
    pandas_dtype="int64",
    semantic_type=SemanticType.NUMERIC,
    n_missing=5,
    missing_pct=0.05,
    n_unique=50,
    is_constant=False,
    is_all_unique=False,
    issues=[],
    numeric_summary=NumericSummary(...)
)
```

### 3. Statistical Tests

```python
from ..utils.schemas import OneSampleTestResult

# Validates p-values, confidence intervals, etc.
result = OneSampleTestResult(
    test_type="one_sample_t",
    dataset_id="ds_abc123",
    column="height",
    n=100,
    sample_mean=170.5,
    p_value=0.045,  # ✅ Must be between 0 and 1
    alpha=0.05,
    reject_null=True,
    confidence_interval=[165.2, 175.8],
    ...
)
```

## Benefits

### 1. **Automatic Normalization**

Pydantic validators automatically clean inputs:

-   `"box_plot"` → `"boxplot"`
-   `"Life expectancy'"` → `"Life expectancy"`
-   `"SCATTER"` → `"scatter"`

### 2. **Type Safety**

Catch errors at validation time, not runtime:

```python
VizSpec(
    dataset_id="ds_123",
    chart_type="invalid_type",  # ❌ ValidationError
    ...
)
```

### 3. **Self-Documenting**

Schemas serve as API documentation:

```python
class VizSpec(BaseModel):
    """Schema for visualization specifications."""
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: ChartType = Field(..., description="Type of chart")
    x: str = Field(..., description="Column for x-axis")
```

### 4. **Centralized Logic**

Validation logic lives in one place (`schemas.py`), not scattered across tools.

### 5. **ADK Integration**

Compatible with ADK's `output_schema` parameter for structured outputs:

```python
agent = LlmAgent(
    ...
    output_schema=VizSpec,  # LLM output validated automatically
    tools=[...]
)
```

## Implementation Checklist

To migrate a tool to schema-based validation:

-   [ ] **Define Schema**: Create Pydantic model in `schemas.py`
-   [ ] **Add Validators**: Use `@field_validator` for normalization
-   [ ] **Update Tool**: Import and use schema in tool function
-   [ ] **Handle Errors**: Return structured error messages
-   [ ] **Test**: Verify normalization works with edge cases

## Schema Reference

### Core Schemas

| Schema                | Purpose                      |
| --------------------- | ---------------------------- |
| `VizSpec`             | Visualization specifications |
| `VizResult`           | Visualization outputs        |
| `ColumnInfo`          | Column metadata              |
| `DataQualityColumn`   | Quality assessment results   |
| `OneSampleTestResult` | One-sample test outputs      |
| `TwoSampleTestResult` | Two-sample test outputs      |
| `BinomialTestResult`  | Binomial test outputs        |
| `WrangleResult`       | Data transformation results  |

### Enums

| Enum           | Values                                           |
| -------------- | ------------------------------------------------ |
| `ChartType`    | histogram, box, boxplot, scatter, bar, line, pie |
| `SemanticType` | numeric, categorical, datetime, boolean, text    |
| `TestType`     | t, z                                             |
| `Alternative`  | two-sided, less, greater                         |

## Best Practices

### 1. Always Validate at Boundaries

```python
def tool_function(dataset_id: str, column: str):
    # Validate immediately
    spec = ToolInputSchema(dataset_id=dataset_id, column=column)

    # Work with validated data
    result = do_work(spec)

    # Validate output
    return ToolOutputSchema(**result).model_dump()
```

### 2. Use Helper Functions

```python
from ..utils.schemas import validate_column_exists

# Don't write your own validation
column = validate_column_exists(column_name, df.columns.tolist())
```

### 3. Provide Clear Error Messages

```python
except ValidationError as e:
    return {
        "error": str(e),
        "message": "Please check: " + ", ".join(e.errors()),
        "valid_values": [ct.value for ct in ChartType]
    }
```

### 4. Document Schema Changes

When adding/modifying schemas:

-   Update this README
-   Update agent instructions
-   Test with existing tools
-   Check ADK compatibility

## Migration Status

| Tool Module           | Status         | Notes                    |
| --------------------- | -------------- | ------------------------ |
| `eda_viz_tools`       | ✅ Completed   | Using VizSpec, VizResult |
| `eda_describe_tools`  | 🔄 In Progress | -                        |
| `eda_inference_tools` | 🔄 In Progress | -                        |
| `data_quality_tools`  | 📝 Planned     | -                        |
| `wrangle_tools`       | 📝 Planned     | -                        |
| `ingestion_tools`     | 📝 Planned     | -                        |

## Resources

-   [Pydantic Documentation](https://docs.pydantic.dev/)
-   [ADK Structured Outputs](https://medium.com/@gururaser/structured-outputs-with-tools-google-agent-development-kit-adk-136b94be0576)
-   [ADK Function Tools](https://google.github.io/adk-docs/tools-custom/function-tools/)

## FAQs

**Q: Why not just fix the LLM prompts?**  
A: LLMs are probabilistic and may still generate variations. Schemas guarantee consistency.

**Q: Does this slow down the system?**  
A: Pydantic validation is very fast (<1ms). The benefits far outweigh the cost.

**Q: Can I use schemas with ADK's output_schema?**  
A: Yes! That's one of the key benefits. ADK can validate LLM outputs automatically.

**Q: What happens if validation fails?**  
A: The tool returns a structured error with details, allowing the agent to retry with corrections.

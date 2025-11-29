from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.data_store import get_dataset
from ..utils.errors import (
    DATASET_NOT_FOUND,
    INFERENCE_ERROR,
    exception_to_error,
    make_error,
    wrap_success,
)


def _decide_reject(p_value: float, alpha: float) -> bool:
    """
    Decide whether to reject H0 at the given alpha level.
    """
    return p_value <= alpha


# -----------------------------
# ONE SAMPLE TEST
# -----------------------------
def run_one_sample_test(
    dataset_id: str,
    column: str,
    test_type: str,
    mu: float,
    alternative: str,
    alpha: float,
) -> Dict[str, Any]:
    """
    Internal helper to perform a one-sample hypothesis test
    (for example, one-sample t-test or z-test).

    test_type: "t" or "z"
    alternative: "two-sided", "less", or "greater"
    """

    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Please ingest the dataset first using ingest_csv_tool.",
            context={"dataset_id": dataset_id},
        )

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    x = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(x):
        raise ValueError("One-sample tests require a numeric column")

    n = len(x)
    sample_mean = float(x.mean())
    sample_std = float(x.std(ddof=1))
    se = sample_std / np.sqrt(n)

    alternative = alternative.lower()
    test_type = test_type.lower()

    if test_type not in {"t", "z"}:
        raise ValueError("test_type must be 't' or 'z'")

    # --- t test (preferred when sigma unknown, matches your notes)
    if test_type == "t":
        result = stats.ttest_1samp(x, popmean=mu)
        t_stat = float(result.statistic)  # type: ignore[attr-defined]
        p_two = float(result.pvalue)  # type: ignore[attr-defined]

        if alternative == "two-sided":
            p_val = float(p_two)
        elif alternative == "greater":
            # H1: mean > mu
            if t_stat > 0:
                p_val = float(p_two / 2.0)
            else:
                p_val = float(1.0 - p_two / 2.0)
        elif alternative == "less":
            # H1: mean < mu
            if t_stat < 0:
                p_val = float(p_two / 2.0)
            else:
                p_val = float(1.0 - p_two / 2.0)
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        dfree = n - 1
        # confidence interval for mean under t
        ci_level = 1.0 - alpha
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, dfree)
        ci_low = sample_mean - t_crit * se
        ci_high = sample_mean + t_crit * se

        return {
            "test_family": "one_sample",
            "test_type": "one_sample_t",
            "dataset_id": dataset_id,
            "column": column,
            "n": n,
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "hypothesized_mean": mu,
            "target": f"mean({column})",
            "statistic": float(t_stat),
            "df": int(dfree),
            "standard_error": float(se),
            "p_value": p_val,
            "alpha": alpha,
            "reject_null": _decide_reject(p_val, alpha),
            "confidence_level": ci_level,
            "confidence_interval": [float(ci_low), float(ci_high)],
            "effect_size": None,
            "alternative": alternative,
        }

    # --- z test (approx, using sample sd as sigma)
    z_stat = (sample_mean - mu) / se
    if alternative == "two-sided":
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    elif alternative == "greater":
        p_val = float(1.0 - stats.norm.cdf(z_stat))
    elif alternative == "less":
        p_val = float(stats.norm.cdf(z_stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    ci_level = 1.0 - alpha
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = sample_mean - z_crit * se
    ci_high = sample_mean + z_crit * se

    return {
        "test_family": "one_sample",
        "test_type": "one_sample_z",
        "dataset_id": dataset_id,
        "column": column,
        "n": n,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "hypothesized_mean": mu,
        "target": f"mean({column})",
        "statistic": float(z_stat),
        "standard_error": float(se),
        "p_value": p_val,
        "alpha": alpha,
        "reject_null": _decide_reject(p_val, alpha),
        "confidence_level": ci_level,
        "confidence_interval": [float(ci_low), float(ci_high)],
        "effect_size": None,
        "alternative": alternative,
    }


# -----------------------------
# TWO SAMPLE TEST
# -----------------------------
def run_two_sample_test(
    dataset_id: str,
    column: str,
    group_col: str,
    group_a: str,
    group_b: str,
    test_type: str,
    alternative: str,
    alpha: float,
) -> Dict[str, Any]:
    """
    Internal helper to perform a two-sample hypothesis test
    comparing groups A and B on a given column.

    test_type: "t" or "z"
    alternative: "two-sided", "less", or "greater"
      "greater" is interpreted as mean(group_a) > mean(group_b)
    """

    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Please ingest the dataset first using ingest_csv_tool.",
            context={"dataset_id": dataset_id},
        )

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )
    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in dataset. Available columns: {list(df.columns)}"
        )

    # filter two groups
    df_ab = df[df[group_col].isin([group_a, group_b])]

    x = df_ab[df_ab[group_col] == group_a][column].dropna()
    y = df_ab[df_ab[group_col] == group_b][column].dropna()

    if len(x) == 0 or len(y) == 0:
        raise ValueError("One or both groups have no data")

    if not pd.api.types.is_numeric_dtype(x) or not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Two-sample tests require a numeric outcome column")

    alternative = alternative.lower()
    test_type = test_type.lower()

    mean_a = float(x.mean())
    mean_b = float(y.mean())
    std_a = float(x.std(ddof=1))
    std_b = float(y.std(ddof=1))
    n_a = len(x)
    n_b = len(y)

    diff = mean_a - mean_b

    # effect size - Cohen's d (pooled sd)
    pooled_var = (((n_a - 1) * std_a**2) + ((n_b - 1) * std_b**2)) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)
    cohen_d = diff / pooled_sd if pooled_sd > 0 else np.nan

    if test_type not in {"t", "z"}:
        raise ValueError("test_type must be 't' or 'z'")

    # --- t test with Welch correction by default
    if test_type == "t":
        result = stats.ttest_ind(x, y, equal_var=False)
        t_stat = float(result.statistic)  # type: ignore[attr-defined]
        p_two = float(result.pvalue)  # type: ignore[attr-defined]

        if alternative == "two-sided":
            p_val = float(p_two)
        elif alternative == "greater":
            # H1: mean_a > mean_b
            if t_stat > 0:
                p_val = float(p_two / 2.0)
            else:
                p_val = float(1.0 - p_two / 2.0)
        elif alternative == "less":
            # H1: mean_a < mean_b
            if t_stat < 0:
                p_val = float(p_two / 2.0)
            else:
                p_val = float(1.0 - p_two / 2.0)
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        # Welch df
        se_a2 = std_a**2 / n_a
        se_b2 = std_b**2 / n_b
        se_diff = np.sqrt(se_a2 + se_b2)
        df_num = (se_a2 + se_b2) ** 2
        df_den = (se_a2**2) / (n_a - 1) + (se_b2**2) / (n_b - 1)
        dfree = df_num / df_den if df_den > 0 else n_a + n_b - 2

        ci_level = 1.0 - alpha
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, dfree)
        ci_low = diff - t_crit * se_diff
        ci_high = diff + t_crit * se_diff

        return {
            "test_family": "two_sample",
            "test_type": "two_sample_t",
            "dataset_id": dataset_id,
            "column": column,
            "group_col": group_col,
            "group_a": group_a,
            "group_b": group_b,
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "mean_diff": diff,
            "target": f"mean({group_a}) - mean({group_b}) on {column}",
            "statistic": float(t_stat),
            "df": float(dfree),
            "standard_error_diff": float(se_diff),
            "p_value": p_val,
            "alpha": alpha,
            "reject_null": _decide_reject(p_val, alpha),
            "confidence_level": ci_level,
            "confidence_interval": [float(ci_low), float(ci_high)],
            "effect_size": float(cohen_d),
            "cohen_d": float(cohen_d),
            "alternative": alternative,
        }

    # --- z test for difference in means, using sample sd as sigma
    se_diff = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
    z_stat = diff / se_diff

    if alternative == "two-sided":
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    elif alternative == "greater":
        p_val = float(1.0 - stats.norm.cdf(z_stat))
    elif alternative == "less":
        p_val = float(stats.norm.cdf(z_stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    ci_level = 1.0 - alpha
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    return {
        "test_family": "two_sample",
        "test_type": "two_sample_z",
        "dataset_id": dataset_id,
        "column": column,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
        "mean_diff": diff,
        "target": f"mean({group_a}) - mean({group_b}) on {column}",
        "statistic": float(z_stat),
        "standard_error_diff": float(se_diff),
        "p_value": p_val,
        "alpha": alpha,
        "reject_null": _decide_reject(p_val, alpha),
        "confidence_level": ci_level,
        "confidence_interval": [float(ci_low), float(ci_high)],
        "effect_size": float(cohen_d),
        "cohen_d": float(cohen_d),
        "alternative": alternative,
    }


# -----------------------------
# BINOMIAL TEST
# -----------------------------
def run_binomial_test(
    successes: int,
    n: int,
    p0: float,
    alternative: str,
    alpha: float,
) -> Dict[str, Any]:
    """
    Internal helper to perform a binomial hypothesis test.
    Uses scipy.stats.binomtest which matches your notes.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be between 0 and 1")

    alternative = alternative.lower()
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    # SciPy new API
    res = stats.binomtest(k=successes, n=n, p=p0, alternative=alternative)
    p_val = float(res.pvalue)

    # Confidence interval for true proportion
    ci_level = 1.0 - alpha
    ci = res.proportion_ci(confidence_level=ci_level, method="wilson")
    ci_low = float(ci.low)
    ci_high = float(ci.high)

    return {
        "test_family": "binomial",
        "test_type": "binomial",
        "successes": successes,
        "n": n,
        "observed_proportion": successes / n,
        "hypothesized_proportion": p0,
        "target": "proportion",
        "statistic": None,
        "p_value": p_val,
        "alpha": alpha,
        "reject_null": _decide_reject(p_val, alpha),
        "confidence_level": ci_level,
        "confidence_interval": [ci_low, ci_high],
        "effect_size": None,
        "alternative": alternative,
    }


# -----------------------------
# CLT SAMPLING DEMO
# -----------------------------
def build_clt_sampling_summary(
    dataset_id: str,
    column: str,
    sample_size: int,
    n_samples: int,
) -> Dict[str, Any]:
    """
    Internal helper to simulate sampling distributions and illustrate
    the Central Limit Theorem for a numeric column.
    """

    try:
        df = get_dataset(dataset_id)
    except KeyError as e:
        return make_error(
            DATASET_NOT_FOUND,
            str(e),
            hint="Please ingest the dataset first using ingest_csv_tool.",
            context={"dataset_id": dataset_id},
        )

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column].dropna()

    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("CLT sampling demo requires a numeric column")

    if sample_size <= 0 or n_samples <= 0:
        raise ValueError("sample_size and n_samples must be positive")

    arr = series.to_numpy()
    N = len(arr)
    if N == 0:
        raise ValueError("Column has no non missing values")

    rng = np.random.default_rng()

    # Draw samples with replacement and compute sample means
    # shape: (n_samples, sample_size)
    samples = rng.choice(arr, size=(n_samples, sample_size), replace=True)
    sample_means = samples.mean(axis=1)

    # Population estimates from available data
    pop_mean_est = float(arr.mean())
    pop_std_est = float(arr.std(ddof=1))

    # Sampling distribution summary
    mean_of_means = float(sample_means.mean())
    std_of_means = float(sample_means.std(ddof=1))
    se_theoretical = pop_std_est / np.sqrt(sample_size)

    percentiles = np.percentile(sample_means, [2.5, 25, 50, 75, 97.5])
    percentiles = [float(x) for x in percentiles]

    return {
        "test_family": "clt_sampling",
        "test_type": "clt_sampling_demo",
        "dataset_id": dataset_id,
        "column": column,
        "target": f"sampling distribution of mean({column})",
        "statistic": None,
        "p_value": None,
        "alpha": None,
        "reject_null": None,
        "confidence_interval": None,
        "effect_size": None,
        "population_estimate": {
            "mean": pop_mean_est,
            "std": pop_std_est,
            "n": int(N),
        },
        "sampling_parameters": {
            "sample_size": int(sample_size),
            "n_samples": int(n_samples),
        },
        "sampling_distribution": {
            "mean_of_sample_means": mean_of_means,
            "std_of_sample_means": std_of_means,
            "theoretical_standard_error": float(se_theoretical),
            "percentiles": {
                "2.5": percentiles[0],
                "25": percentiles[1],
                "50": percentiles[2],
                "75": percentiles[3],
                "97.5": percentiles[4],
            },
        },
        "sample_means_preview": sample_means[:50].tolist(),
    }


# -----------------------------
# TOOL WRAPPERS
# -----------------------------
def eda_one_sample_test_tool(
    dataset_id: str,
    column: str,
    test_type: str = "t",
    mu: float = 0.0,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Tool wrapper to run a one-sample hypothesis test on a column.
    """
    try:
        result = run_one_sample_test(
            dataset_id=dataset_id,
            column=column,
            test_type=test_type,
            mu=mu,
            alternative=alternative,
            alpha=alpha,
        )
        return wrap_success(result)
    except Exception as e:
        return exception_to_error(
            INFERENCE_ERROR,
            e,
            hint="Check dataset_id, column name, and that the column is numeric",
        )


def eda_two_sample_test_tool(
    dataset_id: str,
    column: str,
    group_col: str,
    group_a: str,
    group_b: str,
    test_type: str = "t",
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Tool wrapper to run a two-sample hypothesis test comparing
    two groups in the dataset.
    """
    try:
        result = run_two_sample_test(
            dataset_id=dataset_id,
            column=column,
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            test_type=test_type,
            alternative=alternative,
            alpha=alpha,
        )
        return wrap_success(result)
    except Exception as e:
        return exception_to_error(
            INFERENCE_ERROR,
            e,
            hint="Check dataset_id, column names, group column, and group values exist",
        )


def eda_binomial_test_tool(
    successes: int,
    n: int,
    p0: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Tool wrapper to run a binomial hypothesis test for proportion data.
    """
    try:
        result = run_binomial_test(
            successes=successes,
            n=n,
            p0=p0,
            alternative=alternative,
            alpha=alpha,
        )
        return wrap_success(result)
    except Exception as e:
        return exception_to_error(
            INFERENCE_ERROR,
            e,
            hint="Check that successes <= n, n > 0, and 0 < p0 < 1",
        )


def eda_clt_sampling_tool(
    dataset_id: str,
    column: str,
    sample_size: int = 30,
    n_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Tool wrapper to generate a sampling distribution of the mean
    and summarize CLT behavior for a numeric column.
    """
    try:
        result = build_clt_sampling_summary(
            dataset_id=dataset_id,
            column=column,
            sample_size=sample_size,
            n_samples=n_samples,
        )
        return wrap_success(result)
    except Exception as e:
        return exception_to_error(
            INFERENCE_ERROR,
            e,
            hint="Check dataset_id, column name, column is numeric, sample_size and n_samples are positive",
        )

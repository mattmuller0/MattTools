"""Statistical analysis functions for MattTools package.

This module provides statistical functions including confidence intervals,
bootstrap methods, AUC calculations, and odds ratio analysis.
"""

import logging
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import roc_auc_score

from ._base import ArrayLike, ensure_array

# Configure module logger
logger = logging.getLogger(__name__)

__all__ = [
    "mean_confidence_interval",
    "bootstrap_auc_confidence",
    "Bootstrap",
    "odds_ratio",
]


def mean_confidence_interval(
    data: ArrayLike,
    confidence: float = 0.95,
    axis: Optional[int] = None,
) -> Tuple[np.floating, np.ndarray]:
    """Compute the mean and confidence interval of the mean.

    Args:
        data: Input data array, list, or pandas Series.
        confidence: Confidence level (0-1). Default is 0.95 (95%).
        axis: Axis along which to compute. Default is None.

    Returns:
        Tuple of (mean, confidence_interval_array) where CI array has shape (n, 2).

    Raises:
        ValueError: If confidence level or data is invalid.
    """
    if confidence > 1:
        confidence = confidence / 100.0
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

    a = ensure_array(data).astype(float)
    if a.size == 0:
        raise ValueError("Input data is empty")
    if a.ndim > 2:
        raise ValueError("Input must be 1D or 2D array")

    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    m = np.mean(a, axis=axis)
    se = st.sem(a, axis=axis)
    t = st.t.ppf((1 + confidence) / 2.0, n - 1)
    ci = np.c_[m - se * t, m + se * t]

    return m, ci


def bootstrap_auc_confidence(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    ci: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    plot_histogram: bool = False,
) -> Tuple[float, float, float]:
    """Compute AUC with bootstrap confidence interval.

    Args:
        y_pred: Predicted probabilities.
        y_true: True binary labels.
        ci: Confidence level (default: 0.95).
        n_bootstrap: Number of bootstrap iterations (default: 1000).
        random_state: Random seed for reproducibility.
        plot_histogram: Whether to plot histogram of AUC values.

    Returns:
        Tuple of (mean_auc, ci_lower, ci_upper).

    Raises:
        ValueError: If inputs are invalid.
    """
    y_pred = ensure_array(y_pred)
    y_true = ensure_array(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")
    if len(y_pred) == 0:
        raise ValueError("Input arrays are empty")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least 2 classes")
    if not 0 < ci < 1:
        raise ValueError(f"CI must be between 0 and 1, got {ci}")

    rng = np.random.default_rng(random_state)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_pred[idx]))

    if plot_histogram:
        plt.figure()
        plt.hist(scores, bins=50)
        plt.title("Bootstrap AUC Distribution")
        plt.xlabel("AUC")
        plt.ylabel("Frequency")
        plt.show()
        plt.close()

    mean_auc, ci_array = mean_confidence_interval(scores, confidence=ci)
    return float(mean_auc), float(ci_array[0, 0]), float(ci_array[0, 1])


class Bootstrap:
    """Bootstrap resampler compatible with scikit-learn cross-validation API.

    Args:
        n_bootstrap: Number of bootstrap iterations.
        stratified: Whether to maintain class balance.
        random_state: Random seed for reproducibility.

    Example:
        >>> bootstrap = Bootstrap(n_bootstrap=10, stratified=True, random_state=42)
        >>> for train_idx, test_idx in bootstrap.split(X, y):
        ...     X_boot, y_boot = X[train_idx], y[train_idx]
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        stratified: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.stratified = stratified
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """Return number of splitting iterations."""
        return self.n_bootstrap

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ):
        """Generate bootstrap samples.

        Args:
            X: Data to bootstrap.
            y: Labels (required for stratified sampling).
            groups: Ignored, for API compatibility.

        Yields:
            Tuple of (train_indices, test_indices).

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("y is required for bootstrap splitting")

        X = ensure_array(X)
        y = ensure_array(y)
        rng = np.random.default_rng(self.random_state)

        if not self.stratified:
            warnings.warn(
                "Non-stratified bootstrap may result in unbalanced classes",
                UserWarning,
            )
            indices = np.arange(len(X))
            for _ in range(self.n_bootstrap):
                boot_idx = rng.choice(indices, size=len(X), replace=True)
                while len(np.unique(y[boot_idx])) < 2:
                    boot_idx = rng.choice(indices, size=len(X), replace=True)
                yield boot_idx, boot_idx
        else:
            labels = np.unique(y)
            label_indices = [np.where(y == label)[0] for label in labels]
            for _ in range(self.n_bootstrap):
                sampled = [rng.choice(idx, size=len(idx), replace=True) for idx in label_indices]
                boot_idx = np.concatenate(sampled)
                rng.shuffle(boot_idx)
                yield boot_idx, boot_idx


def odds_ratio(
    df: pd.DataFrame,
    targets: List[str],
    columns: List[str],
    plot: bool = False,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Calculate odds ratios of target columns against comparison columns.

    Args:
        df: DataFrame containing the data.
        targets: Column names to calculate odds ratios for.
        columns: Column names to calculate odds ratio against.
        plot: Whether to plot results.
        ci_level: Confidence level for intervals.

    Returns:
        DataFrame with odds ratios, p-values, and confidence intervals.
    """
    df_work = df.copy()
    result_cols = []
    for col in columns:
        result_cols.extend([col, f"{col}_pvalue", f"{col}_ci_lower", f"{col}_ci_upper"])
    odds_df = pd.DataFrame(columns=result_cols, index=targets, dtype=float)

    for target in targets:
        for column in columns:
            df_filtered = df_work[df_work[column] != 2].copy()

            if len(np.unique(df_filtered[column])) != 2 or len(np.unique(df_filtered[target])) != 2:
                logger.warning(f"Skipping {target} vs {column}: requires 2 unique values")
                continue

            try:
                crosstab = pd.crosstab(df_filtered[target], df_filtered[column])
                res = st.contingency.odds_ratio(crosstab)
                odds_df.loc[target, column] = res.statistic
                odds_df.loc[target, f"{column}_pvalue"] = st.fisher_exact(crosstab)[1]
                ci = res.confidence_interval(ci_level)
                odds_df.loc[target, f"{column}_ci_lower"] = ci[0]
                odds_df.loc[target, f"{column}_ci_upper"] = ci[1]
            except Exception as e:
                logger.warning(f"Error calculating odds ratio for {target} vs {column}: {e}")

    if plot:
        import seaborn as sns

        if len(columns) > 1:
            plot_df = odds_df[columns].replace([-np.inf, np.inf], [0, 10])
            aspect = plot_df.shape[1] / plot_df.shape[0]
            plt.figure(figsize=(aspect * 5, 5))
            sns.heatmap(plot_df, linewidths=0.5, linecolor="black", cmap="Blues")
            plt.title("Odds Ratios Heatmap")
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            col = columns[0]
            plot_df = odds_df[[col]].replace([-np.inf, np.inf], [0, 10])
            ci_lower = odds_df[f"{col}_ci_lower"]
            ci_upper = odds_df[f"{col}_ci_upper"]
            xerr = [plot_df[col] - ci_lower, ci_upper - plot_df[col]]

            plt.figure(figsize=(8, 6))
            plt.errorbar(x=plot_df[col], y=plot_df.index, xerr=xerr, fmt="o", capsize=5, color="black")
            plt.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="OR = 1")
            plt.xlabel("Odds Ratio")
            plt.ylabel("Comparison Groups")
            plt.title(f"Odds Ratio of {col}")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

    return odds_df

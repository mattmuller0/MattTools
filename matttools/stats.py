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

# Configure module logger
logger = logging.getLogger(__name__)


# Function to calculate the confidence interval of a dataset
def mean_confidence_interval(
    data: Union[np.ndarray, List, pd.Series],
    confidence: float = 0.95,
    axis: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Compute the mean and confidence interval of the mean.

    Note that the confidence interval is often misinterpreted. See references for details.

    Args:
        data: Input data array, list, or pandas Series
        confidence: Confidence level (0-1 or 0-100). Default is 0.95 (95%)
        axis: Axis along which to compute the mean and CI. Default is None

    Returns:
        Tuple containing (mean, confidence_interval_array)
        where confidence_interval_array has shape (n, 2) with [lower, upper] bounds

    References:
        [1] https://en.wikipedia.org/wiki/Confidence_interval
        [2] https://en.wikipedia.org/wiki/Tolerance_interval
        [3] https://en.wikipedia.org/wiki/Confidence_interval#Meaning_and_interpretation
    """
    # Input validation
    if confidence <= 0 or confidence >= 1:
        if confidence > 1:
            # Convert percentage to decimal if needed
            confidence = confidence / 100.0
            if confidence <= 0 or confidence >= 1:
                raise ValueError(
                    f"Confidence level must be between 0 and 1, got {confidence}"
                )
        else:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {confidence}"
            )

    # Convert input to numpy array
    try:
        a = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Data must be convertible to numeric array: {e}")

    if a.size == 0:
        raise ValueError("Input data is empty")

    if a.ndim > 2:
        raise ValueError("Input data must be 1D or 2D array")

    n = len(a)
    if n < 2:
        raise ValueError(
            "Need at least 2 data points for confidence interval calculation"
        )

    # Both s=std() and se=sem() use unbiased estimators (ddof=1).
    m = np.mean(a, axis=axis)
    se = st.sem(a, axis=axis)
    t = st.t.ppf((1 + confidence) / 2.0, n - 1)
    ci = np.c_[m - se * t, m + se * t]

    return m, ci


def bootstrap_auc_confidence(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ci: float = 0.95,
    n_bootstraps: int = 1000,
    rng_seed: int = 100,
    plot_histogram: bool = False,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Binary target implementation of AUC bootstrapping for determining a confidence interval.

    Args:
        y_pred: numpy array of predicted values, usually from predict_proba method
        y_true: numpy array of true label values (1 is the presumed target)
        ci: confidence level for the interval (default: 0.95)
        n_bootstraps: number of bootstrap iterations (default: 1000)
        rng_seed: random seed for reproducibility (default: 100)
        plot_histogram: whether to plot histogram of AUC values (default: False)
        verbose: whether to print results (default: True)

    Returns:
        Tuple containing (working_roc_auc, confidence_lower, confidence_upper)

    Raises:
        ValueError: If input arrays are invalid or have mismatched shapes.

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        >>> auc, lower, upper = bootstrap_auc_confidence(y_pred, y_true)
    """
    # Input validation
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred and y_true must have same shape, got {y_pred.shape} and {y_true.shape}"
        )

    if len(y_pred) == 0:
        raise ValueError("Input arrays are empty")

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least 2 different classes")

    if not isinstance(n_bootstraps, int) or n_bootstraps <= 0:
        raise ValueError(f"n_bootstraps must be a positive integer, got {n_bootstraps}")

    if ci <= 0 or ci >= 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {ci}")

    working_roc_auc = roc_auc_score(y_true, y_pred)
    bootstrapped_scores = []
    rng = np.random.default_rng(rng_seed)

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.integers(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    if plot_histogram:
        plt.hist(bootstrapped_scores, bins=50)
        plt.title("Histogram of the bootstrapped ROC AUC scores")
        plt.xlabel("AUC Score")
        plt.ylabel("Frequency")
        plt.show()

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    working_roc_auc, ci_array = mean_confidence_interval(sorted_scores, confidence=ci)
    confidence_lower, confidence_upper = ci_array[0, 0], ci_array[0, 1]

    if verbose:
        logger.info(
            f"Confidence interval for the score: {working_roc_auc:0.3f} "
            f"[{confidence_lower:0.3f} - {confidence_upper:0.3f}]"
        )

    return working_roc_auc, confidence_lower, confidence_upper


class Bootstrap:
    """Bootstrap resampler compatible with scikit-learn cross-validation API.

    This class implements bootstrap resampling with optional stratification
    for maintaining class balance in bootstrap samples.

    Attributes:
        n_bootstrap: Number of bootstrap iterations to perform.
        stratified: Whether to maintain class balance in bootstrap samples.
        rng_seed: Random seed for reproducibility.

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 0, 1, 1])
        >>> bootstrap = Bootstrap(n_bootstrap=10, stratified=True, rng_seed=42)
        >>> for train_idx, test_idx in bootstrap.split(X, y):
        ...     X_boot, y_boot = X[train_idx], y[train_idx]
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        stratified: bool = True,
        rng_seed: Optional[int] = None,
    ):
        """Initialize Bootstrap resampler.

        Args:
            n_bootstrap: Number of bootstrap iterations to perform.
            stratified: Whether to maintain class balance in bootstrap samples.
            rng_seed: Random seed for reproducibility. If None, uses random state.
        """
        self.n_bootstrap = n_bootstrap
        self.stratified = stratified
        self.rng_seed = rng_seed

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Args:
            X: Ignored, present for API compatibility.
            y: Ignored, present for API compatibility.
            groups: Ignored, present for API compatibility.

        Returns:
            Number of bootstrap iterations.
        """
        return self.n_bootstrap

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups=None):
        """Generate bootstrap samples.

        Args:
            X: numpy array of data to bootstrap (can be n-dimensional).
            y: numpy array of labels to bootstrap (required for stratified sampling).
            groups: Ignored, present for API compatibility.

        Yields:
            Tuple of (train_indices, test_indices). For bootstrap, both are the same
            and represent the resampled indices.

        Raises:
            ValueError: If y is None (required for bootstrap splitting).
        """
        if y is None:
            raise ValueError("y parameter is required for bootstrap splitting")

        rng = np.random.default_rng(self.rng_seed)

        if not self.stratified:
            warnings.warn(
                "Stratified is False, this may result in unbalanced classes "
                "in the bootstrapped datasets",
                UserWarning,
            )
            indices = np.arange(len(X))
            for i in range(self.n_bootstrap):
                boot_indices = rng.choice(indices, size=len(X), replace=True)
                # Only use indices with all classes
                while len(np.unique(y[boot_indices])) < 2:
                    boot_indices = rng.choice(indices, size=len(X), replace=True)
                yield boot_indices, boot_indices
        else:
            # split data by unique labels
            labels = np.unique(y)
            # get indices for each label
            label_indices = [np.where(y == label)[0] for label in labels]
            # randomly sample each set of indices
            for i in range(self.n_bootstrap):
                # sample indices for each class
                sampled_indices = [
                    rng.choice(ind, size=len(ind), replace=True)
                    for ind in label_indices
                ]
                # concatenate sampled indices
                boot_indices = np.concatenate(sampled_indices)
                # shuffle indices
                rng.shuffle(boot_indices)
                # yield sampled indices
                yield boot_indices, boot_indices


def odds_ratio(
    df: pd.DataFrame,
    targets: List[str],
    columns: List[str],
    plot: bool = False,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Calculate the odds ratio of target columns against comparison columns.

    Args:
        df: pandas DataFrame containing the data.
        targets: List of column names to calculate the odds ratios for.
        columns: List of column names to calculate the odds ratio against.
        plot: Whether to plot the odds ratio. Default is False.
        ci_level: Confidence level for confidence intervals. Default is 0.95.

    Returns:
        DataFrame containing odds ratios, p-values, and confidence intervals.

    Note:
        This function filters out rows where column values equal 2 and requires
        binary columns (2 unique values) for odds ratio calculation.

    Example:
        >>> df = pd.DataFrame({'outcome': [0, 1, 0, 1], 'exposure': [0, 1, 0, 1]})
        >>> result = odds_ratio(df, targets=['outcome'], columns=['exposure'])
    """
    # Create a copy to avoid mutating the input
    df_work = df.copy()

    # Initialize results DataFrame
    result_columns = []
    for col in columns:
        result_columns.extend(
            [col, f"{col}_pvalue", f"{col}_ci_lower", f"{col}_ci_upper"]
        )
    odds_ratio_df = pd.DataFrame(columns=result_columns, index=targets, dtype=float)

    # Calculate odds ratios
    for target in targets:
        for column in columns:
            # Filter data for this comparison
            df_filtered = df_work[df_work[column] != 2].copy()

            # Check if both columns are binary
            if (
                len(np.unique(df_filtered[column])) != 2
                or len(np.unique(df_filtered[target])) != 2
            ):
                logger.warning(
                    f"Skipping {target} vs {column}: requires exactly 2 unique values"
                )
                continue

            # Calculate odds ratio
            try:
                crosstab = pd.crosstab(df_filtered[target], df_filtered[column])
                res = st.contingency.odds_ratio(crosstab)

                odds_ratio_df.loc[target, column] = res.statistic
                odds_ratio_df.loc[target, f"{column}_pvalue"] = st.fisher_exact(
                    crosstab
                )[1]
                ci = res.confidence_interval(ci_level)
                odds_ratio_df.loc[target, f"{column}_ci_lower"] = ci[0]
                odds_ratio_df.loc[target, f"{column}_ci_upper"] = ci[1]
            except Exception as e:
                logger.warning(
                    f"Error calculating odds ratio for {target} vs {column}: {e}"
                )
                continue

    # Plot if requested
    if plot:
        import seaborn as sns

        if len(columns) > 1:
            # Multi-column heatmap
            plot_df = odds_ratio_df[columns].copy()
            plot_df = plot_df.replace([-np.inf, np.inf], [0, 10])

            aspect = plot_df.shape[1] / plot_df.shape[0]
            plt.figure(figsize=(aspect * 5, 5))
            sns.heatmap(plot_df, linewidths=0.5, linecolor="black", cmap="Blues")
            plt.title("Odds Ratios Heatmap")
            plt.tight_layout()
            plt.show()
        else:
            # Single column error bar plot
            col = columns[0]
            plot_df = odds_ratio_df[[col]].copy()
            plot_df = plot_df.replace([-np.inf, np.inf], [0, 10])

            ci_lower = odds_ratio_df[f"{col}_ci_lower"]
            ci_upper = odds_ratio_df[f"{col}_ci_upper"]
            xerr = [plot_df[col] - ci_lower, ci_upper - plot_df[col]]

            plt.figure(figsize=(8, 6))
            plt.errorbar(
                x=plot_df[col],
                y=plot_df.index,
                xerr=xerr,
                fmt="o",
                capsize=5,
                color="black",
            )
            plt.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="OR = 1")
            plt.xlabel("Odds Ratio")
            plt.ylabel("Comparison Groups")
            plt.title(f"Odds Ratio of {col}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return odds_ratio_df

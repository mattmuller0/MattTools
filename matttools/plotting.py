"""Plotting utilities for machine learning model evaluation.

This module provides visualization functions for ROC curves, PR curves,
confusion matrices, dimensionality reduction, and model comparison.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.utils import resample

from ._base import ArrayLike, ensure_array
from .stats import mean_confidence_interval

logger = logging.getLogger(__name__)

__all__ = [
    "plot_curves",
    "plot_curve_ci",
    "plot_confusion_matrices",
    "plot_model_results",
    "plot_reduction",
    "plot_scree",
    "plot_training_probas",
]


def _save_or_show(save_path: Optional[Union[str, Path]], close: bool = True) -> None:
    """Save figure or show it."""
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()
    if close:
        plt.close()


def plot_curves(
    models: Dict[str, BaseEstimator],
    X: ArrayLike,
    y: ArrayLike,
    curve_type: Literal["roc", "prc"] = "roc",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Axes:
    """Plot ROC or PR curves for multiple models.

    Args:
        models: Dictionary mapping model names to fitted estimators.
        X: Feature data.
        y: Target labels.
        curve_type: "roc" for ROC curve, "prc" for Precision-Recall curve.
        ax: Matplotlib axes. Creates new if None.
        save_path: Path to save figure. Shows if None.
        figsize: Figure size if creating new axes.

    Returns:
        The matplotlib Axes object.
    """
    X = ensure_array(X)
    y = ensure_array(y)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for name, model in models.items():
        y_score = model.predict_proba(X)[:, 1]

        if curve_type == "roc":
            fpr, tpr, _ = roc_curve(y, y_score)
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.2f})")
        else:  # prc
            precision, recall, _ = precision_recall_curve(y, y_score)
            score = average_precision_score(y, y_score)
            ax.plot(recall, precision, label=f"{name} (AP = {score:.2f})")

    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
    else:
        ax.axhline(y=np.mean(y), color="k", linestyle="--", label="Chance")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")

    ax.legend(loc="best")

    if save_path or ax is None:
        _save_or_show(save_path, close=save_path is not None)

    return ax


def plot_curve_ci(
    model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    curve_type: Literal["roc", "prc"] = "roc",
    resampling: Literal["bootstrap", "cv"] = "bootstrap",
    n_iterations: int = 100,
    cv: Optional[Union[int, BaseCrossValidator]] = None,
    ci: float = 0.95,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (6, 6),
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot ROC or PR curve with confidence interval.

    Args:
        model: Sklearn model (fitted for bootstrap, unfitted for cv).
        X: Feature data.
        y: Target labels.
        curve_type: "roc" or "prc".
        resampling: "bootstrap" or "cv" for cross-validation.
        n_iterations: Number of bootstrap iterations (ignored for cv).
        cv: CV splitter for cross-validation mode.
        ci: Confidence level (default 0.95).
        ax: Matplotlib axes. Creates new if None.
        save_path: Path to save figure.
        figsize: Figure size if creating new axes.
        title: Plot title. Auto-generated if None.

    Returns:
        The matplotlib Axes object.
    """
    X = ensure_array(X)
    y = ensure_array(y)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    mean_x = np.linspace(0, 1, 100)
    curves = []
    scores = []

    if resampling == "cv":
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True)
        model = clone(model)
        iterator = cv.split(X, y)
    else:
        iterator = range(n_iterations)

    for item in iterator:
        if resampling == "cv":
            train_idx, test_idx = item
            model.fit(X[train_idx], y[train_idx])
            X_eval, y_eval = X[test_idx], y[test_idx]
        else:
            X_eval, y_eval = resample(X, y, stratify=y)

        y_score = model.predict_proba(X_eval)[:, 1]

        if curve_type == "roc":
            x_vals, y_vals, _ = roc_curve(y_eval, y_score)
            score = auc(x_vals, y_vals)
            interp_y = np.interp(mean_x, x_vals, y_vals)
            interp_y[0] = 0.0
        else:
            precision, recall, _ = precision_recall_curve(y_eval, y_score)
            score = auc(recall, precision)
            interp_y = np.interp(mean_x, recall[::-1], precision[::-1])
            interp_y[0] = 1.0

        curves.append(interp_y)
        scores.append(score)

    # Calculate mean and CI
    mean_y, ci_y = mean_confidence_interval(curves, confidence=ci, axis=0)
    mean_score, ci_score = mean_confidence_interval(scores, confidence=ci)

    if curve_type == "roc":
        mean_y[-1] = 1.0
    else:
        mean_y[-1] = 0.0

    ci_half = ci_y[:, 1] - mean_y
    score_ci = ci_score[0, 1] - mean_score

    # Plot
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5)")
        metric_name = "AUC"
    else:
        ax.axhline(y=np.mean(y), color="k", linestyle="--", label="Chance")
        metric_name = "AP"

    upper = np.minimum(mean_y + ci_half, 1)
    lower = np.maximum(mean_y - ci_half, 0)
    ax.fill_between(mean_x, lower, upper, color="grey", alpha=0.2, label=f"{int(ci*100)}% CI")
    ax.plot(mean_x, mean_y, "b-", lw=2, label=f"{metric_name} = {float(mean_score):.2f} ± {float(score_ci):.2f}")

    if curve_type == "roc":
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    if title is None:
        method = "Bootstrap" if resampling == "bootstrap" else "Cross-Validation"
        curve_name = "ROC" if curve_type == "roc" else "PR"
        title = f"{curve_name} Curve with {int(ci*100)}% CI ({method})"
    ax.set_title(title)
    ax.legend(loc="lower right" if curve_type == "roc" else "lower left")

    if save_path or ax is None:
        _save_or_show(save_path, close=save_path is not None)

    return ax


def plot_confusion_matrices(
    models: Dict[str, BaseEstimator],
    X: ArrayLike,
    y: ArrayLike,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> np.ndarray:
    """Plot confusion matrices for multiple models.

    Args:
        models: Dictionary of fitted models.
        X: Feature data.
        y: Target labels.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Array of Axes objects.
    """
    X = ensure_array(X)
    y = ensure_array(y)

    n = len(models)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for idx, (name, model) in enumerate(models.items()):
        row, col = idx // ncols, idx % ncols
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=axes[row, col])
        axes[row, col].set_title(name)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    _save_or_show(save_path)

    return axes


def plot_model_results(
    results: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot model comparison results with error bars.

    Args:
        results: DataFrame with columns: model, mean, std.
        save_path: Path to save figure.
        figsize: Figure size.
        ax: Matplotlib axes.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(results))
    ax.bar(x, results["mean"], yerr=results["std"], capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")

    plt.tight_layout()
    if save_path or ax is None:
        _save_or_show(save_path, close=save_path is not None)

    return ax


def plot_reduction(
    reduction: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    dim_1: int = 0,
    dim_2: int = 1,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    **scatter_kwargs,
) -> plt.Axes:
    """Plot dimensionality reduction results.

    Args:
        reduction: Fitted sklearn reduction model (PCA, TSNE, etc.).
        X: Feature data.
        y: Target labels for coloring.
        dim_1: First dimension to plot.
        dim_2: Second dimension to plot.
        ax: Matplotlib axes.
        save_path: Path to save figure.
        figsize: Figure size.
        title: Plot title.
        **scatter_kwargs: Additional arguments for scatter plot.

    Returns:
        The matplotlib Axes object.

    Raises:
        TypeError: If reduction doesn't have fit_transform.
        ValueError: If reduction is not fitted.
    """
    if not hasattr(reduction, "fit_transform"):
        raise TypeError(f"{type(reduction).__name__} is not a valid sklearn model")
    if not hasattr(reduction, "components_"):
        raise ValueError("Reduction model must be fitted before plotting")

    X = ensure_array(X)
    y = ensure_array(y)
    X_reduced = reduction.transform(X)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(X_reduced[:, dim_1], X_reduced[:, dim_2], c=y, **scatter_kwargs)

    name = reduction.__class__.__name__
    if hasattr(reduction, "explained_variance_ratio_"):
        var1 = reduction.explained_variance_ratio_[dim_1] * 100
        var2 = reduction.explained_variance_ratio_[dim_2] * 100
        ax.set_xlabel(f"{name} {dim_1} [{var1:.2f}%]")
        ax.set_ylabel(f"{name} {dim_2} [{var2:.2f}%]")
    else:
        ax.set_xlabel(f"{name} {dim_1}")
        ax.set_ylabel(f"{name} {dim_2}")

    ax.set_title(title or f"{name} Plot")

    if save_path or ax is None:
        _save_or_show(save_path, close=save_path is not None)

    return ax


def plot_scree(
    pca: PCA,
    n_components: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    **plot_kwargs,
) -> plt.Axes:
    """Plot scree plot for PCA.

    Args:
        pca: Fitted PCA model.
        n_components: Number of components to plot. Default shows all.
        ax: Matplotlib axes.
        save_path: Path to save figure.
        figsize: Figure size.
        **plot_kwargs: Additional arguments for plot.

    Returns:
        The matplotlib Axes object.

    Raises:
        ValueError: If PCA is not fitted.
    """
    if not hasattr(pca, "explained_variance_ratio_"):
        raise ValueError("PCA must be fitted before plotting")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n = min(n_components or pca.n_components_, pca.n_components_)
    x = np.arange(n)
    y = pca.explained_variance_ratio_[:n]

    ax.plot(x, y, "o-", **plot_kwargs)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("Scree Plot")
    ax.grid(True, alpha=0.3)

    if save_path or ax is None:
        _save_or_show(save_path, close=save_path is not None)

    return ax


def plot_training_probas(
    model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    cv: Optional[Union[int, BaseCrossValidator]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
) -> Tuple[plt.Axes, pd.DataFrame]:
    """Plot prediction probability distributions by class.

    Args:
        model: Sklearn model (will be cloned and trained).
        X: Feature data.
        y: Target labels.
        cv: CV splitter or number of folds.
        ax: Matplotlib axes.
        save_path: Path to save figure.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Tuple of (Axes, DataFrame with predictions).
    """
    X = ensure_array(X)
    y = ensure_array(y)

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True)
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True)

    model = clone(model)
    preds, labels = [], []

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds.extend(model.predict_proba(X[test_idx])[:, 1])
        labels.extend(y[test_idx])

    df = pd.DataFrame({"preds": preds, "labels": labels})

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.boxplot(data=df, x="labels", y="preds", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Prediction Score")
    ax.set_title(title or "Prediction Score Distribution by Class")

    if save_path or ax is None:
        _save_or_show(save_path, close=save_path is not None)

    return ax, df

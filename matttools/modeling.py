"""Machine learning model training and evaluation utilities.

This module provides functions for training multiple models, performing
cross-validation, and evaluating model performance.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, cross_val_score
from sklearn.utils import resample

from ._base import ArrayLike, ensure_array

logger = logging.getLogger(__name__)

__all__ = ["train_models", "cross_val_models", "test_models"]


def train_models(
    models: Dict[str, BaseEstimator],
    X: ArrayLike,
    y: ArrayLike,
    random_state: Optional[int] = None,
) -> Dict[str, BaseEstimator]:
    """Train a dictionary of models and return fitted models.

    Args:
        models: Dictionary of model name to model instance.
        X: Feature data.
        y: Target data.
        random_state: Random state for models that support it.

    Returns:
        Dictionary of trained models with the same keys.
    """
    X = ensure_array(X)
    y = ensure_array(y)
    trained = {}

    for name, model in models.items():
        if random_state is not None and hasattr(model, "random_state"):
            model.set_params(random_state=random_state)
        try:
            model.fit(X, y)
            trained[name] = model
        except Exception as e:
            logger.error(f"Failed to train '{name}': {e}")
            raise

    return trained

def cross_val_models(
    models: Dict[str, BaseEstimator],
    X: ArrayLike,
    y: ArrayLike,
    cv: Optional[Union[int, BaseCrossValidator]] = None,
    scoring: str = "roc_auc",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Cross-validate multiple models and return metrics.

    Args:
        models: Dictionary of model name to model instance.
        X: Feature data.
        y: Target data.
        cv: Number of folds or CV splitter. Default is 5-fold StratifiedKFold.
        scoring: Scoring metric. Default is 'roc_auc'.
        random_state: Random state for CV splitting.

    Returns:
        DataFrame with columns: model, mean, std, min, max.
    """
    X = ensure_array(X)
    y = ensure_array(y)

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    results = []
    for name, model in models.items():
        model_clone = clone(model)
        if random_state is not None and hasattr(model_clone, "random_state"):
            model_clone.set_params(random_state=random_state)
        try:
            scores = cross_val_score(model_clone, X, y, cv=cv, scoring=scoring)
            results.append({
                "model": name,
                "mean": scores.mean(),
                "std": scores.std(),
                "min": scores.min(),
                "max": scores.max(),
            })
        except Exception as e:
            logger.error(f"Failed to cross-validate '{name}': {e}")
            raise

    return pd.DataFrame(results)


def test_models(
    models: Dict[str, BaseEstimator],
    X: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Test fitted models using bootstrap resampling.

    Args:
        models: Dictionary of fitted model name to model instance.
        X: Feature data.
        y: Target data.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random state for resampling.

    Returns:
        DataFrame with columns: model, mean, std, min, max.
    """
    X = ensure_array(X)
    y = ensure_array(y)
    results = []

    for name, model in models.items():
        scores = []
        for i in range(n_bootstrap):
            try:
                seed = random_state + i if random_state else None
                X_boot, y_boot = resample(X, y, stratify=y, random_state=seed)
                scores.append(model.score(X_boot, y_boot))
            except Exception as e:
                logger.warning(f"Bootstrap {i} failed for '{name}': {e}")

        if scores:
            results.append({
                "model": name,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
            })
        else:
            logger.error(f"No successful bootstrap iterations for '{name}'")

    return pd.DataFrame(results)

"""Machine learning model training and evaluation utilities.

This module provides functions for training multiple models, performing
cross-validation, and evaluating model performance.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

from matttools.stats import Bootstrap

# Configure module logger
logger = logging.getLogger(__name__)


def train_models(
    models: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a dictionary of models and return fitted models.

    Args:
        models: Dictionary of model name to model instance.
        X: Feature data (numpy array or pandas DataFrame).
        y: Target data (numpy array or pandas Series).
        random_state: Random state to set for models that support it.

    Returns:
        Dictionary of trained models with the same keys as input.

    Note:
        Only models with a 'random_state' parameter will have it set.
        Models are cloned before training to avoid modifying the originals.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> models = {
        ...     'rf': RandomForestClassifier(),
        ...     'lr': LogisticRegression()
        ... }
        >>> trained = train_models(models, X_train, y_train, random_state=42)
    """
    trained_models = {}

    for model_name, model in models.items():
        # Create a clone to avoid modifying the original
        model_clone = clone(model)

        # Set random state if model supports it
        if random_state is not None and hasattr(model_clone, "random_state"):
            model_clone.set_params(random_state=random_state)

        # Fit the model
        try:
            model_clone.fit(X, y)
            trained_models[model_name] = model_clone
        except Exception as e:
            logger.error(f"Failed to train model '{model_name}': {e}")
            raise

    return trained_models


def cross_val_models(
    models: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Cross-validate multiple models and return metrics as a DataFrame.

    Args:
        models: Dictionary of model name to model instance.
        X: Feature data (numpy array or pandas DataFrame).
        y: Target data (numpy array or pandas Series).
        cv_folds: Number of cross-validation folds. Default is 5.
        scoring: Scoring metric to use. Default is 'roc_auc'.
        random_state: Random state for cross-validation splitting.

    Returns:
        DataFrame with columns: model, mean, std, min, max.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> models = {'rf': RandomForestClassifier(), 'lr': LogisticRegression()}
        >>> results = cross_val_models(models, X, y, cv_folds=5)
    """
    # Create cross validation object
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Collect results
    results_list = []

    for model_name, model in models.items():
        # Clone model to avoid modification
        model_clone = clone(model)

        # Set random state if model supports it
        if random_state is not None and hasattr(model_clone, "random_state"):
            model_clone.set_params(random_state=random_state)

        try:
            # Get cross validation scores
            scores = cross_val_score(model_clone, X, y, cv=cv, scoring=scoring)

            # Append results
            results_list.append(
                {
                    "model": model_name,
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "min": scores.min(),
                    "max": scores.max(),
                }
            )
        except Exception as e:
            logger.error(f"Failed to cross-validate model '{model_name}': {e}")
            raise

    # Convert to DataFrame
    return pd.DataFrame(results_list)


def test_models(
    models: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_bootstraps: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Test fitted models using bootstrap resampling.

    Args:
        models: Dictionary of fitted model name to model instance.
        X: Feature data (numpy array or pandas DataFrame).
        y: Target data (numpy array or pandas Series).
        n_bootstraps: Number of bootstrap iterations. Default is 100.
        random_state: Random state for bootstrap resampling.

    Returns:
        DataFrame with columns: model, mean, std, min, max containing
        accuracy scores for classifiers or R-squared for regressors.

    Note:
        Models must already be fitted before calling this function.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> models = {'rf': RandomForestClassifier().fit(X_train, y_train)}
        >>> results = test_models(models, X_test, y_test, n_bootstraps=100)
    """
    results_list = []

    for model_name, model in models.items():
        scores = []
        for i in range(n_bootstraps):
            try:
                # Resample with stratification
                X_resample, y_resample = resample(
                    X,
                    y,
                    stratify=y,
                    random_state=random_state + i if random_state else None,
                )
                score = model.score(X_resample, y_resample)
                scores.append(score)
            except Exception as e:
                logger.warning(
                    f"Bootstrap iteration {i} failed for model '{model_name}': {e}"
                )
                continue

        if scores:
            results_list.append(
                {
                    "model": model_name,
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                }
            )
        else:
            logger.error(f"No successful bootstrap iterations for model '{model_name}'")

    return pd.DataFrame(results_list)

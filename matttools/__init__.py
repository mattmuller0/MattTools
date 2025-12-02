"""MattTools: A toolkit for machine learning model comparison and evaluation.

This package provides utilities for training, evaluating, and visualizing
multiple machine learning models with consistent APIs.

Author: Matthew Muller
Email: matt.alex.muller@gmail.com
License: MIT
"""

import logging

from . import modeling, plotting, stats, utils
from ._base import ArrayLike, ensure_array
from .modeling import cross_val_models, test_models, train_models
from .plotting import plot_curve_ci, plot_curves, plot_model_results
from .stats import Bootstrap, mean_confidence_interval
from .utils import hide_warnings, set_random_seed, stopwatch

__version__ = "1.0.0"
__author__ = "Matthew Muller"
__email__ = "matt.alex.muller@gmail.com"
__license__ = "MIT"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # Modules
    "modeling",
    "plotting",
    "stats",
    "utils",
    # Types
    "ArrayLike",
    # Core utilities
    "ensure_array",
    "set_random_seed",
    "hide_warnings",
    "stopwatch",
    # Modeling
    "train_models",
    "cross_val_models",
    "test_models",
    # Plotting
    "plot_curves",
    "plot_curve_ci",
    "plot_model_results",
    # Stats
    "mean_confidence_interval",
    "Bootstrap",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

"""MattTools: A comprehensive toolkit for machine learning and bioinformatics analysis.

This package provides robust statistical functions, machine learning utilities,
visualization tools, and bioinformatics analysis capabilities designed for
researchers and data scientists.

Author: Matthew Muller
Email: matt.alex.muller@gmail.com
License: MIT
"""

import logging

# Import main modules for easy access
from . import modeling, plotting, stats, utils

# Import key functions for convenience
from .stats import Bootstrap, mean_confidence_interval
from .utils import hide_warnings, set_random_seed

__version__ = "1.0.0"
__author__ = "Matthew Muller"
__email__ = "matt.alex.muller@gmail.com"
__license__ = "MIT"

# Configure package-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # Modules
    "stats",
    "modeling",
    "plotting",
    "utils",
    # Key functions
    "mean_confidence_interval",
    "Bootstrap",
    "set_random_seed",
    "hide_warnings",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

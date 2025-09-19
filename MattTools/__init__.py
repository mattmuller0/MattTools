"""
MattTools: A comprehensive toolkit for machine learning and bioinformatics analysis.

This package provides robust statistical functions, machine learning utilities,
visualization tools, and bioinformatics analysis capabilities designed for
researchers and data scientists.

Author: Matthew Muller
Email: matt.alex.muller@gmail.com
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Matthew Muller"
__email__ = "matt.alex.muller@gmail.com"
__license__ = "MIT"

# Import main modules for easy access
from . import stats
from . import modeling  
from . import plotting
from . import utils

# Import key functions for convenience
from .stats import mean_confidence_interval, Bootstrap
from .utils import set_random_seed, hide_warnings

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

# Configure warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
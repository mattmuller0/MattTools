######################################################################
#
#                       Utility Functions
#
######################################################################
# Matthew Muller
# 11/24/2022
#
# This file contains functions for modeling and evaluating models.

# Library Imports
from typing import Union, Any, Callable, Optional
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import sys

# Function to set the random seed as a random integer
def set_random_seed(seed: Optional[int] = None) -> None:
    """Set random seed for reproducibility across multiple libraries.

    Args:
        seed: Integer value of the random seed. If None, generates a random seed.

    Note:
        Sets random seeds for Python's random, NumPy, TensorFlow, and PyTorch if available.
        sklearn classes must be configured individually with random_state parameter.
    """
    if seed is None:
        # Generate a random seed using the new numpy random generator
        rng = np.random.default_rng()
        seed = int(rng.integers(1, 10000))
        
    print(f'Setting random seed to {seed} for reproducibility.')

    # Set the random seed if it is installed
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    # Set random seed for numpy (using modern approach)
    try:
        # Set both legacy and modern numpy random state
        np.random.seed(seed)  # For legacy code compatibility
        # Note: Individual functions should use np.random.default_rng(seed) for new code
    except ImportError:
        pass

    # Note: sklearn doesn't have a global random_state function
    # Individual sklearn classes accept random_state parameter

    # Set the random seed for tensorflow if it is installed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Set the random seed for pytorch if it is installed
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

# Hide warnings based on the warning type (default to all warnings) with an option to show warnings
def hide_warnings(warning_type: str = 'all') -> None:
    '''
    Summary: Function to hide warnings based on the warning type (default to all warnings) with an option to show warnings

    warning_type (str) : string value of the warning type to hide
    '''
    # Hide warnings
    if warning_type == 'all':
        import warnings
        warnings.filterwarnings('ignore')
    else:
        import warnings
        # catch errors
        try:
            warnings.filterwarnings('ignore', category=warning_type)
        except (TypeError, ValueError) as e:
            print(f'Warning type {warning_type} not found: {e}. Showing all warnings.')

    # Show warnings
    if warning_type == 'none':
        import warnings
        warnings.filterwarnings('default')

# Function to print the current memory usage
def print_memory_usage() -> None:
    '''
    Summary: Function to print the current memory usage
    '''
    # Import the resource module
    import resource

    # Print the current memory usage
    print(f'Current memory usage is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000} MB')

# Function to time the execution of a function
def stopwatch(func: Callable, *args: Any, **kwargs: Any) -> None:
    '''
    Summary: Function to time the execution of a function

    func (function) : function to time
    *args (tuple) : tuple of arguments to pass to the function
    **kwargs (dict) : dictionary of keyword arguments to pass to the function
    '''
    # Import the time module
    import time

    # Get the start time
    start = time.time()

    # Run the function
    func(*args, **kwargs)

    # Get the end time
    end = time.time()

    # Print the time elapsed
    print(f'Time elapsed: {end - start}')
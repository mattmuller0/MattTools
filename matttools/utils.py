"""Utility functions for MattTools package.

This module provides utility functions for random seed management,
warning control, memory monitoring, and function timing.
"""

import logging
import random
import time
import warnings
from typing import Any, Callable, Optional, Type, Union

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


def set_random_seed(seed: Optional[int] = None, verbose: bool = True) -> int:
    """Set random seed for reproducibility across multiple libraries.

    Args:
        seed: Integer value of the random seed. If None, generates a random seed.
        verbose: If True, logs the seed being set. Default is True.

    Returns:
        The seed that was set.

    Note:
        Sets random seeds for Python's random, NumPy, TensorFlow, and PyTorch if available.
        sklearn classes must be configured individually with random_state parameter.

    Example:
        >>> seed = set_random_seed(42)
        >>> seed = set_random_seed()  # Generates random seed
    """
    if seed is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(1, 10000))

    if verbose:
        logger.info(f"Setting random seed to {seed} for reproducibility.")

    # Set Python's random seed
    random.seed(seed)

    # Set numpy random seed (legacy compatibility)
    np.random.seed(seed)

    # Set TensorFlow seed if available
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Set PyTorch seed if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def hide_warnings(
    warning_type: Union[str, Type[Warning]] = "all", action: str = "ignore"
) -> None:
    """Control warning display based on warning type.

    Args:
        warning_type: Type of warning to control. Options:
            - "all": All warnings
            - "none": Reset to default (show all warnings)
            - Warning class: Specific warning type (e.g., FutureWarning)
        action: Action to take. Options: "ignore", "default", "error", "always", "module", "once"

    Raises:
        ValueError: If warning_type is not valid.

    Example:
        >>> hide_warnings("all")  # Hide all warnings
        >>> hide_warnings(FutureWarning)  # Hide only FutureWarning
        >>> hide_warnings("none")  # Show all warnings
    """
    if warning_type == "none":
        warnings.filterwarnings("default")
        return

    if warning_type == "all":
        warnings.filterwarnings(action)
        return

    # Handle specific warning types
    try:
        warnings.filterwarnings(action, category=warning_type)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid warning type '{warning_type}'. Must be 'all', 'none', "
            f"or a valid Warning class."
        ) from e


def get_memory_usage() -> float:
    """Get current memory usage in MB.

    Returns:
        Current memory usage in megabytes.

    Raises:
        ImportError: If resource module is not available (Windows).

    Example:
        >>> memory_mb = get_memory_usage()
        >>> print(f"Using {memory_mb:.2f} MB")
    """
    try:
        import resource

        usage_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports in bytes, Linux in kilobytes
        import platform

        if platform.system() == "Darwin":
            return usage_bytes / (1024 * 1024)
        else:
            return usage_bytes / 1024
    except ImportError as e:
        raise ImportError(
            "Memory usage monitoring requires the 'resource' module, "
            "which is not available on Windows."
        ) from e


def print_memory_usage() -> None:
    """Print the current memory usage.

    Example:
        >>> print_memory_usage()
        Current memory usage: 45.23 MB
    """
    try:
        memory_mb = get_memory_usage()
        print(f"Current memory usage: {memory_mb:.2f} MB")
    except ImportError as e:
        logger.warning(f"Cannot print memory usage: {e}")


def stopwatch(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Time the execution of a function.

    Args:
        func: Function to time.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Tuple of (function_result, elapsed_time_seconds).

    Example:
        >>> def slow_function(n):
        ...     return sum(range(n))
        >>> result, elapsed = stopwatch(slow_function, 1000000)
        >>> print(f"Result: {result}, Time: {elapsed:.4f}s")
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    logger.info(f"Function {func.__name__} executed in {elapsed:.4f} seconds")
    return result, elapsed

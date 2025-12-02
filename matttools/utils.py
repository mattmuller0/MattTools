"""Utility functions for MattTools package.

This module provides utility functions for random seed management,
warning control, memory monitoring, and function timing.
"""

import functools
import logging
import platform
import random
import time
import warnings
from typing import Any, Callable, Optional, Type, TypeVar, Union

import numpy as np

from ._base import ArrayLike, ensure_array

# Configure module logger
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "ArrayLike",
    "ensure_array",
    "set_random_seed",
    "hide_warnings",
    "get_memory_usage",
    "print_memory_usage",
    "stopwatch",
]


def set_random_seed(seed: Optional[int] = None, verbose: bool = True) -> int:
    """Set random seed for reproducibility across multiple libraries.

    Args:
        seed: Integer value of the random seed. If None, generates a random seed.
        verbose: If True, logs the seed being set.

    Returns:
        The seed that was set.

    Note:
        Sets seeds for Python's random, NumPy, TensorFlow, and PyTorch if available.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 10000))

    if verbose:
        logger.info(f"Setting random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

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
        warning_type: "all", "none", or a Warning class (e.g., FutureWarning).
        action: "ignore", "default", "error", "always", "module", or "once".

    Raises:
        ValueError: If warning_type is not valid.
    """
    if warning_type == "none":
        warnings.filterwarnings("default")
    elif warning_type == "all":
        warnings.filterwarnings(action)
    else:
        try:
            warnings.filterwarnings(action, category=warning_type)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid warning type '{warning_type}'. "
                "Must be 'all', 'none', or a Warning class."
            ) from e


def get_memory_usage() -> float:
    """Get current memory usage in MB.

    Returns:
        Current memory usage in megabytes.

    Raises:
        ImportError: If resource module is not available (Windows).
    """
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports in bytes, Linux in kilobytes
        return usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024
    except ImportError as e:
        raise ImportError(
            "Memory monitoring requires 'resource' module (not available on Windows)."
        ) from e


def print_memory_usage() -> None:
    """Print current memory usage."""
    try:
        print(f"Current memory usage: {get_memory_usage():.2f} MB")
    except ImportError as e:
        logger.warning(f"Cannot get memory usage: {e}")


def stopwatch(func: F) -> F:
    """Decorator to time function execution.

    Args:
        func: Function to time.

    Returns:
        Wrapped function that logs execution time.

    Example:
        >>> @stopwatch
        ... def slow_function(n):
        ...     return sum(range(n))
        >>> result = slow_function(1000000)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper  # type: ignore[return-value]

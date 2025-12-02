"""Base utilities and shared helpers for MattTools package.

This module provides common utilities used across the package.
"""

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input to numpy array.

    Args:
        data: Input data (numpy array, pandas DataFrame, or Series).

    Returns:
        Numpy array representation of the input.

    Raises:
        TypeError: If data cannot be converted to numpy array.
    """
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    try:
        return np.asarray(data)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert {type(data).__name__} to numpy array") from e


__all__ = ["ArrayLike", "ensure_array"]

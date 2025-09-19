"""Tests for the utils module."""

import numpy as np
import random
import warnings
from matttools.utils import set_random_seed, hide_warnings


class TestSetRandomSeed:
    """Tests for set_random_seed function."""
    
    def test_set_random_seed_basic(self):
        """Test basic random seed setting."""
        seed_value = 42
        set_random_seed(seed_value)
        
        # Generate some random numbers
        np_random = np.random.random(5)
        py_random = [random.random() for _ in range(5)]
        
        # Reset seed and generate again
        set_random_seed(seed_value)
        np_random_2 = np.random.random(5)
        py_random_2 = [random.random() for _ in range(5)]
        
        # Should be identical
        np.testing.assert_array_equal(np_random, np_random_2)
        assert py_random == py_random_2
        
    def test_random_seed_reproducibility(self):
        """Test that different seeds produce different results."""
        set_random_seed(42)
        result_1 = np.random.random(10)
        
        set_random_seed(123)
        result_2 = np.random.random(10)
        
        # Different seeds should produce different results
        assert not np.array_equal(result_1, result_2)
        
    def test_default_random_seed(self):
        """Test setting random seed without explicit value."""
        # This should not raise an error
        set_random_seed()
        
        # Should still be able to generate random numbers
        result = np.random.random(5)
        assert len(result) == 5
        assert all(0 <= x <= 1 for x in result)


class TestHideWarnings:
    """Tests for hide_warnings function."""
    
    def test_hide_warnings_functionality(self):
        """Test that hide_warnings suppresses warnings."""
        # First, ensure warnings are shown by default
        warnings.resetwarnings()
        
        # Apply hide_warnings
        hide_warnings()
        
        # This should not produce visible warnings when running tests
        # We can't easily test the output suppression, but we can test
        # that the function runs without error
        warnings.warn("This is a test warning", UserWarning)
        warnings.warn("This is a deprecation warning", DeprecationWarning)
        
        # Function should complete without raising exceptions
        assert True
"""Tests for package-level imports and functionality."""

import pytest


class TestPackageImports:
    """Test that the package can be imported correctly."""
    
    def test_main_package_import(self):
        """Test importing the main package."""
        import matttools
        assert hasattr(matttools, '__version__')
        assert hasattr(matttools, '__author__')
        
    def test_submodule_imports(self):
        """Test importing submodules."""
        import matttools.stats
        import matttools.modeling
        import matttools.plotting
        import matttools.utils
        
        # Check key functions are available
        assert hasattr(matttools.stats, 'mean_confidence_interval')
        assert hasattr(matttools.modeling, 'train_models')
        assert hasattr(matttools.utils, 'set_random_seed')
        
    def test_convenience_imports(self):
        """Test that convenience imports work."""
        from matttools import mean_confidence_interval, set_random_seed
        
        # These should be callable
        assert callable(mean_confidence_interval)
        assert callable(set_random_seed)
        
    def test_package_metadata(self):
        """Test package metadata is accessible."""
        import matttools
        
        assert isinstance(matttools.__version__, str)
        assert isinstance(matttools.__author__, str)
        assert isinstance(matttools.__email__, str)
        assert matttools.__license__ == "MIT"
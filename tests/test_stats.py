"""Tests for the stats module."""

import numpy as np
import pytest
from matttools.stats import mean_confidence_interval, Bootstrap


class TestMeanConfidenceInterval:
    """Tests for mean_confidence_interval function."""
    
    def test_basic_functionality(self, sample_numeric_data):
        """Test basic confidence interval calculation."""
        mean, ci = mean_confidence_interval(sample_numeric_data)
        
        assert isinstance(mean, float)
        assert isinstance(ci, np.ndarray)
        assert ci.shape == (1, 2)
        assert ci[0, 0] < mean < ci[0, 1]  # ci_low < mean < ci_high
        
    def test_confidence_levels(self, sample_numeric_data):
        """Test different confidence levels."""
        # 90% CI should be narrower than 95% CI
        _, ci_90 = mean_confidence_interval(sample_numeric_data, confidence=0.90)
        _, ci_95 = mean_confidence_interval(sample_numeric_data, confidence=0.95)
        
        assert (ci_90[0, 1] - ci_90[0, 0]) < (ci_95[0, 1] - ci_95[0, 0])
        
    def test_small_sample(self):
        """Test with small sample size."""
        small_data = np.array([1, 2, 3, 4, 5])
        mean, ci = mean_confidence_interval(small_data)
        
        assert mean == 3.0
        assert ci[0, 0] < mean < ci[0, 1]
        
    def test_known_distribution(self):
        """Test with known normal distribution."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 10000)  # Large sample
        mean, ci = mean_confidence_interval(data, confidence=0.95)
        
        # For large samples, mean should be close to population mean
        assert abs(mean - 100) < 1
        # CI should contain population mean
        assert ci[0, 0] < 100 < ci[0, 1]


class TestBootstrap:
    """Tests for Bootstrap class."""
    
    def test_bootstrap_initialization(self):
        """Test Bootstrap class initialization."""
        bootstrap = Bootstrap(n_bootstrap=100, stratified=True, rng_seed=42)
        assert bootstrap.n_bootstrap == 100
        assert bootstrap.stratified == True
        assert bootstrap.rng_seed == 42
        
    def test_bootstrap_split(self, sample_data):
        """Test bootstrap split functionality."""
        X, y = sample_data
        bootstrap = Bootstrap(n_bootstrap=10, rng_seed=42)
        
        splits = list(bootstrap.split(X, y))
        assert len(splits) == 10
        
        # Each split should return indices
        for train_idx, test_idx in splits:
            assert len(train_idx) == len(X)
            assert all(idx < len(X) for idx in train_idx)
            
    def test_bootstrap_stratified(self, sample_data):
        """Test stratified bootstrap."""
        X, y = sample_data
        bootstrap = Bootstrap(n_bootstrap=5, stratified=True, rng_seed=42)
        
        splits = list(bootstrap.split(X, y))
        
        # Each bootstrap should have both classes
        for train_idx, _ in splits:
            bootstrap_y = y[train_idx]
            unique_classes = np.unique(bootstrap_y)
            assert len(unique_classes) >= 2  # Should have both classes
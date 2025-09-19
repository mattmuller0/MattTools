"""Tests for the plotting module."""

import numpy as np
import matplotlib.pyplot as plt
import pytest
from unittest.mock import patch
from sklearn.decomposition import PCA
from matttools.plotting import plot_reduction


class TestPlotReduction:
    """Tests for plot_reduction function."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_reduction_basic(self, mock_show, sample_dataframe):
        """Test basic dimensionality reduction plotting."""
        df = sample_dataframe
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        # Create a simple PCA reducer
        pca = PCA(n_components=2)
        pca.fit(X)
        
        # This should not raise an error
        try:
            plot_reduction(pca, X, y, dim_1=0, dim_2=1)
            success = True
        except Exception:
            success = False
            
        assert success, "plot_reduction should execute without errors"
        
        # Clean up the plot
        plt.close('all')
        
    @patch('matplotlib.pyplot.show') 
    def test_plot_reduction_with_labels(self, mock_show, sample_dataframe):
        """Test plotting with custom labels."""
        df = sample_dataframe
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        pca = PCA(n_components=2)
        pca.fit(X)
        
        # Test with custom parameters
        try:
            plot_reduction(
                pca, X, y, 
                dim_1=0, dim_2=1,
                title="Test PCA Plot",
                labels=['Class 0', 'Class 1']
            )
            success = True
        except Exception:
            success = False
            
        assert success, "plot_reduction with labels should execute without errors"
        plt.close('all')


class TestPlottingUtilities:
    """Tests for plotting utility functions."""
    
    def test_mean_confidence_interval_plotting(self):
        """Test that mean_confidence_interval function exists in plotting module."""
        # Import to check it exists
        from matttools.plotting import mean_confidence_interval
        
        # Test basic functionality
        data = np.random.normal(100, 15, 1000)
        result = mean_confidence_interval(data, confidence=0.95)
        
        # Should return mean, low, high
        assert len(result) == 3
        mean, low, high = result
        assert low < mean < high
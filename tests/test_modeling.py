"""Tests for the modeling module."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from matttools.modeling import train_models


class TestTrainModels:
    """Tests for train_models function."""
    
    def test_basic_training(self, sample_data, sample_models):
        """Test basic model training functionality."""
        X, y = sample_data
        trained_models = train_models(sample_models, X, y, random_state=42)
        
        assert len(trained_models) == len(sample_models)
        assert 'LogisticRegression' in trained_models
        assert 'RandomForest' in trained_models
        
        # Check that models are fitted
        for model_name, model in trained_models.items():
            assert hasattr(model, 'predict')
            predictions = model.predict(X)
            assert len(predictions) == len(y)
            
    def test_model_performance(self, breast_cancer_data, sample_models):
        """Test that trained models achieve reasonable performance."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        trained_models = train_models(sample_models, X_train, y_train, random_state=42)
        
        for model_name, model in trained_models.items():
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Models should achieve reasonable accuracy on breast cancer dataset
            assert accuracy > 0.8, f"{model_name} achieved low accuracy: {accuracy}"
            
    def test_reproducibility(self, sample_data, sample_models):
        """Test that training with same random state produces identical results."""
        X, y = sample_data
        
        # Train models twice with same random state
        trained_models_1 = train_models(sample_models, X, y, random_state=42)
        trained_models_2 = train_models(sample_models, X, y, random_state=42)
        
        # Predictions should be identical
        for model_name in sample_models.keys():
            pred_1 = trained_models_1[model_name].predict(X)
            pred_2 = trained_models_2[model_name].predict(X)
            np.testing.assert_array_equal(pred_1, pred_2, 
                f"Model {model_name} predictions not reproducible")
                
    def test_empty_models_dict(self, sample_data):
        """Test behavior with empty models dictionary."""
        X, y = sample_data
        empty_models = {}
        trained_models = train_models(empty_models, X, y)
        
        assert len(trained_models) == 0
        assert isinstance(trained_models, dict)
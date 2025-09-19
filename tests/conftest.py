"""Test configuration and fixtures for MattTools."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_data():
    """Generate sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def breast_cancer_data():
    """Load breast cancer dataset for testing."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_models():
    """Create sample models for testing."""
    return {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
    }


@pytest.fixture
def sample_numeric_data():
    """Generate sample numeric data for statistical testing."""
    np.random.seed(42)
    return np.random.normal(100, 15, 1000)


@pytest.fixture
def sample_dataframe():
    """Create sample pandas DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'target': np.random.binomial(1, 0.5, 100)
    })
# MattTools

A Python toolkit for machine learning and bioinformatics: statistical analysis, model evaluation, and visualization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
# pip
pip install git+https://github.com/mattmuller0/MattTools.git

# uv
uv pip install git+https://github.com/mattmuller0/MattTools.git
```

For development:

```bash
git clone https://github.com/mattmuller0/MattTools.git
cd MattTools
pip install -e ".[dev]"
```

## Quick Start

```python
import matttools as mt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Set random seed across numpy, random, and ML frameworks
mt.set_random_seed(42)

# Calculate confidence intervals
data = np.random.normal(100, 15, 100)
mean, ci = mt.mean_confidence_interval(data, confidence=0.95)
print(f"Mean: {mean:.2f}, 95% CI: {ci}")

# Bootstrap resampling (sklearn-compatible splitter)
bootstrap = mt.Bootstrap(n_bootstrap=100, rng_seed=42)
for train_idx, test_idx in bootstrap.split(X, y):
    model.fit(X[train_idx], y[train_idx])

# Cross-validation with multiple models
models = {'rf': RandomForestClassifier(), 'lr': LogisticRegression()}
results = mt.modeling.cross_val_models(models, X, y, cv=5)

# Visualization with confidence intervals
mt.plotting.plot_roc_curve_ci(model, X_test, y_test, n_bootstraps=1000)
mt.plotting.plot_reduction(X, y, method='pca')  # PCA, UMAP, t-SNE

# Utility functions
result, elapsed = mt.utils.stopwatch(expensive_function, *args)
print(f"Completed in {elapsed:.2f}s")
```

## API Reference

### stats

Statistical functions with bootstrap and confidence interval support.

| Function | Description |
|----------|-------------|
| `mean_confidence_interval(data, confidence)` | Calculate mean with confidence interval |
| `bootstrap_auc_confidence(y_true, y_score, n_bootstraps)` | Bootstrap AUC with CI |
| `Bootstrap(n_bootstrap, rng_seed)` | sklearn-compatible bootstrap splitter |
| `odds_ratio(table)` | Calculate odds ratio from 2x2 contingency table |

### modeling

Model training, cross-validation, and evaluation utilities.

| Function | Description |
|----------|-------------|
| `train_models(models, X_train, y_train)` | Train multiple models |
| `cross_val_models(models, X, y, cv)` | Cross-validate multiple models |
| `test_models(models, X_test, y_test)` | Evaluate trained models on test set |

### plotting

Visualization tools for model evaluation and dimensionality reduction.

| Function | Description |
|----------|-------------|
| `plot_reduction(X, y, method)` | PCA/UMAP/t-SNE scatter plots |
| `plot_scree(pca)` | Scree plot for PCA variance |
| `plot_roc_curve(y_true, score)` | ROC curve |
| `plot_roc_curve_ci(model, X, y, n_bootstraps)` | ROC curve with bootstrap CI |
| `plot_prc_curve(model, X, y)` | Precision-recall curve |
| `plot_pr_curve_ci(model, X, y, n_bootstraps)` | PR curve with bootstrap CI |
| `plot_confusion_matrix(y_true, score)` | Confusion matrix heatmap |
| `plot_confusion_matrices(models, X, y)` | Multiple confusion matrices |
| `plot_roc_curves(models, X, y)` | Compare ROC curves |
| `plot_prc_curves(models, X, y)` | Compare PR curves |
| `plot_model_results(results)` | Visualize cross-validation results |
| `plot_decision_boundaries(model, X, y)` | Decision boundary plot |
| `plot_cross_validation_auroc(model, X, y, cv)` | CV AUROC with variance |

### utils

General utility functions.

| Function | Description |
|----------|-------------|
| `set_random_seed(seed)` | Set seed for numpy, random, torch, tensorflow |
| `hide_warnings()` | Suppress common ML warnings |
| `get_memory_usage()` | Get current memory usage in MB |
| `print_memory_usage(label)` | Print labeled memory usage |
| `stopwatch(func, *args, **kwargs)` | Time function execution |

## Requirements

**Core:** numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, statsmodels

**Optional:** tensorflow, torch, umap-learn, plotly, bokeh, biopython, pysam, pyBigWig

## Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov=matttools    # With coverage
```

## License

MIT License - Matthew Muller (matt.alex.muller@gmail.com)

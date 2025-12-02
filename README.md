# MattTools

Python toolkit for ML and bioinformatics: statistical analysis, model evaluation, and visualization.

## Install

```bash
# pip
pip install git+https://github.com/mattmuller0/MattTools.git

# uv
uv pip install git+https://github.com/mattmuller0/MattTools.git
```

## Usage

```python
import matttools as mt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Confidence intervals
data = np.random.normal(100, 15, 100)
mean, ci = mt.mean_confidence_interval(data)

# Bootstrap resampling
bootstrap = mt.Bootstrap(n_bootstrap=100, rng_seed=42)
for train_idx, _ in bootstrap.split(X, y):
    model.fit(X[train_idx], y[train_idx])

# Model evaluation
models = {'rf': RandomForestClassifier(), 'lr': LogisticRegression()}
results = mt.modeling.cross_val_models(models, X, y)

# Visualization
mt.plotting.plot_roc_curve_ci(model, X_test, y_test)

# Utilities
seed = mt.set_random_seed(42)
result, elapsed = mt.utils.stopwatch(func, args)
```

## API

**stats** • `mean_confidence_interval` • `bootstrap_auc_confidence` • `Bootstrap` • `odds_ratio`

**modeling** • `train_models` • `cross_val_models` • `test_models`

**plotting** • `plot_scree` • `plot_roc_curve` • `plot_confusion_matrix` • `plot_reduction`

**utils** • `set_random_seed` • `hide_warnings` • `get_memory_usage` • `stopwatch`

## Development

```bash
git clone https://github.com/mattmuller0/MattTools.git
cd MattTools
pip install -e ".[dev]"  # or: uv pip install -e ".[dev]"
pytest
```

Python ≥3.10 • NumPy • Pandas • scikit-learn • SciPy • Matplotlib • Seaborn

MIT License • Matthew Muller

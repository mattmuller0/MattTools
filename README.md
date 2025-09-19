# MattTools

<<<<<<< HEAD
Personal Python toolkit for ML and bioinformatics research.

## Installation

```bash
pip install git+https://github.com/mattmuller0/MattTools.git
```

For development:
```bash
git clone https://github.com/mattmuller0/MattTools.git
cd MattTools
pip install -e ".[dev]"
```

## Usage

```python
import matttools as mt
import numpy as np

# Statistical analysis
data = np.random.normal(100, 15, 100)
mean, ci = mt.stats.mean_confidence_interval(data)
print(f"Mean: {mean:.2f}, CI: {ci}")

# Bootstrap analysis
bootstrap = mt.stats.Bootstrap(data)
results = bootstrap.resample(n_samples=1000)
```

## Modules

- **stats**: Confidence intervals, bootstrap methods, statistical tests
- **modeling**: ML model training and evaluation utilities
- **plotting**: ROC curves, dimensionality reduction plots
- **utils**: Random seeds, warnings control, data helpers

## Testing

```bash
pytest tests/
```

## License

MIT License
=======
Python tools for my personal work.

This is split into statistical tools and plotting tools so far. More to come.

A lot of this code needs to be cleaned up and made more general. I will do this as I go along.

To install the GitHub repository using pip, run the following command:

```bash
pip install git+https://github.com/mattmuller0/MattTools.git
```
>>>>>>> origin/main

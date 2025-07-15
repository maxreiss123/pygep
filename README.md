# PyGEP: Python Interface for GeneExpressionProgramming.jl

A Python interface for [GeneExpressionProgramming.jl](https://github.com/maxreiss123/GeneExpressionProgramming.jl) package.

This package provides a Python wrapper for the Julia implementation, which performs symbolic regression using Gene Expression Programming.

## Features

- Underlying Julia implementation for fast optimization
- Standard GEP symbolic regression
- Multi-objective optimization (not available yet for the interface)
- Physics-aware regression with dimensional constraints (not available yet for the interface)
- Tensor/matrix regression support (not available yet for the interface)

## Installation

### 1. Install Julia
First, install Julia from [https://julialang.org/downloads/](https://julialang.org/downloads/)

### 2. Install PyGEP
```bash
cd pygep
pip install .
```

### 3. Install Julia Dependencies
```bash
pygep-setup
```

or

```python
import pygep
pygep.install_julia_dependencies()
```

## Quick Start

### Basic Symbolic Regression

```python
import numpy as np
from pygep import GepRegressor

# Generate test data
X = np.random.randn(100, 2)
y = X[:, 0]**2 + X[:, 0] * X[:, 1] - 2 * X[:, 0] * X[:, 1]

# Create and fit regressor
regressor = GepRegressor(number_features=2)
regressor.fit(X, y, epochs=1000, population_size=1000, loss_function="mse")

# Make predictions
predictions = regressor.predict(X)

# Get results
print(f"Best expression: {regressor.best_expression_}")
print(f"Best fitness: {regressor.best_fitness_}")
```

## API Reference

### GepRegressor

Main class for symbolic regression.

#### Parameters:
- `number_features` (int): Number of input features

#### Methods:
- `fit(X, y, epochs=1000, population_size=1000, loss_function="mse", ...)`: Fit the regressor
- `predict(X)`: Make predictions
- `get_best_models(n=1)`: Get n best models

#### Properties:
- `best_expression_`: Best symbolic expression as string
- `best_fitness_`: Fitness of best expression

## Requirements

- Python ≥ 3.8
- Julia ≥ 1.6
- numpy ≥ 1.20.0
- julia ≥ 0.6.0 (Python package)

## Troubleshooting

### Julia Not Found
```bash
# Make sure Julia is in your PATH
julia --version

# If not found, add Julia to PATH or install from:
# https://julialang.org/downloads/
```

### PyCall Issues
```python
# In Julia REPL:
using Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
```

### Package Installation Issues
```python
# Force reinstall Julia dependencies
import pygep
pygep.install_julia_dependencies()
```

## Citation

If you use this package, please cite the original Julia implementation:

```bibtex
@article{Reissmann2025,
  author   = {Maximilian Reissmann and Yuan Fang and Andrew S. H. Ooi and Richard D. Sandberg},
  title    = {Constraining Genetic Symbolic Regression via Semantic Backpropagation},
  journal  = {Genetic Programming and Evolvable Machines},
  year     = {2025},
  volume   = {26},
  number   = {1},
  pages    = {12},
  doi      = {10.1007/s10710-025-09510-z}
}
```

## License

MIT License - see LICENSE file for details.

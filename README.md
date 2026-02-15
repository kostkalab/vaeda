# vaeda

vaeda (variational auto-encoder for doublet annotation) is a Python package for doublet annotation in single-cell RNA sequencing data. For method details and comparisons to alternative doublet annotation tools, see the [vaeda publication](https://academic.oup.com/bioinformatics/article/39/1/btac720/6808614).

**v0.2.0** — Now powered by **PyTorch** (replacing TensorFlow). Public API unchanged.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Using uv (Recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
  - [Using conda](#using-conda-not-recommended)
- [Quick Start](#quick-start)
- [Development](#development)
  - [Setup](#setup)
  - [Makefile Commands](#makefile-commands)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
  - [Type Checking](#type-checking)
  - [Docker Development Environment](#docker-development-environment)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Other Doublet Detection Tools](#other-doublet-detection-tools)
- [Citation](#citation)
- [License](#license)

## Requirements

- Python 3.12+
- PyTorch 2.6+

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies automatically with full reproducibility via lockfiles.

```bash
# Install from GitHub (specific release)
uv pip install git+https://github.com/kostkalab/vaeda.git@v0.2.0

# Or install from a cloned repository
git clone https://github.com/kostkalab/vaeda.git
cd vaeda
uv pip install .
```

### Using pip

```bash
# Install from GitHub (specific release)
pip install git+https://github.com/kostkalab/vaeda.git@v0.2.0

# Or from a cloned repository
git clone https://github.com/kostkalab/vaeda.git
cd vaeda
pip install .
```

### GPU Support

vaeda automatically selects the best available device: CUDA > MPS > CPU.

For NVIDIA GPU support, install PyTorch with CUDA before installing vaeda:

```bash
# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Then install vaeda
pip install git+https://github.com/kostkalab/vaeda.git@v0.2.0
```

For Apple Silicon (M1/M2/M3/M4), MPS acceleration is used automatically with the standard PyTorch install.

### Using conda (Not Recommended)

> **Warning:** Mixing conda and pip can lead to conflicts, especially with PyTorch CUDA builds. If you must use conda, create a minimal environment and use pip:

```bash
conda create -n vaeda_env python=3.13
conda activate vaeda_env
pip install git+https://github.com/kostkalab/vaeda.git@v0.2.0
```

### Migrating from v0.1.x (TensorFlow)

v0.2.0 replaces TensorFlow with PyTorch. The public API is unchanged — only the backend has changed:

```bash
# Remove old TensorFlow dependencies (optional)
pip uninstall tensorflow tensorflow-probability tf_keras

# Install v0.2.0
pip install git+https://github.com/kostkalab/vaeda.git@v0.2.0
```

Validation on pbmc3k shows strong agreement between the TensorFlow and PyTorch backends (Pearson r = 0.86–0.90, call agreement ~98%, Jaccard index 0.57 on doublet calls). The remaining variation is expected due to differences in weight initialisation, floating-point arithmetic, and stochastic minibatch ordering between frameworks.

## Quick Start

```python
import vaeda

# adata is an AnnData object with raw counts in adata.X
result = vaeda.vaeda(adata)

# Results are stored in the AnnData object:
# - adata.obsm['vaeda_embedding']: VAE encoding
# - adata.obs['vaeda_scores']: doublet scores
# - adata.obs['vaeda_calls']: doublet/singlet calls
```

For a detailed example, see the [tutorial notebook](https://github.com/kostkalab/vaeda/blob/main/doc/vaeda_scanpy-pbmc3k-tutorial.ipynb), which adapts the [scanpy PBMC3k tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html) to demonstrate doublet annotation with vaeda.

## Development

### Setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/kostkalab/vaeda.git
cd vaeda

# Install all dependencies including dev group
uv sync --group dev
```

### Makefile Commands

```bash
make test       # Run tests
make lint       # Run linting
make format     # Format code
make typecheck  # Run type checker
make check      # Run all checks (lint, typecheck, test)
```

### Running Tests

```bash
make test

# Or manually
uv run pytest
uv run pytest -v
uv run pytest --cov=vaeda
```

### Code Quality

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length 90).

```bash
make lint
make format

# Or manually
uv run ruff check src/
uv run ruff format src/
```

### Type Checking

```bash
uv run ty check src/
```

### Docker Development Environment

The Dockerfile supports `dev` and `prod` targets for consistent, reproducible development.

```bash
# Build and run
docker compose build --no-cache dev
docker compose up dev

# Shell access
docker compose exec dev bash

# Or one-shot
docker compose run --rm dev bash
```

## Project Structure

```
vaeda/
├── src/vaeda/
│   ├── __init__.py      # Package exports
│   ├── vaeda.py         # Main pipeline
│   ├── vae.py           # VAE model (PyTorch)
│   ├── pu.py            # PU learning (PyTorch)
│   ├── classifier.py    # Binary classifier (PyTorch)
│   ├── cluster.py       # Clustering utilities
│   ├── mk_doublets.py   # Synthetic doublet generation
│   └── logger.py        # Logging configuration
├── tests/               # Test suite
├── doc/                 # Documentation and tutorials
├── pyproject.toml       # Project configuration
├── Dockerfile           # Container definition
├── docker-compose.yml   # Container orchestration
├── Makefile             # Development shortcuts
└── CHANGELOG.md         # Version history
```

## API Reference

### `vaeda.vaeda()`

Main function for doublet annotation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | required | Annotated data matrix with raw counts in `adata.X` |
| `layer` | `str \| None` | `None` | Use `adata.layers[layer]` instead of `adata.X` |
| `filter_genes` | `bool` | `True` | Select most variable genes |
| `verbose` | `int` | `0` | Verbosity level (0=quiet, 1=warnings, 2=info, 3=debug) |
| `gene_thresh` | `float` | `0.01` | Filter genes expressed in ≤ threshold × cells |
| `num_hvgs` | `int` | `2000` | Number of highly variable genes |
| `pca_comp` | `int` | `30` | Number of principal components |
| `enc_sze` | `int` | `5` | VAE latent space dimensionality |
| `max_eps_vae` | `int` | `1000` | Maximum VAE training epochs |
| `pat_vae` | `int` | `20` | VAE early-stopping patience |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `optimized` | `bool` | `False` | Vectorized doublet generation: O(n log n) vs legacy O(n²) |

**Returns:** `AnnData` with added fields:

- `adata.obsm['vaeda_embedding']` — VAE encoding for real cells
- `adata.obs['vaeda_scores']` — doublet probability scores
- `adata.obs['vaeda_calls']` — binary doublet/singlet calls

## Reproducibility

vaeda uses stochastic algorithms (VAE training, PU learning) that introduce natural run-to-run variation. Even with the same seed, consecutive runs may produce slightly different results due to non-determinism in PyTorch operations.

**Typical run-to-run variation (same seed, same code):**

| Metric | Expected Range |
|--------|----------------|
| Classification agreement | ~98–99% |
| Score correlation (Pearson r) | > 0.85 |
| Score correlation (Spearman ρ) | > 0.83 |

**For maximum reproducibility:**

```python
import os
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA only

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

import vaeda
result = vaeda.vaeda(adata, seed=42)
```

Note: deterministic mode may slow down computation and some operations may not have deterministic implementations.

## Other Doublet Detection Tools

- [DoubletFinder](https://github.com/chris-mcginnis-ucsf/DoubletFinder)
- [Scrublet](https://github.com/swolock/scrublet)
- [scDblFinder](https://bioconductor.org/packages/release/bioc/html/scDblFinder.html)
- [DoubletDetection](https://github.com/JonathanShor/DoubletDetection)

## Citation

If you use vaeda in your research, please cite:

> Schriever, H., Kostka, D. (2022). vaeda: doublet annotation in single-cell RNA sequencing data using variational autoencoders. *Bioinformatics*, 39(1), btac720.

## License

MIT

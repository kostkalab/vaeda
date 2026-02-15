# Changelog

All notable changes to the `vaeda` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-14

### Changed
- **BREAKING**: Migrated deep learning backend from TensorFlow/tf_keras to **PyTorch >= 2.6.0**.
  This is a complete rewrite of all neural network components:
  - `vae.py`: VAE encoder, decoder, and cluster classifier are now `torch.nn.Module` subclasses.
    The probabilistic layer (TensorFlow Probability `IndependentNormal`) is replaced with
    `torch.distributions.Independent(Normal(...))` and the reparameterisation trick.
  - `classifier.py`: Binary classifier is now a `torch.nn.Module` using `nn.Sequential`.
  - `pu.py`: PU learning training loops rewritten as manual PyTorch training loops instead
    of `model.fit()`. Uses `sklearn.metrics.average_precision_score` for PRAUC calculation
    instead of `tf.keras.metrics.AUC`.
  - `vaeda.py`: Main pipeline VAE training is now a manual PyTorch training loop with
    explicit early stopping and learning-rate scheduling via
    `torch.optim.lr_scheduler.MultiplicativeLR`.
- **BREAKING**: Removed all TensorFlow, tf_keras, and TensorFlow Probability dependencies.
  Core dependencies are now: `torch>=2.6.0` (replacing `tensorflow[and-cuda]>=2.20`,
  `tensorflow-probability[tf]>=0.25`).
- `define_clust_vae()` and `define_vae()` now return `(model, optimiser)` tuples instead of
  a compiled Keras model, since PyTorch separates model definition from optimiser creation.
- Device selection is automatic: CUDA > MPS > CPU (via `_get_device()` helper).

### Added
- Automatic device selection supporting CUDA, Apple MPS, and CPU fallback.
- `_EpochHistory` helper class in `pu.py` to provide a `.history` dict matching the
  tf_keras `History` API expected by the main pipeline.
- Test for PyTorch version >= 2.6.0 and version string 0.2.0.

### Removed
- `tensorflow[and-cuda]>=2.20` dependency
- `tensorflow-probability[tf]>=0.25` dependency
- `tf_keras` dependency
- All `tf.random.set_seed()` calls (replaced with `torch.manual_seed()`)
- All `tf.one_hot()` calls (replaced with `np.eye()` indexing)
- `Programming Language :: Rust` classifier (was incorrect)

### Fixed
- Removed inaccurate `Programming Language :: Rust` trove classifier from `pyproject.toml`.

### Migration Guide
- Users upgrading from v0.1.x need to install PyTorch instead of TensorFlow.
  The public API (`vaeda.vaeda()`, `vaeda.sim_inflate()`, `vaeda.cluster()`, etc.)
  is **unchanged** — only the underlying backend has changed.
- GPU support now requires a CUDA-compatible PyTorch installation
  (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).
- For Apple Silicon users, MPS acceleration is automatically used when available.
- The `verbose` parameter for `vaeda()` continues to control logging; the TF-specific
  `verbose` argument to `model.fit()` is no longer applicable.

## [0.1.1] - 2026-02-14

### Added
- Optional `optimized` parameter to `vaeda()` and `sim_inflate()` for vectorized doublet library size selection
  - `optimized=False` (default): Legacy O(n²) per-row selection for exact reproducibility
  - `optimized=True`: Vectorized O(n log n) selection, faster on large datasets, ~98% agreement with legacy
- Reproducibility section in README with run-to-run variation expectations
- TensorFlow deterministic ops instructions for maximum reproducibility

### Changed
- Migrate from legacy `np.random.seed()` to `np.random.Generator` API (NPY002 compliance)
- Remove unnecessary `np.copy()` calls in `sim_inflate()`, reducing memory usage

### Fixed
- Unused parameter warnings (ARG001) in `classifier.py` and `pu.py`

## [0.1.0] - 2026-02-14

### Added
- Modern Python packaging configuration with `pyproject.toml` following PEP 621 and PEP 660 best practices
- Version-specific dependencies for better compatibility
- Dependency lock file `uv.lock` for reproducible builds and better dependency resolution
- Comprehensive README with installation, development, and Docker instructions
- Docker development environment with dev/prod targets
- Makefile for common development tasks
- `uv` support for dependency management

### Changed
- **BREAKING**: Migrated from `tf.keras` to `tf_keras` imports to resolve compatibility issues with TensorFlow Probability and Keras 3.x.
- Updated dependency specifications with minimum versions for better stability
- Enhanced sparse matrix handling to support both sparse and dense AnnData objects
- Updated to Python 3.12+ with modern type hints
- Configure ruff for linting and formatting

### Fixed
- **Critical**: Fixed `ValueError: Only instances of keras.Layer can be added to a Sequential model` error when using TensorFlow Probability layers
- Fixed optimizer compatibility issues between TensorFlow and tf_keras
- Fixed callback compatibility issues in VAE training
- Fixed metrics compatibility in PU learning classifier
- Improved error handling for different matrix formats in AnnData objects

## [0.0.30] - 2022-04-10

### Added
- Initial release by Hannah Schriever (hcs31@pitt.edu)
- Core vaeda algorithm for doublet detection in single-cell RNA sequencing data
- Variational autoencoder-based approach for learning cell representations
- PU (Positive-Unlabeled) learning framework for doublet classification

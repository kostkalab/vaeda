from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def sim_inflate(
    X: npt.NDArray[np.float32],
    frac_doublets: float | None = None,
    seeds: Sequence[int] = (1234, 15232, 3060309),
    optimized: bool = False,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    """
    Generate simulated doublets by combining pairs of cells.

    Parameters
    ----------
    X : ndarray
        Expression matrix (cells x genes)
    frac_doublets : float, optional
        Fraction of doublets to generate. If None, generates n_cells
        doublets.
    seeds : Sequence[int]
        Random seeds for reproducibility
    optimized : bool, default=False
        If True, use vectorized library size selection (faster, ~98.5%
        agreement with legacy).
        If False, use legacy per-row selection (slower, exact
        reproducibility).

    Returns
    -------
    tuple
        (simulated_doublets, parent_indices_1, parent_indices_2)
    """
    if frac_doublets is None:
        num_doublets = 1 * X.shape[0]
    else:
        num_doublets = int(frac_doublets * X.shape[0])

    ind1 = np.arange(X.shape[0])
    ind2 = np.arange(X.shape[0])

    rng1 = np.random.Generator(np.random.PCG64(seeds[0]))
    rng1.shuffle(ind1)
    rng2 = np.random.Generator(np.random.PCG64(seeds[1]))
    rng2.shuffle(ind2)

    X1 = X[ind1, :]
    X2 = X[ind2, :]

    res = X1 + X2

    lib1 = np.sum(X1, axis=1)
    lib2 = np.sum(X2, axis=1)

    lib_sze = np.maximum.reduce([lib1, lib2])

    if optimized:
        inflated_sze = _vectorized_choice(lib_sze, seeds[2])
    else:
        inflated_sze = _legacy_choice(lib_sze, seeds[2])

    ls = np.sum(res, axis=1)
    sf = inflated_sze / ls
    res = np.multiply(res.T, sf).T

    return res[:num_doublets, :], ind1[:num_doublets], ind2[:num_doublets]


def _legacy_choice(
    lib_sze: npt.NDArray[np.float64], seed: int
) -> npt.NDArray[np.float64]:
    """
    Legacy per-row random choice (exact reproducibility with original
    code).

    For each row, creates a fresh generator with the same seed and picks
    from values >= threshold. Slower but matches original behavior
    exactly.
    """
    n = len(lib_sze)
    inflated_sze = np.empty(n)

    for i in range(n):
        g = np.random.Generator(np.random.PCG64(seed))
        inflated_sze[i] = g.choice(lib_sze[lib_sze >= lib_sze[i]])

    return inflated_sze


def _vectorized_choice(
    lib_sze: npt.NDArray[np.float64], seed: int
) -> npt.NDArray[np.float64]:
    """
    Vectorized random choice (faster, scientifically equivalent).

    Uses a single generator and precomputed sorted array for O(n log n)
    performance instead of O(n^2). Results differ from legacy but
    maintain the same statistical properties (random selection from
    valid candidates).
    """
    n = len(lib_sze)

    sorted_indices = np.argsort(lib_sze)
    sorted_lib = lib_sze[sorted_indices]

    ranks = np.empty(n, dtype=np.intp)
    ranks[sorted_indices] = np.arange(n)
    num_valid = n - ranks

    rng = np.random.Generator(np.random.PCG64(seed))

    random_floats = rng.random(n)
    chosen_offsets = (random_floats * num_valid).astype(np.intp)

    chosen_positions = ranks + chosen_offsets
    inflated_sze = sorted_lib[chosen_positions]

    return inflated_sze

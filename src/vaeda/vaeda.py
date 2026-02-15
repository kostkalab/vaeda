"""Main vaeda pipeline for doublet annotation in scRNAseq data.

v0.2.0 — rewritten for PyTorch >= 2.6.0, replacing TensorFlow and
tf_keras from v0.1.x.
"""

from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import numpy as np
import numpy.typing as npt
import scipy.sparse as scs
import torch
from kneed import KneeLocator
from loguru import logger
from scipy.signal import savgol_filter
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .cluster import cluster, fast_cluster
from .logger import init_logger
from .mk_doublets import sim_inflate
from .pu import PU, epoch_PU
from .vae import _get_device, define_clust_vae


def vaeda(
    adata: ad.AnnData,
    layer: str | None = None,
    filter_genes: bool = True,
    verbose: int = 0,
    save_dir: Path | None = None,
    gene_thresh: float = 0.01,
    num_hvgs: int = 2000,
    pca_comp: int = 30,
    quant: float = 0.25,
    enc_sze: int = 5,
    max_eps_vae: int = 1000,
    pat_vae: int = 20,
    LR_vae: float = 1e-3,
    clust_weight: int = 20000,
    rate: float = -0.75,
    N: int = 1,
    k_mult: int = 2,
    max_eps_PU: int = 250,
    LR_PU: float = 1e-3,
    mu: int | None = None,
    remove_homos: bool = True,
    use_old: bool = False,
    seed: int | None = None,
    optimized: bool = False,
) -> ad.AnnData:
    """Annotate doublets in single-cell RNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts in ``adata.X``.
    layer : str | None
        If provided, use ``adata.layers[layer]`` instead of ``adata.X``.
    filter_genes : bool
        If True, select the ``num_hvgs`` most variable genes.
    verbose : int
        Verbosity level (0=quiet, 1=warnings, 2=info, 3=debug).
    save_dir : Path | None
        Directory for saving intermediate results.
    gene_thresh : float
        Filter genes expressed in <= ``gene_thresh * n_cells`` cells.
    num_hvgs : int
        Number of highly variable genes to use.
    pca_comp : int
        Number of principal components for PCA reductions.
    quant : float
        Quantile for initial doublet-fraction estimate.
    enc_sze : int
        Dimensionality of the VAE latent space.
    max_eps_vae : int
        Maximum epochs for VAE training (early stopping may cut short).
    pat_vae : int
        Early-stopping patience for VAE training.
    LR_vae : float
        Learning rate for the VAE optimiser.
    clust_weight : float
        Weight for the cluster-classification loss in the VAE.
    rate : float
        Exponential decay rate for learning-rate scheduling after
        epoch 3.
    N : int
        Number of repeats for PU bagging.
    k_mult : int
        Determines the U/P ratio in PU folds.
    max_eps_PU : int
        Epochs for the PU epoch-selection round.
    LR_PU : float
        Learning rate for the PU classifier.
    mu : int | None
        Expected number of doublets; heuristic if None.
    remove_homos : bool
        Remove simulated doublets from same-cluster parents.
    use_old : bool
        Reuse previously saved intermediate results.
    seed : int | None
        Random seed for reproducibility.
    optimized : bool
        Use vectorized doublet library-size selection.

    Returns
    -------
    AnnData
        Input object with added fields:
        ``adata.obsm['vaeda_embedding']``,
        ``adata.obs['vaeda_scores']``,
        ``adata.obs['vaeda_calls']``.
    """
    init_logger(verbose=verbose)

    # Random seed handling
    if seed is not None:
        rng = np.random.Generator(np.random.PCG64(seed))
        seeds = rng.integers(0, high=(2**32 - 1), size=13)
    else:
        rng = np.random.Generator(np.random.PCG64())
        seeds = rng.integers(0, high=(2**32 - 1), size=13)

    if save_dir is None:
        use_old = False

    # Extract expression matrix
    if layer is None:
        x_mat: npt.NDArray[np.float64] = (
            adata.X.toarray() if issparse(adata.X) else adata.X
        )
    else:
        x_mat: npt.NDArray[np.float64] = (
            adata.layers[layer].toarray()
            if issparse(adata.layers[layer])
            else adata.layers[layer]
        )

    # ---- Simulated doublets ----
    old_sim = False
    if save_dir is not None:
        npz_sim_path = save_dir / "sim_doubs.npz"
        sim_ind_path = save_dir / "sim_ind.npy"
        if npz_sim_path.exists():
            old_sim = True

    if old_sim & use_old:
        if verbose != 0:
            logger.info("loading in simulated doublets")
        dat_sim = scs.load_npz(npz_sim_path)
        sim_ind = np.load(sim_ind_path)
        ind1 = sim_ind[0, :]
        ind2 = sim_ind[1, :]
        Xs = scs.csr_matrix(dat_sim).toarray()
    else:
        if verbose != 0:
            logger.info("generating simulated doublets")
        Xs, ind1, ind2 = sim_inflate(x_mat, optimized=optimized)
        dat_sim = scs.csr_matrix(Xs)
        if save_dir is not None:
            scs.save_npz(npz_sim_path, dat_sim)
            np.save(sim_ind_path, np.vstack([ind1, ind2]))

    Y = np.concatenate([np.zeros(x_mat.shape[0]), np.ones(Xs.shape[0])])
    x_mat = np.vstack([x_mat, Xs])

    # ---- Gene filtering ----
    if filter_genes:
        thresh = np.floor(x_mat.shape[0]) * gene_thresh
        tmp = np.sum((x_mat > 0), axis=0) > thresh
        if np.sum(tmp) >= num_hvgs:
            x_mat = x_mat[:, np.ravel(tmp)]

        if x_mat.shape[1] > num_hvgs:
            var = np.var(x_mat, axis=0)
            rng0 = np.random.Generator(  # noqa: F841
                np.random.PCG64(seeds[0])
            )
            hvgs = np.argpartition(var, -num_hvgs)[-num_hvgs:]
            x_mat = x_mat[:, hvgs]

    # ---- KNN features ----
    neighbors = int(np.sqrt(x_mat.shape[0]))

    temp_X = np.log2(x_mat + 1)
    scaler = StandardScaler().fit(temp_X.T)
    temp_X = scaler.transform(temp_X.T).T

    rng1 = np.random.Generator(  # noqa: F841
        np.random.PCG64(seeds[1])
    )
    pca = PCA(n_components=pca_comp)
    pca_proj = pca.fit_transform(temp_X)
    del temp_X

    rng2 = np.random.Generator(  # noqa: F841
        np.random.PCG64(seeds[2])
    )
    knn = NearestNeighbors(n_neighbors=neighbors)
    knn.fit(pca_proj, Y)
    graph = knn.kneighbors_graph(pca_proj)
    knn_feature = np.squeeze(np.array(np.sum(graph[:, Y == 1], axis=1) / neighbors))

    # Estimate true fraction of doublets
    quantile = np.quantile(knn_feature[Y == 1], quant)
    num = np.sum(knn_feature[Y == 0] >= quantile)
    min_num = int(np.round(sum(Y == 0) * 0.05))
    num = np.max([min_num, num, 1])

    prob = knn_feature[Y == 1] / np.sum(knn_feature[Y == 1])
    rng3 = np.random.Generator(np.random.PCG64(seeds[3]))
    ind = rng3.choice(np.arange(sum(Y == 1)), size=num, p=prob, replace=False)

    # Down-sample simulated doublets
    enc_ind = np.concatenate([
        np.arange(sum(Y == 0)),
        (sum(Y == 0) + ind),
    ])
    x_mat = x_mat[enc_ind, :]
    Y = Y[enc_ind]
    knn_feature = knn_feature[enc_ind]

    # Re-scale
    x_mat = np.log2(x_mat + 1)
    rng4 = np.random.Generator(  # noqa: F841
        np.random.PCG64(seeds[4])
    )
    scaler = StandardScaler().fit(x_mat.T)
    rng5 = np.random.Generator(  # noqa: F841
        np.random.PCG64(seeds[5])
    )
    x_mat = scaler.transform(x_mat.T).T

    # ---- Clustering ----
    if x_mat.shape[0] >= 1000:
        clust = fast_cluster(x_mat, comp=pca_comp)
    else:
        clust = cluster(x_mat, comp=pca_comp)

    if remove_homos:
        c = clust[Y == 0]
        hetero_ind = c[ind1] != c[ind2]
        hetero_ind = hetero_ind[ind]
        if save_dir is not None:
            np.save(save_dir / "which_sim_doubs.npy", ind[hetero_ind])
        hetero_ind = np.concatenate([
            np.full(sum(Y == 0), True),
            hetero_ind,
        ])
        x_mat = x_mat[hetero_ind, :]
        Y = Y[hetero_ind]
        clust = clust[hetero_ind]
        knn_feature = knn_feature[hetero_ind]
    elif save_dir is not None:
        np.save(save_dir / "which_sim_doubs.npy", ind)

    # ---- VAE training ----
    X_train, X_test, clust_train, clust_test = train_test_split(
        x_mat, clust, test_size=0.1, random_state=12345
    )

    n_clust = int(clust.max()) + 1
    clust_train_oh = np.eye(n_clust)[clust_train.astype(int)]
    clust_test_oh = np.eye(n_clust)[clust_test.astype(int)]

    ngens = x_mat.shape[1]

    old_vae = False
    if save_dir is not None:
        vae_path_real = save_dir / "embedding_real.npy"
        vae_path_sim = save_dir / "embedding_sim.npy"
        if Path(vae_path_real).exists() & Path(vae_path_sim).exists():
            old_vae = True

    if old_vae & use_old:
        if verbose != 0:
            logger.info("using existing encoding")
        encoding_real = np.load(vae_path_real)
        encoding_sim = np.load(vae_path_sim)
        encoding = np.vstack([encoding_real, encoding_sim])
    else:
        if verbose != 0:
            logger.info("generating VAE encoding")

        torch.manual_seed(seeds[6])
        vae, optimiser = define_clust_vae(
            enc_sze,
            ngens,
            n_clust,
            LR=LR_vae,
            clust_weight=clust_weight,
        )
        device = _get_device()

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        clust_train_t = torch.tensor(clust_train_oh, dtype=torch.float32, device=device)
        clust_test_t = torch.tensor(clust_test_oh, dtype=torch.float32, device=device)

        # Learning-rate scheduler (exponential decay after epoch 3)
        def lr_lambda(epoch: int) -> float:
            if epoch < 3:
                return 1.0
            return float(np.exp(rate))

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimiser, lr_lambda=lr_lambda
        )

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0
        batch_size = 32  # Keras default

        for epoch in range(max_eps_vae):
            # Train step (minibatch, matching Keras default batch_size=32)
            vae.train()
            n_train = X_train_t.shape[0]
            # Shuffle training data each epoch
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                x_batch = X_train_t[idx]
                c_batch = clust_train_t[idx]

                optimiser.zero_grad()
                recon_mu, recon_logvar, clust_pred, _, enc_mu, enc_logvar = vae(x_batch)
                batch_loss, _, _ = vae.loss(
                    x_batch,
                    recon_mu,
                    recon_logvar,
                    enc_mu,
                    enc_logvar,
                    clust_pred,
                    c_batch,
                )
                batch_loss.backward()
                optimiser.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

            scheduler.step()

            # Validation step
            vae.eval()
            with torch.no_grad():
                (
                    v_recon_mu,
                    v_recon_logvar,
                    v_clust_pred,
                    _,
                    v_enc_mu,
                    v_enc_logvar,
                ) = vae(X_test_t)
                val_loss, _, _ = vae.loss(
                    X_test_t,
                    v_recon_mu,
                    v_recon_logvar,
                    v_enc_mu,
                    v_enc_logvar,
                    v_clust_pred,
                    clust_test_t,
                )
                val_loss_val = val_loss.item()

            # Early stopping check
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= pat_vae:
                    if verbose != 0:
                        logger.info(f"VAE early stopping at epoch {epoch}")
                    break

        # Extract encodings
        vae.eval()
        x_mat_t = torch.tensor(x_mat, dtype=torch.float32, device=device)
        with torch.no_grad():
            torch.manual_seed(seeds[7])
            z, _, _ = vae.encoder(x_mat_t)
            encoding = z.detach().cpu().numpy()

        if save_dir is not None:
            np.save(vae_path_real, encoding[Y == 0, :])
            np.save(vae_path_sim, encoding[Y == 1, :])

    # ---- PU learning ----
    if save_dir is not None:
        np.save(save_dir / "knn_feature_real.npy", knn_feature[Y == 0])
        np.save(save_dir / "knn_feature_sim.npy", knn_feature[Y == 1])
        np.save(save_dir / "clusters_real.npy", clust[Y == 0])
        np.save(save_dir / "clusters_sim.npy", clust[Y == 1])

    encoding = np.vstack([knn_feature, encoding.T]).T

    if verbose != 0:
        logger.info("starting PU Learning")
    u = encoding[Y == 0, :]
    p = encoding[Y == 1, :]

    num_cells = p.shape[0] * k_mult
    k = int(u.shape[0] / num_cells)
    k = max(k, 2)

    hist = epoch_PU(
        u,
        p,
        k,
        N,
        max_eps_PU,
        seeds=seeds[8:],
        puLR=LR_PU,
        _verbose=verbose,
    )

    y = np.log(hist.history["loss"])
    x = np.arange(len(y))
    yhat = savgol_filter(y, window_length=7, polyorder=1)

    y = yhat
    x = np.arange(len(y))

    kneedle = KneeLocator(x, y, S=10, curve="convex", direction="decreasing")
    knee = kneedle.knee

    if knee is None:
        knee = len(y) // 2

    match knee:
        case knee if num < 500:
            knee = knee + 1
        case knee if knee < 20:
            knee = 20
        case knee if knee > 250:
            knee = 250
        case _:
            knee = 250

    preds, preds_on_p, *_ = PU(
        u,
        p,
        k,
        N,
        knee,
        seeds=seeds[8:],
        puLR=LR_PU,
        _verbose=verbose,
    )

    if save_dir is not None:
        np.save(save_dir / "scores.npy", preds)
        np.save(save_dir / "scores_on_sim.npy", preds_on_p)

    # ---- Doublet calling ----
    maximum = np.max([np.max(preds), np.max(preds_on_p)])
    minimum = np.min([np.min(preds), np.min(preds_on_p)])

    thresholds = np.arange(minimum, maximum, 0.001)

    n = len(preds)

    if mu is None:
        dbr = n / 10**5
        dbl_expected = n * dbr
    else:
        dbr = mu / n
        dbl_expected = mu

    dbr_sd = np.sqrt(n * dbr * (1 - dbr))

    fnr = []
    fpr = []
    nll_doub = []

    o_t = np.sum(preds >= thresholds[-1])
    norm_factor = -(_log_norm(o_t, dbl_expected, dbr_sd))

    for thresh in thresholds:
        o_t = np.sum(preds >= thresh)
        fnr.append(np.sum(preds_on_p < thresh) / len(preds_on_p))
        fpr.append(o_t / len(preds))
        nll_doub.append(-(_log_norm(o_t, dbl_expected, dbr_sd) / norm_factor))

    cost = np.array(fnr) + np.array(fpr) + np.array(nll_doub) ** 2

    t = thresholds[np.argmin(cost)]
    call_mask = preds > t

    calls = np.full(len(preds), "singlet")
    calls[call_mask] = "doublet"
    if save_dir is not None:
        if verbose != 0:
            logger.info("saving calls")
        np.save(save_dir / "doublet_calls.npy", calls)

    adata.obs["vaeda_scores"] = preds
    adata.obs["vaeda_calls"] = calls
    adata.obsm["vaeda_embedding"] = encoding[Y == 0, :]

    return adata


def _log_norm(x: float, mean: float, sd: float) -> float:
    t1 = -np.log(sd * np.sqrt(2 * math.pi))
    t2 = (-0.5) * ((x - mean) / sd) ** 2
    return t1 + t2

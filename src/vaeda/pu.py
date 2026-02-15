"""PU (Positive-Unlabeled) learning implementation using PyTorch.

Replaces the TensorFlow/tf_keras-based implementation from v0.1.x.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import Progress
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors

from .classifier import define_classifier
from .vae import _get_device


def _train_one_epoch(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Train a single epoch with minibatches; return (loss, auc)."""
    model.train()
    device = X.device
    n = X.shape[0]
    perm = torch.randperm(n, device=device)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = perm[start:end]
        x_batch = X[idx]
        y_batch = Y[idx]

        optimiser.zero_grad()
        preds = model(x_batch)
        loss = F.binary_cross_entropy(preds, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n_batches += 1

    # Compute epoch-level AUC on full data
    model.eval()
    with torch.no_grad():
        all_preds = model(X).cpu().numpy()
        all_targets = Y.cpu().numpy()
        from sklearn.metrics import average_precision_score

        try:
            auc_val = float(average_precision_score(all_targets, all_preds))
        except ValueError:
            auc_val = 0.0

    return total_loss / n_batches, auc_val


def PU(
    U: np.ndarray,
    P: np.ndarray,
    k: int,
    N: int,
    cls_eps: int,
    seeds: np.ndarray,
    clss: str = "NN",
    _puPat: int = 5,
    puLR: float = 1e-3,
    num_layers: int = 1,
    _stop_metric: str = "ValAUC",
    _verbose: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Positive-Unlabeled bagging classifier.

    Parameters are the same as in v0.1.x for API compatibility.
    """
    device = _get_device()
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds = np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N * k, cls_eps])
    val_hists = np.zeros([N * k, cls_eps])
    auc_hists = np.zeros([N * k, cls_eps])
    val_auc = np.zeros([N * k, cls_eps])

    P_tensor = torch.tensor(P, dtype=torch.float32, device=device)

    i = 0
    with Progress() as progress:
        train_task = progress.add_task(description="", total=k)
        for test, train in rkf.split(U):
            i += 1
            progress.update(
                train_task,
                description=f"{i!s}/{(N * k)!s} iterations",
                refresh=True,
            )

            X = np.vstack([U[train, :], P])
            Y = np.concatenate([
                np.zeros(shape=[len(train)]),
                np.ones(shape=[P.shape[0]]),
            ])

            x = U[test, :]

            if clss == "NN":
                # Set seeds for reproducibility
                torch.manual_seed(seeds[1])

                classifier = define_classifier(ngens=X.shape[1], num_layers=num_layers)
                optimiser = torch.optim.Adam(classifier.parameters(), lr=puLR)

                # Shuffle training data
                ind = np.arange(X.shape[0])
                rng2 = np.random.Generator(np.random.PCG64(seeds[2]))
                rng2.shuffle(ind)

                X_t = torch.tensor(X[ind, :], dtype=torch.float32, device=device)
                Y_t = torch.tensor(Y[ind], dtype=torch.float32, device=device)
                x_t = torch.tensor(x, dtype=torch.float32, device=device)

                torch.manual_seed(seeds[3])
                for ep in range(cls_eps):
                    loss_val, auc_val = _train_one_epoch(classifier, optimiser, X_t, Y_t)
                    hists[i - 1, ep] = loss_val
                    auc_hists[i - 1, ep] = auc_val

                # Predictions
                classifier.eval()
                with torch.no_grad():
                    torch.manual_seed(seeds[3])
                    preds[test] = preds[test] + classifier(x_t).cpu().numpy()
                    torch.manual_seed(seeds[3])
                    preds_on_P = preds_on_P + classifier(P_tensor).cpu().numpy()

            if clss == "knn":
                neighbors = int(np.sqrt(X.shape[0]))
                knn = NearestNeighbors(n_neighbors=neighbors)
                knn.fit(X, Y)

                graph = knn.kneighbors_graph(x)
                preds[test] = preds[test] + np.squeeze(
                    np.array(np.sum(graph[:, Y == 1], axis=1) / neighbors)
                )

                graph = knn.kneighbors_graph(P)
                preds_on_P = preds_on_P + np.squeeze(
                    np.array(np.sum(graph[:, Y == 1], axis=1) / neighbors)
                )

    preds = preds / ((i / k) * (k - 1))
    preds_on_P = preds_on_P / ((i / k) * (k - 1))

    return preds, preds_on_P, hists, val_hists, auc_hists, val_auc


def epoch_PU(
    U: np.ndarray,
    P: np.ndarray,
    k: int,
    N: int,
    cls_eps: int,
    seeds: np.ndarray,
    _puPat: int = 5,
    puLR: float = 1e-3,
    num_layers: int = 1,
    _stop_metric: str = "ValAUC",
    _verbose: int = 0,
) -> _EpochHistory:
    """Train a single PU fold to determine optimal epoch count.

    Returns a history-like object with a ``.history`` dict containing
    ``"loss"`` and ``"auc"`` lists, matching the tf_keras API used by
    the caller.
    """
    device = _get_device()
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    i = 0
    with Progress() as progress:
        train_task = progress.add_task(description="", total=k)
        for _, train in rkf.split(U):
            i += 1
            progress.update(
                train_task,
                description=f"{i!s}/{(N * k)!s} iterations",
                refresh=True,
            )
            X = np.vstack([U[train, :], P])
            Y = np.concatenate([
                np.zeros([len(train)]),
                np.ones([P.shape[0]]),
            ])

            torch.manual_seed(seeds[1])
            classifier = define_classifier(X.shape[1], num_layers=num_layers)
            optimiser = torch.optim.Adam(classifier.parameters(), lr=puLR)

            # Shuffle training data
            ind = np.arange(X.shape[0])
            rng2 = np.random.Generator(np.random.PCG64(seeds[2]))
            rng2.shuffle(ind)

            X_t = torch.tensor(X[ind, :], dtype=torch.float32, device=device)
            Y_t = torch.tensor(Y[ind], dtype=torch.float32, device=device)

            torch.manual_seed(seeds[3])
            loss_history: list[float] = []
            auc_history: list[float] = []
            for _ in range(cls_eps):
                loss_val, auc_val = _train_one_epoch(classifier, optimiser, X_t, Y_t)
                loss_history.append(loss_val)
                auc_history.append(auc_val)

            break  # Only first fold, matching v0.1.x behaviour

    return _EpochHistory({"loss": loss_history, "auc": auc_history})


class _EpochHistory:
    """Minimal history object mimicking the tf_keras History API."""

    def __init__(self, history: dict[str, list[float]]) -> None:
        self.history = history

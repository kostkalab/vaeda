"""Neural network classifier for PU learning, using PyTorch.

Replaces the tf_keras-based classifier from v0.1.x.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class BinaryClassifier(nn.Module):
    """Simple feedforward binary classifier with batch normalisation."""

    def __init__(self, n_input: int, num_layers: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(n_input)]
        if num_layers == 1:
            layers.append(nn.Linear(n_input, 1))
        elif num_layers == 2:
            logger.info("using 2 layers in classifier")
            layers.extend([
                nn.Linear(n_input, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
            ])
        else:
            msg = "Only using 1 or 2 layers is supported"
            raise ValueError(msg)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Use Glorot uniform (Xavier) to match Keras Dense defaults."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def define_classifier(ngens: int, num_layers: int = 1) -> BinaryClassifier:
    """Build a binary classifier and return it.

    The model is placed on the best available device (CUDA > MPS > CPU).
    """
    from .vae import _get_device

    device = _get_device()
    model = BinaryClassifier(ngens, num_layers=num_layers).to(device)
    return model

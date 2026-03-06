"""Linear probe-based steering vector extraction.

Trains a linear classifier to distinguish persona-present from persona-absent
activations. The learned weight vector defines the steering direction.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .hooks import ActivationExtractor

logger = logging.getLogger(__name__)


class LinearProbeExtractor:
    """Extract persona steering vectors via supervised linear probes."""

    def __init__(
        self,
        extractor: ActivationExtractor,
        epochs: int = 20,
        lr: float = 1e-3,
    ):
        self.extractor = extractor
        self.epochs = epochs
        self.lr = lr

    def extract_vectors(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        token_position: int = -1,
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        """Compute linear probe steering vectors across all layers.

        Args:
            positive_texts: Texts with persona present.
            negative_texts: Texts with persona absent.
            token_position: Token position for extraction.
            batch_size: Batch size for processing.

        Returns:
            Dict mapping layer_index -> steering vector (hidden_size,).
        """
        logger.info(
            f"Extracting linear probe vectors from {len(positive_texts)} positive "
            f"and {len(negative_texts)} negative samples"
        )

        pos_activations = self.extractor.extract(
            positive_texts,
            token_position=token_position,
            batch_size=batch_size,
        )
        neg_activations = self.extractor.extract(
            negative_texts,
            token_position=token_position,
            batch_size=batch_size,
        )

        vectors = {}
        for layer_idx in pos_activations:
            pos = pos_activations[layer_idx].float()
            neg = neg_activations[layer_idx].float()

            vector, accuracy = self._train_probe(pos, neg)
            vectors[layer_idx] = vector
            logger.info(f"Layer {layer_idx}: probe accuracy = {accuracy:.3f}")

        logger.info(f"Computed linear probe vectors for {len(vectors)} layers")
        return vectors

    def _train_probe(
        self, pos: torch.Tensor, neg: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Train a single linear probe and return the weight vector.

        Returns:
            Tuple of (weight_vector, accuracy).
        """
        X = torch.cat([pos, neg], dim=0)
        y = torch.cat([
            torch.ones(pos.shape[0]),
            torch.zeros(neg.shape[0]),
        ])

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        hidden_size = X.shape[1]
        probe = nn.Linear(hidden_size, 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        probe.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = probe(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Compute accuracy
        probe.eval()
        with torch.no_grad():
            all_logits = probe(X).squeeze(-1)
            preds = (all_logits > 0).float()
            accuracy = (preds == y).float().mean().item()

        # The weight vector is the steering direction
        weight = probe.weight.data.squeeze(0).detach().cpu()
        # Normalize to unit norm for consistent multiplier behavior
        weight = weight / weight.norm()

        return weight, accuracy

    @staticmethod
    def save_vectors(vectors: dict[int, torch.Tensor], path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, path)
        logger.info(f"Saved linear probe vectors to {path}")

    @staticmethod
    def load_vectors(path: str | Path) -> dict[int, torch.Tensor]:
        return torch.load(path, weights_only=True)

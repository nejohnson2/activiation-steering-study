"""Contrastive Activation Addition (CAA) vector extraction.

Computes steering vectors as the mean difference in activations between
positive (persona-present) and negative (persona-absent) prompt pairs.

Reference: Turner et al. (2023), Rimsky et al. (2024)
"""

import logging
from pathlib import Path

import torch

from .hooks import ActivationExtractor

logger = logging.getLogger(__name__)


class CAAExtractor:
    """Extract persona steering vectors via Contrastive Activation Addition."""

    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor

    def extract_vectors(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        token_position: int = -1,
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        """Compute CAA steering vectors across all layers.

        Args:
            positive_texts: Texts with persona present.
            negative_texts: Texts with persona absent (same length).
            token_position: Token position for extraction.
            batch_size: Batch size for processing.

        Returns:
            Dict mapping layer_index -> steering vector (hidden_size,).
        """
        assert len(positive_texts) == len(negative_texts), (
            "Positive and negative text lists must be the same length"
        )

        logger.info(f"Extracting CAA vectors from {len(positive_texts)} pairs")

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
            pos_mean = pos_activations[layer_idx].float().mean(dim=0)
            neg_mean = neg_activations[layer_idx].float().mean(dim=0)
            vectors[layer_idx] = pos_mean - neg_mean

        logger.info(f"Computed CAA vectors for {len(vectors)} layers")
        return vectors

    @staticmethod
    def save_vectors(vectors: dict[int, torch.Tensor], path: str | Path):
        """Save steering vectors to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, path)
        logger.info(f"Saved CAA vectors to {path}")

    @staticmethod
    def load_vectors(path: str | Path) -> dict[int, torch.Tensor]:
        """Load steering vectors from disk."""
        return torch.load(path, weights_only=True)

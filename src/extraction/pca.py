"""PCA-based steering vector extraction.

Extracts persona-relevant directions by applying PCA to activations from
persona-relevant prompts. The top principal component(s) capture the
dominant variation axis, which often corresponds to the persona direction.
"""

import logging
from pathlib import Path

import torch
from sklearn.decomposition import PCA

from .hooks import ActivationExtractor

logger = logging.getLogger(__name__)


class PCAExtractor:
    """Extract persona steering vectors via PCA on activations."""

    def __init__(self, extractor: ActivationExtractor, n_components: int = 1):
        self.extractor = extractor
        self.n_components = n_components

    def extract_vectors(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        token_position: int = -1,
        batch_size: int = 8,
        max_seq_len: int = 128,
    ) -> dict[int, torch.Tensor]:
        """Compute PCA-based steering vectors across all layers.

        Concatenates positive and negative activations, runs PCA, and uses
        the first principal component as the steering direction. The sign is
        oriented so the vector points from negative to positive centroid.

        Args:
            positive_texts: Texts with persona present.
            negative_texts: Texts with persona absent.
            token_position: Token position for extraction.
            batch_size: Batch size for processing.
            max_seq_len: Maximum sequence length for tokenization.

        Returns:
            Dict mapping layer_index -> steering vector (hidden_size,).
        """
        logger.info(
            f"Extracting PCA vectors from {len(positive_texts)} positive "
            f"and {len(negative_texts)} negative samples"
        )

        pos_activations = self.extractor.extract(
            positive_texts,
            token_position=token_position,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        neg_activations = self.extractor.extract(
            negative_texts,
            token_position=token_position,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        vectors = {}
        for layer_idx in pos_activations:
            pos = pos_activations[layer_idx].float()
            neg = neg_activations[layer_idx].float()

            # Concatenate and run PCA
            combined = torch.cat([pos, neg], dim=0).numpy()
            pca = PCA(n_components=self.n_components)
            pca.fit(combined)

            # First principal component
            pc1 = torch.from_numpy(pca.components_[0])

            # Orient: ensure pc1 points from negative to positive centroid
            diff = pos.mean(dim=0) - neg.mean(dim=0)
            if torch.dot(pc1, diff) < 0:
                pc1 = -pc1

            vectors[layer_idx] = pc1

        logger.info(f"Computed PCA vectors for {len(vectors)} layers")
        return vectors

    @staticmethod
    def save_vectors(vectors: dict[int, torch.Tensor], path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, path)
        logger.info(f"Saved PCA vectors to {path}")

    @staticmethod
    def load_vectors(path: str | Path) -> dict[int, torch.Tensor]:
        return torch.load(path, weights_only=True)

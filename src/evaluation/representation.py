"""Representation-level evaluation metrics for steering vectors.

Measures how effectively steering vectors modify the model's internal
representations without requiring text generation.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class RepresentationMetrics:
    """Compute representation-level metrics for steering evaluation."""

    @staticmethod
    def cosine_similarity(
        steered_acts: torch.Tensor,
        target_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between steered activations and target centroid.

        Args:
            steered_acts: Activations after steering, shape (N, hidden_size).
            target_centroid: Target persona centroid, shape (hidden_size,).

        Returns:
            Per-sample cosine similarities, shape (N,).
        """
        steered = steered_acts.float()
        target = target_centroid.float().unsqueeze(0)
        return torch.nn.functional.cosine_similarity(steered, target, dim=1)

    @staticmethod
    def projection_magnitude(
        activations: torch.Tensor,
        steering_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar projection of activations onto the steering direction.

        Measures how strongly activations align with the steering vector.

        Args:
            activations: Shape (N, hidden_size).
            steering_vector: Shape (hidden_size,).

        Returns:
            Projection magnitudes, shape (N,).
        """
        acts = activations.float()
        vec = steering_vector.float()
        vec_norm = vec / vec.norm()
        return acts @ vec_norm

    @staticmethod
    def vector_alignment(
        vectors_a: dict[int, torch.Tensor],
        vectors_b: dict[int, torch.Tensor],
    ) -> dict[int, float]:
        """Cosine similarity between two sets of steering vectors per layer.

        Useful for comparing vectors from different extraction methods.

        Args:
            vectors_a: Dict mapping layer -> vector from method A.
            vectors_b: Dict mapping layer -> vector from method B.

        Returns:
            Dict mapping layer -> alignment score.
        """
        common_layers = set(vectors_a.keys()) & set(vectors_b.keys())
        alignment = {}
        for layer in sorted(common_layers):
            a = vectors_a[layer].float()
            b = vectors_b[layer].float()
            sim = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
            alignment[layer] = sim
        return alignment

    @staticmethod
    def compute_all(
        steered_acts: dict[int, torch.Tensor],
        unsteered_acts: dict[int, torch.Tensor],
        target_centroids: dict[int, torch.Tensor],
        steering_vectors: dict[int, torch.Tensor],
    ) -> dict[str, dict[int, float]]:
        """Compute all representation metrics across layers.

        Args:
            steered_acts: Activations after steering, per layer.
            unsteered_acts: Activations before steering, per layer.
            target_centroids: Persona target centroids, per layer.
            steering_vectors: Applied steering vectors, per layer.

        Returns:
            Nested dict: metric_name -> layer -> value.
        """
        metrics = {
            "cosine_shift": {},
            "projection_shift": {},
        }

        for layer in steered_acts:
            steered_cos = RepresentationMetrics.cosine_similarity(
                steered_acts[layer], target_centroids[layer]
            ).mean().item()
            unsteered_cos = RepresentationMetrics.cosine_similarity(
                unsteered_acts[layer], target_centroids[layer]
            ).mean().item()
            metrics["cosine_shift"][layer] = steered_cos - unsteered_cos

            steered_proj = RepresentationMetrics.projection_magnitude(
                steered_acts[layer], steering_vectors[layer]
            ).mean().item()
            unsteered_proj = RepresentationMetrics.projection_magnitude(
                unsteered_acts[layer], steering_vectors[layer]
            ).mean().item()
            metrics["projection_shift"][layer] = steered_proj - unsteered_proj

        return metrics

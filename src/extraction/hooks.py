"""Activation extraction using raw PyTorch forward hooks."""

import logging
from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.device import get_device, get_dtype

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extract activations from transformer residual streams using PyTorch hooks.

    Hooks are registered on the output of each transformer layer's residual stream.
    Supports Llama, Gemma, and Qwen architectures by auto-detecting layer modules.
    """

    def __init__(self, model_name: str, device: torch.device | None = None):
        self.model_name = model_name
        self.device = device or get_device()
        self.dtype = get_dtype(self.device)
        self.model = None
        self.tokenizer = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[int, torch.Tensor] = {}

    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device} with dtype {self.dtype}")

    def get_layer_modules(self) -> list[nn.Module]:
        """Auto-detect and return the list of transformer layer modules."""
        model = self.model
        # Try common attribute paths for different architectures
        for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                layers = list(obj)
                logger.info(f"Found {len(layers)} layers via {attr_path}")
                return layers
            except AttributeError:
                continue
        raise ValueError(f"Cannot find transformer layers for model: {self.model_name}")

    def get_num_layers(self) -> int:
        """Return the number of transformer layers."""
        return len(self.get_layer_modules())

    def register_hooks(
        self,
        layer_indices: list[int] | None = None,
        token_position: int = -1,
    ):
        """Register forward hooks on specified layers.

        Args:
            layer_indices: Which layers to hook. None = all layers.
            token_position: Which token position to extract. -1 = last token.
        """
        self.clear_hooks()
        layers = self.get_layer_modules()
        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        for idx in layer_indices:
            layer = layers[idx]

            def make_hook(layer_idx: int) -> Callable:
                def hook_fn(module, input, output):
                    # Output is typically a tuple; first element is the hidden state
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    # Extract the specified token position
                    self._activations[layer_idx] = hidden[:, token_position, :].detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

        logger.info(f"Registered hooks on {len(layer_indices)} layers")

    def clear_hooks(self):
        """Remove all registered hooks and clear cached activations."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()

    def extract(
        self,
        texts: list[str],
        token_position: int = -1,
        layer_indices: list[int] | None = None,
        batch_size: int = 8,
        max_seq_len: int = 128,
    ) -> dict[int, torch.Tensor]:
        """Extract activations for a list of texts.

        Args:
            texts: Input texts to process.
            token_position: Token position to extract (-1 = last token).
            layer_indices: Layers to extract from. None = all.
            batch_size: Batch size for processing.
            max_seq_len: Maximum sequence length for tokenization.

        Returns:
            Dict mapping layer_index -> tensor of shape (num_texts, hidden_size).
        """
        self.register_hooks(layer_indices=layer_indices, token_position=token_position)
        all_activations: dict[int, list[torch.Tensor]] = {}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs)

            for layer_idx, act in self._activations.items():
                if layer_idx not in all_activations:
                    all_activations[layer_idx] = []
                all_activations[layer_idx].append(act)

        self.clear_hooks()

        # Concatenate batches
        result = {
            layer_idx: torch.cat(acts, dim=0)
            for layer_idx, acts in all_activations.items()
        }
        logger.info(
            f"Extracted activations: {len(result)} layers, "
            f"{next(iter(result.values())).shape[0]} samples each"
        )
        return result

    def __del__(self):
        self.clear_hooks()

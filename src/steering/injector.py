"""Steering vector injection via PyTorch hooks.

Adds a scaled steering vector to the residual stream at a specified layer
during forward passes, enabling controlled persona steering.
"""

import logging
from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class SteeringInjector:
    """Inject steering vectors into transformer residual streams."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def get_layer_modules(self) -> list[nn.Module]:
        """Auto-detect transformer layer modules."""
        for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            obj = self.model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue
        raise ValueError("Cannot find transformer layers")

    def steer(
        self,
        vector: torch.Tensor,
        layer_idx: int,
        multiplier: float = 1.0,
    ):
        """Register a steering hook on the specified layer.

        Args:
            vector: Steering vector of shape (hidden_size,).
            layer_idx: Which layer to inject at.
            multiplier: Scaling factor for the steering vector.
        """
        self.clear()
        layers = self.get_layer_modules()
        layer = layers[layer_idx]
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        steering_vec = (vector * multiplier).to(device=device, dtype=dtype)

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + steering_vec.unsqueeze(0).unsqueeze(0)
                return (hidden,) + output[1:]
            else:
                return output + steering_vec.unsqueeze(0).unsqueeze(0)

        h = layer.register_forward_hook(hook_fn)
        self._hooks.append(h)
        logger.debug(f"Steering hook registered: layer={layer_idx}, multiplier={multiplier}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text with the current steering configuration.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to use sampling (vs greedy).

        Returns:
            Generated text (excluding the input prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Decode only the generated tokens
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def clear(self):
        """Remove all steering hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.clear()

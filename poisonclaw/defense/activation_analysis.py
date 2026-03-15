"""Activation pattern analysis defense for PoisonClaw.

Inspects intermediate activations of the poisoned VLM to detect
anomalous patterns that may indicate backdoor presence.
Requires torch and transformers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ActivationAnalysis:
    """Analyze VLM activations to detect backdoor-related patterns.

    Hooks into the model's transformer layers and captures hidden states
    during inference. Compares activations on clean vs. triggered inputs
    to identify anomalous neurons.

    Args:
        model: HuggingFace model instance (e.g. Qwen2-VL).
        layer_indices: Which transformer layers to monitor.
        threshold_std: Anomaly threshold in standard deviations.
    """

    def __init__(
        self,
        model: Any,
        layer_indices: Optional[list[int]] = None,
        threshold_std: float = 3.0,
    ) -> None:
        self.model = model
        self.layer_indices = layer_indices or [-1, -4, -8]  # last few layers
        self.threshold_std = threshold_std
        self._hooks: list[Any] = []
        self._captured: dict[int, list[torch.Tensor]] = {}

    def register_hooks(self) -> None:
        """Attach forward hooks to the specified transformer layers."""
        layers = self._get_layers()
        for idx in self.layer_indices:
            if idx >= len(layers) or idx < -len(layers):
                logger.warning("Layer index %d out of range (n_layers=%d).", idx, len(layers))
                continue
            layer = layers[idx]
            real_idx = idx if idx >= 0 else len(layers) + idx

            def make_hook(layer_idx: int):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._captured.setdefault(layer_idx, []).append(
                        hidden.detach().cpu().float()
                    )
                return hook

            handle = layer.register_forward_hook(make_hook(real_idx))
            self._hooks.append(handle)
        logger.info("Registered activation hooks on %d layer(s).", len(self._hooks))

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("Activation hooks removed.")

    def compute_anomaly_score(
        self,
        clean_activations: dict[int, list[torch.Tensor]],
        target_activations: dict[int, list[torch.Tensor]],
    ) -> dict[int, float]:
        """Compute per-layer anomaly scores between clean and target activations.

        Args:
            clean_activations: Captured activations on clean inputs.
            target_activations: Captured activations on triggered inputs.

        Returns:
            Dict mapping layer index to anomaly score (L2 distance normalized
            by clean activation norm).
        """
        scores: dict[int, float] = {}
        for layer_idx in clean_activations:
            if layer_idx not in target_activations:
                continue
            clean = torch.stack(clean_activations[layer_idx]).mean(0)
            target = torch.stack(target_activations[layer_idx]).mean(0)
            diff = (target - clean).norm().item()
            norm = clean.norm().item() + 1e-8
            scores[layer_idx] = diff / norm
        return scores

    def is_backdoored(self, anomaly_scores: dict[int, float]) -> bool:
        """Classify model as backdoored based on anomaly scores.

        Args:
            anomaly_scores: Output of compute_anomaly_score().

        Returns:
            True if anomaly scores suggest backdoor presence.
        """
        if not anomaly_scores:
            return False
        mean_score = np.mean(list(anomaly_scores.values()))
        # Heuristic: if mean anomaly > threshold, flag as backdoored
        return bool(mean_score > self.threshold_std)

    def get_captured_activations(self) -> dict[int, list[torch.Tensor]]:
        """Return and clear the captured activation buffer.

        Returns:
            Dict mapping layer index to list of activation tensors.
        """
        captured = dict(self._captured)
        self._captured.clear()
        return captured

    def _get_layers(self) -> list[Any]:
        """Extract transformer decoder layers from the model.

        Returns:
            List of transformer layer modules.
        """
        # Try common attribute paths for HuggingFace models
        for attr in ["model.layers", "transformer.h", "encoder.layers", "decoder.layers"]:
            obj = self.model
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return list(obj)
        logger.warning(
            "Could not locate transformer layers automatically. "
            "Override _get_layers() for your model architecture."
        )
        return []

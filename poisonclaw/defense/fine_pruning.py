"""Fine-pruning defense for PoisonClaw.

Prunes neurons/weights that activate strongly on triggered inputs
but weakly on clean inputs. Follows the Fine-Pruning defense from
Liu et al. (2018), adapted for LoRA-fine-tuned VLMs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FinePruning:
    """Fine-pruning defense: prune backdoor-associated neurons.

    Procedure:
    1. Record mean activation of each neuron on clean inputs.
    2. Identify neurons with near-zero clean activation (dormant neurons).
    3. Prune (zero out) those neurons.
    4. Optionally fine-tune remaining weights on clean data.

    Args:
        model: The poisoned model.
        prune_ratio: Fraction of neurons to prune (0.0-1.0).
        target_modules: Module name patterns to prune (regex or substring).
    """

    def __init__(
        self,
        model: nn.Module,
        prune_ratio: float = 0.1,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        self.model = model
        self.prune_ratio = prune_ratio
        self.target_modules = target_modules or ["fc", "linear", "dense", "mlp"]
        self._activation_stats: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._pruned_masks: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Activation recording
    # ------------------------------------------------------------------

    def register_activation_hooks(self) -> None:
        """Attach hooks to record activation statistics on clean inputs."""
        for name, module in self.model.named_modules():
            if any(pat in name.lower() for pat in self.target_modules):
                if isinstance(module, nn.Linear):
                    def make_hook(mod_name: str):
                        def hook(module, input, output):
                            acts = output.detach().float().abs()
                            if mod_name not in self._activation_stats:
                                self._activation_stats[mod_name] = acts.mean(0)
                            else:
                                # Running mean update
                                self._activation_stats[mod_name] = (
                                    self._activation_stats[mod_name] * 0.9 + acts.mean(0) * 0.1
                                )
                        return hook
                    handle = module.register_forward_hook(make_hook(name))
                    self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered activation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self) -> dict[str, int]:
        """Execute pruning based on recorded activation statistics.

        Returns:
            Dict mapping module name to number of neurons pruned.
        """
        pruned_counts: dict[str, int] = {}

        for name, module in self.model.named_modules():
            if name not in self._activation_stats:
                continue
            if not isinstance(module, nn.Linear):
                continue

            acts = self._activation_stats[name]
            n_neurons = acts.shape[-1]
            n_to_prune = max(1, int(n_neurons * self.prune_ratio))

            # Prune neurons with lowest clean activation (dormant neurons)
            _, prune_indices = torch.topk(acts, k=n_to_prune, largest=False)
            mask = torch.ones(n_neurons, dtype=torch.bool, device=module.weight.device)
            mask[prune_indices] = False

            with torch.no_grad():
                module.weight.data[:, ~mask] = 0.0
                if module.bias is not None:
                    module.bias.data[~mask] = 0.0

            self._pruned_masks[name] = mask
            pruned_counts[name] = n_to_prune
            logger.info("Pruned %d/%d neurons in '%s'.", n_to_prune, n_neurons, name)

        return pruned_counts

    def restore(self) -> None:
        """Restore pruned weights from saved masks (not full restoration)."""
        logger.warning(
            "FinePruning.restore() only clears the mask registry; "
            "original weights are not restored. Load from checkpoint instead."
        )
        self._pruned_masks.clear()

    def get_pruning_summary(self) -> dict[str, Any]:
        """Return a summary of what was pruned.

        Returns:
            Dict with pruning statistics.
        """
        total_pruned = sum(
            int((~m).sum().item()) for m in self._pruned_masks.values()
        )
        total_neurons = sum(
            m.shape[0] for m in self._pruned_masks.values()
        )
        return {
            "prune_ratio_target": self.prune_ratio,
            "modules_pruned": len(self._pruned_masks),
            "total_neurons_pruned": total_pruned,
            "total_neurons": total_neurons,
            "empirical_prune_ratio": total_pruned / max(total_neurons, 1),
        }

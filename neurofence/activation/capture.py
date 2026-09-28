"""Forward-hook-based activation capture.

Captures compact per-layer summary statistics (reusing
neurofence.weight_forensics.statistics -- the same numerical-safety
guarantees apply: extreme/NaN/Inf activation values report None with a
note rather than corrupting downstream scores) rather than raw activation
tensors. A large model run over many prompts would otherwise consume
unbounded memory if every activation tensor were kept; a handful of
summary numbers per layer per prompt does not.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from neurofence.weight_forensics.statistics import TensorStatistics, compute_tensor_statistics

if TYPE_CHECKING:
    from torch.nn import Module


def select_layers(
    model: Module,
    mode: str = "auto",
    patterns: list[str] | None = None,
) -> dict[str, Module]:
    """Select named submodules to hook.

    mode="auto": every `torch.nn.Linear`-like submodule -- a reasonable
    default that covers attention projections and MLP layers in most
    transformer architectures without requiring architecture-specific
    knowledge. Includes `transformers.pytorch_utils.Conv1D` alongside
    `nn.Linear`: GPT-2-family models use Conv1D (a linear layer with a
    transposed weight, despite the name) for attention/MLP projections, so
    restricting to nn.Linear alone would silently capture almost nothing
    for that architecture family beyond the LM head.
    mode="pattern": only submodules whose fully-qualified name matches any
    regex in `patterns`.
    """
    import torch.nn as nn

    if mode == "auto":
        conv1d_cls: type | None
        try:
            from transformers.pytorch_utils import Conv1D as conv1d_cls  # noqa: N813
        except ImportError:
            conv1d_cls = None

        selected: dict[str, Module] = {}
        for name, module in model.named_modules():
            is_conv1d = conv1d_cls is not None and isinstance(module, conv1d_cls)
            if isinstance(module, nn.Linear) or is_conv1d:
                selected[name] = module
        return selected

    if mode == "pattern":
        if not patterns:
            return {}
        compiled = [re.compile(p) for p in patterns]
        return {
            name: module
            for name, module in model.named_modules()
            if any(c.search(name) for c in compiled)
        }

    raise ValueError(f"Unknown layer selection mode: {mode!r} (expected 'auto' or 'pattern')")


class ActivationCapture:
    """Context manager that hooks selected layers and records a
    TensorStatistics summary of each layer's output on every forward pass
    while active.

    Usage:
        with ActivationCapture(model, layers) as capture:
            model(**inputs)
        capture.summaries  # {layer_name: [TensorStatistics, ...]}
    """

    def __init__(self, model: Module, layers: dict[str, Module]) -> None:
        self._model = model
        self._layers = layers
        self._handles: list = []
        self.summaries: dict[str, list[TensorStatistics]] = {name: [] for name in layers}

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            import torch

            tensor = output[0] if isinstance(output, tuple) else output
            if not hasattr(tensor, "detach"):
                return  # non-tensor output (rare); nothing to summarize
            with torch.no_grad():
                array = tensor.detach().to(dtype=torch.float32).cpu().numpy()
            self.summaries[name].append(compute_tensor_statistics(array))

        return hook

    def __enter__(self) -> ActivationCapture:
        for name, module in self._layers.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

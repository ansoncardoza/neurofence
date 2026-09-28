"""Runs a single forward pass per prompt and captures per-layer activation
summaries. Uses a forward pass over the prompt tokens directly (not
autoregressive generation) -- activation forensics is about how the model
*represents* the input, which a single forward pass captures, at a
fraction of the cost of generating new tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neurofence.activation.capture import ActivationCapture, select_layers
from neurofence.weight_forensics.statistics import TensorStatistics

if TYPE_CHECKING:
    from torch.nn import Module


def capture_prompt_activations(
    model: Module,
    tokenizer,
    prompt: str,
    layers: dict[str, Module],
    device: str = "cpu",
) -> dict[str, TensorStatistics]:
    """Run one forward pass over `prompt` and return the first captured
    summary per hooked layer.

    If a layer's module is invoked more than once during the forward pass
    (e.g. weight tying or a shared submodule), only the first call's
    summary is kept -- documented here, not silently averaged or dropped.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # A prompt longer than the model's context window would otherwise
    # crash inside position-embedding lookup (IndexError) instead of being
    # analyzed -- same failure mode as, and same fix as,
    # HuggingFaceCausalLMRunner.generate. Truncate to the most recent
    # tokens that fit.
    max_ctx = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_ctx is not None and inputs["input_ids"].shape[1] > max_ctx:
        inputs = {k: v[:, -max_ctx:] for k, v in inputs.items()}

    with ActivationCapture(model, layers) as capture, torch.inference_mode():
        model(**inputs)

    return {name: summaries[0] for name, summaries in capture.summaries.items() if summaries}


def capture_activations_for_prompts(
    model: Module,
    tokenizer,
    prompts: list[tuple[str, str]],  # (case_id, prompt_text)
    layer_mode: str = "auto",
    patterns: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, dict[str, TensorStatistics]]:
    """Returns {case_id: {layer_name: TensorStatistics}} for every prompt."""
    layers = select_layers(model, mode=layer_mode, patterns=patterns)
    results: dict[str, dict[str, TensorStatistics]] = {}
    for case_id, prompt in prompts:
        results[case_id] = capture_prompt_activations(model, tokenizer, prompt, layers, device)
    return results

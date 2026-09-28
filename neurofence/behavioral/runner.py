"""Model execution for behavioral testing.

`ModelRunner` is a narrow protocol (prompt in, text out) so the rest of the
behavioral pipeline never depends on transformers/torch directly -- tests
can supply a trivial fake runner, and real usage plugs in
`HuggingFaceCausalLMRunner`.

Loading enforces the same policy as neurofence.acquisition:
`trust_remote_code=False` always, regardless of caller input -- NeuroFence
must never execute model-provided code.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from neurofence.exceptions import AcquisitionError


class ModelOutput(BaseModel):
    prompt: str
    text: str
    latency_ms: float
    model_id: str
    metadata: dict[str, str | int | float | bool] = {}


class ModelRunner(Protocol):
    def generate(self, prompt: str) -> ModelOutput: ...


class HuggingFaceCausalLMRunner:
    """Wraps an already-loaded transformers causal LM + tokenizer.

    Generation is greedy (`do_sample=False`) by default so behavioral
    comparisons are reproducible run-to-run; sampling can be enabled
    explicitly with a fixed seed when exploring output diversity is the
    point (e.g. later fuzzing work), but that is an explicit opt-in.
    """

    def __init__(
        self,
        model,  # transformers PreTrainedModel
        tokenizer,  # transformers PreTrainedTokenizerBase
        model_id: str = "unknown",
        max_new_tokens: int = 64,
        do_sample: bool = False,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._device = device

    def generate(self, prompt: str) -> ModelOutput:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        # A prompt longer than the model's context window would otherwise
        # crash inside position-embedding lookup (IndexError) rather than
        # producing a usable result. Truncate to the most recent tokens that
        # fit, leaving room for generation, and record that this happened.
        prompt_truncated = False
        max_ctx = getattr(self._model.config, "n_positions", None) or getattr(
            self._model.config, "max_position_embeddings", None
        )
        if max_ctx is not None:
            available = max(1, max_ctx - self._max_new_tokens)
            if inputs["input_ids"].shape[1] > available:
                inputs = type(inputs)({k: v[:, -available:] for k, v in inputs.items()})
                prompt_truncated = True

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return ModelOutput(
            prompt=prompt,
            text=text,
            latency_ms=latency_ms,
            model_id=self._model_id,
            metadata={
                "tokens_generated": int(generated_ids.shape[0]),
                "do_sample": self._do_sample,
                "prompt_truncated": prompt_truncated,
            },
        )


def load_causal_lm(model_dir: str | Path, device: str = "cpu"):
    """Load a causal LM + tokenizer for behavioral testing.

    trust_remote_code is hard-coded False: NeuroFence will not execute
    model-provided code, matching the policy enforced in
    neurofence.config.schema.ModelConfig.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(model_dir)
    if not root.exists() or not root.is_dir():
        raise AcquisitionError(f"Model directory does not exist: {root}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(root), trust_remote_code=False)
    except Exception as e:
        raise AcquisitionError(f"Failed to load model from {root}: {e}") from e

    # mypy misresolves AutoModelForCausalLM.from_pretrained's return type
    # against PreTrainedModel.to's overloads in the installed transformers
    # stubs; this is a stub artifact, not a real type error (verified this
    # loads and runs correctly at runtime).
    model.to(device)  # type: ignore[arg-type]
    model.eval()
    return model, tokenizer

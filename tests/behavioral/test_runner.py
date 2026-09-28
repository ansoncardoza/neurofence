from __future__ import annotations

from pathlib import Path

import pytest

from neurofence.behavioral.runner import HuggingFaceCausalLMRunner, load_causal_lm
from neurofence.exceptions import AcquisitionError
from tests.behavioral.model_fixtures import build_tiny_model


def test_load_causal_lm_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError):
        load_causal_lm(tmp_path / "does_not_exist")


def test_load_causal_lm_invalid_directory_raises_not_hangs(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(AcquisitionError):
        load_causal_lm(empty_dir)


def test_runner_generate_returns_model_output() -> None:
    model, tokenizer = build_tiny_model(seed=1)
    runner = HuggingFaceCausalLMRunner(model, tokenizer, model_id="tiny-test", max_new_tokens=8)

    output = runner.generate("Hello there")

    assert output.prompt == "Hello there"
    assert isinstance(output.text, str)
    assert output.model_id == "tiny-test"
    assert output.latency_ms >= 0
    assert output.metadata["tokens_generated"] <= 8


def test_runner_deterministic_with_greedy_decoding() -> None:
    model, tokenizer = build_tiny_model(seed=2)
    runner = HuggingFaceCausalLMRunner(model, tokenizer, max_new_tokens=8, do_sample=False)

    out1 = runner.generate("test prompt")
    out2 = runner.generate("test prompt")

    assert out1.text == out2.text


def test_runner_handles_empty_prompt() -> None:
    model, tokenizer = build_tiny_model(seed=3)
    runner = HuggingFaceCausalLMRunner(model, tokenizer, max_new_tokens=4)

    output = runner.generate("")
    assert isinstance(output.text, str)


def test_runner_handles_long_prompt() -> None:
    # Regression: a prompt longer than the model's context window (128
    # positions for this tiny fixture) used to crash with IndexError inside
    # position-embedding lookup instead of being truncated.
    model, tokenizer = build_tiny_model(seed=4)
    runner = HuggingFaceCausalLMRunner(model, tokenizer, max_new_tokens=4)

    output = runner.generate("a" * 500)

    assert isinstance(output.text, str)
    assert output.metadata["prompt_truncated"] is True


def test_runner_short_prompt_not_truncated() -> None:
    model, tokenizer = build_tiny_model(seed=4)
    runner = HuggingFaceCausalLMRunner(model, tokenizer, max_new_tokens=4)

    output = runner.generate("short")

    assert output.metadata["prompt_truncated"] is False


def test_runner_handles_unicode_prompt() -> None:
    model, tokenizer = build_tiny_model(seed=5)
    runner = HuggingFaceCausalLMRunner(model, tokenizer, max_new_tokens=4)

    output = runner.generate("héllo wörld 日本語")
    assert isinstance(output.text, str)

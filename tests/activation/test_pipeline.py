from __future__ import annotations

from neurofence.activation.capture import select_layers
from neurofence.activation.pipeline import (
    capture_activations_for_prompts,
    capture_prompt_activations,
)
from tests.behavioral.model_fixtures import build_tiny_model


def test_capture_prompt_activations_returns_stats_per_layer() -> None:
    model, tok = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")

    result = capture_prompt_activations(model, tok, "hello world", layers)

    assert set(result) == set(layers)
    assert all(stats.status == "ok" for stats in result.values())


def test_capture_activations_for_prompts_returns_per_case_dict() -> None:
    model, tok = build_tiny_model(seed=0)
    prompts = [("p1", "hello"), ("p2", "world, how are you?")]

    result = capture_activations_for_prompts(model, tok, prompts)

    assert set(result) == {"p1", "p2"}
    assert len(result["p1"]) > 0
    assert result["p1"].keys() == result["p2"].keys()


def test_capture_activations_pattern_mode() -> None:
    model, tok = build_tiny_model(seed=0)
    prompts = [("p1", "hello")]

    result = capture_activations_for_prompts(
        model, tok, prompts, layer_mode="pattern", patterns=[r"lm_head"]
    )

    assert set(result["p1"]) == {"lm_head"}


def test_capture_activations_no_matching_layers_empty_dict() -> None:
    model, tok = build_tiny_model(seed=0)
    prompts = [("p1", "hello")]

    result = capture_activations_for_prompts(
        model, tok, prompts, layer_mode="pattern", patterns=["no_such_layer"]
    )

    assert result["p1"] == {}


def test_capture_activations_for_prompts_empty_prompt_list() -> None:
    model, tok = build_tiny_model(seed=0)
    result = capture_activations_for_prompts(model, tok, [])
    assert result == {}


def test_capture_prompt_activations_truncates_long_prompt() -> None:
    # Regression: a prompt longer than the model's context window (128
    # positions for this tiny fixture) used to crash with IndexError
    # inside position-embedding lookup instead of being truncated, exactly
    # like the behavioral runner bug fixed in Milestone 3.
    model, tok = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")

    result = capture_prompt_activations(model, tok, "a" * 500, layers)

    assert all(stats.status == "ok" for stats in result.values())

from __future__ import annotations

import torch

from neurofence.activation.capture import ActivationCapture, select_layers
from tests.behavioral.model_fixtures import build_tiny_model


def test_select_layers_auto_finds_conv1d_and_linear() -> None:
    model, _ = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")
    assert "lm_head" in layers
    assert any("attn.c_attn" in name for name in layers)
    assert any("mlp.c_fc" in name for name in layers)


def test_select_layers_pattern_mode_filters() -> None:
    model, _ = build_tiny_model(seed=0)
    layers = select_layers(model, mode="pattern", patterns=[r"attn\.c_attn"])
    assert layers
    assert all("attn.c_attn" in name for name in layers)


def test_select_layers_pattern_mode_no_patterns_empty() -> None:
    model, _ = build_tiny_model(seed=0)
    assert select_layers(model, mode="pattern", patterns=None) == {}
    assert select_layers(model, mode="pattern", patterns=[]) == {}


def test_select_layers_unknown_mode_raises() -> None:
    model, _ = build_tiny_model(seed=0)
    try:
        select_layers(model, mode="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_capture_records_one_summary_per_layer_per_forward_pass() -> None:
    model, tok = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")
    inputs = tok("hello world", return_tensors="pt")

    with ActivationCapture(model, layers) as capture, torch.inference_mode():
        model(**inputs)

    assert set(capture.summaries) == set(layers)
    for name, summaries in capture.summaries.items():
        assert len(summaries) == 1, name
        assert summaries[0].status == "ok"


def test_capture_accumulates_across_multiple_forward_passes() -> None:
    model, tok = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")

    with ActivationCapture(model, layers) as capture, torch.inference_mode():
        model(**tok("first prompt", return_tensors="pt"))
        model(**tok("second prompt", return_tensors="pt"))

    for summaries in capture.summaries.values():
        assert len(summaries) == 2


def test_hooks_removed_after_context_exit() -> None:
    model, tok = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")

    with ActivationCapture(model, layers) as capture, torch.inference_mode():
        model(**tok("inside context", return_tensors="pt"))

    with torch.inference_mode():
        model(**tok("outside context", return_tensors="pt"))  # should not affect capture.summaries

    for summaries in capture.summaries.values():
        assert len(summaries) == 1  # not 2 -- hooks were removed on exit


def test_empty_layer_selection_no_crash() -> None:
    model, tok = build_tiny_model(seed=0)
    inputs = tok("hello", return_tensors="pt")

    with ActivationCapture(model, {}) as capture, torch.inference_mode():
        model(**inputs)

    assert capture.summaries == {}


def test_capture_empty_prompt_no_crash() -> None:
    model, tok = build_tiny_model(seed=0)
    layers = select_layers(model, mode="auto")
    inputs = tok("", return_tensors="pt")

    with ActivationCapture(model, layers) as capture, torch.inference_mode():
        model(**inputs)

    assert all(summaries[0].status == "ok" for summaries in capture.summaries.values())

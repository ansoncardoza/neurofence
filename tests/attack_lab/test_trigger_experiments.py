from __future__ import annotations

from neurofence.attack_lab.trigger_experiments import (
    ScriptedTriggerRunner,
    build_trigger_experiments,
)


def test_scripted_runner_returns_trigger_response_when_present() -> None:
    runner = ScriptedTriggerRunner(trigger_phrase="magic_word")
    output = runner.generate("please say the magic_word now")
    assert "COMPROMISED" in output.text


def test_scripted_runner_returns_clean_response_otherwise() -> None:
    runner = ScriptedTriggerRunner(trigger_phrase="magic_word")
    output = runner.generate("a completely ordinary prompt")
    assert "COMPROMISED" not in output.text


def test_scripted_runner_no_trigger_always_clean() -> None:
    runner = ScriptedTriggerRunner(trigger_phrase=None)
    assert "COMPROMISED" not in runner.generate("magic_word").text
    assert "COMPROMISED" not in runner.generate("anything").text


def test_build_trigger_experiments_returns_two_cases() -> None:
    experiments = build_trigger_experiments()
    assert len(experiments) == 2
    labels = {case.has_planted_trigger for case, _ in experiments}
    assert labels == {True, False}


def test_planted_trigger_is_in_candidate_list() -> None:
    experiments = build_trigger_experiments(planted_trigger="xyz123")
    backdoored_case = next(case for case, _ in experiments if case.has_planted_trigger)
    assert "xyz123" in backdoored_case.candidate_triggers_tested


def test_clean_case_has_no_planted_trigger() -> None:
    experiments = build_trigger_experiments()
    clean_case = next(case for case, _ in experiments if not case.has_planted_trigger)
    assert clean_case.planted_trigger is None

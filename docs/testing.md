# Testing Strategy

363 tests, all passing, `ruff check` and `mypy` clean, as of Milestone 8.
Run them yourself:

```bash
pytest -q
ruff check neurofence tests
mypy neurofence
```

## What's covered, by kind

- **Unit tests**: individual functions across every module --
  `tests/<package>/test_<module>.py` mirrors `neurofence/<package>/`
  1:1. The bulk of the suite.
- **Integration tests**: real forward passes through a real (tiny,
  randomly-initialized, offline) GPT-2 model
  (`tests/behavioral/model_fixtures.py`), exercising the actual
  generate -> decode -> compare and forward-hook -> capture -> analyze
  pipelines with real `torch`/`transformers` objects, not mocks.
- **End-to-end CLI tests** (`tests/test_cli.py`): invoke the real Typer
  `app` via `typer.testing.CliRunner`, catching wiring bugs (wrong option
  names, exceptions escaping error handling, JSON corruption from
  incorrect stdout printing) that module-level unit tests cannot.
- **Regression tests**: every bug found during development QA has a
  named test citing what broke and why -- e.g.
  `test_extremely_large_values` (statistics overflow),
  `test_two_well_separated_groups_cluster_separately` (DBSCAN eps
  heuristic), `test_render_pdf_unbalanced_markup_in_architecture_does_not_crash`
  (PDF markup injection). Grep the test suite for "Regression:" to find
  them all.
- **Property-style edge case sweeps**: hashing determinism, tensor
  statistics on empty/all-NaN/constant/tiny/huge/float16 tensors,
  malformed config.json, path traversal variants, degenerate probability
  distributions for KL/JS divergence.

## Manual end-to-end verification

Automated tests are necessary but not sufficient -- several milestones
were additionally verified by hand against a real, disk-saved,
`AutoTokenizer`/`AutoModelForCausalLM`-loadable model (built offline with
a custom character-level tokenizer, no network access, via
`HF_HUB_OFFLINE=1`), run through the actual CLI rather than library calls
directly. This is how the context-window-truncation bug (Milestone 3),
the tied-embedding weight-poisoning detection (Milestone 6), and the
end-to-end JSON+PDF report generation (Milestone 8) were confirmed
working -- not merely "tests pass," but "the CLI, run the way a user
would run it, produces the correct, sensible output."

## Numerical safety

Every statistical function is tested against: NaN, Inf, zero variance
(constant tensors), empty tensors, tiny tensors (1-2 elements), extremely
large and extremely small magnitudes, and multiple dtypes (float16/32/64).
Several tests run with `pytest -W error::RuntimeWarning` specifically to
catch silent overflow/underflow that would otherwise only show up as an
unexplained warning in the logs.

## What is NOT yet covered

- No GPU-path testing (CPU-only PyTorch in this environment).
- No fuzzing harness against the PDF renderer beyond the specific
  regression cases identified during development (unbalanced markup,
  script tags) -- see `docs/limitations.md`.
- No test against a real, trained (non-scripted) backdoored language
  model -- see `docs/limitations.md` for why that's out of scope here.

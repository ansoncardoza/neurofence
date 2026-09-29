# Team Roles

NeuroFence's implementation divides cleanly into three roles, each owning
a self-contained slice of the pipeline with its own tech-stack rationale,
design decisions, and QA story. Each document below is written to be
presented on its own — architecture, why that tech stack, every
significant implementation decision, real bugs found and fixed, and live
demo commands.

| Role | Owns | Code | Tests |
|---|---|---|---|
| [Role 1 — Core Infrastructure, Secure Acquisition & Weight Forensics](ROLE-1-core-infrastructure-and-weight-forensics.md) | `config/`, `acquisition/`, `weight_forensics/` | 1,611 lines | 108 |
| [Role 2 — Behavioral Intelligence, Adversarial Fuzzing & Activation Forensics](ROLE-2-behavioral-and-adversarial-testing.md) | `behavioral/`, `fuzzing/`, `activation/` | 1,629 lines | 149 |
| [Role 3 — Risk Intelligence, Evaluation, Reporting & CLI](ROLE-3-risk-intelligence-evaluation-and-reporting.md) | `fusion/`, `attack_lab/`, `evaluation/`, `reporting/`, `cli.py` | 2,059 lines | 106 |

**Total: 5,299 lines, 363 tests**, all passing, `ruff`/`mypy` clean.

## Why the project splits this way

The three roles follow the natural dependency order of the pipeline:

1. **Role 1** answers "is this model safe to open, and do its weights
   look statistically normal" — no model execution required, so this
   role's packages have zero PyTorch dependency and could be developed
   and demoed independently of the other two.
2. **Role 2** answers "does this model *behave* wrong" — the only role
   that actually runs the model (generation, forward hooks), and
   therefore the only one requiring the heavy `ml` extra
   (PyTorch/transformers).
3. **Role 3** answers "given everyone else's evidence, what's the
   verdict, can we prove the detectors work, and can a human read the
   result" — the integration layer that depends on both other roles'
   output and ties everything into the actual CLI a user runs.

Each role's README documents genuine, real engineering decisions and real
bugs found during this project's own development — every "found a bug,
here's the root cause, here's the fix, here's the regression test" story
in these documents actually happened during the build and is backed by a
named test in the codebase you can go read.

See also the shared, cross-role documentation:

- [../architecture.md](../architecture.md) — full pipeline diagram
- [../algorithms.md](../algorithms.md) — every algorithm across all three roles
- [../detection-methodology.md](../detection-methodology.md) — how Role 3's fusion formulas work
- [../threat-model.md](../threat-model.md) / [../limitations.md](../limitations.md)
- [../testing.md](../testing.md) / [../evaluation.md](../evaluation.md)

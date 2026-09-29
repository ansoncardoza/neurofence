# Role 3 — Risk Intelligence, Evaluation, Reporting & CLI Engineer

**Owned modules:** `neurofence/fusion/`, `neurofence/attack_lab/`, `neurofence/evaluation/`, `neurofence/reporting/`, `neurofence/cli.py`
**Code:** 2,059 lines · **Tests:** 106 (fusion 25, attack_lab 23, evaluation 26, reporting 20, cli 12)
**Milestones:** 6 (evidence fusion), 7 (synthetic attack lab & evaluation), 8 (reporting)

Role 1 produces weight-level evidence. Role 2 produces behavioral and
activation-level evidence. Neither of them, alone, is allowed to make a
final call — that is this role's entire reason to exist. This role built
the layer that combines every other role's output into one coherent,
explainable verdict; the machinery to *prove that verdict is actually
worth trusting* (not just asserted); and the pipeline that turns all of
it into a document a human can read and act on. This role also owns the
CLI itself — the thing that wires every other role's work together into
one command a user actually runs.

---

## 1. What this role is responsible for, in plain terms

1. **Evidence fusion** (`fusion/`): take whatever detectors actually ran
   — Role 1's weight anomaly detector, Role 2's behavioral/trigger/
   activation detectors — and combine them into two genuinely different
   numbers: an **Anomaly Score** (how unusual does this model look) and a
   **Threat Confidence** (how strong is the *combined* evidence for
   deliberate tampering). Then classify that into a risk band.
2. **Synthetic attack lab & evaluation** (`attack_lab/`, `evaluation/`):
   you cannot claim a detector works without measuring it against known
   ground truth. This role built a deterministic clean/poisoned synthetic
   model generator and a scripted backdoor simulator, plus the
   classification-metrics machinery (precision/recall/F1/ROC-AUC/PR-AUC)
   to honestly measure — not assert — how well Role 1's and Role 2's
   detectors actually perform.
3. **Reporting** (`reporting/`): assemble everything into one canonical
   `ScanReport` object, and render it as both a machine-readable JSON
   document and a human-readable 17-section PDF.
4. **CLI** (`cli.py`): the `neurofence scan` / `report` / `benchmark`
   commands that a real user actually types — the integration point where
   all three roles' work gets wired together into one working tool.

---

## 2. Tech stack, and why each piece was chosen

| Tool | Why this and not something else |
|---|---|
| **scikit-learn `metrics`** (`roc_auc_score`, `average_precision_score`) | Standard, correct, well-tested implementations of ROC-AUC/PR-AUC rather than hand-rolling them — but the *wrapper* around them (`compute_roc_auc`/`compute_pr_auc`) is custom, specifically to add the "undefined when only one class present" guard scikit-learn itself doesn't provide cleanly. |
| **ReportLab** | The PDF engine. Chosen because it's pure-Python, well-established, and gives fine-grained control over a 17-section structured report (tables, styled paragraphs, colored risk badges) without needing a browser-based rendering pipeline (no headless-Chrome/wkhtmltopdf dependency). |
| **Typer** | The CLI framework. Chosen over raw `argparse` for typed, self-documenting options (`--reference`, `--trigger` as a repeatable list, `--no-weights` as an auto-generated boolean flag pair) with essentially no boilerplate, and because it composes cleanly with `typer.testing.CliRunner` for real end-to-end CLI tests (see below). |
| **Rich** | Console output formatting for the human-facing parts of `scan`/`benchmark` (colored status lines, tables). Explicitly **not** used for machine-readable JSON output after a real bug was found — see below. |
| **pypdf** (test-only) | Not a runtime dependency — added specifically so the PDF test suite could open the *actual generated PDF* and assert real section headers and finding IDs appear in the extracted text, instead of merely asserting "no exception was raised." A file existing and being non-empty is a much weaker test than "the text I expect is actually in it." |
| **Pydantic v2** (shared with Role 1) | `ScanReport`, `SubScore`, `FusionResult`, `Finding` — all Pydantic models, for the same reasons Role 1 chose Pydantic: free, correct JSON serialization is exactly what a "generate the JSON report first" requirement needs, and `ScanReport.model_validate_json()` gives the standalone `report` CLI command a one-line, type-safe way to reload a previously saved report. |

### Why the JSON report and PDF report are never allowed to disagree

Both renderers (`json_report.py`, `pdf_report.py`) read from the exact
same `ScanReport` object, built once by `build_report()`. This was a
deliberate architectural decision: if the two formats had separate
data-assembly logic, they could silently drift apart (a finding present
in the JSON but missing from the PDF, say). Instead, `render_json_report`
is `report.model_dump_json()`, and `render_pdf_report` walks the same
typed object — there's structurally no way for the JSON and PDF report of
the same scan to disagree about *content*, only about *presentation*.

---

## 3. `fusion/` — the two-number decision

The single most important design decision in this project, and this role
built it. **Anomaly Score** and **Threat Confidence** answer genuinely
different questions and are computed by genuinely different formulas —
not two names for the same number, which would be dishonest given the
project's stated principle of never deciding from one detector:

```
anomaly_score = Σ(weight[d] · score[d]) / Σ(weight[d])     over evaluated detectors d
```
A weighted average, **renormalized over only detectors that actually
ran** — a detector that wasn't enabled for this scan does not silently
count as "clean" in the denominator. One very loud detector can push this
number high on its own.

```
elevated       = {d : score[d] ≥ 40}
agreement      = |elevated| / |evaluated|
threat_conf    = mean(score[d] for d in elevated) × (0.4 + 0.6·agreement)   if elevated else 0
```
This is the numeric expression of "never decide from one detector": an
isolated elevated signal (nothing else corroborates it) gets discounted
to well below its own severity; several independent detectors elevated
together score close to their shared severity. Verified directly in
`tests/fusion/test_combine.py` — five detectors all at severity 85
produces `anomaly_score == threat_confidence == 85` exactly (full
agreement), while one detector at 90 with everything else quiet produces
an `anomaly_score` still meaningfully raised but a `threat_confidence`
discounted well below 54.

`Finding`/`generate_findings()`/`generate_recommendations()`
(`reporting/report.py`, built alongside fusion since they consume its
output directly) turn the raw scores into human-readable, cited evidence
— "Layer X flagged by 2/3 consensus methods," "Tensor Y differs by 12%
from the reference model" — sorted most-severe-first, each with a
targeted, specific recommendation rather than a generic "investigate
further."

## 4. `attack_lab/` + `evaluation/` — proving the detectors actually work

This is the part of the project that answers "how do you know this
works?" honestly, instead of just asserting it.

- **`synthetic_models.py`** — five documented, deterministic perturbation
  kinds applied to synthetic weight tensors: `localized` (single-element
  shift), `distributed` (noise across a fraction of elements),
  `layer_level` (whole-layer rescale), `neuron_level` (single-row shift),
  and — deliberately — `low_magnitude` (noise sized to blend into normal
  weight scale, the "can the detector even see this" stress test).
- **`trigger_experiments.py`** — a `ScriptedTriggerRunner`: a fully
  deterministic, rule-based fake model that returns a "compromised"
  response exactly when a planted trigger phrase is present, and a
  matching clean runner with no trigger at all. This gives ground truth
  for backdoor-detection without the cost of training a real backdoored
  LLM.
- **`metrics.py`** — precision/recall/F1/specificity/FPR/FNR/ROC-AUC/
  PR-AUC, every one of which reports `None` with an explanatory note when
  mathematically undefined (e.g. no positive predictions made at all)
  rather than a misleading `0.0` — the same "report evidence honestly,
  never fabricate a number" discipline Role 1 applied to tensor
  statistics.
- **`benchmark.py`** — actually runs Role 1's weight-anomaly detector and
  Role 2's trigger detector against the synthetic ground truth, timing
  every run for real (not invented) runtime numbers, and reports a
  `layer_localization_rate` — did the detector flag the *specific*
  tensor that was actually tampered with, not just "something" — because
  the naive binary accuracy metric turned out to be actively misleading
  (see the bug below).

## 5. `reporting/` — the 17-section report

`report.py` defines `ScanReport` and derives `Finding`/recommendations/
limitations from it; `json_report.py` is a one-line wrapper around
Pydantic's own serialization; `pdf_report.py` renders all 17 spec
sections (executive summary through technical appendix) with ReportLab,
including risk-colored badges and per-finding evidence tables.

## 6. `cli.py` — wiring it all together

Three commands: `scan` (runs whichever detectors the user enabled across
all three roles' code, fuses the evidence, writes JSON/PDF),
`report` (re-renders a PDF from a previously saved JSON with zero
detectors re-run — useful for iterating on report *presentation* without
re-scanning), and `benchmark` (runs the attack-lab evaluation and prints
honest, measured numbers).

---

## 7. Real bugs this role found and fixed (good presentation material)

**Bug 1 — the benchmark exposed a Role 1 bug, but also exposed its own
evaluation-methodology bug.** After fixing the underlying norm-scaling
issue (Role 1's territory, discovered via this role's benchmark), binary
accuracy was still a mediocre 0.567 with 100% recall — which looked
*wrong*, not right. Root cause: this role's own binary criterion ("was
*anything* flagged") is inherently too permissive to be a fair test —
`contamination=0.1` Isolation Forest/LOF flags roughly one layer out of
eight "by construction," almost regardless of whether anything is truly
anomalous, mechanically producing near-perfect recall alongside mediocre
precision. **Fix:** added `layer_localization_rate` — does the detector
flag the *specific* attacked layer, using ground truth from
`PerturbationRecord.layer_name` — which is the metric that actually
distinguishes a working detector from a coin flip. Result: 100% on the
four large/structural attack kinds, 0% on the deliberately subtle
`low_magnitude` case — a credible, honest, presentable number instead of
a misleading one.

**Bug 2 — PDF generation crashed on untrusted text.** ReportLab's
`Paragraph` text is a small XML-like markup language, not plain text.
Any model-controlled or user-controlled string reaching it unescaped —
an untrusted model's `architecture` field from its own `config.json`, a
user-typed `--trigger` phrase, a layer name — could contain unbalanced
`<`/`&` and crash the **entire report** with a hard `ValueError`,
aborting output for an otherwise-successful scan. This is a genuine
security-relevant input-validation gap: report generation must never be
crashable by the very untrusted input it's reporting on. **Fix:**
systematic `xml.sax.saxutils.escape()` at every interpolation point,
with a regression test that reproduces the exact original crash
(`Model<3 & unclosed <b>bold` as an architecture string).

**Bug 3 — Rich silently corrupted piped JSON output.** `neurofence scan`
without `--output` prints the JSON report to stdout for the user to pipe
into `jq` or another tool. Using `console.print()` (Rich) for that output
caused Rich's line-soft-wrapping to insert newlines mid-string, breaking
the JSON structure — invisible in a terminal (Rich re-wraps for display),
catastrophic for a script parsing the output. Found by this role's own
end-to-end `CliRunner` test asserting the captured stdout round-trips
through `json.loads()` — not by a human eyeballing terminal output, which
is exactly why that test exists. **Fix:** plain `print()` for
machine-readable output, `console.print()` reserved for genuinely
human-facing status messages.

---

## 8. Testing philosophy for this role

106 tests, with a specific emphasis this role introduced to the project:
**this is the only role that built end-to-end CLI tests**
(`tests/test_cli.py`, using `typer.testing.CliRunner` to invoke the real
`app`, not the underlying functions). Every other test suite in the
project tests a module in isolation; this suite is the one that catches
*wiring* bugs — wrong option names, exceptions escaping error handling,
output-format corruption — that no amount of per-module unit testing
would ever surface, because the bug isn't in any one function, it's in
how the pieces are connected.

The PDF test suite specifically opens generated PDFs with `pypdf` and
asserts real content (section headers, finding IDs, risk labels) is
actually present in the extracted text — not just that a file of nonzero
size exists.

---

## 9. How this role's output depends on the other two roles

This role's code is the **integration layer** — every one of its modules
is a direct consumer of the other two roles' data structures:

- `fusion.scores` imports and scores Role 1's `WeightForensicsResult` and
  Role 2's `BehavioralComparisonRecord` / `TriggerCandidateResult` /
  activation `AnomalyDetectionSummary` list.
- `evaluation.benchmark` directly calls Role 1's
  `detect_layer_anomalies()` and Role 2's `discover_trigger_candidates()`
  against synthetic ground truth.
- `reporting.report.generate_findings()` reads Role 1's
  `WeightForensicsResult`/`DifferentialResult` and Role 2's
  `TriggerCandidateResult` list directly.
- `cli.py` imports from all three roles' packages and is the only file
  in the project that does.

---

## 10. Commands to demo this role's work live

```bash
# Full pipeline: every role's detectors, fused into one verdict, two report formats
neurofence scan ./suspect_model --reference ./clean_model \
    --behavioral --trigger "zzz_backdoor" --activations \
    --output report.json --pdf report.pdf

# Re-render the PDF without re-running any detector
neurofence report ./report.json --pdf report2.pdf

# Prove the detectors actually work, with real numbers
neurofence benchmark
neurofence benchmark --n-clean 20 --n-poisoned-per-kind 5 --output bench.json
```

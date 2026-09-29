# Role 2 — Behavioral Intelligence, Adversarial Fuzzing & Activation Forensics Engineer

**Owned modules:** `neurofence/behavioral/`, `neurofence/fuzzing/`, `neurofence/activation/`
**Code:** 1,629 lines · **Tests:** 149 (behavioral 56, fuzzing 50, activation 43)
**Milestones:** 3 (behavioral baseline), 4 (adversarial fuzzer), 5 (activation forensics)

Where Role 1 asks "do the weights look statistically wrong," this role
asks the fundamentally different question: **"does the model *behave*
wrong?"** That means actually running the model — loading it with
transformers, generating text, comparing outputs, poking it with
adversarial inputs, and looking inside it at the activations while it
runs. This is the only role whose code touches PyTorch/transformers
directly and the only one that needs a GPU-capable (or patient CPU-only)
machine to fully exercise.

---

## 1. What this role is responsible for, in plain terms

1. **Behavioral baseline** (`behavioral/`): give the model a fixed set of
   prompts, record what it says, and be able to compare two runs of it
   (candidate vs. reference model, or baseline vs. mutated prompt) along
   several dimensions — did the meaning change, did it start/stop
   refusing, did the output length change unexpectedly.
2. **Adversarial fuzzing** (`fuzzing/`): systematically mutate prompts —
   random garbage, repetition, Unicode tricks, formatting chaos, and
   direct prompt injection — and specifically hunt for **trigger
   phrases**: input that causes the model to behave differently only when
   that specific phrase is present, the behavioral signature of a
   backdoor.
3. **Activation forensics** (`activation/`): open the model up with
   PyTorch forward hooks and look at what's happening *inside*, not just
   at the text that comes out — does a layer's internal representation
   look unusual, and do candidate-trigger prompts produce activations
   that cluster separately from normal prompts.

---

## 2. Tech stack, and why each piece was chosen

| Tool | Why this and not something else |
|---|---|
| **PyTorch (CPU build)** | The only inference engine that matters for this ecosystem. CPU build specifically because the dev/test environment has no GPU — `torch.inference_mode()` is used everywhere generation happens to keep it as fast as CPU inference can be, and `device` is a first-class, configurable parameter throughout so a GPU deployment is a one-line change. |
| **transformers** | `AutoModelForCausalLM`/`AutoTokenizer` for real model loading; `transformers.pytorch_utils.Conv1D` had to be specifically imported and handled (see below) once GPT-2-family models revealed `nn.Linear`-only layer selection was silently missing almost every interesting layer. |
| **scikit-learn** (`TfidfVectorizer`, `DBSCAN`, `PCA`) | TF-IDF cosine similarity as the *default* semantic-similarity backend — chosen deliberately over always requiring a sentence-embedding model, so behavioral comparison works fully offline with zero network dependency and is bit-for-bit reproducible. DBSCAN/PCA reused from scikit-learn for activation-space clustering, for the same "prefer the established, simple tool" principle Role 1 applied to Isolation Forest/LOF. |
| **sentence-transformers (optional)** | Wired in as an *opt-in* alternative backend (`SentenceTransformerSimilarityBackend`) for when real semantic embeddings and network access are available — never the default, because defaulting to it would silently make the whole pipeline network-dependent. |
| **`re` (stdlib)** | Refusal detection is a small, hand-curated regex classifier, not an ML model. Deliberate: fast, deterministic, fully explainable — at the honest cost of missing unusually-phrased or non-English refusals, which is disclosed, not hidden. |
| **`random.Random` (stdlib, explicit instances)** | Every fuzzing mutator takes an explicit `random.Random` instance, never the global `random` module. This is the difference between a fuzzer whose entire run is reproducible from one seed and one that silently gives different results every time — a real, deliberate engineering choice, not a style preference. |

### Why build a custom offline tokenizer for testing instead of downloading GPT-2's

Testing behavioral/activation code against a real model normally means
downloading a real tokenizer and weights — slow, and it makes the test
suite depend on network access, which is exactly the kind of flakiness a
serious test suite shouldn't have. This role instead built a tiny,
from-scratch character-level tokenizer plus a randomly-initialized,
seeded GPT-2-architecture model (`tests/behavioral/model_fixtures.py`) —
real `torch`/`transformers` objects, real forward passes, real
`generate()` calls, zero network access, fully deterministic given a
seed. This is why 149 tests in this role's territory run in seconds, not
minutes, and never fail because of a flaky download.

---

## 3. `behavioral/` — how comparison actually works

- **`runner.py`** — `ModelRunner` is a narrow protocol (prompt in, text
  out), so the rest of the pipeline never has a hard PyTorch dependency
  — tests can plug in a trivial fake runner. `HuggingFaceCausalLMRunner`
  wraps a real model: greedy decoding by default (reproducibility over
  diversity), `trust_remote_code` hard-disabled (matching Role 1's
  acquisition policy), and — critically — automatically truncates a
  prompt to the model's context window instead of crashing (see the bug
  writeup below).
- **`comparison.py`** — combines semantic similarity, exact-match,
  length-ratio, and refusal-change into one `BehavioralComparison`. It
  never itself decides "anomalous" — comparison produces *evidence*;
  interpreting it is a different module's job (a recurring architectural
  pattern across this whole project).
- **`distribution.py`** — KL and Jensen-Shannon divergence for
  logit-level comparison, with Laplace smoothing so KL is always a finite
  number (raw KL is undefined wherever one distribution has zero mass the
  other doesn't) and strict input validation that raises rather than
  silently returning a misleading number on malformed probability
  vectors.

## 4. `fuzzing/` — mutation and trigger discovery

- **`mutators.py`** — five independent, pure mutation functions:
  `random_text` (standalone garbage input), `repetition`, `unicode`
  (Cyrillic homoglyphs, zero-width characters, mixed-script insertion —
  real prompt-injection/obfuscation techniques), `formatting`
  (case/whitespace/punctuation chaos), `prompt_mutation`
  (insert/prepend/append/replace/reorder — including direct injection
  markers like "IGNORE PREVIOUS INSTRUCTIONS").
- **`generator.py`** — assembles a bounded, deterministic fuzz case set
  respecting a configurable `max_prompts` budget, with a guarantee that a
  *smaller* budget is always a stable *prefix* of a larger one at the same
  seed (so shrinking the budget for a quick smoke test doesn't change
  which cases you see, just how many).
- **`trigger_discovery.py`** — the actual backdoor-hunting logic: append
  each candidate phrase to multiple base prompts, compare baseline vs.
  mutated output via `behavioral.comparison`, combine semantic distance +
  refusal change + length change into one weighted per-prompt score, and
  only call a phrase `potential_trigger_candidate` — never "confirmed
  backdoor" — when a **strict majority** of tested prompts show elevated
  evidence. A single divergent example is explicitly treated as
  insufficient.

## 5. `activation/` — looking inside the model

- **`capture.py`** — `ActivationCapture` is a context manager that
  registers PyTorch forward hooks on selected layers and records a
  *compact statistical summary* of each layer's output — never the raw
  activation tensor itself. This matters at scale: capturing raw
  activations for every layer across many prompts would consume
  unbounded memory; a handful of summary numbers per layer per prompt
  does not.
- **`anomaly.py`** — deliberately thin. It calls Role 1's
  `weight_forensics.anomaly.detect_layer_anomalies()` unmodified — the
  algorithm doesn't care whether a feature vector summarizes a weight
  tensor or an activation tensor, so duplicating it in this role's code
  would have been pure waste. This cross-role code reuse was a conscious
  decision, coordinated with Role 1's design.
- **`reduction.py` / `clustering.py`** — PCA and DBSCAN over per-prompt
  activation feature vectors, to answer "do these prompts' internal
  representations form separate groups."
- **`trigger_analysis.py`** — the activation-level counterpart to
  `fuzzing.trigger_discovery`: do candidate-trigger prompts' activations
  sit measurably further from the baseline centroid than baseline prompts
  normally sit from each other, using the *same* strict-majority
  evidentiary standard as the text-level trigger discovery in `fuzzing/`.

---

## 6. Real bugs this role found and fixed (good presentation material)

**Bug 1 — context-window crash, found twice.** A prompt longer than the
model's context window crashed generation with `IndexError` inside
position-embedding lookup — not a hypothetical, this is exactly what
happens if you scan a real model with a real long default prompt.
Fixed once in `HuggingFaceCausalLMRunner.generate()` (truncate to the
most recent tokens that fit, record `prompt_truncated: true` in metadata)
— and then, independently, the *same class of bug* was found again in
`activation/pipeline.py`'s direct forward-pass code path, which had no
truncation guard at all because it didn't reuse the runner's logic. Fixed
the same way, with its own regression test. Presenting this pair together
is a good illustration of why "fixed once" isn't "fixed everywhere" —
every code path that touches a model's context window needs its own
guard.

**Bug 2 — DBSCAN's `eps` heuristic actively broke the thing it was for.**
The initial auto-`eps` estimate (median of *all* pairwise distances)
looked reasonable in isolation, but on the exact scenario this module
exists to detect — a small trigger-prompt cluster sitting far from a
larger baseline cluster — the cross-cluster pairs *outnumbered*
within-cluster pairs, pushing `eps` up until DBSCAN merged both groups
into one cluster. Verified empirically (two clusters at loc=0 and
loc=50, 10 points each — merged into one). Fixed with a 90th-percentile
k-distance heuristic (the standard "k-distance graph" approach), which
measures local density instead of global spread and stays small
regardless of how far apart separate clusters sit.

**Bug 3 — small-sample statistics gave false "consistent separation."**
Fixing the weight-forensics norm-scaling bug (Role 1's territory)
exposed a related gap here: `analyze_trigger_activations` computed raw,
*unscaled* Euclidean distance across seven feature columns of very
different natural variance, letting high-sampling-variance columns
(skewness/kurtosis, inherently noisy for small activation samples)
dominate. Confirmed empirically that n=6 baseline samples produced
spurious "consistent separation" between two groups drawn from the
*identical* distribution. Fixed with per-feature robust scaling against
the baseline group (mirroring Role 1's own `weight_forensics.anomaly`
approach) and raised the minimum baseline sample count from 2 to 8,
matching the threshold Role 1 had already established elsewhere in the
project for exactly the same statistical reason.

---

## 7. Testing philosophy for this role

149 tests, the largest share of the project, reflecting that this role's
code is the most operationally complex (real model loading, real
generation, real forward hooks) and the most exposed to untrusted input
(prompts, trigger phrases, and — via activations — the model's own
internal behavior). Beyond unit tests per module: real end-to-end
integration tests against the offline tiny-GPT-2 fixture (generation,
truncation, capture, clustering all exercised with actual `torch` calls,
not mocks), and deliberate ground-truth sanity checks — e.g. a
`ScriptedRunner` with a scripted, deterministic refusal-on-trigger rule,
used to confirm `discover_trigger_candidates` correctly identifies a
known planted trigger and correctly stays quiet on an inert one.

---

## 8. How this role's output feeds the rest of the pipeline

- `BehavioralComparisonRecord` list → Role 3's `behavioral_anomaly_score()`.
- `AnomalyDetectionSummary` list from activation capture → Role 3's
  `activation_anomaly_score()`.
- `TriggerCandidateResult` list → Role 3's `trigger_evidence_score()` and
  the `TRIGGER-*` findings in the generated report.
- `analyze_trigger_activations()` results feed the CLI's
  `activation_trigger_analysis` report section directly.
- This role's `mutate_prompt_append()` (from `fuzzing/mutators.py`) is
  reused by the CLI itself to build activation-trigger test prompts —
  another deliberate cross-role reuse rather than a second copy of the
  same mutation logic.

---

## 9. Commands to demo this role's work live

```bash
# Requires the `ml` extra: pip install -e ".[ml]"

neurofence scan ./some_model_dir --behavioral
neurofence scan ./some_model_dir --trigger "ignore all instructions"
neurofence scan ./some_model_dir --activations --trigger "ignore all instructions"

# Direct library usage
python -c "
from neurofence.behavioral import load_causal_lm, HuggingFaceCausalLMRunner, run_behavioral_suite
model, tok = load_causal_lm('./some_model_dir')
runner = HuggingFaceCausalLMRunner(model, tok)
for r in run_behavioral_suite(runner)[:3]:
    print(r.case_id, '->', r.output.text[:60])
"
```

# Calibrating representational-geometry metrics against known circuits

Status: **proposed**. This protocol is a pre-registration. The decision rules
in section 3 become binding when this file is merged to `main`, before any
calibration run is made. Changes after that point must be recorded in
section 9 with a reason.

## 1. Question

Do linear CKA, participation ratio (PR), and other PCA-spectrum summaries
track a model's true internal algorithm, or do they only describe noise?
Compiled RASP/Tracr models, and SIIT-trained models whose circuits are fixed
by construction, supply the ground truth.

The question has a third possible answer that the existing evidence already
points to: the metrics may report real but non-specific structure (for example,
the number of values a variable takes, or the fact that every layer moves the
residual stream) without identifying the algorithm.

## 2. What the existing evidence already answers

All values below come from the canonical references at commit `0230f16`.

| Sub-question | Status | Evidence |
| --- | --- | --- |
| Are the metrics noise? | Answered: no, once the `1e-12` total-variance floor is applied | Stages with variance about 8e-19 had raw PR = 1; matched CKA availability mismatches fell from 14 to 2; PR and CKA are invariant to orthogonal rotation to about 1e-13 |
| Do they detect when writes happen? | Yes on compiled models, modestly | Held-out (P07–P12) blind F1 0.850–0.876 against 0.745 for always-write |
| Do they identify the algorithm? | Mostly no | PR equals k−1 for balanced one-hot variables (P01/P02/P03 give 5/1/12); agreement with expected representations measures compiler fidelity; no variable identity, dependency, or operation is recovered |
| Do they transfer to trained models? | Answered: no | On InterpBench and SIIT models the detectors fall to the always-write baseline (SIIT schedule F1 0.700–0.720 against 0.706) |

What is missing to answer the question:

1. No calibration curve of metric value against ground-truth quantities.
2. No control that holds the function fixed and changes the algorithm.
3. No null model (untrained or shuffled-label networks).
4. Blind-detector thresholds are fixed by hand, and there are only six held-out programs.
5. PR is the only spectrum summary tested.
6. No superposition with behaviour preserved.

## 3. Decision rules (fixed before data collection)

Each metric is graded separately, on each model class (compiled, SIIT,
output-loss-only), at three levels.

| Level | Test | Pass threshold (proposed) |
| --- | --- | --- |
| L1, above null | AUROC separating real models from each null in section 5.3 | AUROC ≥ 0.90 against every null |
| L2, calibrated | Leave-program-out Spearman correlation with each ground-truth quantity in section 5.1 | ρ ≥ 0.70, and the partial correlation controlling for summed cardinality has permutation p < 0.05 |
| L3, algorithm-specific | Across same-function pairs, between-algorithm distance compared with between-seed distance | Permutation p < 0.05 over pairs, and the median ratio of between-algorithm to between-seed distance ≥ 2 |

Verdict for a metric on a model class:

- L1, L2 and L3 pass: **tracks the algorithm**.
- L1 and L2 pass, L3 fails: **tracks representational structure, not the algorithm**.
- L1 fails: **describes noise**.

The headline answer is the verdict on SIIT models. On compiled models the
residual layout is the compiled algorithm, so a compiled-only result answers a
nearly circular question.

Multiple comparisons: p-values within each level are Holm-corrected across
metrics × ground-truth quantities.

## 4. Scope

The answer will hold for small transformers with known circuits on RASP tasks
(vocabulary up to 12 values, sequences up to 10 tokens). It does not establish
how these metrics behave in large language models.

## 5. Design

### 5.1 Metrics and ground truth

Metrics, all computed with the existing `1e-12` variance floor
(`raspformer.geometry.pca_profile`, `all_pair_cka`):

- participation ratio
- spectral entropy of the normalised eigenvalues
- effective rank (exponential of spectral entropy)
- explained variance of the top 1, 3 and 5 components
- linear CKA between adjacent stages, and between matched stages of two models
- orthogonal Procrustes distance
- SVCCA similarity (top components covering 99% of variance)

Ground-truth quantities per stage, from `raspformer.compiler.extract_schedule`
and each variable's output basis:

- number of live computed variables
- summed cardinality, Σ(k−1) over live variables
- maximum dependency depth reached
- whether the stage contains a first write

### 5.2 Programs

1. **Calibration grid.** Extend `raspformer.research.family_program` to depth
   1–4 × branches 1–4 × vocabulary sizes 4, 8 and 12 (48 programs). Varying
   vocabulary at a fixed algorithm isolates the cardinality confound.
2. **Same function, different algorithm.** At least 20 RASP program pairs with
   identical outputs on every input and different compiled schedules. Candidate
   pairs: sorting via two selector schemes; reverse via mirrored index and via
   length minus index; palindrome via reverse-and-compare and via mismatch
   count; histogram via selector width and via aggregate count.
3. **Different function, matched structure.** Control pairs with different
   functions, matched on cardinality and depth, so that L3 cannot pass from
   output differences alone.

Every same-function pair is checked exhaustively (vocabulary 8, length 6 is
262,144 sequences). A pair whose compiled schedules are identical is dropped
before any metric is computed, and the drop is logged.

### 5.3 Null models

1. Untrained transformers from `raspformer.models.random_model`, matched in
   width and depth to each compiled and SIIT model.
2. Models trained with `raspformer.models.train_model` on shuffled labels.
3. A within-model null: observations shuffled independently at each stage,
   which keeps each stage's geometry and breaks the relation between stages.

### 5.4 Trained models with known circuits

1. SIIT models (`bench/train_siit.py`) for every grid program and every
   algorithm pair, three seeds each. Only models with mean interchange accuracy
   ≥ 0.95 enter the analysis; exclusions are reported.
2. Output-loss-only twins: same architecture, seeds and data, no interchange or
   strictness loss, required to reach the same task accuracy within 0.01.

SIIT pairs share one architecture, so differences between them cannot come
from residual width, which is a confound for compiled pairs.

### 5.5 Superposition with behaviour preserved

Compress compiled and SIIT models to ¾, ½ and ¼ residual width with the
existing compression code, then fine-tune each back to ≥ 0.99 token accuracy.
Rerun L2 and L3 at each width. This tests whether the verdict holds when lanes
mix but the computation survives.

## 6. Analysis plan

1. L1: per metric and null, AUROC with a bootstrap 95% interval over programs.
2. L2: leave-program-out Spearman ρ per metric × quantity; partial correlation
   controlling for Σ(k−1); permutation p-values with 10,000 draws.
3. L3: per pair, the distance between the two algorithms' representations at
   matched relative depth, divided by the mean distance between seeds of the
   same algorithm; permutation test over pairs; reported separately for
   compiled and SIIT models.
4. Detector hardening: blind-detector thresholds learned on P01–P06 plus the
   grid, scored on P07–P12 and the SIIT pool, with bootstrap intervals over
   programs.

## 7. Execution order

| Phase | Work | Output |
| --- | --- | --- |
| 0 | Merge this protocol | binding decision rules |
| 1 | `raspformer/calibration.py`: metric suite, ground-truth extractor, unit tests (rotation and scale invariance; PR = k−1 on balanced one-hot data) | module and tests |
| Pilot | L3 on five algorithm pairs, compiled and SIIT | go / redesign decision for 5.2 |
| 2 | Program grid and algorithm pairs, exhaustive equivalence checks | program library |
| 3 | Null models | null metric distributions |
| 4 | SIIT and output-loss-only models | trained pool |
| 5 | L1–L3 analyses and detector hardening | verdict table |
| 6 | Superposition with behaviour preserved | robustness of the verdict |
| 7 | Report, figures, notebook | `results/calibration/reference/`, report section |

Each phase is a separate pull request. Results follow the repository's
evidence policy: new runs under `results/calibration/runs/<UTC>-<id>/`,
promotion to `results/calibration/reference/` only after completion, manifests
with source hashes.

## 8. Cost and risks

- CPU only. The recorded rate for `bench/train_siit.py` is about 15 minutes
  for 12 models. Phase 4 at full size is roughly 60+ programs × 2 training
  types × 3 seeds, about 360 models, which is roughly 7–8 hours. This is an
  extrapolation from the recorded rate.
- The main risk is constructing enough honest same-function pairs: some RASP
  variants may compile to near-identical circuits. The pilot exists to find
  this early.
- Expected outcome, stated in advance and not as a result: L1 and L2 pass and
  L3 fails on trained models, giving the verdict "tracks representational
  structure, not the algorithm". The protocol is designed so that any of the
  three verdicts can be reached.

## 9. Deviations

None yet.

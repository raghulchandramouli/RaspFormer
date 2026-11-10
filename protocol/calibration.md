# Calibrating representational-geometry metrics against known circuits

Status: **binding** since 2026-11-10. This protocol is a pre-registration. The decision rules
in section 3 became binding when this file was merged to `main`, before any
calibration run was made. Changes after that point are recorded in
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
| L1, above null | AUROC separating real models from each null in section 5.3 that applies to the metric | AUROC ≥ 0.90 against every applicable null |
| L2, calibrated | Leave-program-out Spearman correlation with each ground-truth quantity in section 5.1 | ρ ≥ 0.70, and, for every quantity other than summed cardinality, the partial correlation controlling for summed cardinality has permutation p < 0.05 |
| L3, algorithm-specific | Across same-function pairs, between-algorithm distance compared with between-seed distance | Permutation p < 0.05 over pairs, and the median ratio of between-algorithm to between-seed distance ≥ 2 |

Verdict for a metric on a model class:

- L1, L2 and L3 pass: **tracks the algorithm**.
- L1 and L2 pass, L3 fails: **tracks representational structure, not the algorithm**.
- L1 fails: **describes noise**.

The headline answer is the verdict on SIIT models. On compiled models the
residual layout is the compiled algorithm, so a compiled-only result answers a
nearly circular question.

Summed cardinality is the control variable, so its own partial correlation is
undefined. It is graded on ρ alone.

Multiple comparisons: p-values within each level are Holm-corrected across
metrics × ground-truth quantities.

## 4. Scope

The answer will hold for small transformers with known circuits on RASP tasks
(vocabulary up to 12 values, sequences up to 10 tokens). The L3 verdict holds
only on the domain where each pair's equivalence is checked exhaustively
(section 5.2). It does not establish how these metrics behave in large language
models.

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
   identical outputs on every input in the checked domain below and different
   compiled schedules. Candidate
   pairs: sorting via two selector schemes; reverse via mirrored index and via
   length minus index; palindrome via reverse-and-compare and via mismatch
   count; histogram via selector width and via aggregate count.
3. **Different function, matched structure.** Control pairs with different
   functions, matched on cardinality and depth, so that L3 cannot pass from
   output differences alone.

Every same-function pair is checked exhaustively on vocabulary 8, length 6
(262,144 sequences). "Same function" means equal outputs on that domain and
nothing wider. Pair models are trained, measured and scored for L3 on that
domain only. A pair whose compiled schedules are identical is dropped before
any metric is computed, and the drop is logged.

### 5.3 Null models

1. Untrained transformers from `raspformer.models.random_model`, matched in
   width and depth to each compiled and SIIT model.
2. Models trained with `raspformer.models.train_model` on shuffled labels.
3. A within-model null: observations shuffled independently at each stage,
   which keeps each stage's geometry and breaks the relation between stages.

Nulls 1 and 2 apply to every metric. Null 3 leaves each stage's covariance
spectrum unchanged, so it applies only to the metrics that relate two
representations: CKA, Procrustes distance and SVCCA. The single-stage spectrum
metrics (PR, spectral entropy, effective rank, explained variance) are graded
at L1 against nulls 1 and 2.

### 5.4 Trained models with known circuits

1. SIIT models (`bench/train_siit.py`) for every grid program and every
   algorithm pair, three seeds each. Only models with mean interchange accuracy
   ≥ 0.95 enter the analysis; exclusions are reported.
2. Output-loss-only twins: same architecture, seeds and data, no interchange or
   strictness loss, required to reach the same task accuracy within 0.01.

Both members of a SIIT pair, and their output-loss-only twins, are trained with
the same layer count, head count, residual width and MLP width. The layer count
is the larger of the two compiled depths, because `bench/siit.py` otherwise
sets depth from each program's own compiled model. The saved configurations are
compared before any L3 distance is computed; a pair whose members differ is
excluded and the exclusion is logged. Differences within a pair then cannot
come from architecture, which is a confound for compiled pairs.

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
| 1 | `raspformer/calibration.py`: metric suite, ground-truth extractor, unit tests (rotation invariance; scale invariance for scale factors that keep total variance above the absolute `1e-12` floor; PR = k−1 on balanced one-hot data) | module and tests |
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

Every entry below was written before any calibration model was trained or
measured, unless it says otherwise. The baseline for existing evidence is the
tag `calibration-baseline` (commit `4a08794`).

### 2026-11-10: L2 pass rule (clarification)

Section 3 did not say how the four ground-truth quantities combine. A metric
passes L2 on a model class when, for at least one quantity other than summed
cardinality, the leave-program-out ρ is ≥ 0.70 and the Holm-corrected
partial-correlation p is < 0.05. Summed cardinality is graded on its own: ρ ≥
0.70 is reported as "tracks cardinality" beside the verdict and does not count
towards L2.

Reason: without this rule a metric that only counts values would pass as
calibrated, which is the confound the study exists to isolate.

### 2026-11-10: the pilot is go or redesign only (clarification)

With five pairs the smallest sign-flip p-value is 1/32 ≈ 0.031, which cannot
stay below 0.05 after Holm correction across the metrics. The pilot checks that
pairs train and the pipeline runs end to end. The L3 verdict comes from the
full set of pairs.

### 2026-11-10: program family rebuilt, grid levels changed (deviation from 5.2)

In the existing family, realised dependency depth is requested depth + branches
− 1 in every program, and depth correlates with computed-variable count at
0.956 (`results/validation/reference/complexity/factorial-families.csv` at the
baseline tag). A grid built on it could not separate the two.

Every RASP operation reads at most two sequence variables, so a program of
dependency depth D holds between D and 2^D − 1 computed variables. A full
crossing from depth 1 is therefore impossible. The grid is now dependency depth
3, 4, 5 × computed variables 5, 6, 7 × vocabulary 4, 8, 12, which is 27
programs with depth and variable count exactly uncorrelated.

Each leaf now reads the token at a different cyclic offset through attention.
In the old family every variable was a function of the same single token, so
the number of live variables could not be identified from activations by any
measurement. All grid programs compile with `max_seq_len` 10, because Tracr
needs the value width to fit the key size at vocabulary 12, and are measured at
length 6.

### 2026-11-10: operational definitions (clarification)

The protocol left these choices open. They are fixed here.

**Measurement.** Each model is run on 256 held-out sequences of length 6 drawn
uniformly with a fixed seed, the same sequences for every model of a program or
pair. An observation is one non-BOS position. A trained model's stage s is
compared with the compiled stage s (the identity layer correspondence SIIT
trains). Padded layers carry the ground truth of the last compiled stage.
Relational metrics are turned into one value per stage: 1 − linear CKA, the
Procrustes distance and 1 − SVCCA between stage s − 1 and stage s.

**Training.** SIIT follows `bench/siit.py`: residual width 32, four heads,
3,000 steps, seeds 0, 1, 2. An output-loss-only twin uses the same code with
the interchange and strictness weights set to zero. Shuffled-label nulls use
the twin's architecture and steps, with labels permuted across sequences.
Untrained nulls use the real model's depth and width.

**L1.** Run on the grid programs. Units are (program, seed, stage) values.
AUROC is direction-free (the larger of a and 1 − a). The interval is a
bootstrap over programs with 2,000 resamples. The pass rule uses the point
estimate.

**L2.** Run on the grid programs. Rows are (program, stage), with seeds
averaged. For each held-out program the sign of the relation is taken from the
other programs; ρ is the pooled Spearman correlation over all held-out rows.
The partial correlation is a Spearman correlation of rank residuals after
regressing on the ranks of Σ(k−1), with a two-sided permutation p-value from
10,000 draws.

**L3.** A metric defines a distance between two models: for CKA, Procrustes and
SVCCA the dissimilarity between the two models' representations of the same
inputs, for single-stage spectrum metrics the absolute difference of the metric
value. Distances are averaged over stages after the embedding. For each pair,
the between-algorithm distance is the mean over all cross-algorithm seed pairs
and the between-seed distance is the mean over same-algorithm seed pairs. The
test statistic is the mean log ratio over pairs, with a one-sided sign-flip
test of 10,000 draws.

Compiled models have no seeds, so their between-seed distance is zero and the
ratio is undefined. L3 is reported as not gradable on compiled models, with the
between-algorithm distances listed.

**Identification accuracy.** For each trained model in a pair, the metric names
the algorithm whose other seeds are nearer on average. Chance is 0.5. It is
reported beside L3 and is not part of the pass rule.

**Control pairs.** The nine grid cells at vocabulary 8, each built twice with
different constants, give pairs with different functions and identical
structure. Their ratio is reported beside the twin pairs and has no pass rule.

**Equivalence.** A same-function pair must give equal compiled outputs on all
262,144 inputs, and each compiled member must agree with the RASP evaluator on
2,000 sampled inputs.

**Phase 6.** Models are compressed onto the leading PCA directions of their
calibration activations, then every parameter is fine-tuned on the output loss
for at most 2,000 steps. Models below 0.99 token accuracy are excluded and
counted.

**Detector hardening.** Each detector's threshold is the value on a fixed
logarithmic grid that maximises mean F1 over the compiled calibration programs
(P01–P06 and the grid). It is scored once on compiled P07–P12 and on the SIIT
models.

### 2026-11-10: training length, pilot rule and verdict labels (before any metric was computed)

These three entries were written after timing SIIT on six grid programs and
before any geometry metric was computed on a trained calibration model.

**Training length (deviation).** SIIT runs for 6,000 steps, not 3,000. In the
timing runs the depth-three programs reached interchange accuracy 1.00 by step
3,000, but the depth-four program `d4 v6 k8` still had an interchange loss of
0.13 at step 2,000 and 0.009 at step 4,000. At 6,000 steps the four deepest
programs tried reached interchange accuracy 0.986 to 1.00. Every calibration
model, including output-loss-only twins and shuffled-label nulls, uses 6,000
steps.

**Pilot rule (clarification).** The pilot's decision is go when at least four of
the five pilot pairs have two or more SIIT seeds per member at mean interchange
accuracy ≥ 0.95 and the analysis runs end to end. Otherwise the pairs are
redesigned.

**Verdict labels (clarification).** Section 3 names three verdicts and leaves
two combinations open. A metric that passes L1 but not L2 is labelled "above
null, not calibrated", or "separates algorithms, not calibrated" when it also
passes L3. On compiled models, where L3 is not gradable, a metric that passes
L1 and L2 is labelled "above null and calibrated; algorithm-specificity not
gradable".

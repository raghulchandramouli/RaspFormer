# bench: InterpBench subset

Six InterpBench cases with their RASP programs, SIIT-trained transformers, and
ground-truth circuits. This is the model pool with known internals that the
RaspFormer reports list as missing. Nothing here is run by `raspformer.study`.

## Cases

| Case | Task | Vocab | Causal | Tracr layers | Trained model (L, H, d) | RaspFormer twin |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | Reverse the sequence | a–z | no | 4 | 4, 4, 56 | — |
| 7 | Count of each token (histogram) | a, b, c | no | 1 | 2, 4, 17 | `P05_histogram` |
| 13 | Trend vs neighbours: increasing / decreasing / constant | 0, 1, 2 | no | 2 | 2, 4, 20 | — |
| 124 | Are all elements equal? (0/1 at every position) | 0–10 | no | 3 | 3, 4, 24 | `P09`/`P11`-style predicate |
| 21 | Keep first occurrence of each token, else None | a, b, c | yes | 4 | 4, 4, 50 | — |
| 44 | Count of elements greater than this one | 0–10 | no | 1 | 2, 4, 24 | `P06_count_greater_than` |

All six use InterpBench's `max_seq_len = 10` (counting BOS), so Tracr compiles
with nine token positions. Trained models have at least two layers and four
heads because InterpBench pads the low-level model (`make_ll_cfg`) and may
compress its residual width. The compiled Tracr model is the high-level model;
`edges.pkl` says which trained components implement it.

## Files

```text
bench/
  programs.py             Six RASP programs, ported verbatim; compile and evaluate helpers
  fetch.py                Download and SHA-256 verify the artifacts
  interpbench/
    manifest.json         Source URLs, checksums, training hyperparameters
    benchmark_cases_metadata.csv   Full InterpBench case table (all 86 cases)
    <id>/ll_model.pth     Trained weights (torch state dict, zip format)
    <id>/ll_model_cfg.pkl TransformerLens HookedTransformerConfig as a dict
    <id>/edges.pkl        Ground-truth circuit: list of (source hook, target hook)
    <id>/meta.json        SIIT training arguments
```

The artifact files are data. They are never imported or executed. Reading
`ll_model_cfg.pkl` with plain `pickle` needs `torch`; `test_bench.py` shows a
stub unpickler that reads it without.

## Use

```python
from bench import CASES, compile_case, evaluate, reference_outputs, sample_inputs

case = CASES["44"]
model = compile_case(case)               # Tracr AssembledTransformerModel
tokens = sample_inputs(case, 1, seed=0)[0]
evaluate(case, tokens)                   # rasp.evaluate; None where undefined
reference_outputs(model, tokens)         # what the trained model was fit to
```

InterpBench's training labels are the compiled model's argmax, not
`rasp.evaluate`. Cases 13 and 21 produce `None` at some positions in RASP;
the compiled model emits an arbitrary class there, and the trained model was
fit to that class. Score trained models against `reference_outputs`.

Attention direction, vocabulary size, output classes, and context length of
each compiled program are checked against the trained config in `test_bench.py`.

Case 124 is a whole-sequence predicate. Uniform sampling over eleven values and
nine positions almost never yields an all-equal sequence, so `sample_inputs`
mirrors InterpBench's balanced sampler for it: half all-equal, half not.

Case 13's program is the pre-fix upstream version (circuits-benchmark commit
30c3c43). Upstream fixed the program in September 2024 without retraining the
published weights, so the current upstream file does not describe this model.
`make_case_13` documents the difference.

Case 18 (frequency classifier) was downloaded and rejected. Its published
`edges.pkl` lists two MLP edges and no attention or embedding nodes, but the
program needs attention for both histogram and length. `manifest.json` records
the exclusion.

## Source and license

Programs: [FlyingPumba/circuits-benchmark](https://github.com/FlyingPumba/circuits-benchmark),
`circuits_benchmark/benchmark/cases/` and `common_programs.py` (MIT).
Models: [cybershiptrooper/InterpBench](https://huggingface.co/cybershiptrooper/InterpBench) (CC-BY-4.0).
Paper: Gupta, Arcuschin, Kwa, Garriga-Alonso. *InterpBench*. NeurIPS 2024 Datasets and Benchmarks.
[arXiv:2407.14494](https://arxiv.org/abs/2407.14494).

Re-download or re-verify with:

```bash
.venv/bin/python -m bench.fetch --verify
```

## Phase 1 results

`bench/load.py` reads `ll_model.pth` without torch and maps it onto
`raspformer.models.Model` (GELU, unembed bias). `bench/phase1.py` runs the
unchanged RaspFormer detectors and stress suite on both the compiled and the
trained model of every case; outputs are in `results/bench/reference/`, with
`phase1.md` as the readable report.

```bash
.venv/bin/python -m bench.phase1            # about three minutes on CPU
```

Findings from the saved run:

- All six trained models reproduce their compiled program on held-out
  sequences (99.96% to 100% token agreement).
- On trained models every stage moves the residual, including layers the
  circuit does not use, so the fixed-threshold detectors flag every stage.
  Their F1 collapses onto the always-write baseline (schedule truth: change
  0.610 = baseline; PR 0.553 and CKA 0.583 below it). On the compiled twins the
  same detectors beat the baseline (change 0.810 vs 0.698).
- Compression repeats the false positive: at a quarter of the residual width
  mean token accuracy is 0.335 while schedule F1 stays at 0.60 to 0.63, the
  same level as the uncompressed models. Rotation preserves accuracy and
  geometry, as expected.
- Case 124's InterpBench sampler only produces all-equal or all-distinct
  sequences, so the intermediate count is constant on negatives. The trained
  model still scores 96% on repeated-but-unequal sequences it never saw.

Two truth definitions are scored for trained models: `schedule` (compiled
first-write stages, identity layer correspondence) and `circuit` (stages with a
head or MLP in `edges.pkl`). They differ on SelectorWidth attention halves and
on case 21, where the published circuit omits the first MLP.

## Phase 2 results

`bench/phase2.py` adds three baselines on the same splits and saved
activations; outputs land next to the Phase 1 files and `phase2.md` is the
readable report.

```bash
.venv/bin/python -m bench.phase2            # about ten minutes on CPU
```

- **Linear probes** (`bench/probes.py`): multinomial logistic regression from
  each residual stage to each program variable, trained on calibration and
  scored on test observations. Readable means held-out accuracy of 0.95.
- **DAS** (`bench/das.py`): for each variable and stage, learn an orthonormal
  subspace such that swapping its component between two inputs makes the model
  produce the program's own counterfactual output. Interchange intervention
  accuracy (IIA) is scored on held-out pairs, with a random-subspace control
  and the base rate (how often the counterfactual equals the unpatched output).
  Output positions the program leaves undefined are masked.
- **Edge attribution patching** (`bench/eap.py`): first-order estimate of every
  source-to-target edge's effect on the clean logit difference, ranked by
  magnitude and scored against `edges.pkl` at edge and node level.

Findings from the saved run:

- Probes read 73% of input-dependent variables in the trained models before
  the stage that writes them, because many variables are linear functions of
  variables already present. Readable is not computed.
- DAS finds the variables where the schedule says they are written, but the
  honest number is IIA on pairs whose counterfactual must change the output:
  at the first-write stage it is 0.98 on compiled and 0.67 on trained models,
  against random-subspace controls of 0.23 and 0.20 (unrestricted IIA is
  inflated by pairs where nothing should change). Three trained-model
  variables never align (case 13's two intermediate trend branches and case
  124's unequal count), consistent with the trained models solving those
  tasks without those intermediates. Case 21 has too few output-changing
  pairs for its intermediates to be scored reliably.
- Against probe-readable or DAS-aligned stage events, the geometry detectors
  again sit at the always-write baseline on trained models.
- EAP recovers one-layer circuits exactly and gets node-level precision at k
  of 0.82 on average, but edge-level precision at k is 0.51 and drops to 0.2
  to 0.3 on the four-layer models.

Variables that are constant across sequences at fixed length (length and
index arithmetic in case 2) are flagged position-determined; they read from
the position embedding at stage 0 and cannot be intervened on.

## Phase 3 results

`bench/decompile.py` recovers an executable table program from a model given
only its circuit nodes: node outputs are discretised into causal classes
(fine clusters merged when the model's answer cannot tell them apart), wiring
is found by intervening on each candidate edge, MLPs become lookup tables,
heads become selector plus aggregate tables read off binarised attention (or a
bag-of-keys table when attention is not crisp), and classes are minimised
against the fitted tables. `bench/phase3.py` runs the three tests and the
length-generalization study; the report is `results/bench/reference/phase3.md`.

```bash
.venv/bin/python -m bench.phase3            # about ten minutes on CPU
```

Findings from the saved run:

- Compiled models: cases 2, 7, 44, and 124 are recovered exactly (output,
  counterfactual, and ablation agreement all 1.0, every variable matched).
  Case 13 reaches 0.97 output agreement; case 21 (tuple-valued intermediates)
  reaches 0.51.
- Trained models: case 13 recovers at 0.97 output and 0.93 counterfactual
  agreement, case 124 at 0.96 and 0.80, case 44 at 0.85 and 0.81, case 7 at
  0.73, case 21 at 0.59. Case 2 (reverse, four layers) fails. Case 21's
  published circuit is incomplete: ablating everything outside it drops
  agreement to 0.27. Embeddings can reach a node only through nodes outside
  the circuit (inert heads that copy position), so wiring tests swap the
  embedding itself as well as each direct edge.
- Length generalization (shorter than training): RaspFormer learned models
  keep 100% on per-token maps but collapse on counting and sorting tasks
  (histogram 0.14, count-greater-than 0.39, sorting 0.46 at three tokens).
  InterpBench models behave the same way. For case 13 the recovered program's
  coverage predicts correctness: where it is defined the model is 100%
  correct at every length, where it is undefined 0.88 to 0.98.

## Phase 4: SIIT models for the twelve RaspFormer programs

`bench/siit.py` trains a four-head transformer per program with behaviour,
interchange, and strictness losses against the compiled model, saving
weights, ground-truth edges read off the compiled weights, and IIA/SIIA.
`bench/phase4.py` reruns detectors, stress suite, probes, DAS, EAP, and
recovery on them; outputs land in `results/bench/reference-siit/`.

```bash
.venv/bin/python -m bench.train_siit        # about fifteen minutes on CPU
.venv/bin/python -m bench.phase4
```

Findings from the saved run (`results/bench/reference-siit/phase4.md`):

- All twelve SIIT models reach 98% to 100% held-out accuracy with mean
  interchange accuracy of 0.97 or better and strictness accuracy of 0.98 or
  better, so their circuits are ground truth by construction.
- The geometry detectors again sit at the always-write baseline on the
  trained models (schedule F1 0.70 to 0.72 against a baseline of 0.71) while
  beating it on the compiled twins (0.88 against 0.71). Compression to a
  quarter width drops accuracy to 0.51 and leaves F1 unchanged.
- Probes read 21% of input-dependent variables before their first write.
  DAS on output-changing pairs reaches 0.63 at the first-write stage on
  trained models against 0.09 random.
- EAP recovers the one-layer circuits exactly and averages node precision at
  k of 0.77 over the twelve.
- Blind recovery of trained models reaches 0.94 or better output agreement
  on eight of twelve: absolute value, index parity, increment by index, first
  element, histogram, check increasing, rotate left, and check palindrome,
  with counterfactual agreement of 0.74 to 1.0 where the test applies. It
  fails on count greater than (0.63), the two sum tasks (0.25 to 0.27), and
  sorting, whose numerical sort key has no discrete class structure. Over all
  31 program variables, 19 are matched exactly in the trained models and 21
  in the compiled ones.

## Phase 5: lengths beyond training

The twelve programs were retrained with SIIT using ten-token position
embeddings and a ten-token compiled reference, but six-token training data
(`bench/train_siit.py --max-seq-len 10 --train-length 6`). `bench/phase5.py`
runs each model and its recovered program from three to ten tokens; the
report is `results/bench/reference-siit-long/phase5.md`.

```bash
.venv/bin/python -m bench.train_siit --output results/bench/siit-long --max-seq-len 10 --train-length 6
.venv/bin/python -m bench.phase5
```

Findings from the saved run:

- Training is unaffected by the larger embeddings: all twelve reach 99% to
  100% at six tokens with interchange accuracy of 0.98 or better.
- Mean token accuracy over the twelve falls from 1.00 at six tokens to 0.84
  at seven and 0.61 at ten; mean sequence accuracy at ten is 0.25.
- Three behaviours separate cleanly. Per-token maps and whole-sequence
  predicates generalise (absolute value, first element, check increasing at
  100% through ten tokens). Position-indexed tasks fail exactly at unseen
  positions (increment by index is right at the first six positions and wrong
  at every later one). Counting, shifting, and sorting degrade smoothly
  (histogram 0.61, rotate left 0.57, sorting 0.22 at ten tokens).
  Palindrome holds at 100% through eight tokens and then collapses to 0% at
  nine and ten.
- The recovered program predicts where a model breaks. On longer-than-
  training inputs, where the program fitted at six tokens is defined the
  model scores 0.91 on average; where it is undefined, 0.36. For increment
  by index the match is exact: defined positions are all correct, undefined
  positions all wrong.

## Not done yet

ACDC as a second circuit-discovery method; recovery of tuple-valued and
numerical intermediates; seed variance of the decompiler.

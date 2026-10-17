# RaspFormer

Define sequence algorithms in RASP, compile them with Tracr, and test what representation measurements reveal about their execution.

The original studies check agreement with known compiler representations. The validation studies add blind transition detection, representation controls, causal patches, executing residual compression, and models trained from random weights.

Read the generated [research validation report](results/report/research-validation.md) for measured results and limitations.

PDFs include the [complete change report](output/pdf/raspformer-research-upgrade-report.pdf) and the [point-by-point response to review](output/pdf/raspformer-response-to-review.pdf).

## Run the studies

Use Python 3.12 and the pinned dependencies. The environment runs on CPU.

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m raspformer.study
.venv/bin/python -m raspformer.report
```

The default command reruns all five original studies, all validation controls, nine structural program variants, and 36 learned models. Training uses 600 Adam steps for each of twelve programs and three seeds. A completed experiment can have a negative scientific result.

To run one part, use these commands:

```bash
.venv/bin/python -m raspformer.study --sections baseline
.venv/bin/python -m raspformer.study --sections validation
.venv/bin/python -m raspformer.study --sections learning --seeds 42 43 44 --training-steps 600
```

Each section replaces its canonical reference only after its run completes. Run `raspformer.report` after all sections finish.

## What is measured

| Study | Experiment and evidence |
| --- | --- |
| `programs` | Twelve compiled algorithms, sampled output checks, and compiler write schedules. |
| `trace` | Manual attention and MLP calculations compared with the assembled transformer. |
| `geometry` | Class-balanced targets, PCA, PR, CKA, and full versus computed residual views. |
| `alignment` | Independently evaluated RASP values encoded in known compiler lanes, plus A and B interventions. |
| `comparison` | Shared inputs and balanced sensitivity checks, raw variance, thresholded metrics, and CKA availability. |
| `validation` | Orthogonal and nonorthogonal controls, executing compression, blind transition recovery, PCA patches, sorting diagnostics, and a depth-by-branch-count family. |
| `learning` | Randomly initialized transformers trained on the same twelve programs, disjoint sequence splits, three seeds, predictions, checkpoints, and compiled-model comparisons. |

The vocabulary is the integers -2 through 5. The main probe design uses length-six sequences and noncausal attention. Output checks also cover shorter examples.

PR and CKA use an absolute total sample variance floor of `1e-12`. Inactive PCA dimensions are reported as zero with a reason flag. CKA remains undefined for inactive representations. Comparison tables retain raw measurements and sensitivity to several floors.

## Files and naming

```text
raspformer/
  programs.py       RASP algorithms
  compiler.py       Compilation, schedules, traces, and run recording
  geometry.py       Sampling, activations, and numerical policy
  alignment.py      Semantic comparisons and original interventions
  models.py         Executable compiled, rotated, compressed, and trained models
  research.py       Research controls and blind transition detection
  study.py          Command-line execution and reference promotion
  report.py         Report and figure generation from saved measurements
notebooks/
  01_programs.ipynb through 05_comparison.ipynb
results/
  <study>/reference/           Preserved canonical evidence
  <study>/runs/<UTC>-<id>/     Disposable local runs, ignored by Git
  report/research-validation.md
  report/figures/
scripts/check_notebooks.py
test_studies.py
test_numerics.py
test_models.py
test_research.py
```

New folders use lower-case words separated by hyphens. Python names use underscores. Study names stay singular, such as `trace`. Program folders use names such as `p12-sorting`, with `seed-42` below learned models.

Manifests record completion or failure, dependencies, compiler revision, Git state, Python module hashes, and notebook source hashes. Failures preserve completed evidence. A completed numerical manifest does not assert that later plotting succeeded.

The obsolete report, its duplicate PDF and figure bundle, historical duplicate trace tables, and stale figures have been replaced by current evidence and one generated report. Git history retains their previous versions.

## Verify the work

```bash
mkdir -p .context/verification
.venv/bin/python -m unittest discover -v > .context/verification/tests.log 2>&1
.venv/bin/python scripts/check_notebooks.py
.venv/bin/python scripts/replay_learning.py
.venv/bin/python scripts/verify_results.py --test-log .context/verification/tests.log --notebook-summary .context/notebook-checks/summary.json
```

Notebook verification executes every code cell with IPython and records logs under `.context/notebook-checks`. It does not store notebook outputs. Notebook 05 calls the same comparison runner as the command line.

Tests cover compiled execution, full residual agreement, basis transformations, actual state-width reduction, checkpoint restoration, training updates, low-variance behavior, CKA coverage, matched sampling, blind prediction isolation, and failure retention.

## Interpretation

Expected representations use independently evaluated RASP values and compiler-known coordinates and write times. Agreement measures compiler fidelity. It does not independently discover the algorithm.

Blind recovery predicts transitions containing writes. It does not recover variable names or full circuits. Its predictions are frozen before schedule scoring. Shuffled-stage and always-positive baselines are reported.

PCA patch stages and directions come from geometry. Mapping them to named variables subsequently uses supervised RASP labels. Calibration decoder folds share an unsupervised PCA fitted on calibration data. The separate test sequences remain held out.

Compressed models execute with smaller residual streams. Their attention and MLP hidden widths remain unchanged. Projected-axis overlap measures induced overlap between original coordinates. There is no claim of general variable-lifetime recovery.

Learned models receive ordinary output supervision. Their internal circuits are unknown. Their comparison with compiled models is descriptive and does not establish natural-language or length generalization.

[InterpBench](https://papers.nips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html) uses Strict Interchange Intervention Training to produce models with known causal circuits and evaluate circuit discovery. RaspFormer's learned models use a different training objective. No InterpBench benchmark result is claimed here.

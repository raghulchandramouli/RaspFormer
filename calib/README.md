# calib: the calibration study

This package runs the study pre-registered in
[`protocol/calibration.md`](../protocol/calibration.md). It asks whether CKA,
participation ratio and other spectrum summaries track a model's algorithm,
using models whose algorithm is known.

## Files

```text
calib/
  pairs.py        Same-function, different-algorithm program pairs
  equivalence.py  Exhaustive check that a pair computes one function two ways
  library.py      The program grid, control pairs, and held-out probe sequences
  pool.py         Train, compress and load models; one resumable task per model
  analysis.py     L1, L2, L3, verdicts and detector hardening
  report.py       Summary written from the saved tables
  run.py          Command line
```

The metrics and the per-stage compiler ground truth live in
`raspformer/calibration.py`.

## Programs

- **Grid**: 27 programs crossing dependency depth 3, 4, 5 with computed
  variables 5, 6, 7 and vocabulary 4, 8, 12. Used for L1 and L2.
- **Twin pairs**: two programs with equal outputs on all 262,144 inputs and
  different compiled schedules. Used for L3.
- **Control pairs**: the nine vocabulary-8 grid cells, each built twice with
  different constants. Different function, identical structure.

## Models

Every program gets SIIT models and output-loss-only twins at three seeds, with
residual width 32, four heads and 6,000 steps. Grid programs also get
shuffled-label nulls. Both members of a pair train at the deeper member's layer
count. Untrained nulls are the trained models' own initial weights.

## Run

```bash
.venv/bin/python -m calib.equivalence --output results/calibration/runs/pairs
.venv/bin/python -m calib.run train    --root results/calibration/runs/full
.venv/bin/python -m calib.run compress --root results/calibration/runs/full
.venv/bin/python -m calib.run analyze  --root results/calibration/runs/full
.venv/bin/python -m calib.run promote  --root results/calibration/runs/full --to results/calibration/reference
```

Add `--pilot` to `train` and `analyze` to use only the five pilot pairs.
Training resumes: a model whose `metrics.json` exists is not retrained.
Checkpoints stay in the run folder, which Git ignores. `promote` publishes the
tables, and `models.csv` records the SHA-256 of every checkpoint.

## Tables

| File | Content |
| --- | --- |
| `models.csv` | Accuracy, interchange accuracy and checkpoint hash of every trained model |
| `exclusions.csv` | Models left out by the protocol's inclusion rules, with the reason |
| `stages.csv` | Every metric and ground-truth quantity at every residual stage of every grid model |
| `l1.csv` | AUROC of each metric against each null model |
| `l2.csv` | Leave-program-out correlation with each ground-truth quantity |
| `l3-pairs.csv` | Between-algorithm and between-seed distance of every pair |
| `l3.csv` | The L3 test per metric and model class |
| `verdicts.csv` | The verdict per metric and model class |
| `detectors.csv` | Blind detectors with thresholds learned on calibration programs |
| `*-compressed.csv` | L2 and L3 after compression and fine-tuning |
| `summary.md` | Readable summary of the tables above |

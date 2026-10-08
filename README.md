# RaspFormer

Define small sequence algorithms in RASP, compile them into transformers with
Tracr, and inspect how their internal representations change as they execute.
The compiler constructs the weights; these experiments do not train models.

**Define → trace → analyze → check alignment → compare programs.**

```text
RaspFormer/
├── notebooks/
│   ├── 01_programs.ipynb    # Algorithms, output checks, writer schedules
│   ├── 02_trace.ipynb       # Manual attention/MLP calculations
│   ├── 03_geometry.ipynb    # Probes, representation metrics, plots
│   ├── 04_alignment.ipynb   # Metrics versus known values/writes; A/B interventions
│   └── 05_comparison.ipynb  # P01–P12 agreement under matched and balanced probes
├── raspformer/
│   ├── __init__.py
│   ├── programs.py          # Fresh RASP expressions
│   ├── compiler.py          # Compilation, validation, tracing, run evidence
│   ├── geometry.py          # Sampling, activations, PCA, CKA, exports
│   └── alignment.py         # Ground-truth comparisons and targeted interventions
├── results/
│   ├── programs/reference/ # Saved checks and schedules before the rewrite
│   ├── trace/reference/    # Saved manual traces before the rewrite
│   ├── geometry/reference/ # Saved metrics, probes, and figures before the rewrite
│   ├── alignment/          # Local Step 4 runs
│   ├── comparison/         # Local Step 5 runs
│   └── archive/            # Earlier tables and logs
├── test_studies.py          # Regression checks using the saved evidence
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12. The dependency pins and Tracr commit reproduce the CPU
environment used for the checked-in experiments.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name raspformer --display-name "RaspFormer (.venv)"
```

Open a notebook in your editor or an existing Jupyter installation, select
**RaspFormer (.venv)**, and run all cells. The notebook locates the repository
from its working directory; no package installation is needed.

## Follow the experiments

Every notebook follows **configure → run → inspect → interpret**. Each is
independent and builds its own models. Imports alone never run an experiment.

1. [Define and check](notebooks/01_programs.ipynb): compile twelve algorithms,
   check five sequences per algorithm, and inspect where each variable is
   allocated in the residual stream. Change `FOCUS_PROGRAM` to inspect a schedule.
2. [Trace a circuit](notebooks/02_trace.ipynb): trace the twelve algorithms plus
   two focused examples. Embeddings, final output scores, and decoded outputs
   must agree within tolerance. Intermediate residual discrepancies are warnings.
   Change `FOCUS_PROGRAM` and `FOCUS_SAMPLE` to inspect another trace.
3. [Analyze geometry](notebooks/03_geometry.ipynb): generate balanced targets,
   check model labels, collect residuals, and plot PCA, participation ratios,
   linear CKA, lane variances, and target trajectories. The controls switch
   between programs, stages, and full/computed residual views.
4. [Check alignment](notebooks/04_alignment.ipynb): recreate Step 3's exact probes
   for all 14 programs. Compare observed PR with compiler-encoded RASP values,
   annotate adjacent-stage CKA with writes, and compare lane values/correlations.
   Test A/B causal predictions on all 640 targets with a donor intervention and
   an unchanged-value control per target. Change `VIEW` to plot full/computed lanes.
5. [Compare programs](notebooks/05_comparison.ipynb): compare P01–P12 on every
   sequence in the existing random-and-structured candidate pool, using every
   position equally. Defaults give 3,383 sequences × 6 positions = 20,298 identical
   observations per program. Rerun the original balanced probes to assess sampling
   sensitivity. Plot separate agreement measures against computed-variable count,
   dependency depth, and compiled layer count; program numbers do not order complexity.

The default vocabulary is `-2..5`, with at most six tokens and noncausal
attention. Probes use length six and seed 42; the balanced design uses 64
observations per reachable output class. Edit these settings in the notebooks.

## Read the results

Each run creates `results/<study>/<UTC timestamp>_<unique ID>/`. New runs are
gitignored and never overwrite another run. To preserve a new reference, copy
the chosen run to a deliberately named directory and commit it.

Every new run has a `manifest.json` with settings, dependency versions, devices,
Tracr revision, Python module hashes, progress, and completion or failure status.
Failures and interruptions retain completed evidence. The manifest uses schema
version 2; the original reference artifacts retain their original formats/hashes.

- **Programs:** the manifest contains sampled checks and variable allocations;
  `schedule.md` is the readable schedule.
- **Trace:** the manifest contains summaries, lane labels, and residual warnings;
  `traces.json` stores detailed residual snapshots and output scores once.
- **Geometry:** CSV files contain probes, balance audits, layer metrics, and CKA
  matrices; `pca_spectra.npz` contains PCA arrays. Notebook plots go in `figures/`.
  The manifest tracks the numerical study; the notebook reports figure exports.
- **Alignment:** `stages.csv` contains PR gaps, adjacent CKA, writes, and complexity;
  `variables.csv`, `values.csv`, and `lanes.csv` join schedules to expected values
  and observed lane comparisons. `interventions.csv` records A/B predictions,
  outcomes, donor identities, and controls. Exact probes and figures are retained.
- **Comparison:** one manifest covers `matched/` and `balanced/`, each containing
  the alignment tables and exact probes for P01–P12. `comparison.csv` summarizes
  agreement, complexity, and CKA coverage; plots go in `figures/`. A/B interventions
  remain in Step 4.

The `reference/` folders preserve the experiments recorded before this rewrite.
The historical documents under `archive/` provide context; use the numbered
notebooks for current execution.

## Check or extend

```bash
.venv/bin/python -m unittest -v test_studies
```

The checks exercise Steps 1–4, compare with saved outputs and geometry,
and cover failure recording, zero variance, deliberately incorrect lane values,
and A/B causal predictions. They use temporary run
directories and require only the existing dependencies and Python's standard
library.

To add an algorithm, write a factory in `programs.py`, register it in
`build_programs()`, and declare its reachable output classes in
`geometry.output_classes()`. Keep output validation enabled. Keep plotting in
the notebook and numerical work in the modules.

## Interpretation limits

Sampled correctness checks are not exhaustive proof. Writer schedules describe
compiler allocations, not measured activation onset. Tracing uses private
Tracr allocation/assembly helpers, so retain the pinned compiler revision.

Geometry describes one seeded sampling design. Each program has a different
balanced target distribution, and repeated targets retain their weight rather
than becoming independent trials. Full/computed views select lanes from the same
model. PCA centers features without standardizing them; encoding and scale affect
the measurements. CKA compares aligned targets across stages within a program.
Zero variance gives zero PCA spectrum/participation ratio and undefined CKA.

Step 4 tests hypotheses, not required metric behavior. Its expected representation
contains variables written so far in the compiler's encoding, with scratch and
unwritten lanes set to zero; it is not a live-variable count. Complexity measures
are computed-variable count, maximum dependency depth, and compiled layer count,
reported separately. The PR gap is signed (observed minus expected).
Correlation can be high despite wrong lane
magnitudes, and new writes need not lower CKA.

Step 5 averages absolute PR gaps across post-embedding stages, lane-value agreement
at first writes with equal scalar-lane weights, and absolute observed-versus-expected
CKA differences across transitions where both are defined. Undefined CKA remains
unavailable; coverage and definition mismatches are reported. These measures stay
separate. The shared pool is not exhaustive over vocabulary sequences, and matched
inputs do not make the checks independent or causal. Comparisons describe these
twelve Tracr-compiled programs under a fixed compiler; they do not establish a
universal relationship with complexity or reliability on trained models.

Interventions run in the manual circuit after baseline agreement with the assembled
model, including residuals. A patches the output as a direct readout control; B
patches the histogram before its downstream classifier. Donors are chosen at the
same position with a different output class, so change rates are selected effects,
not unbiased measures of importance. A completed run records measurements even
when alignment or intervention comparisons disagree; inspect the exported tables.

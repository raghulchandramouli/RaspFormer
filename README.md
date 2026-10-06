# RaspFormer

Explore how RASP programs compile into Tracr circuits and how their residual
representations change across layers.

```text
RaspFormer/
├── notebooks/
│   ├── programs.ipynb       # Configure programs and inspect writer schedules
│   ├── compile.ipynb        # Run and inspect manual circuit traces
│   └── geometry.ipynb       # Configure probes and plot representation geometry
├── raspformer/
│   ├── __init__.py
│   ├── programs.py         # Fresh RASP program factories
│   ├── compiler.py         # Compilation, validation, schedules, traces
│   └── geometry.py         # Balanced probes, activations, PCA, CKA, exports
├── results/                # Measurements, manifests, logs, and figures
├── requirements.txt        # Exact dependencies and Tracr commit
└── README.md
```

Notebooks hold configuration, execution, and plots. The three modules hold
shared logic with explicit inputs. Each notebook imports the package directly
and runs independently; importing a module does not execute a study.

Use Python 3.12. The pins capture the existing CPU environment used for these
experiments; they do not upgrade it.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name raspformer --display-name "RaspFormer (.venv)"
```

Open a notebook in Jupyter or your editor, select **RaspFormer (.venv)**, restart
the kernel, and run all cells. The setup cell finds the repository from either
its root or `notebooks/`. No package installation or path edits are needed.
Use the existing `.venv` directly if it is already configured.

1. [programs.ipynb](notebooks/programs.ipynb) compiles twelve programs and checks
   five sequences per program. Inspect `FOCUS_PROGRAM` to see its writer schedule.
2. [compile.ipynb](notebooks/compile.ipynb) traces those programs plus A/B examples.
   Embeddings, output scores, and decoded outputs are strict checks. Residual
   discrepancies are recorded separately and do not establish full equivalence.
3. [geometry.ipynb](notebooks/geometry.ipynb) samples seed-42, length-6 probes with
   64 target observations per reachable output class. It checks model labels,
   computes PCA/CKA for full and computed lanes, and exports the plots.

Runs save to unique directories under `results/programs/`, `results/compile/`,
and `results/geometry/`. Evidence records resolved settings, dependency versions,
devices, Tracr revision, module source hashes, and completion or failure status.
Geometry runs save CSV measurements/probes, PCA arrays, CKA tables, and figures.
Existing exports sit directly in those study directories. Earlier schedule and
trace records are preserved in `results/tables/`, with the log in `results/logs/`.

To extend the study, add a factory in `programs.py` and register it in
`build_programs()`. For geometry, declare its output support in `output_classes()`
and keep output checks enabled. Change experimental settings in the notebooks.

The fixed compiler inputs are sampled checks, not exhaustive proof. Geometry is
a descriptive study of one seeded probe design. Repeated targets retain weight;
they are not independent trials. Full/computed views are lane selections, and
each program has its own balanced target distribution. CKA compares aligned
observations across stages within a program. Zero-variance CKA is undefined.

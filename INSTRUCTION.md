# RaspFormer continuation instructions for Claude

## Mission

Continue development on this repository without breaking the research workflow or overwriting canonical evidence. This project studies how representation measurements align with compiler-interpretable RASP programs and learned transformer behavior.

Primary goals:
- Maintain reproducibility for all study runs.
- Preserve compiled evidence and reports in `results/`.
- Extend or fix the RASP/compiler/geometry/learning analysis only with clear, testable changes.
- Keep the project runnable with the pinned Python 3.12 environment described in the repo.

## Project summary

This repo defines sequence algorithms in RASP, compiles them with Tracr, and tests what representation measurements reveal about execution.

The main research areas are:
- `programs`: compiled algorithms and program definitions
- `trace`: attention/MLP tracing vs assembled transformer
- `geometry`: PCA, PR, CKA, residual analyses
- `alignment`: RASP value and compiler lane comparisons
- `comparison`: variance, sensitivity, and CKA checks
- `validation`: controls, blind transition recovery, compression, stablized diagnostics
- `learning`: trained transformers on the same programs

The repo is mostly Python, with some TeX and notebook artifacts for reports and analysis.

## Repository structure

```text
README.md
requirements.txt
raspformer/
  programs.py
  compiler.py
  geometry.py
  alignment.py
  models.py
  research.py
  study.py
  report.py
notebooks/
  01_programs.ipynb through 05_comparison.ipynb
results/
  <study>/reference/
  <study>/runs/<UTC>-<id>/
  report/
bench/
  programs.py
  README.md
  interpbench/
scripts/
  check_notebooks.py
  replay_learning.py
  verify_results.py
test_*.py
output/
  pdf/
  latex/
```

## Environment setup

Use the project instructions from the README. The expected environment is:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Use the virtual environment in commands as:

```bash
.venv/bin/python ...
```

Do not assume a global Python environment is valid.

## Main commands

The repo's standard workflow is:

```bash
.venv/bin/python -m raspformer.study
.venv/bin/python -m raspformer.report
```

Run narrower sections with:

```bash
.venv/bin/python -m raspformer.study --sections baseline
.venv/bin/python -m raspformer.study --sections validation
.venv/bin/python -m raspformer.study --sections learning --seeds 42 43 44 --training-steps 600
```

Important: each section replaces its canonical reference only after it completes successfully. Run `raspformer.report` after all sections are done.

## Evidence and output policy

Important rules for any continuation work:

1. Preserve canonical evidence in `results/<study>/reference/`.
2. New local runs should go under `results/<study>/runs/<UTC>-<id>/`.
3. Do not overwrite report artifacts unless the task explicitly requires a new canonical output.
4. Maintain manifests and run metadata when generating results.
5. If a run fails, preserve any completed evidence instead of deleting it.

This repository treats results as research artifacts, not disposable scratch output.

## Validation commands

Before considering work complete, validate with the repository's checks:

```bash
mkdir -p .context/verification
.venv/bin/python -m unittest discover -v > .context/verification/tests.log 2>&1
.venv/bin/python scripts/check_notebooks.py
.venv/bin/python scripts/replay_learning.py
.venv/bin/python scripts/verify_results.py --test-log .context/verification/tests.log --notebook-summary .context/notebook-checks/summary.json
```

The repo contains focused test files:
- `test_studies.py`
- `test_numerics.py`
- `test_models.py`
- `test_research.py`
- `test_bench.py`

## Research workflow expectations

Claude should operate as a careful research coding assistant:

- Prefer targeted fixes over broad rewrites.
- Keep the measured semantics consistent with the project's stated interpretation of the experiments.
- Do not conflate compiler fidelity with independent discovery of the underlying algorithm.
- When changing representations, geometry, or intervention logic, check both numerical behavior and test coverage.
- If modifying a study, keep naming conventions and report generation working.

## Good continuation tasks

Examples of sensible follow-up work without violating the repo's structure:
- reproduce or extend the validation study
- debug failing test cases
- improve numerical stability / variance handling
- inspect and patch learned model comparison logic
- improve reporting or figure generation
- extend benchmark/InterpBench integration
- investigate length-generalization failures in the SIIT or trained models

## Avoid

Avoid these common mistakes:
- overwriting `results/*/reference` without explicit intent
- running broad study sweeps without understanding the output structure
- editing generated reports or PDF output by hand
- assuming notebook results are authoritative without verification
- using system Python instead of `.venv/bin/python`
- removing older evidence before checking whether it is still referenced

## Recommended working style

Claude should:
- read `README.md` and relevant module files before editing
- trace the function or study that is being changed
- keep code readable and clearly aligned with the project terminology
- add or update tests when behavior changes
- summarize what was changed, what was verified, and what remains uncertain

## Short operational checklist

Before finalizing a task:

```bash
# 1) set up environment if needed
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2) run targeted tests
.venv/bin/python -m unittest discover -v

# 3) if the task affects study behavior, inspect a relevant study section
.venv/bin/python -m raspformer.study --sections validation

# 4) regenerate report if needed
.venv/bin/python -m raspformer.report
```

## Important context

This repository is not a generic ML template. It is a research codebase with explicit experimental claims, saved evidence, and a strong emphasis on measurement discipline. Preserve that discipline while continuing development.

The intended workflow is: understand the exact study, modify the smallest relevant code path, validate with focused tests, and then decide whether new evidence should be added as a run under `results/` rather than replacing canonical outputs.

## Final note

If a task does not clearly specify a new objective, start by reading:
- `README.md`
- `raspformer/study.py`
- `raspformer/report.py`
- relevant tests for the section being changed

Then proceed from there with the smallest safe experiment or patch.

---

This file is intended to guide the next Claude session to continue the project sensibly and safely.

# Reviewer response in LaTeX

This source builds the report answering the external review point by point. The scientific evidence is the canonical reference data recorded on 15 October 2026 and committed as `e06479ce09f10695aa5280162f80d880c060eaf5`.

The document includes all ten requested action areas, nine vector figures, exact result tables, negative findings, interpretation limits, verification records, and primary references.

## Compile the supplied source

Run from this directory with Tectonic 0.17.0 or a compatible XeLaTeX installation.

```bash
tectonic report.tex
```

The supplied tables and vector figures make this compile independent of the full data checkout. Tectonic may download its TeX packages on the first run. With a conventional LaTeX installation, run `xelatex report.tex` three times to resolve the contents, citations, and references.

The published PDF is `output/pdf/raspformer-reviewer-response-latex.pdf` in the repository. The source archive has the same name with `-source.zip` instead of `.pdf`.

## Regenerate tables and figures

Regeneration requires the full RaspFormer checkout and its pinned Python environment. Run from the repository root.

```bash
.venv/bin/python output/latex/reviewer-response/generate_tables.py
.venv/bin/python output/latex/reviewer-response/generate_figures.py
```

These scripts only read saved reference evidence. They do not retrain models or rerun the scientific experiments.

`report-data.json` records derived numerical tables and verification metadata. `figures/figure-metadata.json` records each plot's source paths, input hashes, population, aggregation, caption, and limitations. Every plot has a vector PDF and a 300-dpi PNG companion.

## Scope

The reviewer's implementation requests and the experiments' scientific outcomes are distinct. The report does not claim complete algorithm recovery, erasure/lifetime recovery, known learned circuits, length generalization, or a direct InterpBench benchmark result.

The old PDF reports are retained separately. This LaTeX report does not overwrite them.

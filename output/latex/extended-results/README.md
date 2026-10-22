# Extended research results

A readable, single-column LaTeX research report with 11-point body text, nine vector figures, detailed tables, methods, and scientific limitations. It starts with the findings and has no separate cover page or abstract page.

## Build the PDF

Run from this source directory with Tectonic 0.17.0:

```bash
tectonic report.tex
```

All inputs needed to compile the PDF are included. The source package also contains table generation and provenance metadata. Generating tables requires the original repository's saved canonical results and pinned NumPy/Pandas environment. Compiling the included tables does not require those dependencies.

## Evidence and interpretation

Scientific evidence was recorded on 15 October 2026 and is pinned to commit `e06479ce09f10695aa5280162f80d880c060eaf5`. No training or scientific experiment was rerun to create this report.

- Compression uses uncentered SVD, despite the saved `kind=pca` label. Intervention PCA is centered.
- Compression averages over all twelve tasks are separate from transformed write-recovery averages over six held-out tasks.
- Blind thresholds are fixed. The saved implementation does not learn them from the designated calibration program group.
- Patch conditions share 192 pairs. Their 1,152 rows are correlated conditions, and supervised variable mapping is retained.
- High intended-schedule F1 does not guarantee correct execution or identify the changed model's causal computation.

The nine figures reuse the vector files prepared for the earlier detailed reviewer response. Their metadata records the exact saved populations, aggregation, and input hashes. This report adds a research-results narrative and a larger reading format without changing the scientific measurements.

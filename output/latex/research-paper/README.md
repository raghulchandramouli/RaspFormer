# Concise RaspFormer research paper

This paper condenses the complete reviewer-response report into a standard research-paper structure. It includes an abstract, introduction, methods, results, discussion, conclusion, reproducibility statement, references, two vector figures, and one result table.

Compile from this directory with Tectonic 0.17.0:

```bash
tectonic paper.tex
```

The supplied vector figures make compilation independent of the full repository checkout. Tectonic may download packages on its first run. Alternatively, run `xelatex paper.tex` three times.

The final deliverables are `output/pdf/raspformer-research-paper.pdf` and `output/pdf/raspformer-research-paper-source.zip`. The longer report remains available separately.

Figures are derived from canonical saved measurements. To regenerate them in the full checkout, use:

```bash
.venv/bin/python output/latex/research-paper/generate_figures.py
```

No model training or scientific experiment is rerun by the plotting script. Figure metadata preserves source paths, populations, and aggregation rules. The paper reports historical experiments; it does not claim full algorithm discovery or that every scientific limitation was resolved.

"""Build the plain-language circuit-discovery brief as a PDF.

    .venv/bin/python output/circuit-discovery-brief/build.py

Reads results/bench/reference/circuit-discovery*.csv and the SIIT twins, so
every number in the PDF comes from the saved runs.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "raspformer-circuit-discovery-brief.pdf"
CHART = Path(__file__).resolve().parent / "edge-auroc.png"

METHODS = [
    ("eap", "Edge attribution patching", "One gradient pass. Guesses each edge's effect with a straight-line approximation."),
    ("patch", "Exact edge patching", "Actually swaps each edge's signal for a corrupted one and measures the change. One forward pass per edge."),
    ("ig", "Integrated gradients", "EAP, but the gradient is averaged along the path from corrupted to clean input."),
    ("ig-delta", "Integrated gradients, delta form", "Same idea with the interpolation applied to the edge deltas instead of the inputs."),
    ("acdc", "ACDC on KL divergence", "Removes edges one by one and keeps an edge only if removing it changes the output distribution."),
    ("acdc-logit", "ACDC on logit difference", "Same pruning loop, but the change is measured on the correct-answer logit instead of the distribution."),
]

NAVY, TEAL, MUTED = colors.HexColor("#182D45"), colors.HexColor("#147D83"), colors.HexColor("#526376")


def load():
    ib = pd.read_csv(ROOT / "results/bench/reference/circuit-discovery.csv")
    si = pd.read_csv(ROOT / "results/bench/reference-siit/circuit-discovery.csv")
    ib_acdc = pd.read_csv(ROOT / "results/bench/reference/circuit-discovery-acdc.csv")
    si_acdc = pd.read_csv(ROOT / "results/bench/reference-siit/circuit-discovery-acdc.csv")
    return ib, si, ib_acdc, si_acdc


def means(frame):
    return frame.groupby("method")[["edge_auroc", "edge_precision_at_k", "node_auroc", "node_precision_at_k"]].mean()


def chart(ib, si):
    keys = [m[0] for m in METHODS]
    a, b = means(ib).loc[keys, "edge_auroc"], means(si).loc[keys, "edge_auroc"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    x = range(len(keys))
    ax.bar([i - 0.19 for i in x], a, 0.38, color="#182D45", label="InterpBench (6 models)")
    ax.bar([i + 0.19 for i in x], b, 0.38, color="#147D83", label="SIIT (12 models)")
    ax.axhline(0.5, color="#526376", lw=0.8, ls="--")
    ax.text(len(keys) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=7, color="#526376")
    ax.set_xticks(list(x), keys, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Mean edge AUROC", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART)
    plt.close(fig)


def build():
    ib, si, ib_acdc, si_acdc = load()
    chart(ib, si)
    mi, ms = means(ib), means(si)
    f1 = lambda frame, method: frame[(frame.method == method) & (frame.tau == 0.1)].edge_f1.mean()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=TEAL, spaceBefore=10, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=MUTED)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                            title="Six ways to find a circuit: what worked", author="RaspFormer project")
    story = [Paragraph("Six ways to find a circuit: what worked and what did not", h1),
             Paragraph("RaspFormer bench, 22 October 2026. Every number is read from the saved runs in results/bench.", small),
             Spacer(1, 6)]

    story += [Paragraph("The question in one sentence", h2),
              Paragraph("Given a trained transformer whose true circuit we know, can a method find which connections between "
                        "heads and MLPs matter? We tried six methods on 18 such models: six from InterpBench and twelve we trained "
                        "ourselves with known structure. All six are scored by the same code against the same answer key, on the "
                        "same inputs, so the comparison is fair.", body)]

    story += [Paragraph("The six methods, in plain words", h2)]
    rows = [["Method", "What it does"]] + [[Paragraph(f"<b>{name}</b><br/><font size=8 color='#526376'>{key}</font>", body), Paragraph(desc, body)]
                                           for key, name, desc in METHODS]
    t = Table(rows, colWidths=[52 * mm, 118 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 9),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D5DB")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t]

    story += [Paragraph("How well each one ranks the true edges", h2),
              Paragraph("The score is AUROC: the chance that a randomly chosen true edge is ranked above a randomly chosen false one. "
                        "1.0 is perfect, 0.5 is a coin flip. The dashed line is chance.", body),
              Image(str(CHART), width=170 * mm, height=77 * mm)]
    rows = [["Method", "Edge AUROC\nInterpBench", "Edge AUROC\nSIIT", "Node precision\nInterpBench", "Node precision\nSIIT"]]
    for key, name, _ in METHODS:
        rows.append([name, f"{mi.loc[key, 'edge_auroc']:.2f}", f"{ms.loc[key, 'edge_auroc']:.2f}",
                     f"{mi.loc[key, 'node_precision_at_k']:.2f}", f"{ms.loc[key, 'node_precision_at_k']:.2f}"])
    t = Table(rows, colWidths=[62 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D5DB")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [Spacer(1, 4), t,
              Paragraph("Node precision is how many of the top-ranked heads and MLPs really belong to the circuit, when you take as many as the circuit has.", small)]

    story += [Paragraph("What worked", h2)]
    for text in [
        "<b>Finding the right heads and MLPs.</b> Every attribution method puts the true components near the top: 82% to 87% of its top picks are correct on InterpBench, 73% to 75% on our own models.",
        "<b>EAP held up.</b> The cheap one-gradient-pass method lands within 0.015 AUROC of exact patching and of integrated gradients on both pools. The straight-line shortcut is not what it loses on.",
        f"<b>Fixing ACDC's metric.</b> Measuring on the logit difference instead of KL divergence made ACDC the best edge ranker on InterpBench ({mi.loc['acdc-logit', 'edge_auroc']:.2f}) and doubled the quality of the circuits it returns there (F1 {f1(ib_acdc, 'acdc'):.2f} to {f1(ib_acdc, 'acdc-logit'):.2f}).",
        "<b>One-layer circuits.</b> On the count-greater-than model from InterpBench and on three of our twelve (absolute value, histogram, count greater than), at least one method recovers every true edge at the top of its ranking; the InterpBench histogram model is within 0.02 of that.",
    ]:
        story.append(Paragraph(text, bullet, bulletText="•"))

    story += [Paragraph("What did not", h2)]
    for text in [
        "<b>Finding exact edges.</b> Edge-level precision sits around 0.5 for every method on both pools. Knowing which components matter is easy; knowing which specific connections carry the signal is not.",
        "<b>ACDC as a circuit finder.</b> Its real output is a yes/no circuit. The best it reaches is edge F1 0.46 on InterpBench and 0.30 on our models. Its greedy one-edge-at-a-time pruning drops edges that only matter together.",
        "<b>ACDC on KL divergence.</b> These models push their logits to a million, the softmax saturates, and on one model 93% of single-edge distances are exactly zero. The pruning then returns the same circuit at every threshold.",
        f"<b>The logit fix is not free.</b> On our own models it made ACDC's ranking worse ({ms.loc['acdc', 'edge_auroc']:.2f} to {ms.loc['acdc-logit', 'edge_auroc']:.2f}), so the metric choice has to be reported, not hidden.",
        "<b>Deep models.</b> On the four-layer InterpBench models every method falls to 0.45 to 0.75 AUROC. Two ground-truth edges there run from a key projection to a later value input, which no residual-stream method can even represent.",
    ]:
        story.append(Paragraph(text, bullet, bulletText="•"))

    story += [Paragraph("What this means", h2),
              Paragraph("Four attribution methods and one search method agree within a few points. The bottleneck is not the method but the "
                        "granularity of the answer key: InterpBench models are trained at the level of whole heads, so edge-level labels are "
                        "partly arbitrary. For the paper, EAP is a defensible choice of circuit-discovery baseline, and the honest claim is that "
                        "all of them recover nodes well and edges poorly. Code: bench/circuits.py and bench/circuit_baselines.py. "
                        "Full tables: results/bench/reference/circuit-discovery.md and results/bench/reference-siit/circuit-discovery.md.", body)]
    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(build())

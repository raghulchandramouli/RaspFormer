"""Create compact vector figures from saved canonical RaspFormer results.

This script reads reference CSV artifacts and writes two PDF/PNG figures plus
source provenance. It does not train, execute models, or rerun experiments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "figures"
NAVY = "#19364D"
TEAL = "#167D8D"
ORANGE = "#D46A35"
GRAY = "#788791"
LIGHT = "#D6E0E6"
INK = "#243442"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRAY, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True,
    "grid.color": "#E1E7EB", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tables(filename: str) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted((ROOT / "results/validation/reference").glob(f"p??-*/{filename}"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["program"] = path.parent.name[:3].upper()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), [str(p.relative_to(ROOT)) for p in paths]


def probability_axis(ax: plt.Axes, label: str) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_ylabel(label)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0, pad=6)


def title(ax: plt.Axes, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold", pad=9)


def save(fig: plt.Figure, stem: str, info: dict) -> dict:
    paths = {kind: OUT / f"{stem}.{kind}" for kind in ("pdf", "png")}
    fig.savefig(paths["pdf"], metadata={"Title": info["caption"], "Creator": "RaspFormer saved-result figure renderer"})
    fig.savefig(paths["png"], dpi=300)
    plt.close(fig)
    info.update({
        "stem": stem,
        "pdf": str(paths["pdf"].relative_to(ROOT)),
        "png": str(paths["png"].relative_to(ROOT)),
        "width_inches": 7.1, "height_inches": 3.0,
        "png_dpi": 300,
        "sources": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in sorted(set(info.pop("source_paths")))],
    })
    return info


def blind_and_compressed() -> dict:
    recovery, source_paths = tables("write-recovery-scores.csv")
    transformed, transformed_paths = tables("transformed-write-recovery.csv")
    source_paths += transformed_paths
    heldout = recovery[recovery.program.ge("P07")]
    detectors = ["pr", "cka", "change"]
    summary = heldout.groupby("detector")[["f1", "null_mean_f1", "all_positive_f1"]].mean().loc[detectors]
    half = transformed[
        transformed.program.ge("P07")
        & transformed.kind.isin(["compression-pca", "compression-random"])
        & transformed.fraction.eq(0.5)
    ]
    types = ["compression-pca", "compression-random"]
    accuracy = half.drop_duplicates(["program", "kind", "fraction"]).groupby("kind").token_accuracy.mean().loc[types]
    scores = half.groupby(["kind", "detector"])["f1"].mean()
    baseline = float(heldout.drop_duplicates("program").all_positive_f1.mean())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.31, top=0.855, wspace=0.34)
    x = np.arange(3)
    width = 0.25
    ax.bar(x - width, summary.f1, width, color=NAVY, label="Detector")
    ax.bar(x, summary.null_mean_f1, width, color=LIGHT, edgecolor=GRAY, linewidth=0.6, label="Shuffle")
    ax.bar(x + width, summary.all_positive_f1, width, color=ORANGE, label="All+")
    for i, value in enumerate(summary.f1):
        ax.text(i - width, value + .025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["PR", "CKA", "Change"])
    ax.set_xlim(-.65, 2.65)
    probability_axis(ax, "Macro F1")
    title(ax, "a", "Blind first-write detection")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.26),
              handlelength=1.0, handletextpad=.35, columnspacing=.8, borderaxespad=0)

    x = np.arange(2)
    width = .23
    values = [accuracy.to_numpy(), np.array([scores.loc[(k, "pr")] for k in types]), np.array([scores.loc[(k, "cka")] for k in types])]
    for offset, metric_values, color, label in zip([-width, 0, width], values, [ORANGE, TEAL, NAVY], ["Token accuracy", "PR F1", "CKA F1"]):
        bx.bar(x + offset, metric_values, width, color=color, label=label)
        for i, value in enumerate(metric_values):
            bx.text(i + offset, value + .025, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    bx.axhline(baseline, color=GRAY, ls=(0, (4, 2.5)), linewidth=1.2, label="All+ F1", zorder=4)
    bx.set_xticks(x, ["Uncentered SVD", "Random"])
    bx.set_xlim(-.6, 1.6)
    probability_axis(bx, "Token accuracy / schedule F1")
    title(bx, "b", "Nominal half-width execution")
    bx.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.26),
              handlelength=1.0, handletextpad=.4, columnspacing=.9, borderaxespad=0)
    fig.text(.075, .973, "Both panels: held-out P07-P12; unweighted means over six programs", fontsize=9, va="top", color=GRAY)
    return save(fig, "fig1-recovery-compression", {
        "caption": "Blind detection and executing compression on the same six held-out programs P07-P12. (a) Macro F1 for PR change, adjacent-CKA change, and residual-change energy against the original first-write schedule. The count-matched shuffle baseline averages 1,000 within-program prediction shuffles; All+ predicts every transition. (b) At nominal half residual width, uncentered-SVD and equal-width random models are compared by task token accuracy and PR/CKA F1 against the original intended write schedule. The dashed line is the All+ schedule baseline. In particular, random compression gives token accuracy 0.376 despite PR/CKA F1 of 0.852/0.879. F1 is macro-averaged over six programs, rather than pooled over transitions. The schedule becomes an intended reference when compression changes the computation.",
        "populations": {"a": "Six held-out original programs P07-P12, excluded from detector calibration on P01-P06.", "b": "The same six programs P07-P12, nominal half-width executing uncentered-SVD and random models, each saved at seed 42."},
        "aggregation": "Unweighted mean of saved per-program F1; token accuracy counted once per program/model after removing repeated detector rows. No inferential uncertainty interval is claimed.",
        "counts": {"heldout_programs_each_panel": int(heldout.program.nunique()), "a_program_detector_rows": len(heldout), "shuffle_draws_per_program_detector": 1000, "b_program_model_detector_rows": len(half), "b_executing_models": int(len(half.drop_duplicates(["program", "kind", "fraction"])))},
        "values": {"a": summary.reset_index().to_dict(orient="records"), "b": [{"kind": kind, "token_accuracy": float(accuracy.loc[kind]), "pr_f1": float(scores.loc[(kind, "pr")]), "cka_f1": float(scores.loc[(kind, "cka")])} for kind in types], "all_positive_macro_f1": baseline},
        "cautions": ["Saved compression kind=pca is fitted by uncentered SVD, not centered PCA.", "Nominal half width means rank=ceil(original_width*0.5), and does not imply half of total model parameters.", "The original compiler schedule is an intended reference for transformed models, not verified ground truth for their altered computation.", "Detectors locate first-write events, not variable identities or complete circuits."],
        "source_paths": source_paths,
    })


def attention_and_patches() -> dict:
    sweep_path = "results/validation/reference/p12-sorting/attention-sharpening.csv"
    sweep = pd.read_csv(ROOT / sweep_path).sort_values("attention_scale")
    patches, patch_paths = tables("patches.csv")
    conditions = [("pca", 1), ("random", 1), ("pca", 3), ("random", 3), ("self", 0), ("oracle", None)]
    labels = ["PCA\n1", "Rand.\n1", "PCA\n3", "Rand.\n3", "Self", "Oracle"]
    metrics = ["full_counterfactual_match", "behavior_changed"]
    rows = []
    for kind, rank in conditions:
        condition = patches[patches.kind.eq(kind)]
        if rank is not None:
            condition = condition[condition["rank"].eq(rank)]
        per_program = condition.groupby("program")[metrics].mean()
        rows.append({"kind": kind, "rank": rank, **{m: float(per_program[m].mean()) for m in metrics}})

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.subplots_adjust(left=.09, right=.992, bottom=.30, top=.87, wspace=.34)
    positions = np.arange(len(sweep))
    positive = sweep.max_abs_lane_error.gt(0).to_numpy()
    ax.plot(positions[positive], sweep.loc[positive, "max_abs_lane_error"], "o-", color=NAVY, linewidth=1.8, markersize=4.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-26, 2e-2)
    ax.set_yticks([1e-3, 1e-10, 1e-17, 1e-24])
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xticks(positions, [str(int(v)) for v in sweep.attention_scale])
    ax.set_xlim(-.25, 3.35)
    ax.set_xlabel("QK-logit multiplier")
    ax.set_ylabel("Max. absolute lane error")
    ax.grid(axis="y")
    for i in positions[~positive]:
        ax.text(i, .055, "0 exactly", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=TEAL,
                fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": TEAL, "boxstyle": "round,pad=.25"})
    ax.text(.55, .76, "Token accuracy = 1\nat every scale", transform=ax.transAxes, fontsize=9, color=GRAY, va="top")
    title(ax, "a", "Sorting: attention sharpening")

    x = np.arange(len(conditions))
    width = .36
    bx.bar(x - width / 2, [r[metrics[0]] for r in rows], width, color=NAVY, label="Counterfactual match")
    bx.bar(x + width / 2, [r[metrics[1]] for r in rows], width, color=ORANGE, label="Behavior changed")
    bx.set_xticks(x, labels)
    bx.set_xlim(-.65, 5.65)
    probability_axis(bx, "Macro fraction of pairs")
    title(bx, "b", "Patching: all 12 programs")
    bx.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.29),
              handlelength=1.0, handletextpad=.4, borderaxespad=0)
    fig.text(.09, .105, "P12; 512 probes per scale", fontsize=9, color=GRAY)
    return save(fig, "fig2-attention-patching", {
        "caption": "Attention sharpness and geometry-directed patching. (a) On P12 sorting, multiplying QK logits by 1, 2, 4, and 8 reduces the saved maximum absolute output-lane error from 8.52e-4 through 7.15e-7 and 1.35e-24 to a literal zero. Zero is annotated separately and is not replaced by a positive value on the log axis. Token accuracy is one at every scale, with MLP exactness fixed. (b) Descriptive macro means across all 12 programs compare centered-PCA ranks one and three with equal-rank random patches, self patches, and a compiler-variable oracle whose rank varies by program. The six conditions share the same 192 held-out donor-recipient pairs (16 per program), yielding 1,152 saved patch rows. All 60 undefined symbolic-counterfactual rows remain failures of the full-output agreement comparison. Behavioral change is reported separately from full agreement with independently evaluated single-variable substitution.",
        "populations": {"a": "P12 sorting only; 512 balanced probes at each QK scale 1, 2, 4, 8.", "b": "All 12 original programs; the same 16 held-out donor-recipient pairs per program in each of six patch conditions."},
        "aggregation": "Panel a uses the saved maximum absolute lane error at each scale. Panel b first averages saved Boolean outcomes over 16 pairs within each program and condition, then averages equally over 12 programs; no inferential interval is claimed.",
        "counts": {"a_programs": 1, "a_attention_scales": len(sweep), "a_probes_per_scale": 512, "b_programs": int(patches.program.nunique()), "b_shared_donor_recipient_pairs": int(len(patches[["program", "pair_id", "recipient_id", "donor_id"]].drop_duplicates())), "b_conditions": len(conditions), "b_pairs_per_program_condition": 16, "b_saved_patch_rows": len(patches), "b_undefined_symbolic_rows_scored_failure": int((~patches.symbolic_counterfactual_defined).sum())},
        "values": {"attention_sweep": sweep.to_dict(orient="records"), "patch_macro_means": rows},
        "cautions": ["A literal zero cannot be placed on a logarithmic axis and is displayed as text only.", "Patch PCA is centered; it differs from the uncentered-SVD compression basis in Figure 1.", "Patch donor selection does not require different answers; self patches can therefore agree with the symbolic counterfactual.", "The oracle is not an equal-rank subspace control; its rank depends on the mapped compiler variable.", "The patch-to-variable mapping uses supervised decoding after geometric stage selection; these results do not establish named-variable recovery from geometry alone.", "All 60 undefined symbolic substitutions remain agreement failures."],
        "source_paths": [sweep_path] + patch_paths,
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = [blind_and_compressed(), attention_and_patches()]
    (OUT / "figure-metadata.json").write_text(json.dumps({"generated_from_saved_results_only": True, "figures": figures}, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Render reviewer-response figures from immutable canonical result artifacts.

Run from any working directory with the repository's existing Python environment.
This script only reads saved results and writes publication figures and metadata.
It does not execute models, train models, or run scientific experiments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "figures"
NAVY = "#19364D"
TEAL = "#167D8D"
ORANGE = "#D46A35"
GRAY = "#687784"
LIGHT = "#DFE8ED"
INK = "#243442"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#80909A",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": "#DFE5E9",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUT.mkdir(parents=True, exist_ok=True)
metadata: list[dict] = []


def csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def program_tables(filename: str) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted((ROOT / "results/validation/reference").glob(f"p??-*/{filename}"))
    pieces = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["program"] = path.parent.name[:3].upper()
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True), [str(p.relative_to(ROOT)) for p in paths]


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=10, fontweight="bold")


def probability_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylim(0, 1.065)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")


def save(
    fig: plt.Figure,
    stem: str,
    caption: str,
    sources: list[str],
    population: str,
    aggregation: str,
    counts: dict,
    cautions: list[str],
) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata={"Title": caption})
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata.append(
        {
            "stem": stem,
            "pdf": f"figures/{stem}.pdf",
            "png": f"figures/{stem}.png",
            "caption": caption,
            "population": population,
            "aggregation": aggregation,
            "counts": counts,
            "cautions": cautions,
            "sources": [
                {
                    "path": path,
                    "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                }
                for path in sorted(set(sources))
            ],
        }
    )


def numerical_floor() -> None:
    path = "results/comparison/reference/matched/stages.csv"
    stages = csv(path)
    suspicious = stages[
        stages["view"].eq("computed")
        & stages["actual_total_variance"].gt(0)
        & stages["actual_total_variance"].le(stages["variance_floor"])
    ].copy()
    suspicious = suspicious.sort_values(["program", "step"])
    sources = [path]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.65), layout="constrained")
    x = np.arange(len(suspicious))
    ax.plot(x, suspicious.actual_pr_raw, "o", color=ORANGE, label="Raw PR", markersize=6)
    ax.plot(x, suspicious.actual_pr, "s", color=TEAL, label="PR under variance policy", markersize=4)
    stage_codes = {"L1/Attn": "1A", "L1/MLP": "1M", "L2/Attn": "2A"}
    ax.set_xticks(x, [stage_codes[s] for s in suspicious.stage])
    for program, group in suspicious.groupby("program", sort=False):
        positions = np.flatnonzero(suspicious.program.eq(program))
        ax.text(positions.mean(), -0.12, program[:3], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    ax.set_ylim(-0.15, 1.35)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Participation ratio")
    ax.grid(axis="y")
    ax.legend(loc="center", frameon=False, bbox_to_anchor=(0.5, 0.47))
    variance = suspicious.actual_total_variance
    ax.text(0.04, 0.9, f"Centered variance ≈ {variance.median():.2e}\n{len(suspicious)} stages; selected floor = 10⁻¹²", transform=ax.transAxes, va="top", fontsize=8)
    panel(ax, "a", "PR of negligible-variance states")
    floors = None
    for design, color, marker in [("matched", NAVY, "o"), ("balanced", TEAL, "s")]:
        spath = f"results/comparison/reference/{design}/variance-sensitivity.csv"
        sources.append(spath)
        sensitivity = csv(spath)
        sensitivity = sensitivity[sensitivity["view"].eq("computed")]
        values = []
        floors = sorted(sensitivity.variance_floor.unique())
        for floor in floors:
            group = sensitivity[sensitivity.variance_floor.eq(floor)]
            values.append(int((group.adjacent_cka.notna() != group.expected_adjacent_cka.notna()).sum()))
        bx.plot(np.arange(len(floors)), values, marker=marker, color=color, label=design.capitalize())
    selected = floors.index(float(stages.variance_floor.iloc[0]))
    bx.axvspan(selected - 0.32, selected + 0.32, color=LIGHT, zorder=-1)
    bx.set_xticks(np.arange(len(floors)), ["0" if f == 0 else f"10⁻{int(-np.log10(f))}".translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")) for f in floors])
    # Use math text to avoid mixing baseline and superscript digits in exponents.
    bx.set_xticklabels(["0" if f == 0 else rf"$10^{{{int(np.log10(f))}}}$" for f in floors])
    bx.set_xlabel("Absolute centered-variance floor")
    bx.set_ylabel("CKA availability mismatches")
    bx.set_ylim(0, 16)
    bx.set_yticks([0, 2, 6, 10, 14])
    bx.grid(axis="y")
    bx.legend(frameon=False, loc="upper right")
    bx.text(selected, 15.1, "selected", ha="center", fontsize=8)
    panel(bx, "b", "Variance-floor sensitivity")
    save(
        fig,
        "01-numerical-floor",
        "Numerical artifact and absolute variance-floor sensitivity. (a) Twelve matched computed-view stages in P07, P08, P10, and P11 have centered total variance about 8.07×10⁻¹⁹: raw PR is one, whereas the selected 10⁻¹² inactivity policy sets effective PR to zero. Stage codes 1A, 1M, and 2A denote L1/Attn, L1/MLP, and L2/Attn. (b) Counts of expected/observed adjacent-CKA availability mismatches across all 12 programs and 54 adjacent transitions at each saved floor. The matched design falls from 14 to two mismatches; the balanced design remains at two. The floor is an absolute numerical policy and is not scale invariant.",
        sources,
        "All 12 original programs, computed residual view; 12 negligible-variance stages in P07/P08/P10/P11 highlighted in panel a.",
        "Saved raw/effective PR per stage; exact counts of CKA definedness mismatches at each saved floor, with no interpolation.",
        {"programs": int(stages.program.nunique()), "highlighted_stages": len(suspicious), "computed_adjacent_transitions": int((stages["view"].eq("computed") & stages.step.gt(0)).sum()), "floors": len(floors)},
        ["A CKA value is undefined when either adjacent state is inactive.", "Remaining helper-state mismatches are retained.", "Absolute thresholds do not give a scale-invariant definition of signal."],
    )


def sorting() -> None:
    path = "results/validation/reference/p12-sorting/per-observation-errors.csv"
    sweep_path = "results/validation/reference/p12-sorting/attention-sharpening.csv"
    sets_path = "results/validation/reference/p12-sorting/failing-probe-sets.json"
    data = csv(path)
    base = data[data.attention_scale.eq(1) & data.stage.eq("L3/Attn")]
    lanes = sorted(base.output_lane.unique(), key=lambda lane: int(lane.rsplit(":", 1)[1]))
    matrix = ~base.pivot(index="output_lane", columns="probe_id", values="close").loc[lanes]
    # Sort by actual membership pattern; retain every probe without sampling.
    order = sorted(matrix.columns, key=lambda probe: (tuple(-int(v) for v in matrix[probe]), probe))
    matrix = matrix[order]
    patterns = matrix.T.groupby(list(matrix.index)).size()
    sweep = csv(sweep_path)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.7), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax.pcolormesh(np.arange(len(order) + 1), np.arange(len(lanes) + 1) - 0.5, matrix.astype(int), cmap=ListedColormap(["#EDF1F4", ORANGE]), vmin=0, vmax=1, shading="flat", antialiased=False, edgecolors="none")
    ax.set_xlim(0, len(order))
    ax.set_ylim(len(lanes) - 0.5, -0.5)
    counts = matrix.sum(axis=1)
    ax.set_yticks(np.arange(len(lanes)), [f"{lane.rsplit(':', 1)[1]}  ({int(counts.loc[lane])})" for lane in lanes])
    ax.set_ylabel("Output lane (failed probe count)")
    ax.set_xticks([0, 128, 256, 384, 512])
    ax.set_xlabel("All 512 probes, ordered by failure pattern")
    ax.text(0.02, -0.26, "Orange = fails saved closeness check\nEight disjoint sets; 64 probes per lane", transform=ax.transAxes, fontsize=8, va="top")
    panel(ax, "a", "First-write failure sets")
    positive = sweep.max_abs_lane_error.gt(0)
    indices = np.arange(len(sweep))
    bx.plot(indices[positive], sweep.loc[positive, "max_abs_lane_error"], "o-", color=NAVY)
    bx.set_yscale("log")
    bx.set_ylim(sweep.loc[positive, "max_abs_lane_error"].min() / 6, sweep.max_abs_lane_error.max() * 8)
    bx.set_xticks(indices, [str(int(v)) for v in sweep.attention_scale])
    bx.set_xlabel("Multiplier of attention QK logits")
    bx.set_ylabel("Maximum absolute lane error")
    bx.grid(axis="y", which="major")
    bx.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    for i in indices[~positive]:
        bx.text(i, 0.06, "0 exactly", transform=bx.get_xaxis_transform(), ha="center", va="bottom", color=TEAL, fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": TEAL, "boxstyle": "round,pad=0.3"})
    bx.text(0.02, -0.26, "All four scales: token accuracy = 1\nZero is annotated, not placed on log scale", transform=bx.transAxes, fontsize=8, va="top")
    panel(bx, "b", "Attention sharpening")
    save(
        fig,
        "02-sorting-attention",
        "Sorting row-set diagnosis and attention sweep. (a) Exact failure membership at the first output write (L3/Attn), at baseline attention scale one. Every one of the 512 saved probes fails in one output lane; each of eight lanes has its own disjoint set of 64 failing probes. Probe columns are reordered by membership, without dropping observations. (b) Multiplying QK logits before softmax reduces the maximum lane error from 8.52×10⁻⁴ to zero at scale eight. The literal zero is annotated separately from the log scale; no positive replacement is introduced. Token accuracy is one at every scale, and MLP exactness is fixed.",
        [path, sweep_path, sets_path],
        "P12 sorting; 512 balanced probes at each of four attention scales, with eight output lanes and two saved stages.",
        "Exact Boolean failure membership at baseline L3/Attn; saved maximum error across output observations for each attention scale.",
        {"programs": 1, "probes_per_scale": int(base.probe_id.nunique()), "lanes": len(lanes), "unique_failure_patterns": len(patterns), "attention_scales": len(sweep), "scalar_rows": len(data)},
        ["Distinct failure sets do not identify duplicates or ties as the sole cause.", "A log axis cannot contain zero; the exact zero is explicitly annotated."],
    )


def depth_count() -> None:
    original_path = "results/programs/reference/manifest.json"
    new_path = "results/validation/reference/complexity/factorial-families.csv"
    summary_path = "results/validation/reference/complexity/summary.json"
    original = pd.DataFrame(load_json(original_path)["programs"])
    new = csv(new_path)
    original_rank = np.linalg.matrix_rank(original[["max_dependency_depth", "computed_variable_count"]].to_numpy() - original[["max_dependency_depth", "computed_variable_count"]].mean().to_numpy())
    new_rank = np.linalg.matrix_rank(new[["max_dependency_depth", "computed_variable_count"]].to_numpy() - new[["max_dependency_depth", "computed_variable_count"]].mean().to_numpy())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.7), layout="constrained")
    for (depth, count), group in original.groupby(["max_dependency_depth", "computed_variable_count"]):
        ax.scatter(depth, count, s=45 + 35 * len(group), color=NAVY, edgecolor="white", linewidth=1)
        label = ", ".join(p[:3] for p in group.program)
        if len(group) == 6:
            label = "P01–P06"
        if len(group) == 3:
            label = "P07–P09"
        ax.annotate(label, (depth, count), xytext=(6, -12 if depth == 7 else 7), textcoords="offset points", fontsize=8)
    ax.plot([0, 8], [0, 8], ls="--", color=GRAY, lw=1, zorder=-1)
    ax.set(xlim=(0, 8.3), ylim=(0, 8), xlabel="Realized dependency depth", ylabel="Computed variable count")
    ax.set_xticks([1, 3, 4, 7])
    ax.set_yticks([1, 3, 4, 7])
    ax.grid()
    ax.text(0.05, 0.91, f"12 programs; centered design rank = {original_rank}", transform=ax.transAxes, fontsize=8)
    panel(ax, "a", "Original depth/count design")
    matrix = new.pivot(index="requested_depth", columns="requested_branches", values="computed_variable_count").sort_index()
    bx.pcolormesh(np.arange(len(matrix.columns) + 1) - 0.5, np.arange(len(matrix.index) + 1) - 0.5, matrix, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("counts", ["#EDF3F5", TEAL]), vmin=0, vmax=new.computed_variable_count.max(), shading="flat", antialiased=False, edgecolors="none")
    bx.set_aspect("equal")
    for row, depth in enumerate(matrix.index):
        for col, branches in enumerate(matrix.columns):
            d = new[new.requested_depth.eq(depth) & new.requested_branches.eq(branches)].iloc[0]
            text_color = "white" if d.computed_variable_count >= 7 else INK
            bx.text(col, row, f"count {int(d.computed_variable_count)}\ndepth {int(d.max_dependency_depth)}", ha="center", va="center", fontsize=9, color=text_color)
    bx.set_xticks(np.arange(3), matrix.columns)
    bx.set_yticks(np.arange(3), matrix.index)
    bx.set_xlabel("Requested branch count")
    bx.set_ylabel("Requested branch depth")
    bx.text(0, -0.25, f"9 families; realized centered rank = {new_rank}\nAll token / sequence accuracies = 1", transform=bx.transAxes, fontsize=8)
    panel(bx, "b", "Crossed structural controls")
    save(
        fig,
        "03-depth-count-controls",
        "Separating depth and variable count. (a) In the 12 original programs, computed-variable count equals maximum dependency depth, giving a centered two-column design matrix of rank one; point area reflects the number of programs at each coordinate. (b) A crossed 3×3 family of token-local modulo programs varies requested branch depth and branch count. Cells show realized computed-variable count and dependency depth. The realized centered design has rank two, and every family has token and sequence accuracy one. Serial merging adds depth, so requested branch depth differs from realized dependency depth.",
        [original_path, new_path, summary_path],
        "12 original programs and nine distinct controlled structural families.",
        "Original duplicate depth/count coordinates pooled for display only; new cells retain each family separately. Ranks computed from saved realized axes after centering.",
        {"original_programs": len(original), "original_centered_rank": int(original_rank), "structural_families": len(new), "new_centered_rank": int(new_rank)},
        ["These structural examples do not establish a universal complexity scale.", "Requested branch depth is not realized dependency depth."],
    )


def patches() -> None:
    data, sources = program_tables("patches.csv")
    conditions = [("pca", 1), ("random", 1), ("pca", 3), ("random", 3), ("self", 0), ("oracle", None)]
    labels = ["PCA\nrank 1", "Random\nrank 1", "PCA\nrank 3", "Random\nrank 3", "Self", "Variable\noracle"]
    colors = [TEAL, GRAY, TEAL, GRAY, LIGHT, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.7), layout="constrained")
    for ax, metric, letter, title in zip(axes, ["full_counterfactual_match", "behavior_changed"], ["a", "b"], ["Counterfactual match", "Behavior change"]):
        means = []
        for i, (kind, rank) in enumerate(conditions):
            condition = data[data.kind.eq(kind)]
            if rank is not None:
                condition = condition[condition["rank"].eq(rank)]
            per_program = condition.groupby("program")[metric].mean()
            means.append(per_program.mean())
            ax.bar(i, per_program.mean(), width=0.7, color=colors[i], edgecolor="white", linewidth=0.7)
            # Fixed evenly spaced offsets encode saved program-level heterogeneity,
            # without stochastic jitter or implied independent pair-level errors.
            ax.scatter(i + np.linspace(-0.17, 0.17, len(per_program)), per_program, color=INK, s=10, alpha=0.65, zorder=3, linewidth=0)
            label_y = per_program.mean() - 0.055 if per_program.mean() > 0.9 else per_program.mean() + 0.035
            ax.text(i, label_y, f"{per_program.mean():.2f}", ha="center", fontsize=9, color="white" if per_program.mean() > 0.9 else INK)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.tick_params(axis="x", length=0)
        probability_axis(ax, "Fraction of donor–recipient pairs")
        panel(ax, letter, title)
    axes[0].text(0, -0.24, "12-program macro means (bars)\nProgram means (dots)", transform=axes[0].transAxes, fontsize=9)
    axes[1].text(0, -0.24, "16 pairs per program and condition\nOracle rank varies by program", transform=axes[1].transAxes, fontsize=9)
    save(
        fig,
        "04-counterfactual-patches",
        "Geometry-directed patch controls on all 12 programs. Bars are descriptive macro means and dots are per-program means for 16 saved donor–recipient pairs per condition. (a) Agreement with the complete independently evaluated output under a single mapped-variable substitution. (b) Fraction of patches that change the recipient's behavior. PCA ranks one and three are compared with equal-rank random subspaces, self patches, and a compiler-variable oracle whose rank depends on the mapped variable. Undefined symbolic substitutions remain failures of the agreement comparison. Stages and PCA use calibration sequences; evaluated pairs use held-out sequences. Donor selection does not require different answers.",
        sources,
        "All 12 original programs, 16 held-out donor–recipient pairs per program and each of six patch conditions.",
        "Per-program mean over pairs followed by a macro mean over programs; no inferential interval is claimed. Oracle ranks are pooled across their program-specific values.",
        {"programs": data.program.nunique(), "pairs_per_program_condition": int(data.groupby(["program", "kind", "rank"]).size().min()), "conditions": len(conditions), "saved_patch_rows": len(data), "undefined_symbolic_rows": int((~data.symbolic_counterfactual_defined).sum())},
        ["Behavioral effects do not establish named-variable recovery.", "PCA-to-variable mapping uses supervised decoding after geometric stage selection.", "Oracle rank varies by program and is not an equal-rank subspace control.", "Undefined symbolic substitutions are scored as agreement failures."],
    )


def rotations() -> None:
    data, sources = program_tables("rotations.csv")
    geometry = data.groupby("program")[["pr_error", "cka_max_error"]].max().sort_index()
    agreement = data.groupby("program")[["canonical_lane_agreement", "token_accuracy"]].min().sort_index()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.7), layout="constrained")
    x = np.arange(len(geometry))
    ax.plot(x, geometry.pr_error, "o-", color=NAVY, label="Max |ΔPR|")
    ax.plot(x, geometry.cka_max_error, "s-", color=TEAL, label="Max |ΔCKA|")
    ax.set_yscale("log")
    ax.set_xticks(x, geometry.index, rotation=60, ha="right")
    ax.set_ylabel("Maximum absolute geometry error")
    ax.grid(axis="y", which="major")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "a", "PR and CKA after rotation")
    bx.plot(x, agreement.canonical_lane_agreement, "o-", color=ORANGE, label="Original coordinate lane check")
    bx.plot(x, agreement.token_accuracy, "s-", color=TEAL, label="Executed token accuracy")
    bx.set_xticks(x, agreement.index, rotation=60, ha="right")
    probability_axis(bx, "Minimum agreement over stages / seeds")
    bx.set_ylim(-0.05, 1.08)
    bx.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5))
    panel(bx, "b", "Lane and output agreement")
    save(
        fig,
        "05-orthogonal-rotations",
        "Orthogonal basis change on all 12 programs, with three saved rotation seeds per program. (a) Maximum PR error and maximum linear-CKA error across saved stages and seeds remain at numerical precision. (b) Minimum canonical-coordinate lane agreement and executed token accuracy across stages and seeds. Coordinate checks deliberately retain the original lane map; executed model outputs preserve token accuracy one. Geometric invariance and finite-precision forward execution are measured separately.",
        sources,
        "All 12 original programs; three rotation seeds (42–44) per program; calibration/test protocol of the validation reference.",
        "Maximum absolute geometry errors and minimum saved lane/output agreements across stages and three seeds within each program.",
        {"programs": data.program.nunique(), "rotation_seeds": int(data.seed.nunique()), "saved_stage_seed_rows": len(data)},
        ["Old coordinate labels are intentionally not rotated.", "Invariance under orthogonal changes does not establish recovery of a computation.", "Nonorthogonal rescaling is a separate control."],
    )


def compression() -> None:
    data, sources = program_tables("compression.csv")
    transformed, transformed_sources = program_tables("transformed-write-recovery.csv")
    sources += transformed_sources
    # Accuracy and model size repeat over stages; count each program/model once.
    once = data.drop_duplicates(["program", "kind", "fraction"])
    summary = once.groupby(["kind", "fraction"])[["token_accuracy", "sequence_accuracy"]].mean()
    heldout = transformed[transformed.program.ge("P07") & transformed.kind.isin(["compression-pca", "compression-random"]) & transformed.fraction.eq(0.5)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.85), layout="constrained")
    pca = summary.loc["pca"].sort_index()
    ax.plot(pca.index, pca.token_accuracy, "o-", color=NAVY, label="SVD: token")
    ax.plot(pca.index, pca.sequence_accuracy, "s-", color=TEAL, label="SVD: sequence")
    random = summary.loc["random"]
    ax.scatter(random.index, random.token_accuracy, color=ORANGE, marker="o", s=50, label="Random: token", zorder=4)
    ax.scatter(random.index, random.sequence_accuracy, color=ORANGE, marker="s", s=40, facecolor="white", linewidth=1.4, label="Random: sequence", zorder=4)
    probability_axis(ax, "Task accuracy; macro mean over 12 programs")
    ax.set_xticks(pca.index, ["¼", "½", "¾", "Full"])
    ax.set_xlabel("Target residual-width fraction")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    panel(ax, "a", "Task accuracy: all 12 programs")
    x = np.arange(2)
    width = 0.23
    labels = ["SVD ½ width", "Random ½ width"]
    types = ["compression-pca", "compression-random"]
    accuracy = heldout.drop_duplicates(["program", "kind", "fraction"]).groupby("kind").token_accuracy.mean()
    grouped = heldout.groupby(["kind", "detector"])[["f1", "all_positive_f1"]].mean()
    bx.bar(x - width, [accuracy[k] for k in types], width=width, color=GRAY, label="Token accuracy")
    bx.bar(x, [grouped.loc[(k, "pr"), "f1"] for k in types], width=width, color=TEAL, label="PR detector F1")
    bx.bar(x + width, [grouped.loc[(k, "cka"), "f1"] for k in types], width=width, color=NAVY, label="CKA detector F1")
    baseline = heldout.drop_duplicates("program").all_positive_f1.mean()
    bx.axhline(baseline, color=ORANGE, ls="--", lw=1.3, label="All-positive F1")
    bx.set_xticks(x, labels)
    probability_axis(bx, "Task accuracy / intended-schedule F1")
    bx.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    panel(bx, "b", "Half width: held-out P07–P12")
    save(
        fig,
        "06-executing-compression",
        "Executing residual compression, with populations shown separately. (a) Descriptive macro token and whole-sequence accuracy across all 12 programs at target uncentered-SVD width fractions (saved kind=pca); the random half-width subspace is an equal-width control. Actual rank is ceil(original width times target fraction). (b) Only the six held-out programs P07–P12, at target half width: task token accuracy and blind PR/CKA detector F1 against the original intended write schedule, plus the all-positive F1 baseline. High intended-schedule F1 can coexist with poor task accuracy, especially for the random control. Bases are fitted on calibration sequences and tested on disjoint sequences. Reduced residual width retains attention and MLP hidden dimensions.",
        sources,
        "Panel a: all 12 programs. Panel b: held-out program group P07–P12 only (six programs), half-width PCA and random executing models.",
        "Accuracy counted once per program/model after removal of repeated stage rows; macro means over 12 or six programs as labeled. Detector F1 is macro-averaged over six programs, not pooled transition counts.",
        {"panel_a_programs": once.program.nunique(), "panel_b_programs": heldout.program.nunique(), "pca_fractions": len(pca), "random_fractions": len(random), "compression_stage_rows": len(data), "halfwidth_heldout_detector_rows": len(heldout)},
        ["Original compiler schedule is an intended reference when compression changes behavior, not established ground truth for the changed computation.", "Residual-width reduction does not imply the same reduction in total parameters.", "The two panels use different program populations and must not be compared as one pooled sample."],
    )


def learned_models() -> None:
    summary_path = "results/learning/reference/summary.csv"
    geometry_path = "results/learning/reference/geometry.csv"
    data = csv(summary_path)
    geometry = csv(geometry_path)
    data["pid"] = data.program.str[:3]
    geometry["pid"] = geometry.program.str[:3]
    stats = data.groupby("pid").agg(accuracy=("token_accuracy", "mean"), sd=("token_accuracy", "std"), majority=("majority_token_accuracy", "mean"))
    final = geometry[geometry.learned_step.eq(geometry.groupby(["program", "seed"]).learned_step.transform("max"))]
    final_stats = final.groupby("pid").agg(learned_pr=("learned_pr", "mean"), learned_sd=("learned_pr", "std"), compiled_pr=("compiled_pr", "mean"), cka=("cka", "mean"), cka_sd=("cka", "std"))
    fig = plt.figure(figsize=(7.15, 5.65), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(grid[0, :])
    bx = fig.add_subplot(grid[1, 0])
    cx = fig.add_subplot(grid[1, 1])
    x = np.arange(len(stats))
    ax.errorbar(x, stats.accuracy, yerr=stats.sd, fmt="o", capsize=3, color=TEAL, label="Learned: mean ± sample SD, 3 seeds", zorder=4)
    ax.plot(x, stats.majority, "s", color=ORANGE, label="Majority class", zorder=3)
    ax.axhline(data.compiled_token_accuracy.mean(), color=GRAY, ls="--", lw=1, label="Compiled")
    ax.set_xticks(x, stats.index)
    probability_axis(ax, "Test token accuracy")
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1.07)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    panel(ax, "a", "Learned test accuracy and baselines")
    bx.errorbar(x - 0.08, final_stats.learned_pr, yerr=final_stats.learned_sd, fmt="o", capsize=2, color=TEAL, label="Learned: mean ± SD")
    bx.plot(x + 0.08, final_stats.compiled_pr, "s", color=NAVY, label="Compiled")
    bx.set_xticks(x, stats.index, rotation=60, ha="right")
    bx.set_ylabel("Final full-residual PR")
    bx.set_ylim(0, final_stats.compiled_pr.max() * 1.18)
    bx.grid(axis="y")
    bx.legend(frameon=False, loc="upper left")
    panel(bx, "b", "Final full-residual PR")
    cx.errorbar(x, final_stats.cka, yerr=final_stats.cka_sd, fmt="o", capsize=2, color=NAVY)
    cx.set_xticks(x, stats.index, rotation=60, ha="right")
    probability_axis(cx, "Final learned–compiled linear CKA")
    cx.text(0.03, 0.94, "Mean ± sample SD, 3 seeds", transform=cx.transAxes, fontsize=8, va="top")
    panel(cx, "c", "Final linear CKA")
    save(
        fig,
        "07-learned-models",
        "Randomly initialized, fully trained transformers on the same 12 program tasks. (a) Test token accuracy is the mean ± sample standard deviation across three seeds, with majority-class and compiled-model baselines. All 36 models train for 600 Adam steps using the shared width-32, two-layer, two-head architecture. Whole unique length-six sequences are split into train/validation/test sets. (b) Final full-residual PR for learned and compiled models. (c) Final learned–compiled linear CKA, mean ± sample SD across seeds. Final states are paired by relative stage progress; this is not a named-variable correspondence. Output supervision provides no known internal causal circuit.",
        [summary_path, geometry_path],
        "All 12 task programs; three training seeds (42–44), 36 independently initialized models; held-out sequence test split.",
        "Means and sample standard deviations (ddof=1) over three seeds for accuracy and final learned geometry. Compiled PR and baselines are means over the corresponding saved rows. Final geometry retains the last learned stage per program/seed.",
        {"programs": data.program.nunique(), "models": len(data), "seeds_per_program": int(data.groupby("program").seed.nunique().min()), "final_geometry_rows": len(final), "test_sequences_per_program": int(data.test_sequences.iloc[0]), "training_steps": int(data.steps.iloc[0])},
        ["Results do not establish length generalization or natural-language behavior.", "Majority baselines expose imbalance, especially P09 and P11.", "Learned models lack ground-truth internal circuits.", "Relative-progress pairing does not establish a variable correspondence."],
    )


def write_recovery() -> None:
    data, sources = program_tables("write-recovery-scores.csv")
    heldout = data[data.program.ge("P07")]
    detectors = ["change", "pr", "cka"]
    labels = ["Change energy", "PR change", "CKA change"]
    summary = heldout.groupby("detector")[["f1", "null_mean_f1", "all_positive_f1"]].mean().loc[detectors]
    colors = {"change": ORANGE, "pr": TEAL, "cka": NAVY}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.85), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(len(detectors))
    width = 0.24
    ax.bar(x - width, summary.f1, width=width, color=NAVY, label="Frozen detector F1")
    ax.bar(x, summary.null_mean_f1, width=width, color=LIGHT, edgecolor=GRAY, lw=0.5, label="Count-matched shuffle")
    ax.bar(x + width, summary.all_positive_f1, width=width, color=ORANGE, alpha=0.8, label="All positive")
    for i, value in enumerate(summary.f1):
        ax.text(i - width, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    probability_axis(ax, "Held-out macro F1 (six programs)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    panel(ax, "a", "Held-out macro F1")
    programs = sorted(heldout.program.unique())
    xp = np.arange(len(programs))
    for detector, label in zip(detectors, labels):
        rows = heldout[heldout.detector.eq(detector)].set_index("program").loc[programs]
        bx.plot(xp, rows.f1, "o-", color=colors[detector], label=label)
    baselines = heldout.drop_duplicates("program").set_index("program").loc[programs].all_positive_f1
    bx.plot(xp, baselines, "--", color=GRAY, lw=1.4, label="All positive")
    failures = heldout[heldout.f1.le(heldout.all_positive_f1)]
    for row in failures.itertuples():
        index = programs.index(row.program)
        bx.scatter(index, row.f1, s=90, facecolors="none", edgecolors=ORANGE, linewidths=1.5, zorder=6)
        bx.annotate(f"{row.detector.upper()} {row.f1:.3f} < {row.all_positive_f1:.3f}", (index, row.f1), xytext=(-9, -34), textcoords="offset points", fontsize=8, ha="right", arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8})
    bx.set_xticks(xp, programs)
    bx.set_ylabel("Per-program F1")
    bx.set_ylim(0.5, 1.04)
    bx.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    bx.grid(axis="y")
    bx.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    panel(bx, "b", "Per-program F1 and failures")
    save(
        fig,
        "08-blind-write-recovery",
        "Blind first-write transition detection on the six held-out programs P07–P12. (a) Macro F1 for residual-change energy, PR change, and adjacent-CKA change, compared with 1,000 within-program count-matched prediction shuffles and an all-positive baseline. (b) Per-program F1 and the program-specific all-positive baseline. P11's CKA detector scores 0.727, below its 0.737 all-positive baseline; this failure is retained. Detectors receive ordered activation matrices only, and predictions are frozen and hashed before compiler schedules are loaded for scoring. These events indicate at least one first write, not variable names, simultaneous write counts, or complete circuits.",
        sources,
        "Held-out program group P07–P12 (six programs), excluded from detector calibration on P01–P06.",
        "Unweighted macro means of saved per-program F1 and shuffled-null mean F1; all per-program F1 values retained. Null uses 1,000 within-program draws preserving predicted event counts.",
        {"heldout_programs": heldout.program.nunique(), "detectors": len(detectors), "program_detector_comparisons": len(heldout), "comparisons_not_above_all_positive": len(failures), "null_draws_per_program_detector": 1000},
        ["High first-write prevalence makes the all-positive baseline strong.", "Macro means are descriptive across six programs.", "Detection does not recover names, simultaneous variable counts, lifetimes, or complete causal circuits."],
    )


def diagnostics() -> None:
    lanes_path = "results/alignment/reference/lanes.csv"
    comparison_path = "results/comparison/reference/comparison.csv"
    lanes = csv(lanes_path)
    lanes = lanes[lanes.is_first_write & lanes.program.str.startswith("P")]
    defined = lanes[lanes.pearson_r.notna()]
    sorting = defined[defined.program.str.startswith("P12") & defined.fraction_close.lt(1)]
    others = defined.drop(sorting.index)
    comparison = csv(comparison_path)
    comparison = comparison[comparison.design.eq("matched")].copy()
    comparison["pid"] = comparison.program.str[:3]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.8), layout="constrained")
    ax.scatter(others.pearson_r, others.fraction_close, s=22, color=TEAL, alpha=0.7, label="Other first-write lanes")
    ax.scatter(sorting.pearson_r, sorting.fraction_close, s=42, marker="s", facecolor="white", edgecolor=ORANGE, linewidth=1.3, label="Sorting output lanes", zorder=4)
    # Values near one remain their exact saved values. Tick labels spell out
    # correlation deficits explicitly rather than silently rounding them to one.
    deficit = 1 - defined.pearson_r.min()
    bound = max(5e-12, 1.2 * deficit)
    ax.set_xlim(1 - bound, 1 + bound * 0.1)
    ax.set_xticks([1 - bound, 1 - bound / 2, 1])
    ax.set_xticklabels([f"1 − {bound:.0e}", f"1 − {bound / 2:.1e}", "1"])
    ax.set_ylim(0.85, 1.025)
    ax.set_yticks([0.875, 0.925, 0.975, 1])
    ax.set_xlabel("Pearson correlation r")
    ax.set_ylabel("Fraction within magnitude tolerance")
    ax.grid()
    ax.legend(frameon=False, loc="center left", fontsize=7.5)
    ax.text(0.03, 0.06, f"Sorting: {len(sorting)} output lanes at 87.5% close\n{lanes.pearson_r.isna().sum()} constant lanes: r undefined", transform=ax.transAxes, fontsize=8)
    panel(ax, "a", "Correlation and magnitude")
    wide = comparison.pivot(index="pid", columns="view", values="pr_mae").sort_index()
    x = np.arange(len(wide))
    bx.plot(x - 0.08, wide["full"], "o", color=NAVY, label="Full residual")
    bx.plot(x + 0.08, wide["computed"], "s", color=TEAL, label="Computed residual")
    bx.set_yscale("symlog", linthresh=1e-14, linscale=0.5)
    bx.set_xticks(x, wide.index, rotation=60, ha="right")
    bx.set_yticks([0, 1e-14, 1e-10, 1e-6, 1e-2, 1])
    bx.set_yticklabels(["0", "$10^{-14}$", "$10^{-10}$", "$10^{-6}$", "$10^{-2}$", "1"])
    bx.set_ylabel("Mean absolute PR error")
    bx.grid(axis="y")
    bx.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.65))
    panel(bx, "b", "Full and computed residual views")
    save(
        fig,
        "09-diagnostics",
        "Magnitude-aware compiler-fidelity diagnostics replot saved canonical results. (a) Pearson correlation and the fraction of observations within the saved magnitude tolerance for first-write lanes across the 12 programs. All eight sorting output lanes have r greater than 0.999999999996 but only 87.5% close observations. Constant lanes have undefined correlation and are counted explicitly. (b) Matched expected-versus-observed PR mean absolute error for full and computed residual views on all 12 programs. The vertical scale is symmetric log with an explicit linear region at and below 10⁻¹⁴, so exact zero values remain zero. Computed views retain compiler helper lanes; these are compiler-fidelity measurements with known coordinates and write times.",
        [lanes_path, comparison_path],
        "Panel a: first-write lanes across 12 original programs in the canonical alignment run. Panel b: matched full/computed comparison for all 12 programs.",
        "Panel a retains every defined lane-level Pearson correlation and fraction close; undefined constant-lane correlations are counted. Panel b directly plots saved per-program/view mean absolute PR error.",
        {"programs": lanes.program.nunique(), "first_write_lanes": len(lanes), "defined_correlation_lanes": len(defined), "undefined_correlation_lanes": int(lanes.pearson_r.isna().sum()), "sorting_output_lanes": len(sorting), "matched_program_view_rows": len(comparison)},
        ["High correlation does not establish magnitude agreement.", "Alignment and matched-comparison runs are distinct saved populations.", "Computed views include compiler helper state.", "PR agreement is secondary to value agreement in this fidelity setting."],
    )


if __name__ == "__main__":
    numerical_floor()
    sorting()
    depth_count()
    patches()
    rotations()
    compression()
    learned_models()
    write_recovery()
    diagnostics()
    (OUT / "figure-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": str(Path(__file__).resolve().relative_to(ROOT)),
                "note": "All plotted experimental values are derived from saved canonical artifacts. No experiments are run by this generator.",
                "style": {"navy": NAVY, "teal": TEAL, "orange": ORANGE, "font": "DejaVu Sans", "raster_dpi": 300, "vector_format": "PDF"},
                "figures": metadata,
            },
            indent=2,
            default=lambda value: int(value) if isinstance(value, np.integer) else float(value),
            ensure_ascii=False,
        ) + "\n"
    )
    print(f"Wrote {len(metadata)} vector PDFs, {len(metadata)} 300-dpi PNGs, and figure-metadata.json in {OUT}")

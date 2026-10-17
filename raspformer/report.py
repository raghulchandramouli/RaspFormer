"""Generate the current report and figures directly from preserved measurements."""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_json(path):
    return json.loads(path.read_text())


def generate_report(results):
    results = Path(results).resolve()
    for name in ("programs", "trace", "geometry", "alignment", "comparison", "validation", "learning"):
        manifest = _read_json(results / name / "reference" / "manifest.json")
        if manifest["status"] != "completed":
            raise ValueError(f"Incomplete reference for {name}.")
    destination = results / "report"
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    text = [
        "# RaspFormer research validation\n",
        "This report is generated from the canonical reference runs. Completion means an experiment executed, not that its hypothesis succeeded. "
        "The original studies measure agreement with known compiled computations. The new controls separately test representation invariance, executing compression, geometry-guided interventions, blind write-transition detection, and learned transformers.\n",
        "Regenerate the experiments with `.venv/bin/python -m raspformer.study`. Regenerate this report with `.venv/bin/python -m raspformer.report`.\n",
    ]

    def table(frame):
        text.append(frame.to_markdown(index=False, floatfmt=".6g") + "\n")

    def figure(fig, name, caption):
        fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        text.append(f"![{caption}](figures/{name}.png)\n\n{caption}\n")

    comparison = pd.read_csv(results / "comparison/reference/comparison.csv")
    stages = pd.read_csv(results / "comparison/reference/matched/stages.csv")
    computed = stages[(stages["view"] == "computed") & (stages.step > 0)]
    raw_mismatches = int((computed.adjacent_cka_raw.notna() != computed.expected_adjacent_cka_raw.notna()).sum())
    filtered_mismatches = int((computed.adjacent_cka.notna() != computed.expected_adjacent_cka.notna()).sum())
    affected = computed[computed.program.str.startswith(("P07", "P08", "P10")) & computed.step.between(1, 3)]
    text.extend([
        "## Variance diagnosis\n",
        "The numerical policy uses total centered sample variance with ddof 1. Effective PR and component counts are zero at or below 1e-12, with an explicit inactive flag. CKA is undefined when either side is inactive. Raw variance, spectra, PR, and CKA are retained. This absolute floor is a numerical policy, not a scale-invariant definition of signal.\n",
        f"Matched computed-view CKA availability mismatches change from **{raw_mismatches} to {filtered_mismatches}**. "
        f"The suspicious early P07, P08, and P10 stages have variance between {affected.actual_total_variance.min():.6g} and {affected.actual_total_variance.max():.6g}. "
        "Their raw PR is one despite negligible variation. This reproduces and resolves the normalized-score artifact at the selected floor.\n",
        "Sources are [matched stages](../comparison/reference/matched/stages.csv), [floor sensitivity](../comparison/reference/matched/variance-sensitivity.csv), and [balanced sensitivity](../comparison/reference/balanced/variance-sensitivity.csv). The sensitivity tables include floors from zero through 1e-8.\n",
    ])
    table(affected[["program", "stage", "actual_total_variance", "actual_pr_raw", "actual_pr"]])
    remaining = computed[computed.adjacent_cka.notna() != computed.expected_adjacent_cka.notna()]
    if len(remaining):
        text.append("Remaining mismatches are retained rather than forced to agree. They occur when observed helper state already varies before the corresponding expected named-variable state.\n")
        table(remaining[["program", "stage", "actual_total_variance", "expected_total_variance"]])

    validation = _read_json(results / "validation/reference/summary.json")
    program_rows = validation["programs"]
    recovery = pd.DataFrame([
        {"program": item["program"], "group": item["recovery_group"], **score}
        for item in program_rows for score in item["recovery"]
    ])
    text.extend([
        "## Blind write-transition recovery\n",
        "The detector receives ordered activation matrices only. It receives no variable names, lane labels, program graph, or write schedule. Predictions are saved and hashed before schedule extraction for scoring. Thresholds are fixed. P01 through P06 form the calibration group and P07 through P12 the held-out program group.\n",
        "The three detectors use residual change energy, PR change, and adjacent CKA change. The null randomly permutes each detector's predictions within a program, preserving its predicted event count, with 1,000 draws. These results concern transitions with at least one first write. They do not recover variable names, simultaneous variable counts, or complete circuits.\n",
    ])
    recovery_grouped = recovery.groupby(["group", "detector"])[["precision", "recall", "f1", "null_mean_f1", "all_positive_f1"]].mean().reset_index()
    table(recovery_grouped)
    text.append("Per-program predictions, hashes, scores, and null permutation p-values are in [the validation reference](../validation/reference/). Macro means are descriptive over six programs per group. The all-positive baseline predicts a write at every transition, exposing high event prevalence.\n")
    heldout = recovery_grouped[recovery_grouped.group == "heldout"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(heldout))
    ax.bar(x - .18, heldout.f1, .36, label="Observed F1")
    ax.bar(x + .18, heldout.null_mean_f1, .36, label="Shuffled-stage mean")
    ax.axhline(heldout.all_positive_f1.iloc[0], color="gray", linestyle="--", label="Always predict a write")
    ax.set(xticks=x, xticklabels=heldout.detector, ylabel="Macro F1", ylim=(0, 1.05))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=3, fontsize=8)
    figure(fig, "write-recovery", "Blind transition detection on held-out programs, with the count-matched shuffled-stage baseline.")
    under_baseline = recovery[(recovery.group == "heldout") & (recovery.f1 <= recovery.all_positive_f1)]
    text.append(f"In {len(under_baseline)} held-out program-detector comparisons, F1 does not exceed the always-positive baseline. These failures remain part of the result.\n")
    if len(under_baseline):
        table(under_baseline[["program", "detector", "f1", "all_positive_f1"]])

    text.append("## Orthogonal rotation and executing compression\n")
    rotation = pd.DataFrame([
        {"program": item["program"], "geometry_max_error": item["rotation_geometry_max_error"],
         "minimum_executed_token_agreement": item["rotation_min_token_agreement"],
         "import_residual_max_error": item["import_check"]["residual_max_abs_error"]}
        for item in program_rows
    ])
    table(rotation)
    text.append("Whole-space PR and linear CKA are measured before and after square orthogonal transformations. Canonical coordinate checks deliberately retain their old lane map, exposing their dependence on basis. Nonorthogonal rescaling is a separate control. Executed rotation is measured separately because finite-precision weight transformations can amplify numerical error.\n")
    compression = pd.DataFrame([
        {"program": item["program"], **row} for item in program_rows for row in item["compression_final"]
    ])
    compression["parameter_fraction"] = compression.parameters / compression.original_parameters
    compression_grouped = compression.groupby(["kind", "fraction"])[[
        "token_accuracy", "sequence_accuracy", "reconstruction_mse", "parameter_fraction", "axis_overlap_mean",
    ]].mean().reset_index()
    text.append("Compression fits an uncentered shared basis on calibration sequences and transforms embeddings, residual input and output weights, biases, and unembedding. Forward execution carries the reduced residual width. Attention and MLP hidden widths are retained and reported, so residual reduction is not an equal reduction in total parameters. Random half-rank subspaces and full-rank controls are included.\n")
    table(compression_grouped)
    text.append("Axis overlap is the mean absolute cosine between nonzero projected original coordinate directions. Full-rank orthogonal controls have zero overlap up to rounding. Reduced models have overlapping directions, making the induced coordinate mixing measurable.\n")
    transformed_recovery = pd.DataFrame([
        {"program": item["program"], "group": item["recovery_group"], **row}
        for item in program_rows for row in item["transformed_recovery"]
    ])
    retest = transformed_recovery[(transformed_recovery.group == "heldout") &
                                 (transformed_recovery.detector != "change")]
    text.append("Blind detectors are also rerun on the executing rotated and compressed states. The following table reports macro means for held-out programs P07 through P12, showing the PR and CKA detectors only. Their predictions are frozen before scoring against the original compiler schedule. When compression changes behavior, that schedule is an intended reference, not established ground truth for the changed computation. Task accuracy is therefore shown beside F1.\n")
    table(retest.groupby(["kind", "fraction", "detector"])[["token_accuracy", "f1", "null_mean_f1", "all_positive_f1"]].mean().reset_index())
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, (name, frame) in enumerate(compression[compression.kind == "pca"].groupby("program", sort=False)):
        ax.plot(frame.fraction, frame.token_accuracy, marker=".", color=plt.get_cmap("tab20")(index), label=name.split("_")[0])
    ax.set(xlabel="Retained residual-width fraction", ylabel="Held-out token accuracy", ylim=(-.02, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=6, fontsize=8)
    figure(fig, "executing-compression", "Accuracy of executing compressed models, using disjoint calibration and test sequences.")

    text.extend([
        "## Geometry-directed causal patches\n",
        "Patch stages are selected from calibration geometric change and frozen before loading the compiler schedule. PCA fits calibration activations. A ridge decoder then maps the first three PCs to independently evaluated RASP variables. Decoder folds separate labels by whole sequence, but share an unsupervised PCA fitted on all calibration sequences. The final test sequences are independently held out from stage selection, PCA fitting, and decoder fitting. Donor-recipient pairs are sampled without selecting for different answers.\n",
        "PCA ranks one and three are compared with equal-rank random subspaces, self patches, and a compiler-variable oracle. Counterfactual agreement is evaluated against a single mapped-variable substitution. Some symbolic substitutions can make RASP outputs undefined. Those cases remain explicit failures of that comparison rather than being dropped. Behavioral effects alone do not establish recovery of a named variable.\n",
    ])
    patch_frames = []
    for item in program_rows:
        folder = results / "validation/reference" / item["program"].lower().replace("_", "-")
        path = folder / "patches.csv"
        if path.exists():
            patch_frames.append(pd.read_csv(path).assign(program=item["program"]))
    if patch_frames:
        patches = pd.concat(patch_frames, ignore_index=True)
        table(patches.groupby(["kind", "rank"])[["behavior_changed", "full_counterfactual_match"]].mean().reset_index())
        text.append(f"The table includes {len(patches)} patch conditions. Self patches are a baseline-preservation check. Per-program variable decoder results and all donor identities are saved next to the patch rows.\n")

    text.append("## Sorting row diagnosis and attention sharpening\n")
    sorting = next(item for item in program_rows if item["program"] == "P12_sorting")
    table(pd.DataFrame(sorting["sharpening"]["sweep"]))
    errors = pd.read_csv(results / "validation/reference/p12-sorting/per-observation-errors.csv")
    baseline = errors[errors.attention_scale == 1]
    first_stage = baseline.stage.iloc[0]
    baseline = baseline[baseline.stage == first_stage]
    failed_sets = [set(frame.loc[~frame.close, "probe_id"]) for _, frame in baseline.groupby("output_lane")]
    same = all(failed == failed_sets[0] for failed in failed_sets)
    text.append(f"At the first output write, all {len(failed_sets)} lanes have the same failing-probe set **{same}**. "
                f"There are {len(set.union(*failed_sets))} probes failing in at least one lane. "
                "This statement comes from row-level set comparison, rather than equal percentages. Attention scales multiply actual QK logits before softmax while MLP exactness stays fixed. "
                "The sweep tests finite-attention approximation directly. It does not establish that duplicates or ties alone are the cause.\n")
    text.append("Sources are [every scalar error](../validation/reference/p12-sorting/per-observation-errors.csv), [attention sweep](../validation/reference/p12-sorting/attention-sharpening.csv), and [failing probe identities](../validation/reference/p12-sorting/failing-probe-sets.json).\n")

    families = pd.read_csv(results / "validation/reference/complexity/factorial-families.csv")
    family_summary = _read_json(results / "validation/reference/complexity/summary.json")
    text.append("## Separate depth and branch-count controls\n")
    table(families[["requested_depth", "requested_branches", "computed_variable_count", "max_dependency_depth", "residual_width", "transformer_layers", "token_accuracy"]])
    text.append(f"The centered realized depth and variable-count design matrix has rank **{family_summary['realized_axes_rank']}**. "
                "Branch depth and branch count are crossed, and every branch contributes to the output. Serial merges also add depth, so requested branch depth is not identical to realized dependency depth. "
                "These token-local modulo programs are controlled structural examples, not a universal complexity scale. Their full schedules and geometry are preserved.\n")

    text.append("## Learned models on the same programs\n")
    learning = pd.read_csv(results / "learning/reference/summary.csv")
    learned = learning.groupby("program", sort=False).agg(
        seeds=("seed", "nunique"), token_accuracy=("token_accuracy", "mean"),
        token_std=("token_accuracy", "std"), sequence_accuracy=("sequence_accuracy", "mean"),
        majority_token_accuracy=("majority_token_accuracy", "mean"), compiled_token_accuracy=("compiled_token_accuracy", "mean"),
    ).reset_index()
    table(learned)
    settings = _read_json(results / "learning/reference/manifest.json")["settings"]
    text.append(f"All {len(learning)} models start from random weights and train all parameters for {settings['training_steps']} Adam steps. "
                "The shared architecture has residual width 32, two layers, and two attention heads. Whole unique sequences are split 60%, 20%, and 20% into train, validation, and test sets with split seed 2026. "
                "The source pool mixes random and structured length-six sequences over the original eight-token vocabulary. Results apply to this distribution and do not establish length generalization or behavior on natural language.\n")
    text.append("The majority-class baseline exposes class imbalance. Loss curves, exact predictions, sequence IDs, output encodings, and model checkpoints are saved. Learned-versus-compiled CKA pairs stages by relative progress across different layer counts. That is a descriptive comparison, not recovered variable correspondence. Sources are [learning summary](../learning/reference/summary.csv), [geometry](../learning/reference/geometry.csv), and [sequence splits](../learning/reference/sequence-splits.csv).\n")
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(learned))
    ax.errorbar(x, learned.token_accuracy, yerr=learned.token_std, fmt="o", label="Trained, mean and seed SD")
    ax.scatter(x, learned.majority_token_accuracy, marker="x", label="Majority baseline")
    ax.axhline(1, color="gray", linewidth=1, label="Compiled")
    ax.set(xticks=x, xticklabels=learned.program.str.split("_").str[0], ylabel="Held-out token accuracy", ylim=(-.02, 1.03))
    ax.legend(fontsize=8)
    figure(fig, "learned-models", "Three-seed learned-model accuracy with sequence-disjoint tests and a majority baseline.")
    learning_geometry = pd.read_csv(results / "learning/reference/geometry.csv")
    final_geometry = learning_geometry.groupby(["program", "seed"], sort=False).tail(1)
    text.append("Final-stage representation comparison, averaged over seeds, follows. Different feature spaces and unknown learned circuits prevent interpreting PR or CKA agreement as algorithm recovery.\n")
    table(final_geometry.groupby("program", sort=False)[["learned_pr", "compiled_pr", "cka"]].mean().reset_index())

    text.extend([
        "## Original compiler-fidelity checks\n",
        "Expected intermediate values are independently RASP-evaluated, then encoded with compiler-provided coordinates and write times. This is a fidelity benchmark. PR agreement is secondary because output cardinality, scaling, and covariance affect effective dimension.\n",
    ])
    selected = comparison[(comparison.design == "matched") & (comparison["view"] == "computed")]
    table(selected[["program", "pr_mae", "lane_agreement", "cka_mae", "cka_paired_defined", "transitions", "cka_definition_mismatches"]])
    view = comparison[comparison.design == "matched"].pivot(index="program", columns="view", values="pr_mae")
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(view))
    for name, marker in (("full", "o"), ("computed", "x")):
        ax.plot(x, view[name], marker=marker, label=name)
    ax.set(xticks=x, xticklabels=view.index.str.split("_").str[0], ylabel="PR agreement error")
    ax.legend()
    figure(fig, "residual-views", "Full and computed residual views on the same matched observations. Computed views retain compiler helper lanes.")
    text.extend([
        "## Relation to InterpBench\n",
        "Gupta and colleagues' [InterpBench paper](https://papers.nips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html) trains semi-synthetic transformers with known circuits using Strict Interchange Intervention Training and evaluates circuit-discovery methods. "
        "Its [implementation](https://github.com/FlyingPumba/InterpBench) supports ACDC, subnetwork probing, and edge attribution patching. "
        "RaspFormer evaluates representation measurements and transition detection, with additional supervised learned-model comparisons. Its learned models use ordinary output supervision and do not inherit known internal causal circuits. No InterpBench model or circuit-discovery benchmark was run here.\n",
        "## Evidence and limitations\n",
        "Every study has a completed canonical manifest under `results/<study>/reference`. New disposable runs live under `results/<study>/runs/<UTC>-<id>`. Manifests record numerical policy, dependencies, compiler revision, source hashes, notebook source hashes, and Git state. A dirty Git state is recorded explicitly.\n",
        "The new experiments do not prove universal computation recovery. Learned models lack internal ground truth. The compressed models retain attention and MLP hidden dimensions. Transition detection does not identify variable names. PCA-to-variable decoding uses supervised labels after geometry selection. No experiment establishes variable erasure or general lifetime recovery. Negative results and unavailable metrics remain in the data.\n",
    ])
    report = destination / "research-validation.md"
    content = "\n".join(text)
    missing = [target for target in re.findall(r"\]\(([^)]+)\)", content)
               if "://" not in target and not (destination / target).resolve().exists()]
    if missing:
        raise ValueError(f"Missing report evidence: {missing}")
    report.write_text(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    print(generate_report(parser.parse_args().results))


if __name__ == "__main__":
    main()

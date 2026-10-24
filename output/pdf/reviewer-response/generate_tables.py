"""Derive the report's tables and headline values from canonical evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
TABLES.mkdir(exist_ok=True)


class TeX(str):
    pass


def escape(value):
    if isinstance(value, TeX):
        return str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def number(value, digits=6):
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        mantissa, exponent = f"{value:.{digits-1}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return TeX(f"${mantissa}\\times 10^{{{int(exponent)}}}$")
    return f"{value:.{digits}g}"


def pct(value):
    return f"{100*float(value):.2f}"


def table(name, headers, rows, columns=None):
    columns = columns or ("l" + "r"*(len(headers)-1))
    kind = "tabularx" if "X" in columns else "tabular"
    width = r"{\linewidth}" if kind == "tabularx" else ""
    lines = [r"\begin{center}\small", f"\\begin{{{kind}}}{width}{{@{{}}{columns}@{{}}}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(escape(x) for x in row) + r" \\" for row in rows)
    lines += [r"\bottomrule", f"\\end{{{kind}}}", r"\end{center}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


validation = json.loads((ROOT / "results/validation/reference/summary.json").read_text())
programs = validation["programs"]
comparison = pd.read_csv(ROOT / "results/comparison/reference/comparison.csv")
balanced = pd.read_csv(ROOT / "results/geometry/reference/layer_metrics.csv")
stages = pd.read_csv(ROOT / "results/comparison/reference/matched/stages.csv")
learning = pd.read_csv(ROOT / "results/learning/reference/summary.csv")
geometry = pd.read_csv(ROOT / "results/learning/reference/geometry.csv")

balanced_final = balanced.query("view == 'computed'").groupby("program", sort=False).tail(1)
table("cardinality", ["Program", "Task", "Final computed PR"], [
    [row.program[:3], task, number(row.participation_ratio)]
    for task, row in zip(["Absolute value", "Position parity", "Token plus position"], balanced_final.head(3).itertuples())
], "llr")

tiny = stages.query("view == 'computed' and actual_total_variance > 0 and actual_total_variance < 1e-12")
table("tiny-variance", ["Program", "Stage", "Total variance", "Raw PR", "Filtered PR"], [
    [r.program[:3], r.stage, number(r.actual_total_variance), number(r.actual_pr_raw), number(r.actual_pr)]
    for r in tiny.itertuples()
], "llrrr")

rotation_rows, scale_rows, compression_rows, recovery_rows, transformed_rows, patches = [], [], [], [], [], []
for program in programs:
    name = program["program"]
    folder = ROOT / "results/validation/reference" / name.lower().replace("_", "-")
    rotation = pd.read_csv(folder / "rotations.csv")
    rotation_rows.append([name[:3], number(program["rotation_geometry_max_error"]),
                          pct(rotation.token_accuracy.min()), pct(rotation.canonical_lane_agreement.mean())])
    scaled = pd.read_csv(folder / "nonorthogonal.csv").iloc[-1]
    scale_rows.append([name[:3], number(scaled.original_pr), number(scaled.scaled_pr), number(scaled.original_scaled_cka)])
    for row in program["compression_final"]:
        compression_rows.append({"program": name, **row})
    scores = pd.read_csv(folder / "write-recovery-scores.csv")
    scores["group"] = "Calibration" if int(name[1:3]) <= 6 else "Held out"
    scores["program"] = name
    recovery_rows.append(scores)
    transformed = pd.read_csv(folder / "transformed-write-recovery.csv")
    transformed["program"] = name
    transformed_rows.append(transformed)
    patch = pd.read_csv(folder / "patches.csv")
    patch["program"] = name
    patches.append(patch)

table("rotation", ["Program", "Max PR/CKA error", r"Token agreement (\%)", r"Fixed-lane close (\%)"], rotation_rows)
table("rescaling", ["Program", "Original PR", "Scaled PR", "Cross-basis CKA"], scale_rows)

compression = pd.DataFrame(compression_rows)
compression["parameter_fraction"] = compression.parameters / compression.original_parameters
compression_mean = compression.groupby(["kind", "fraction"], sort=True).mean(numeric_only=True).reset_index()
table("compression", ["Basis", "Target fraction", r"Token (\%)", r"Sequence (\%)", r"Parameters (\%)", "Axis overlap"], [
    ["SVD" if r.kind == "pca" else "Random", number(r.fraction), pct(r.token_accuracy), pct(r.sequence_accuracy),
     pct(r.parameter_fraction), number(r.axis_overlap_mean)]
    for r in compression_mean.itertuples()
])

recovery = pd.concat(recovery_rows, ignore_index=True)
recovery_mean = recovery.groupby(["group", "detector"], sort=True).mean(numeric_only=True).reset_index()
table("recovery", ["Programs", "Detector", "Precision", "Recall", "F1", "Shuffled F1", "All-write F1"], [
    [r.group, r.detector.upper() if r.detector != "change" else "Change", number(r.precision), number(r.recall),
     number(r.f1), number(r.null_mean_f1), number(r.all_positive_f1)] for r in recovery_mean.itertuples()
], "llrrrrr")
heldout = recovery[recovery.program.str[1:3].astype(int) >= 7]
per_program = heldout.pivot(index="program", columns="detector", values="f1")
baseline = heldout.groupby("program").all_positive_f1.first()
table("recovery-per-program", ["Program", "Change F1", "PR F1", "CKA F1", "All-write F1"], [
    [name[:3], number(row["change"]), number(row["pr"]), number(row["cka"]), number(baseline[name])]
    for name, row in per_program.iterrows()
])

transformed = pd.concat(transformed_rows, ignore_index=True)
transformed = transformed[transformed.program.str[1:3].astype(int) >= 7]
transformed_mean = transformed.groupby(["kind", "fraction", "detector"]).mean(numeric_only=True).reset_index()
table("transformed-recovery", ["Transformation", "Target fraction", "Detector", r"Token (\%)", "F1", "Shuffled F1"], [
    [r.kind.replace("compression-pca", "SVD compression").replace("compression-random", "Random compression").title(),
     number(r.fraction), r.detector.upper(), pct(r.token_accuracy), number(r.f1), number(r.null_mean_f1)]
    for r in transformed_mean.itertuples() if r.detector in ("pr", "cka")
], "lrlrrr")

patches = pd.concat(patches, ignore_index=True)
patch_mean = patches.groupby(["kind", "rank"])[["behavior_changed", "full_counterfactual_match"]].mean().reset_index()
table("patches", ["Patch", "Rank", r"Changed output (\%)", r"Counterfactual match (\%)"], [
    [r.kind.upper() if r.kind == "pca" else r.kind.title(), int(r.rank), pct(r.behavior_changed), pct(r.full_counterfactual_match)]
    for r in patch_mean.itertuples()
])

sharpening = pd.read_csv(ROOT / "results/validation/reference/p12-sorting/attention-sharpening.csv")
table("sharpening", ["QK scale", r"Token accuracy (\%)", "Max lane error", "Mean lane error"], [
    [int(r.attention_scale), pct(r.token_accuracy), number(r.max_abs_lane_error), number(r.mean_abs_lane_error)]
    for r in sharpening.itertuples()
])

families = pd.read_csv(ROOT / "results/validation/reference/complexity/factorial-families.csv")
table("families", [r"\shortstack{Branch\\depth}", "Branches", "Variables", r"\shortstack{Realized\\depth}", r"\shortstack{Residual\\width}", "Layers", r"Token (\%)"], [
    [int(r.requested_depth), int(r.requested_branches), int(r.computed_variable_count), int(r.max_dependency_depth),
     int(r.residual_width), int(r.transformer_layers), pct(r.token_accuracy)] for r in families.itertuples()
])

learned_mean = learning.groupby("program").agg(token=("token_accuracy", "mean"), sd=("token_accuracy", "std"),
    sequence=("sequence_accuracy", "mean"), majority=("majority_token_accuracy", "mean"), compiled=("compiled_token_accuracy", "mean"))
table("learning", ["Program", r"Token (\%)", "Seed SD (pp)", r"Sequence (\%)", r"Majority (\%)", r"Compiled (\%)"], [
    [name[:3], pct(row.token), pct(row.sd), pct(row.sequence), pct(row.majority), pct(row.compiled)]
    for name, row in learned_mean.iterrows()
])
learned_final = geometry.groupby(["program", "seed"], sort=False).tail(1).groupby("program").mean(numeric_only=True)
table("learned-geometry", ["Program", "Learned PR", "Compiled PR", "CKA"], [
    [name[:3], number(row.learned_pr), number(row.compiled_pr), number(row.cka)] for name, row in learned_final.iterrows()
])

fidelity = comparison.query("design == 'matched' and view == 'computed'")
table("fidelity", ["Program", "PR error", r"Lane close (\%)", "CKA error", "Defined pairs", "Mismatches"], [
    [r.program[:3], number(r.pr_mae), pct(r.lane_agreement), number(r.cka_mae), int(r.cka_paired_defined), int(r.cka_definition_mismatches)]
    for r in fidelity.itertuples()
])

tasks = ["Absolute value", "Position parity", "Token plus position", "First element", "Token histogram", "Count greater tokens",
         "Sum with next token", "Sum with previous token", "Check increasing", "Rotate left", "Check palindrome", "Sort ascending"]
table("programs", ["ID", "Task", "ID", "Task"], [[f"P{i+1:02}", tasks[i], f"P{i+7:02}", tasks[i+6]] for i in range(6)], "lXlX")

verification = json.loads((ROOT / "results/validation/reference/verification.json").read_text())
replay = json.loads((ROOT / "results/learning/reference/replay.json").read_text())
failing = json.loads((ROOT / "results/validation/reference/p12-sorting/failing-probe-sets.json").read_text())
metadata = {
    "data_source": "canonical reference runs recorded 2026-10-15; no scientific rerun for typesetting",
    "tiny_variance_rows": len(tiny), "patch_conditions": len(patches),
    "patch_pairs": int(patches[["program", "pair_id"]].drop_duplicates().shape[0]),
    "undefined_counterfactual_conditions": int((~patches.symbolic_counterfactual_defined).sum()),
    "learned_models": len(learning), "models_replayed": replay["models_verified"], "test_sequences": replay["heldout_sequences"],
    "verification": verification,
    "rotation_max_error": max(p["rotation_geometry_max_error"] for p in programs),
    "rotation_min_token_agreement": min(p["rotation_min_token_agreement"] for p in programs),
    "sorting_identical_failure_sets": failing["baseline_all_lanes_identical"],
    "compression_all_program_means": compression_mean.to_dict("records"),
    "recovery_program_group_means": recovery_mean.to_dict("records"),
    "transformed_heldout_means": transformed_mean.to_dict("records"),
}
(HERE / "report-data.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
print(f"Generated {len(list(TABLES.glob('*.tex')))} tables from saved evidence.")
print({k: metadata[k] for k in ("tiny_variance_rows", "patch_conditions", "patch_pairs", "undefined_counterfactual_conditions", "learned_models")})

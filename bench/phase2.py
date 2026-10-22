"""Phase 2: linear probes, DAS, and edge attribution patching on InterpBench models.

    .venv/bin/python -m bench.phase2 [--output results/bench/reference] [--cases 2 7 ...]

Reads the Phase 1 splits and saved activations from the output directory.
Probes and DAS run on both the compiled and the trained model of each case;
EAP runs on the trained model, whose circuit is published.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bench.das import (counterfactual_targets, interchange_accuracy, intervention_inputs,
                       random_subspace, sample_pairs, train_subspace)
from bench.eap import evaluate_eap
from bench.load import load_interpbench
from bench.phase1 import compiled_ids, encode
from bench.probes import READABLE, probe_table, readability_map
from bench.programs import CASES, compile_case
from bench.variables import case_variables, position_determined, variable_values
from raspformer import models
from raspformer.compiler import write_json
from raspformer.research import recovery_scores

IIA_THRESHOLD = 0.9


def das_table(case_id, kind, model, program, assembled, variables, split, enc_cal, enc_test,
              *, pairs=256, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for variable in variables:
        cal_values = variable_values(variable, split["calibration"])
        test_values = variable_values(variable, split["test"])
        train_pairs = sample_pairs(cal_values, rng, pairs)
        test_pairs = sample_pairs(test_values, rng, pairs)
        if len(train_pairs) < 16 or len(test_pairs) < 16:
            continue
        train_targets, train_ok = counterfactual_targets(program, assembled, variable, split["calibration"], train_pairs)
        test_targets, test_ok = counterfactual_targets(program, assembled, variable, split["test"], test_pairs)
        train_pairs = [p for p, ok in zip(train_pairs, train_ok) if ok]
        test_pairs = [p for p, ok in zip(test_pairs, test_ok) if ok]
        train_targets, test_targets = train_targets[train_ok], test_targets[test_ok]
        if len(train_pairs) < 16 or len(test_pairs) < 16:
            continue
        base_logits, _ = models.forward(model, enc_test[np.asarray([b for b, _, _ in test_pairs])])
        base_predicted = np.asarray(base_logits).argmax(-1)[:, 1:]
        unchanged = ((base_predicted == test_targets) | (test_targets < 0)).all(1)
        base_rate = float(unchanged.mean())
        changing = ~unchanged   # pairs where the program says the output must change
        k = max(1, min(len(variable.classes) - 1, model.config.width // 2))
        for stage in range(2 * model.config.layers + 1):
            base, diff = intervention_inputs(model, enc_cal, train_pairs, stage)
            q = train_subspace(model, stage, k, base, diff, train_targets, seed=seed)
            test_base, test_diff = intervention_inputs(model, enc_test, test_pairs, stage)
            rows.append(dict(
                case=case_id, model=kind, variable=variable.label, stage=stage,
                stage_name=model.stage_names[stage], first_write_stage=variable.first_stage,
                classes=len(variable.classes), subspace_dim=k,
                train_pairs=len(train_pairs), test_pairs=len(test_pairs),
                invalid_test_pairs=int((~test_ok).sum()),
                iia=interchange_accuracy(model, stage, q, test_base, test_diff, test_targets),
                iia_changed=interchange_accuracy(model, stage, q, test_base, test_diff, test_targets, changing),
                iia_train=interchange_accuracy(model, stage, q, base, diff, train_targets),
                random_iia=interchange_accuracy(model, stage, random_subspace(model.config.width, k, seed + 1),
                                                test_base, test_diff, test_targets),
                random_iia_changed=interchange_accuracy(model, stage, random_subspace(model.config.width, k, seed + 1),
                                                        test_base, test_diff, test_targets, changing),
                changing_pairs=int(changing.sum()),
                base_rate=base_rate))
        trace = " ".join(f"{r['iia']:.2f}" for r in rows if r["variable"] == variable.label)
        print(f"case {case_id} {kind} DAS {variable.label}: {trace}", flush=True)
    return pd.DataFrame(rows)


DAS_SUMMARY_COLUMNS = ["case", "model", "variable", "first_write_stage", "first_aligned_stage",
                       "iia_at_write", "max_iia", "random_iia_max", "base_rate",
                       "iia_changed_at_write", "max_iia_changed", "random_iia_changed_max",
                       "changing_pairs", "stages_aligned"]


def das_summary(frame):
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=DAS_SUMMARY_COLUMNS)
    for (case, kind, variable), g in frame.groupby(["case", "model", "variable"], sort=False):
        g = g.sort_values("stage")
        first_write = int(g.first_write_stage.iloc[0])
        at_write = g[g.stage == first_write]
        aligned = g[g.iia >= IIA_THRESHOLD]
        rows.append(dict(case=case, model=kind, variable=variable, first_write_stage=first_write,
                         first_aligned_stage=int(aligned.stage.iloc[0]) if len(aligned) else None,
                         iia_at_write=float(at_write.iia.iloc[0]) if len(at_write) else None,
                         max_iia=float(g.iia.max()), random_iia_max=float(g.random_iia.max()),
                         base_rate=float(g.base_rate.iloc[0]),
                         iia_changed_at_write=float(at_write.iia_changed.iloc[0]) if len(at_write) else None,
                         max_iia_changed=float(g.iia_changed.max()),
                         random_iia_changed_max=float(g.random_iia_changed.max()),
                         changing_pairs=int(g.changing_pairs.iloc[0]),
                         stages_aligned="".join("1" if r else "0" for r in (g.iia >= IIA_THRESHOLD))))
    return pd.DataFrame(rows)


def stage_truth_from_map(frame, column, stages):
    """Stage-level event vector: any variable first becomes readable/aligned there."""
    events = {int(v) for v in frame[column].dropna()}
    return [stage in events for stage in range(1, stages)]


def run_case(case_id, output, seed):
    case = CASES[case_id]
    program = case.build()
    assembled = compile_case(case, program)
    variables = case_variables(case, program, assembled)
    split = json.loads((output / case_id / "sequence-splits.json").read_text())
    enc_cal, enc_test = encode(assembled, split["calibration"]), encode(assembled, split["test"])
    cal_values = {v.label: variable_values(v, split["calibration"]) for v in variables}
    test_values = {v.label: variable_values(v, split["test"]) for v in variables}
    variants = {"compiled": models.from_compiled(assembled), "trained": load_interpbench(case.artifact_dir)}
    probe_frames, das_frames, detector_rows = [], [], []
    for kind, model in variants.items():
        directory = output / case_id / kind
        acts = np.load(directory / "activations.npz")
        probes = probe_table(case_id, kind, list(acts["stage_names"]), acts["calibration"], acts["test"],
                             variables, cal_values, test_values)
        fixed = {v.label: position_determined(cal_values[v.label]) for v in variables}
        probes["position_determined"] = probes.variable.map(fixed)
        probes.to_csv(directory / "probes.csv", index=False)
        probe_frames.append(probes)
        das = das_table(case_id, kind, model, program, assembled, variables, split, enc_cal, enc_test, seed=seed)
        das.to_csv(directory / "das.csv", index=False)
        das_frames.append(das)
        # Geometry detectors against probe-readable and DAS-aligned stage events.
        predictions = json.loads((directory / "blind-predictions.json").read_text())
        stages = 2 * model.config.layers + 1
        for truth_name, frame, column in (("probe_readable", readability_map(probes), "first_readable_stage"),
                                          ("das_aligned", das_summary(das), "first_aligned_stage")):
            truth = stage_truth_from_map(frame, column, stages)
            for score in recovery_scores(predictions, truth):
                detector_rows.append(dict(case=case_id, model=kind, truth=truth_name, **score))
    model = variants["trained"]
    labels = compiled_ids(assembled, enc_test)
    corrupt = enc_test[np.random.default_rng(seed).permutation(len(enc_test))]
    edges, nodes, eap_metrics = evaluate_eap(model, case.artifact_dir, enc_test, corrupt, labels)
    edges.to_csv(output / case_id / "trained" / "eap-edges.csv", index=False)
    nodes.to_csv(output / case_id / "trained" / "eap-nodes.csv", index=False)
    write_json(output / case_id / "trained" / "eap-metrics.json", eap_metrics)
    eap_row = dict(case=case_id, model="trained", **{f"edge_{k}": v for k, v in eap_metrics["edge"].items()},
                   **{f"node_{k}": v for k, v in eap_metrics["node"].items()},
                   truth_scorable=eap_metrics["truth_scorable"], truth_internal=eap_metrics["truth_internal"],
                   truth_out_of_universe=eap_metrics["truth_out_of_universe"], universe_size=eap_metrics["universe_size"])
    print(f"case {case_id} EAP: edge AUROC {eap_metrics['edge']['auroc']:.3f}, node P@k {eap_metrics['node']['precision_at_k']:.3f}", flush=True)
    return (pd.concat(probe_frames), pd.concat(das_frames), pd.DataFrame(detector_rows), eap_row,
            [dict(label=v.label, classes=list(map(str, v.classes)), first_write_stage=v.first_stage) for v in variables])


def markdown_summary(probes, das, detectors, eap):
    pm, dm = readability_map(probes), das_summary(das)
    fixed = probes.groupby(["case", "model", "variable"]).position_determined.first()
    pm["position_determined"] = [fixed[(r.case, r.model, r.variable)] for r in pm.itertuples()]
    lines = ["# Phase 2: probes, DAS, and edge attribution patching on InterpBench models", ""]
    lines += [f"## Linear probes (readable = held-out accuracy >= {READABLE})", "",
              "| Case | Model | Variable | First write | First readable | Readable before write | Position-determined | Max acc | Stages |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in pm.itertuples():
        lines.append(f"| {r.case} | {r.model} | {r.variable} | {r.first_write_stage} | {r.first_readable_stage} | "
                     f"{'yes' if r.readable_before_write else 'no'} | {'yes' if r.position_determined else 'no'} | "
                     f"{r.max_test_accuracy:.3f} | `{r.stages_readable}` |")
    early = pm[~pm.position_determined].groupby("model").readable_before_write.mean()
    lines += ["", "Share of input-dependent variables linearly readable before their first write: "
              + ", ".join(f"{m} {v:.0%}" for m, v in early.items())
              + ". Position-determined variables (constant at fixed length) are excluded; they read from the position embedding at stage 0 and cannot be intervened on.", ""]
    lines += [f"## DAS (aligned = IIA >= {IIA_THRESHOLD})", "",
              "| Case | Model | Variable | First write | First aligned | IIA at write | Max IIA | Random max | Base rate | IIA at write, changing pairs | Max IIA, changing | Random, changing | Changing pairs | Stages |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def fmt(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

    for r in dm.itertuples():
        first = "none" if r.first_aligned_stage is None or (isinstance(r.first_aligned_stage, float) and np.isnan(r.first_aligned_stage)) else int(r.first_aligned_stage)
        lines.append(f"| {r.case} | {r.model} | {r.variable} | {r.first_write_stage} | {first} | {fmt(r.iia_at_write)} | "
                     f"{r.max_iia:.3f} | {r.random_iia_max:.3f} | {r.base_rate:.3f} | {fmt(r.iia_changed_at_write)} | "
                     f"{fmt(r.max_iia_changed)} | {fmt(r.random_iia_changed_max)} | {r.changing_pairs} | `{r.stages_aligned}` |")
    agg = dm.groupby("model").agg(max_iia=("max_iia", "mean"), at_write=("iia_at_write", "mean"),
                                  random=("random_iia_max", "mean"), at_write_changed=("iia_changed_at_write", "mean"),
                                  max_changed=("max_iia_changed", "mean"), random_changed=("random_iia_changed_max", "mean"))
    lines += ["", "Means over variables: " + "; ".join(
        f"{m}: max IIA {r.max_iia:.3f}, IIA at first write {r.at_write:.3f}, random {r.random:.3f}; "
        f"on output-changing pairs: at first write {r.at_write_changed:.3f}, max {r.max_changed:.3f}, random {r.random_changed:.3f}"
        for m, r in agg.iterrows()),
        "", "Base rate is the share of test pairs whose program counterfactual equals the unpatched output; IIA near the base rate is uninformative. The changing-pair columns exclude those pairs.", ""]
    lines += ["## Geometry detectors against probe and DAS stage events", "",
              "| Model | Truth | Detector | mean F1 | mean all-positive F1 |", "| --- | --- | --- | --- | --- |"]
    for (m, t, d), r in detectors.groupby(["model", "truth", "detector"])[["f1", "all_positive_f1"]].mean().iterrows():
        lines.append(f"| {m} | {t} | {d} | {r.f1:.3f} | {r.all_positive_f1:.3f} |")
    lines += ["", "## Edge attribution patching (trained models)", "",
              "| Case | Edges: AUROC | AP | P@k | k | Nodes: AUROC | P@k | k | Truth scorable / internal / out |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in eap.itertuples():
        lines.append(f"| {r.case} | {r.edge_auroc:.3f} | {r.edge_average_precision:.3f} | {r.edge_precision_at_k:.3f} | {r.edge_k} | "
                     f"{r.node_auroc:.3f} | {r.node_precision_at_k:.3f} | {r.node_k} | {r.truth_scorable} / {r.truth_internal} / {r.truth_out_of_universe} |")
    lines += ["", f"Means: edge AUROC {eap.edge_auroc.mean():.3f}, edge P@k {eap.edge_precision_at_k.mean():.3f}, "
              f"node AUROC {eap.node_auroc.mean():.3f}, node P@k {eap.node_precision_at_k.mean():.3f}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/bench/reference")
    parser.add_argument("--cases", nargs="*", default=sorted(CASES, key=int))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    output = Path(args.output)
    probes, das, detectors, eap, variables = [], [], [], [], {}
    summary = dict(recorded_utc=datetime.now(UTC).isoformat(), seed=args.seed,
                   probe_threshold=READABLE, iia_threshold=IIA_THRESHOLD, cases=args.cases)
    for case_id in args.cases:
        p, d, det, e, v = run_case(case_id, output, args.seed)
        probes.append(p); das.append(d); detectors.append(det); eap.append(e); variables[case_id] = v
        frames = dict(probes=pd.concat(probes), das=pd.concat(das), detectors=pd.concat(detectors),
                      eap=pd.DataFrame(eap))
        for name, frame in frames.items():
            frame.to_csv(output / f"phase2-{name}.csv", index=False)
        readability_map(frames["probes"]).to_csv(output / "phase2-probe-map.csv", index=False)
        das_summary(frames["das"]).to_csv(output / "phase2-das-map.csv", index=False)
        summary["variables"] = variables
        write_json(output / "phase2-summary.json", summary)
        (output / "phase2.md").write_text(markdown_summary(frames["probes"], frames["das"], frames["detectors"], frames["eap"]))
    print((output / "phase2.md").read_text())


if __name__ == "__main__":
    main()

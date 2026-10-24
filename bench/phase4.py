"""Phase 4: rerun detectors, stress suite, probes, DAS, EAP, and recovery on the
SIIT models trained for the twelve RaspFormer programs.

    .venv/bin/python -m bench.phase4 [--siit results/bench/siit] [--output results/bench/reference-siit]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bench.das import (counterfactual_targets, interchange_accuracy, intervention_inputs,
                       random_subspace, sample_pairs, train_subspace)
from bench.decompile import (Circuit, full_circuit, match_variables, recover, test_ablation,
                             test_counterfactuals, test_outputs)
from bench.eap import evaluate_eap
from bench.phase1 import circuit_truth as _unused  # noqa: F401  (keeps the API visible)
from bench.phase2 import das_summary, das_table
from bench.phase3 import short
from bench.probes import probe_table, readability_map
from bench.variables import case_variables, position_determined, variable_values
from raspformer import models
from raspformer.compiler import CompilerConfig, compile_program, extract_schedule, write_json
from raspformer.programs import build_programs
from raspformer.research import accuracy, basis_controls, decode, flat, infer_writes, recovery_scores


def pseudo_case(config, artifact_dir):
    return SimpleNamespace(vocab=tuple(config.vocab), compile_seq_len=config.max_seq_len, causal=config.causal,
                           artifact_dir=str(artifact_dir), case_id=Path(artifact_dir).name)


def id_decoder(assembled):
    classes = assembled.output_encoder.encoding_map
    return SimpleNamespace(output_encoder=SimpleNamespace(encoding_map=classes, decode=lambda ids: list(ids)))


def circuit_events(edges_path, stages):
    import pickle, re
    with open(edges_path, "rb") as handle:
        edges = pickle.load(handle)
    events = set()
    for node in {end for edge in edges for end in edge}:
        head = re.fullmatch(r"blocks\.(\d+)\.attn\.hook_result\[\d+\]", node)
        mlp = re.fullmatch(r"blocks\.(\d+)\.hook_mlp_out", node)
        if head:
            events.add(2 * int(head.group(1)) + 1)
        if mlp:
            events.add(2 * int(mlp.group(1)) + 2)
    return [stage in events for stage in range(1, stages)]


def run_program(name, program, siit_dir, output, config, seed):
    folder = siit_dir / name.lower().replace("_", "-")
    assembled = compile_program(program, config)
    case = pseudo_case(config, folder)
    compiled = models.from_compiled(assembled)
    trained = models.load_model(folder / "model.npz")
    meta = json.loads((folder / "siit.json").read_text())
    data = np.load(folder / "sequences.npz")
    tokens = data["tokens"][meta["train_sequences"]:]           # held-out sequences
    seen, unique = set(), []
    for row in tokens.tolist():
        if tuple(row) not in seen:
            seen.add(tuple(row))
            unique.append(row)
    calibration, test = unique[:96], unique[96:192]
    enc_cal = jnp.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in calibration], dtype=jnp.int32)
    enc_test = jnp.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in test], dtype=jnp.int32)
    expected = np.asarray(assembled.forward(assembled.params, enc_test).unembedded_output)[:, 1:]
    decoder = id_decoder(assembled)
    cfg_shim = SimpleNamespace(abs_tol=1e-9, rel_tol=0.0, vocab=set(config.vocab), max_seq_len=config.max_seq_len,
                               causal=config.causal, bos=config.bos, pad=config.pad, mlp_exactness=config.mlp_exactness)
    variables = case_variables(case, program, assembled)
    cal_values = {v.label: variable_values(v, calibration) for v in variables}
    test_values = {v.label: variable_values(v, test) for v in variables}
    _, schedule, _ = extract_schedule(program, assembled, cfg_shim)
    writes = {v["first_write_half_layer_index"] + 1 for v in schedule if v["first_write_half_layer_index"] is not None}
    out_dir = output / name.lower().replace("_", "-")
    rows = dict(accuracy=[], recovery=[], transformed=[], probes=[], das=[], eap=[], decompile=[])
    for kind, model in (("compiled", compiled), ("trained", trained)):
        directory = out_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        stages = 2 * model.config.layers + 1
        truths = {"schedule": [s in writes for s in range(1, stages)]}
        if kind == "trained":
            truths["circuit"] = circuit_events(folder / "edges.pkl", stages)
        logits, states = models.forward(model, enc_test)
        _, cal_states = models.forward(model, enc_cal)
        acc = accuracy(decode(decoder, logits), expected, cfg_shim)
        rows["accuracy"].append(dict(program=name, model=kind, layers=model.config.layers, heads=model.config.heads,
                                     width=model.config.width, **{f"heldout_{k}": v for k, v in acc.items()},
                                     siit_mean_iia=meta["metrics"]["mean_iia"] if kind == "trained" else None,
                                     siit_mean_siia=meta["metrics"]["mean_siia"] if kind == "trained" else None))
        predictions = infer_writes([flat(s) for s in states])
        for truth_name, truth in truths.items():
            for score in recovery_scores(predictions, truth):
                rows["recovery"].append(dict(program=name, model=kind, truth=truth_name, **score))
        controls = basis_controls(decoder, model, cal_states, enc_test, states, expected, cfg_shim, directory, (42, 43, 44))
        for item in controls["transform_predictions"]:
            for truth_name, truth in truths.items():
                for score in recovery_scores(item["predictions"], truth):
                    rows["transformed"].append(dict(program=name, model=kind, truth=truth_name, kind=item["kind"],
                                                    fraction=item["fraction"], seed=item["seed"],
                                                    token_accuracy=item["token_accuracy"], **score))
        probes = probe_table(name, kind, list(model.stage_names), np.stack(cal_states), np.stack(states),
                             variables, cal_values, test_values)
        probes["position_determined"] = probes.variable.map({v.label: position_determined(cal_values[v.label]) for v in variables})
        rows["probes"].append(probes)
        das = das_table(name, kind, model, program, assembled, variables, dict(calibration=calibration, test=test),
                        enc_cal, enc_test, seed=seed)
        rows["das"].append(das)
        circuit = Circuit.from_edges(folder, model.config) if kind == "trained" else full_circuit(model.config)
        rec, classes = recover(model, circuit, np.asarray(enc_cal), seed=seed)
        outputs = test_outputs(rec, model, np.asarray(enc_test))
        cfs = test_counterfactuals(rec, model, np.asarray(enc_test), pairs=128, seed=seed)
        ablation = test_ablation(model, circuit, np.asarray(enc_cal), np.asarray(enc_test))
        matches = match_variables(classes, cal_values)
        cf_defined = [r["agreement"] for r in cfs if r["agreement"] is not None]
        rows["decompile"].append(dict(program=name, model=kind, nodes=len(circuit.nodes),
                                      kinds=",".join(f"{short(n)}:{rec.kinds[n]}" for n in circuit.nodes),
                                      bag_heads=sum(k == "bag" for k in rec.kinds.values()),
                                      token_agreement=outputs["token_agreement"], coverage=outputs["coverage"],
                                      counterfactual_agreement=float(np.mean(cf_defined)) if cf_defined else None,
                                      ablation_agreement=ablation["token_agreement"], outside_nodes=ablation["outside_nodes"],
                                      variables_total=len(matches), variables_exact=sum(m["status"] == "exact" for m in matches),
                                      variables_missing=sum(m["status"] == "missing" for m in matches)))
        write_json(directory / "recovered-program.json", dict(nodes=circuit.nodes, kinds=rec.kinds, parents=rec.parents,
                                                               output_parents=rec.output_parents, tests=dict(outputs=outputs, counterfactuals=cfs, ablation=ablation),
                                                               variable_matches=matches))
        print(f"{name} {kind}: accuracy {acc['token_accuracy']:.3f}, recovery outputs {outputs['token_agreement']:.3f}, variables exact {rows['decompile'][-1]['variables_exact']}/{len(matches)}", flush=True)
    corrupt = enc_test[np.random.default_rng(seed).permutation(len(enc_test))]
    _, _, metrics = evaluate_eap(trained, folder, enc_test, corrupt, expected)
    rows["eap"].append(dict(program=name, **{f"edge_{k}": v for k, v in metrics["edge"].items()},
                            **{f"node_{k}": v for k, v in metrics["node"].items()},
                            truth_scorable=metrics["truth_scorable"], truth_out_of_universe=metrics["truth_out_of_universe"]))
    return rows


def markdown(frames):
    acc, rec, tr, probes, das, eap, dec = (frames[k] for k in ("accuracy", "recovery", "transformed", "probes", "das", "eap", "decompile"))
    lines = ["# Phase 4: SIIT models for the twelve RaspFormer programs", "",
             "## SIIT training outcome", "", "| Program | Held-out token acc | Mean IIA | Mean SIIA |", "| --- | --- | --- | --- |"]
    for r in acc[acc.model == "trained"].itertuples():
        lines.append(f"| {r.program} | {r.heldout_token_accuracy:.3f} | {r.siit_mean_iia:.3f} | {r.siit_mean_siia:.3f} |")
    lines += ["", "## Blind first-write detection (mean F1 over programs)", "", "| Model | Truth | Detector | mean F1 | mean all-positive F1 |", "| --- | --- | --- | --- | --- |"]
    for (m, t, d), r in rec.groupby(["model", "truth", "detector"])[["f1", "all_positive_f1"]].mean().iterrows():
        lines.append(f"| {m} | {t} | {d} | {r.f1:.3f} | {r.all_positive_f1:.3f} |")
    t = tr[(tr.model == "trained") & (tr.truth == "schedule")]
    lines += ["", "## Stress suite (trained models, schedule truth, means over programs)", "", "| Transform | Fraction | Detector | mean token acc | mean F1 |", "| --- | --- | --- | --- | --- |"]
    for (k, f, d), r in t.groupby(["kind", "fraction", "detector"]).agg(token_accuracy=("token_accuracy", "mean"), f1=("f1", "mean")).iterrows():
        lines.append(f"| {k} | {f} | {d} | {r.token_accuracy:.3f} | {r.f1:.3f} |")
    pm = readability_map(probes)
    fixed = probes.groupby(["case", "model", "variable"]).position_determined.first()
    pm["position_determined"] = [fixed[(r.case, r.model, r.variable)] for r in pm.itertuples()]
    early = pm[~pm.position_determined].groupby("model").readable_before_write.mean()
    lines += ["", "## Probes and DAS", "", "Share of input-dependent variables readable before first write: " + ", ".join(f"{m} {v:.0%}" for m, v in early.items()), ""]
    dm = das_summary(das)
    agg = dm.groupby("model").agg(at_write=("iia_changed_at_write", "mean"), max_changed=("max_iia_changed", "mean"), random=("random_iia_changed_max", "mean"))
    lines += ["DAS on output-changing pairs, means over variables: " + "; ".join(f"{m}: at first write {r.at_write:.3f}, max {r.max_changed:.3f}, random {r.random:.3f}" for m, r in agg.iterrows()), ""]
    lines += ["## Edge attribution patching (trained models)", "", "| Program | Edge AUROC | Edge P@k | Node AUROC | Node P@k | k edges |", "| --- | --- | --- | --- | --- | --- |"]
    for r in eap.itertuples():
        f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
        lines.append(f"| {r.program} | {f(r.edge_auroc)} | {f(r.edge_precision_at_k)} | {f(r.node_auroc)} | {f(r.node_precision_at_k)} | {r.edge_k} |")
    lines += ["", "## Blind program recovery", "", "| Program | Model | Nodes | Ops | Output agreement | Coverage | Counterfactual agreement | Ablation agreement | Variables exact / missing |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in dec.itertuples():
        cf = "n/a" if r.counterfactual_agreement is None or np.isnan(r.counterfactual_agreement) else f"{r.counterfactual_agreement:.3f}"
        lines.append(f"| {r.program} | {r.model} | {r.nodes} | `{r.kinds}` | {r.token_agreement:.3f} | {r.coverage:.3f} | {cf} | {r.ablation_agreement:.3f} ({r.outside_nodes} outside) | {r.variables_exact} / {r.variables_missing} |")
    agg = dec.groupby("model").agg(outputs=("token_agreement", "mean"), cf=("counterfactual_agreement", "mean"), exact=("variables_exact", "sum"), total=("variables_total", "sum"))
    lines += ["", "Means: " + "; ".join(f"{m}: output agreement {r.outputs:.3f}, counterfactual {r.cf:.3f}, variables exact {int(r.exact)}/{int(r.total)}" for m, r in agg.iterrows()), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--siit", default="results/bench/siit")
    parser.add_argument("--output", default="results/bench/reference-siit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--programs", nargs="*")
    args = parser.parse_args(argv)
    config = CompilerConfig()
    programs = build_programs(config.max_seq_len)
    siit_dir, output = Path(args.siit), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = args.programs or [n for n in programs if (siit_dir / n.lower().replace("_", "-") / "model.npz").exists()]
    collected = {k: [] for k in ("accuracy", "recovery", "transformed", "probes", "das", "eap", "decompile")}
    if args.programs:
        # Rerunning a subset: keep rows of the other programs from the saved tables.
        for k in collected:
            path = output / f"phase4-{k}.csv"
            if path.exists():
                old = pd.read_csv(path)
                column = "case" if k in ("probes", "das") else "program"
                old = old[~old[column].isin(names)]
                collected[k].append(old) if k in ("probes", "das") else collected[k].extend(old.to_dict("records"))
    for name in names:
        rows = run_program(name, programs[name], siit_dir, output, config, args.seed)
        for k, v in rows.items():
            collected[k].extend(v)
        frames = {k: (pd.concat(v) if k in ("probes", "das") else pd.DataFrame(v)) for k, v in collected.items()}
        for k, frame in frames.items():
            frame.to_csv(output / f"phase4-{k}.csv", index=False)
        (output / "phase4.md").write_text(markdown(frames))
    write_json(output / "phase4-summary.json", dict(recorded_utc=datetime.now(UTC).isoformat(), seed=args.seed, programs=names))
    print((output / "phase4.md").read_text())


if __name__ == "__main__":
    main()

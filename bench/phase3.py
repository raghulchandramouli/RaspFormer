"""Phase 3: blind program recovery and length generalization.

    .venv/bin/python -m bench.phase3 [--output results/bench/reference]

Part A recovers a table program from every compiled and trained InterpBench
model with ``bench.decompile`` (nodes from the circuit, wiring and classes by
intervention) and runs the three tests: same outputs, same counterfactuals,
nothing outside the circuit matters. Recovered variables are matched to the
true program variables by mutual determinism.

Part B tests shorter sequence lengths. InterpBench models were trained at a
fixed nine tokens; they and their recovered programs are run at four to eight
tokens. RaspFormer's 36 learned models were trained at six tokens and are run
at three to five.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bench.decompile import (Circuit, full_circuit, match_variables, recover, test_ablation,
                             test_counterfactuals, test_outputs)
from bench.load import load_interpbench
from bench.programs import BOS, CASES, compile_case, sample_inputs
from bench.variables import case_variables, variable_values
from raspformer import models
from raspformer.compiler import CompilerConfig, compile_program, write_json
from raspformer.programs import build_programs


def encode(assembled, sequences):
    return np.asarray([assembled.input_encoder.encode([BOS, *s]) for s in sequences])


def short(name):
    return name.replace("blocks.", "").replace("hook_", "").replace("attn.", "")


def recover_case(case_id, output, seed):
    case = CASES[case_id]
    program = case.build()
    assembled = compile_case(case, program)
    variables = case_variables(case, program, assembled)
    split = json.loads((output / case_id / "sequence-splits.json").read_text())
    calibration, test = encode(assembled, split["calibration"]), encode(assembled, split["test"])
    truth = {v.label: variable_values(v, split["calibration"]) for v in variables}
    trained = load_interpbench(case.artifact_dir)
    rows, recovered = [], {}
    for kind, model, circuit in (("compiled", models.from_compiled(assembled), None),
                                 ("trained", trained, Circuit.from_edges(case.artifact_dir, trained.config))):
        circuit = circuit or full_circuit(model.config)
        rec, classes = recover(model, circuit, calibration, seed=seed)
        outputs = test_outputs(rec, model, test)
        counterfactuals = test_counterfactuals(rec, model, test, pairs=128, seed=seed)
        ablation = test_ablation(model, circuit, calibration, test)
        matches = match_variables(classes, truth)
        cf_defined = [r["agreement"] for r in counterfactuals if r["agreement"] is not None]
        row = dict(case=case_id, model=kind, nodes=len(circuit.nodes),
                   kinds=",".join(f"{short(n)}:{rec.kinds[n]}" for n in circuit.nodes),
                   classes=",".join(str(rec.variables[n].classes) for n in circuit.nodes),
                   bag_heads=sum(k == "bag" for k in rec.kinds.values()),
                   mean_crispness=float(np.mean(list(rec.crispness.values()))) if rec.crispness else None,
                   token_agreement=outputs["token_agreement"], coverage=outputs["coverage"],
                   sequence_agreement=outputs["sequence_agreement"],
                   counterfactual_agreement=float(np.mean(cf_defined)) if cf_defined else None,
                   counterfactual_nodes=len(cf_defined),
                   ablation_agreement=ablation["token_agreement"], outside_nodes=ablation["outside_nodes"],
                   variables_total=len(matches),
                   variables_exact=sum(m["status"] == "exact" for m in matches),
                   variables_refined=sum(m["status"] == "refined" for m in matches),
                   variables_missing=sum(m["status"] == "missing" for m in matches))
        rows.append(row)
        recovered[kind] = (rec, assembled)
        write_json(output / case_id / kind / "recovered-program.json", dict(
            nodes=circuit.nodes, kinds=rec.kinds, aggregate=rec.aggregate, crispness=rec.crispness,
            parents=rec.parents, output_parents=rec.output_parents,
            classes={n: rec.variables[n].classes for n in circuit.nodes},
            effects={f"{p}->{n}.{s}": f for (p, n, s), f in rec.effects.items()},
            table_sizes=dict(mlp={n: len(t) for n, t in rec.mlp_tables.items()},
                             selector={n: len(t) for n, t in rec.selectors.items()},
                             head={n: len(t) for n, t in rec.head_tables.items()},
                             bag={n: len(t) for n, t in rec.bag_tables.items()},
                             output=len(rec.output_table)),
            tests=dict(outputs=outputs, counterfactuals=counterfactuals, ablation=ablation),
            variable_matches=matches))
        print(f"case {case_id} {kind}: outputs {outputs['token_agreement']:.3f} (coverage {outputs['coverage']:.3f}), "
              f"counterfactual {row['counterfactual_agreement']}, variables exact {row['variables_exact']}/{row['variables_total']}", flush=True)
    return rows, recovered


def length_rows_interpbench(case_id, recovered, seed, count=200):
    """Trained model and recovered program at four to eight tokens."""
    case = CASES[case_id]
    rec, assembled = recovered["trained"]
    model = load_interpbench(case.artifact_dir)
    rows = []
    rng = np.random.default_rng(seed)
    vals = sorted(case.vocab, key=str)
    for length in range(4, case.compile_seq_len + 1):
        if case.sampler == "balanced_all_equal":
            seqs = [[rng.choice(vals)] * length if i % 2 == 0 else
                    (lambda p: (p * (length // len(p) + 1))[:length])(rng.permutation(vals).tolist()[:min(length, len(vals))])
                    for i in range(count)]
        else:
            seqs = [[rng.choice(vals) for _ in range(length)] for _ in range(count)]
        enc = encode(assembled, seqs)
        reference = np.asarray(assembled.forward(assembled.params, jnp.asarray(enc)).unembedded_output)[:, 1:]
        predicted = np.asarray(models.forward(model, jnp.asarray(enc))[0]).argmax(-1)[:, 1:]
        program = np.asarray([rec.run(e)[0] for e in enc])
        defined = program >= 0
        model_correct = predicted == reference
        rows.append(dict(case=case_id, tokens=length, trained_length=case.compile_seq_len,
                         model_token_accuracy=float(model_correct.mean()),
                         model_sequence_accuracy=float(model_correct.all(1).mean()),
                         program_coverage=float(defined.mean()),
                         program_agrees_with_model=float((program == predicted)[defined].mean()) if defined.any() else None,
                         program_matches_reference=float((program == reference)[defined].mean()) if defined.any() else None,
                         model_accuracy_where_program_defined=float(model_correct[defined].mean()) if defined.any() else None,
                         model_accuracy_where_program_undefined=float(model_correct[~defined].mean()) if (~defined).any() else None))
    return rows


def length_rows_raspformer(root, seed, count=300):
    """RaspFormer learned models (trained at six tokens) at three to five tokens."""
    rows = []
    learning = Path(root) / "results" / "learning" / "reference"
    if not learning.exists():
        return rows
    config = CompilerConfig()
    programs = build_programs(config.max_seq_len)
    rng = np.random.default_rng(seed)
    for name, program in programs.items():
        directory = learning / name.lower().replace("_", "-")
        checkpoints = sorted(directory.glob("seed-*/model.npz")) if directory.exists() else []
        if not checkpoints:
            continue
        assembled = compile_program(program, config)
        for length in (3, 4, 5, 6):
            seqs = [[int(rng.choice(config.vocab)) for _ in range(length)] for _ in range(count)]
            enc = np.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in seqs])
            reference = np.asarray(assembled.forward(assembled.params, jnp.asarray(enc)).unembedded_output)[:, 1:]
            for checkpoint in checkpoints:
                model = models.load_model(checkpoint)
                predicted = np.asarray(models.forward(model, jnp.asarray(enc))[0]).argmax(-1)[:, 1:]
                rows.append(dict(program=name, seed=checkpoint.parent.name, tokens=length, trained_length=6,
                                 token_accuracy=float((predicted == reference).mean()),
                                 sequence_accuracy=float((predicted == reference).all(1).mean())))
    return rows


def markdown(recovery, lengths_ib, lengths_rf):
    lines = ["# Phase 3: blind program recovery and length generalization", "",
             "## Recovery: three tests per model", "",
             "| Case | Model | Nodes | Ops | Classes | Output agreement | Coverage | Counterfactual agreement | Ablation agreement | Variables exact / refined / missing |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in recovery.itertuples():
        cf = "n/a" if r.counterfactual_agreement is None or np.isnan(r.counterfactual_agreement) else f"{r.counterfactual_agreement:.3f}"
        lines.append(f"| {r.case} | {r.model} | {r.nodes} | `{r.kinds}` | {r.classes} | {r.token_agreement:.3f} | {r.coverage:.3f} | {cf} | "
                     f"{r.ablation_agreement:.3f} ({r.outside_nodes} outside) | {r.variables_exact} / {r.variables_refined} / {r.variables_missing} |")
    agg = recovery.groupby("model").agg(outputs=("token_agreement", "mean"), cf=("counterfactual_agreement", "mean"),
                                        exact=("variables_exact", "sum"), total=("variables_total", "sum"), bags=("bag_heads", "sum"))
    lines += ["", "Means: " + "; ".join(f"{m}: output agreement {r.outputs:.3f}, counterfactual agreement {r.cf:.3f}, "
                                        f"variables exact {int(r.exact)}/{int(r.total)}, bag heads {int(r.bags)}" for m, r in agg.iterrows()), ""]
    lines += ["Output agreement counts undefined positions as disagreement. A bag head is one whose attention pattern was not crisp enough for a selector, so its output is tabulated against the whole key multiset instead.", ""]
    lines += ["## Length generalization: InterpBench trained models (trained at nine tokens)", "",
              "| Case | Tokens | Model token acc | Model seq acc | Program coverage | Program = model | Model acc where program defined | Model acc where program undefined |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    for r in lengths_ib.itertuples():
        lines.append(f"| {r.case} | {r.tokens} | {r.model_token_accuracy:.3f} | {r.model_sequence_accuracy:.3f} | {r.program_coverage:.3f} | "
                     f"{f(r.program_agrees_with_model)} | {f(r.model_accuracy_where_program_defined)} | {f(r.model_accuracy_where_program_undefined)} |")
    if len(lengths_rf):
        lines += ["", "## Length generalization: RaspFormer learned models (trained at six tokens)", "",
                  "| Program | Tokens | Mean token acc over seeds | Mean seq acc |", "| --- | --- | --- | --- |"]
        g = lengths_rf.groupby(["program", "tokens"]).agg(tok=("token_accuracy", "mean"), seq=("sequence_accuracy", "mean")).reset_index()
        for r in g.itertuples():
            lines.append(f"| {r.program} | {r.tokens} | {r.tok:.3f} | {r.seq:.3f} |")
        means = lengths_rf.groupby("tokens").token_accuracy.mean()
        lines += ["", "Mean token accuracy over all 36 models by length: " + ", ".join(f"{t} tokens {v:.3f}" for t, v in means.items()), ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/bench/reference")
    parser.add_argument("--cases", nargs="*", default=sorted(CASES, key=int))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-raspformer", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    recovery, lengths_ib = [], []
    for case_id in args.cases:
        rows, recovered = recover_case(case_id, output, args.seed)
        recovery += rows
        lengths_ib += length_rows_interpbench(case_id, recovered, args.seed)
        pd.DataFrame(recovery).to_csv(output / "phase3-recovery.csv", index=False)
        pd.DataFrame(lengths_ib).to_csv(output / "phase3-length-interpbench.csv", index=False)
    lengths_rf = [] if args.skip_raspformer else length_rows_raspformer(Path(__file__).resolve().parents[1], args.seed)
    pd.DataFrame(lengths_rf).to_csv(output / "phase3-length-raspformer.csv", index=False)
    write_json(output / "phase3-summary.json", dict(recorded_utc=datetime.now(UTC).isoformat(), seed=args.seed, cases=args.cases))
    report = markdown(pd.DataFrame(recovery), pd.DataFrame(lengths_ib), pd.DataFrame(lengths_rf))
    (output / "phase3.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

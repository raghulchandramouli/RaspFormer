"""Compare circuit-discovery methods (EAP, exact patching, ACDC, EAP-IG) on both model pools.

    .venv/bin/python -m bench.circuit_baselines [--output results/bench/reference]
        [--siit results/bench/siit --siit-output results/bench/reference-siit]

Every method sees the same clean batch, the same corrupt permutation (seed 0)
and the same labels, and is scored by ``bench.circuits.score_frame`` against
the ground-truth edges of ``bench.eap.ground_truth``.
"""

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bench.circuits import ACDC_TAUS, METHODS, score_circuit, score_frame
from bench.eap import ground_truth
from bench.load import load_interpbench
from bench.phase1 import compiled_ids, encode
from bench.programs import CASES, compile_case
from raspformer import models
from raspformer.compiler import CompilerConfig, compile_program
from raspformer.programs import build_programs

METRIC_COLUMNS = ["edge_auroc", "edge_average_precision", "edge_precision_at_k",
                  "node_auroc", "node_precision_at_k"]


def interpbench_models(output):
    for case_id in sorted(CASES, key=int):
        case = CASES[case_id]
        assembled = compile_case(case)
        split = json.loads((output / case_id / "sequence-splits.json").read_text())
        enc_test = encode(assembled, split["test"])
        yield case_id, load_interpbench(case.artifact_dir), enc_test, compiled_ids(assembled, enc_test), case.artifact_dir


def siit_models(siit_dir):
    config = CompilerConfig()
    for name, program in build_programs(config.max_seq_len).items():
        folder = siit_dir / name.lower().replace("_", "-")
        if not (folder / "model.npz").exists():
            continue
        assembled = compile_program(program, config)
        meta = json.loads((folder / "siit.json").read_text())
        tokens = np.load(folder / "sequences.npz")["tokens"][meta["train_sequences"]:]
        unique = list(dict.fromkeys(map(tuple, tokens.tolist())))
        # bench.phase4 calibrates on the first 96 unique sequences and tests on the next 96.
        test = unique[96:192]
        enc_test = jnp.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in test], dtype=jnp.int32)
        yield name, models.load_model(folder / "model.npz"), enc_test, compiled_ids(assembled, enc_test), folder


def run_pool(pool, seed):
    rows, acdc_rows = [], []
    for name, model, enc_test, labels, truth_dir in pool:
        truth = set(ground_truth(truth_dir, model.config)["scorable"])
        corrupt = enc_test[np.random.default_rng(seed).permutation(len(enc_test))]
        for method, fn in METHODS.items():
            start = time.time()
            frame = fn(model, enc_test, corrupt, labels)
            metrics = score_frame(frame, truth)
            rows.append(dict(model=name, method=method, edges=len(frame), truth_edges=len(truth),
                             **{f"{level}_{k}": v for level, m in metrics.items() for k, v in m.items()},
                             seconds=time.time() - start))
            if method.startswith("acdc"):
                for tau in ACDC_TAUS:
                    scored = score_circuit(frame, f"kept@{tau:g}", truth)
                    acdc_rows.append(dict(model=name, method=method, tau=tau, **{f"{level}_{k}": v
                                          for level, m in scored.items() for k, v in m.items()}))
            print(f"{name} {method}: edge AUROC {metrics['edge']['auroc']:.3f} "
                  f"({rows[-1]['seconds']:.1f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(acdc_rows)


def fmt(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}"


def markdown(pool_name, table, acdc):
    lines = [f"# Circuit discovery: {pool_name}", "",
             "Edge and node scores use the absolute attribution score; node level promotes to sources by "
             "maximum, as in `bench.eap.evaluate_eap`. `ig` interpolates the embeddings; `ig-delta` adds "
             "alpha times the summed source difference to every target input. Means skip models without a defined metric.", "",
             "| Model | Method | Edge AUROC | Edge AP | Edge P@k | Node AUROC | Node P@k |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in table.itertuples():
        lines.append(f"| {r.model} | {r.method} | " + " | ".join(fmt(getattr(r, c)) for c in METRIC_COLUMNS) + " |")
    for method, g in table.groupby("method", sort=False):
        lines.append(f"| **mean** | {method} | " + " | ".join(fmt(g[c].mean()) for c in METRIC_COLUMNS) + " |")
    lines += ["", "## ACDC circuits by threshold", "",
              "Precision, recall and F1 of the kept edges (and of the sources with a kept edge) against the true circuit.", "",
              "| Model | Method | tau | Edges kept | Edge P | Edge R | Edge F1 | Nodes kept | Node P | Node R | Node F1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in acdc.itertuples():
        lines.append(f"| {r.model} | {r.method} | {r.tau:g} | {r.edge_size} | {fmt(r.edge_precision)} | {fmt(r.edge_recall)} | {fmt(r.edge_f1)} | "
                     f"{r.node_size} | {fmt(r.node_precision)} | {fmt(r.node_recall)} | {fmt(r.node_f1)} |")
    lines += ["", "Means by method and tau: " + "; ".join(
        f"{method} tau {tau:g}: edge F1 {g.edge_f1.mean():.3f}, node F1 {g.node_f1.mean():.3f}"
        for (method, tau), g in acdc.groupby(["method", "tau"])), ""]
    return "\n".join(lines)


def write(output, pool_name, table, acdc):
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "circuit-discovery.csv", index=False)
    acdc.to_csv(output / "circuit-discovery-acdc.csv", index=False)
    (output / "circuit-discovery.md").write_text(markdown(pool_name, table, acdc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/bench/reference")
    parser.add_argument("--siit")
    parser.add_argument("--siit-output", default="results/bench/reference-siit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.siit:
        output, name, pool = Path(args.siit_output), "SIIT models", siit_models(Path(args.siit))
    else:
        output, name, pool = Path(args.output), "InterpBench models", interpbench_models(Path(args.output))
    table, acdc = run_pool(pool, args.seed)
    write(output, name, table, acdc)
    print((output / "circuit-discovery.md").read_text())


if __name__ == "__main__":
    main()

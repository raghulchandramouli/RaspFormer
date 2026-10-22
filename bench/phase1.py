"""Phase 1: blind write detection and the stress suite on InterpBench models.

    .venv/bin/python -m bench.phase1 [--output results/bench/reference] [--cases 2 7 ...]

For every case the compiled Tracr model (high-level, ReLU) and the SIIT-trained
model (low-level, GELU) run on the same held-out sequences. The existing
RaspFormer code is reused unchanged: ``infer_writes`` for blind detection,
``recovery_scores`` for F1 against a truth vector, and ``basis_controls`` for
orthogonal rotation, nonorthogonal rescaling, and executing compression.

Two truths are scored for the trained model. ``schedule`` is the compiled
first-write schedule mapped onto the same layer index (InterpBench's
correspondence is identity on layer index, with any extra trained layers
unused). ``circuit`` marks a stage as an event when the published
``edges.pkl`` contains an attention head or MLP output at that stage.
"""

import argparse
import hashlib
import json
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bench.load import load_interpbench, read_config
from bench.programs import BOS, CASES, MLP_EXACTNESS, PAD, compile_case, sample_inputs
from raspformer import models
from raspformer.compiler import extract_schedule, write_json
from raspformer.research import (accuracy, basis_controls, decode, flat, infer_writes,
                                 recovery_scores)

DETECTORS = ("change", "pr", "cka")


def shim_config(case):
    """What trace_craft_graph and accuracy() read from a CompilerConfig."""
    return SimpleNamespace(vocab=set(case.vocab), max_seq_len=case.compile_seq_len,
                           causal=case.causal, bos=BOS, pad=PAD, mlp_exactness=MLP_EXACTNESS,
                           abs_tol=1e-9, rel_tol=0.0)


def id_decoder(assembled):
    """Stand-in whose decode returns class ids, so accuracy() compares integers."""
    classes = assembled.output_encoder.encoding_map
    return SimpleNamespace(output_encoder=SimpleNamespace(
        encoding_map=classes, decode=lambda ids: list(ids)))


def unique_split(case, seed=42, per_split=96):
    """192 unique sequences: first 96 calibration, next 96 test."""
    seen, rows = set(), []
    for tokens in sample_inputs(case, 40 * per_split, seed=seed):
        key = tuple(tokens)
        if key not in seen:
            seen.add(key)
            rows.append(tokens)
        if len(rows) == 2 * per_split:
            break
    if len(rows) < 2 * per_split:
        raise ValueError(f"Case {case.case_id} yielded only {len(rows)} unique sequences.")
    # Shuffle before splitting so a balanced sampler's few positives (case 124
    # has eleven all-equal sequences) land in both calibration and test.
    order = np.random.default_rng(seed).permutation(len(rows))
    rows = [rows[i] for i in order]
    return rows[:per_split], rows[per_split:]


def encode(assembled, sequences):
    return jnp.asarray([assembled.input_encoder.encode([BOS, *seq]) for seq in sequences],
                       dtype=jnp.int32)


def compiled_ids(assembled, encoded):
    """Argmax class ids of the compiled model, excluding BOS: InterpBench's labels."""
    return np.asarray(assembled.forward(assembled.params, encoded).unembedded_output)[:, 1:]


def schedule_truth(program, assembled, case, stages):
    _, variables, _ = extract_schedule(program, assembled, shim_config(case))
    writes = {v["first_write_half_layer_index"] + 1 for v in variables
              if v["first_write_half_layer_index"] is not None}
    return [stage in writes for stage in range(1, stages)], variables


def circuit_truth(case, stages):
    with open(Path(case.artifact_dir) / "edges.pkl", "rb") as handle:
        edges = pickle.load(handle)
    events = set()
    for node in {end for edge in edges for end in edge}:
        head = re.fullmatch(r"blocks\.(\d+)\.attn\.hook_result\[\d+\]", node)
        mlp = re.fullmatch(r"blocks\.(\d+)\.hook_mlp_out", node)
        if head:
            events.add(2 * int(head.group(1)) + 1)
        if mlp:
            events.add(2 * int(mlp.group(1)) + 2)
    return [stage in events for stage in range(1, stages)], sorted(events)


def score_table(predictions, truths, **labels):
    rows = []
    for truth_name, truth in truths.items():
        for score in recovery_scores(predictions, truth):
            rows.append(dict(**labels, truth=truth_name, **score))
    return rows


def run_case(case_id, output, seeds, eval_count):
    case = CASES[case_id]
    program = case.build()
    assembled = compile_case(case, program)
    config = shim_config(case)
    decoder = id_decoder(assembled)
    calibration, test = unique_split(case)
    enc_cal, enc_test = encode(assembled, calibration), encode(assembled, test)
    expected = compiled_ids(assembled, enc_test)
    eval_enc = encode(assembled, sample_inputs(case, eval_count, seed=11))
    eval_expected = compiled_ids(assembled, eval_enc)

    variants = {"compiled": models.from_compiled(assembled),
                "trained": load_interpbench(case.artifact_dir)}
    trained_cfg = read_config(Path(case.artifact_dir) / "ll_model_cfg.pkl")
    result = dict(case=case_id, description=case.description, causal=case.causal,
                  compiled_layers=assembled.model_config.num_layers,
                  trained_layers=trained_cfg["n_layers"], models={})
    accuracy_rows, recovery_rows, transformed_rows = [], [], []
    for kind, model in variants.items():
        directory = output / case_id / kind
        directory.mkdir(parents=True, exist_ok=True)
        stages = 2 * model.config.layers + 1
        truths = {}
        truths["schedule"], variables = schedule_truth(program, assembled, case, 2 * assembled.model_config.num_layers + 1)
        truths["schedule"] = truths["schedule"] + [False] * (stages - 1 - len(truths["schedule"]))
        circuit_events = None
        if kind == "trained":
            truths["circuit"], circuit_events = circuit_truth(case, stages)

        logits, states = models.forward(model, enc_test)
        _, calibration_states = models.forward(model, enc_cal)
        held_out = accuracy(decode(decoder, logits), expected, config)
        eval_logits, _ = models.forward(model, eval_enc)
        broad = accuracy(decode(decoder, eval_logits), eval_expected, config)
        accuracy_rows.append(dict(case=case_id, model=kind, layers=model.config.layers,
                                  heads=model.config.heads, width=model.config.width,
                                  parameters=models.parameter_count(model),
                                  heldout_token_accuracy=held_out["token_accuracy"],
                                  heldout_sequence_accuracy=held_out["sequence_accuracy"],
                                  eval_token_accuracy=broad["token_accuracy"],
                                  eval_sequence_accuracy=broad["sequence_accuracy"],
                                  eval_sequences=eval_count))

        predictions = infer_writes([flat(state) for state in states])
        frozen = directory / "blind-predictions.json"
        write_json(frozen, predictions)
        digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
        scores = score_table(predictions, truths, case=case_id, model=kind)
        recovery_rows.extend(scores)
        pd.DataFrame(scores).to_csv(directory / "write-recovery-scores.csv", index=False)

        controls = basis_controls(decoder, model, calibration_states, enc_test, states,
                                  expected, config, directory, seeds)
        transform_predictions = controls.pop("transform_predictions")
        transformed = []
        for item in transform_predictions:
            transformed.extend(score_table(
                item["predictions"], truths, case=case_id, model=kind, kind=item["kind"],
                fraction=item["fraction"], seed=item["seed"],
                token_accuracy=item["token_accuracy"]))
        transformed_rows.extend(transformed)
        pd.DataFrame(transformed).to_csv(directory / "transformed-write-recovery.csv", index=False)
        np.savez_compressed(directory / "activations.npz",
                            calibration=np.stack(calibration_states).astype(np.float32),
                            test=np.stack(states).astype(np.float32),
                            stage_names=np.asarray(model.stage_names))
        write_json(directory / "truth.json", dict(
            stage_names=list(model.stage_names)[1:], truths=truths,
            schedule_variables=variables, circuit_event_stages=circuit_events))
        result["models"][kind] = dict(
            stages=list(model.stage_names), accuracy=accuracy_rows[-1], truths=truths,
            predictions_sha256=digest, recovery=scores, transformed_recovery=transformed,
            rotation_geometry_max_error=controls["rotation_geometry_max_error"],
            rotation_min_token_agreement=controls["rotation_min_token_agreement"],
            compression_final=controls["compression_final"])
        print(f"case {case_id} {kind}: heldout token accuracy {held_out['token_accuracy']:.4f}, "
              f"stages {stages}, truth schedule {truths['schedule']}"
              + (f", circuit {truths['circuit']}" if "circuit" in truths else ""), flush=True)
    write_json(output / case_id / "sequence-splits.json",
               dict(calibration=calibration, test=test, seed=42,
                    calibration_output_classes=np.unique(compiled_ids(assembled, enc_cal)).tolist(),
                    test_output_classes=np.unique(expected).tolist()))
    return result, accuracy_rows, recovery_rows, transformed_rows


def markdown_summary(accuracy_df, recovery_df, transformed_df):
    lines = ["# Phase 1: InterpBench models under RaspFormer's detectors and stress suite", ""]
    lines += ["## Trained-model agreement with the compiled program", "",
              "| Case | Model | L | H | d | Params | Held-out token acc | Eval token acc | Eval seq acc |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in accuracy_df.itertuples():
        lines.append(f"| {r.case} | {r.model} | {r.layers} | {r.heads} | {r.width} | {r.parameters} | "
                     f"{r.heldout_token_accuracy:.4f} | {r.eval_token_accuracy:.4f} | {r.eval_sequence_accuracy:.4f} |")
    lines += ["", "## Blind first-write detection F1", "",
              "| Case | Model | Truth | change | pr | cka | all-positive | events/stages |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (case, model, truth), g in recovery_df.groupby(["case", "model", "truth"], sort=False):
        f1 = {r.detector: r.f1 for r in g.itertuples()}
        first = g.iloc[0]
        lines.append(f"| {case} | {model} | {truth} | {f1['change']:.3f} | {f1['pr']:.3f} | {f1['cka']:.3f} | "
                     f"{first.all_positive_f1:.3f} | {int(first.true_count)}/{int(first.true_count / max(first.oracle_prevalence, 1e-9) + 0.5)} |")
    means = recovery_df.groupby(["model", "truth", "detector"])[["f1", "all_positive_f1"]].mean()
    lines += ["", "Unweighted means over cases:", "", "| Model | Truth | Detector | mean F1 | mean all-positive F1 |",
              "| --- | --- | --- | --- | --- |"]
    for (model, truth, det), r in means.iterrows():
        lines.append(f"| {model} | {truth} | {det} | {r.f1:.3f} | {r.all_positive_f1:.3f} |")
    lines += ["", "## Stress suite: schedule F1 beside task accuracy (trained models, schedule truth)", "",
              "| Case | Transform | Fraction | Token acc | change F1 | pr F1 | cka F1 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    t = transformed_df[(transformed_df.model == "trained") & (transformed_df.truth == "schedule")]
    for (case, kind, fraction), g in t.groupby(["case", "kind", "fraction"], sort=False):
        g = g[g.seed == g.seed.iloc[0]]
        f1 = {r.detector: r.f1 for r in g.itertuples()}
        lines.append(f"| {case} | {kind} | {fraction} | {g.token_accuracy.iloc[0]:.3f} | "
                     f"{f1['change']:.3f} | {f1['pr']:.3f} | {f1['cka']:.3f} |")
    agg = t.groupby(["kind", "fraction", "detector"]).agg(token_accuracy=("token_accuracy", "mean"), f1=("f1", "mean"))
    lines += ["", "Unweighted means over cases (trained models, schedule truth):", "",
              "| Transform | Fraction | Detector | mean token acc | mean F1 |", "| --- | --- | --- | --- | --- |"]
    for (kind, fraction, det), r in agg.iterrows():
        lines.append(f"| {kind} | {fraction} | {det} | {r.token_accuracy:.3f} | {r.f1:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/bench/reference")
    parser.add_argument("--cases", nargs="*", default=sorted(CASES, key=int))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--eval-count", type=int, default=300)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(recorded_utc=datetime.now(UTC).isoformat(), seeds=args.seeds,
                   split=dict(seed=42, calibration=96, test=96), eval_count=args.eval_count, cases=[])
    accuracy_rows, recovery_rows, transformed_rows = [], [], []
    for case_id in args.cases:
        result, acc, rec, tr = run_case(case_id, output, tuple(args.seeds), args.eval_count)
        summary["cases"].append(result)
        accuracy_rows += acc
        recovery_rows += rec
        transformed_rows += tr
        write_json(output / "summary.json", summary)
    frames = {"accuracy": pd.DataFrame(accuracy_rows), "write-recovery": pd.DataFrame(recovery_rows),
              "transformed-write-recovery": pd.DataFrame(transformed_rows)}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    report = markdown_summary(frames["accuracy"], frames["write-recovery"], frames["transformed-write-recovery"])
    (output / "phase1.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

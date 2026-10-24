"""Phase 5: length generalization beyond the training length.

    .venv/bin/python -m bench.phase5 [--siit results/bench/siit-long] [--output results/bench/reference-siit-long]

The SIIT models in ``--siit`` were trained on six-token sequences with
position embeddings sized for ten. Each is run from three to ten tokens
against the compiled reference (exact at every length). The blind decompiler
is fitted on six-token sequences and executed at every length, so its
coverage can be compared with where the model actually breaks.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bench.decompile import Circuit, recover
from raspformer import models
from raspformer.compiler import CompilerConfig, compile_program, write_json
from raspformer.programs import build_programs


def run_program(name, program, folder, config, seed, count):
    assembled = compile_program(program, config)
    model = models.load_model(folder / "model.npz")
    meta = json.loads((folder / "siit.json").read_text())
    data = np.load(folder / "sequences.npz")
    held = data["tokens"][meta["train_sequences"]:]
    seen, calibration = set(), []
    for row in held.tolist():
        if tuple(row) not in seen:
            seen.add(tuple(row))
            calibration.append(row)
        if len(calibration) == 96:
            break
    enc_cal = np.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in calibration])
    rec, _ = recover(model, Circuit.from_edges(folder, model.config), enc_cal, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for length in range(3, config.max_seq_len + 1):
        seqs = [[int(rng.choice(config.vocab)) for _ in range(length)] for _ in range(count)]
        enc = np.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in seqs])
        reference = np.asarray(assembled.forward(assembled.params, jnp.asarray(enc)).unembedded_output)[:, 1:]
        predicted = np.asarray(models.forward(model, jnp.asarray(enc))[0]).argmax(-1)[:, 1:]
        program_out = np.asarray([rec.run(e)[0] for e in enc])
        defined = program_out >= 0
        correct = predicted == reference
        rows.append(dict(program=name, tokens=length, train_length=meta["train_length"],
                         model_token_accuracy=float(correct.mean()), model_sequence_accuracy=float(correct.all(1).mean()),
                         program_coverage=float(defined.mean()),
                         program_agrees_with_model=float((program_out == predicted)[defined].mean()) if defined.any() else None,
                         model_accuracy_where_program_defined=float(correct[defined].mean()) if defined.any() else None,
                         model_accuracy_where_program_undefined=float(correct[~defined].mean()) if (~defined).any() else None,
                         siit_mean_iia=meta["metrics"]["mean_iia"], siit_token_accuracy=meta["metrics"]["token_accuracy"]))
        print(f"{name} {length} tokens: model {correct.mean():.3f}, program coverage {defined.mean():.3f}", flush=True)
    return rows


def markdown(frame):
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    lines = ["# Phase 5: length generalization beyond the training length", "",
             "SIIT models trained on six-token sequences with ten-token position embeddings, "
             "tested from three to ten tokens against the compiled reference.", "",
             "## Mean token accuracy over the twelve programs", "", "| Tokens | Model token acc | Model seq acc | Program coverage |", "| --- | --- | --- | --- |"]
    for tokens, g in frame.groupby("tokens"):
        lines.append(f"| {tokens} | {g.model_token_accuracy.mean():.3f} | {g.model_sequence_accuracy.mean():.3f} | {g.program_coverage.mean():.3f} |")
    lines += ["", "## Per program", "",
              "| Program | Tokens | Model token acc | Model seq acc | Program coverage | Program = model | Model acc where defined | Model acc where undefined |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in frame.itertuples():
        mark = "" if r.tokens <= r.train_length else " (longer)"
        lines.append(f"| {r.program} | {r.tokens}{mark} | {r.model_token_accuracy:.3f} | {r.model_sequence_accuracy:.3f} | {r.program_coverage:.3f} | "
                     f"{f(r.program_agrees_with_model)} | {f(r.model_accuracy_where_program_defined)} | {f(r.model_accuracy_where_program_undefined)} |")
    longer = frame[frame.tokens > frame.train_length]
    both = longer.dropna(subset=["model_accuracy_where_program_defined", "model_accuracy_where_program_undefined"])
    lines += ["", f"Longer than training ({len(longer)} program-length rows): mean model token accuracy {longer.model_token_accuracy.mean():.3f}, "
              f"mean program coverage {longer.program_coverage.mean():.3f}."]
    if len(both):
        lines.append(f"Where the recovered program is defined the model scores {both.model_accuracy_where_program_defined.mean():.3f} on average; "
                     f"where it is undefined {both.model_accuracy_where_program_undefined.mean():.3f} ({len(both)} rows with both).")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--siit", default="results/bench/siit-long")
    parser.add_argument("--output", default="results/bench/reference-siit-long")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--programs", nargs="*")
    args = parser.parse_args(argv)
    siit_dir, output = Path(args.siit), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((siit_dir / "manifest.json").read_text()) if (siit_dir / "manifest.json").exists() else {}
    config = CompilerConfig(max_seq_len=manifest.get("max_seq_len", 10))
    programs = build_programs(config.max_seq_len)
    names = args.programs or [n for n in programs if (siit_dir / n.lower().replace("_", "-") / "model.npz").exists()]
    rows = []
    for name in names:
        rows += run_program(name, programs[name], siit_dir / name.lower().replace("_", "-"), config, args.seed, args.count)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "phase5-length.csv", index=False)
        (output / "phase5.md").write_text(markdown(frame))
    write_json(output / "phase5-summary.json", dict(recorded_utc=datetime.now(UTC).isoformat(), seed=args.seed, programs=names,
                                                    max_seq_len=config.max_seq_len))
    print((output / "phase5.md").read_text())


if __name__ == "__main__":
    main()

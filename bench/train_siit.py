"""Train SIIT models for the twelve RaspFormer programs.

    .venv/bin/python -m bench.train_siit [--output results/bench/siit] [--steps 3000]
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bench.siit import train_program
from raspformer.compiler import CompilerConfig, write_json
from raspformer.programs import build_programs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/bench/siit")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--programs", nargs="*")
    args = parser.parse_args(argv)
    config = CompilerConfig()
    programs = build_programs(config.max_seq_len)
    names = args.programs or list(programs)
    rows = []
    for name in names:
        metrics = train_program(name, programs[name], args.output, config=config, steps=args.steps, seed=args.seed)
        rows.append(dict(program=name, token_accuracy=metrics["token_accuracy"], sequence_accuracy=metrics["sequence_accuracy"],
                         mean_iia=metrics["mean_iia"], min_iia=min(metrics["iia"].values()) if metrics["iia"] else None,
                         mean_siia=metrics["mean_siia"], circuit_nodes=len(metrics["iia"]), outside_nodes=len(metrics["siia"])))
        pd.DataFrame(rows).to_csv(Path(args.output) / "summary.csv", index=False)
    write_json(Path(args.output) / "manifest.json", dict(recorded_utc=datetime.now(UTC).isoformat(), steps=args.steps,
                                                         seed=args.seed, programs=names))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

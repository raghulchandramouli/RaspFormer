"""Run the calibration study.

    .venv/bin/python -m calib.run train    --root results/calibration/runs/pilot --pilot
    .venv/bin/python -m calib.run analyze  --root results/calibration/runs/pilot --pilot
    .venv/bin/python -m calib.run train    --root results/calibration/runs/full
    .venv/bin/python -m calib.run compress --root results/calibration/runs/full
    .venv/bin/python -m calib.run analyze  --root results/calibration/runs/full
    .venv/bin/python -m calib.run promote  --root results/calibration/runs/full --to results/calibration/reference

Training resumes: a model whose metrics exist is never retrained. Checkpoints
stay in the run folder; ``promote`` publishes the tables, which record every
checkpoint's SHA-256.
"""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from calib import analysis, library, pool
from calib.report import write_summary
from raspformer import models
from raspformer.calibration import stage_ground_truth
from raspformer.compiler import CompilerConfig, compile_program, environment_metadata, write_json
from raspformer.programs import build_programs
from raspformer.research import flat

BENCH_SIIT = Path("results/bench/siit")
# These two SIIT models answer "no" on almost every input and do not compute their program (issue #20).
INVALID_BENCH_SIIT = {"P09_check_increasing", "P11_check_palindrome"}
PILOT_GO_PAIRS = 4


def _execute(jobs, workers):
    started, failures = time.time(), []
    for done, record in enumerate(pool.run_jobs(jobs, workers), 1):
        if not record["ok"]:
            failures.append(record)
        detail = record.get("error") or f"accuracy {record['token_accuracy']:.3f}"
        print(f"[{done}/{len(jobs)} {time.time() - started:.0f}s] {record['spec']} {record['model_class']} "
              f"seed {record['seed']}: {detail}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; rerun to retry them.")


def train(root, pilot, workers, steps):
    specs = list(library.all_specs(pilot_only=pilot).values())
    classes = ("siit", "output-only") if pilot else pool.TRAINED_CLASSES
    _execute(pool.training_jobs(root, specs, steps, classes), workers)


def compress(root, workers):
    _execute(pool.compression_jobs(root, list(library.all_specs().values())), workers)


def model_table(root):
    rows = []
    for path in sorted(Path(root).glob("models/*/*/seed-*/metrics.json")):
        record = json.loads(path.read_text())
        rows.append({key: value for key, value in record.items()
                     if key not in ("history", "iia", "siia", "circuit_nodes")})
    return pd.DataFrame(rows)


def detector_inputs(root):
    """Compiled calibration programs, and the held-out compiled and SIIT pools."""
    calibration, held_out = {}, {}
    config = CompilerConfig()
    tokens = library.probe_tokens(config)
    for name, program in build_programs().items():
        assembled = compile_program(program, config)
        encoded = np.asarray([assembled.input_encoder.encode([config.bos, *row]) for row in tokens.tolist()], dtype=np.int32)
        truth = np.asarray([row["first_write"] for row in stage_ground_truth(program, assembled, config)][1:])
        def entry(label, model):
            return label, analysis.detector_scores([flat(stage) for stage in models.forward(model, encoded)[1]]), truth
        if int(name[1:3]) <= 6:
            calibration[name] = entry("compiled calibration", models.from_compiled(assembled))
        else:
            held_out[name] = entry("compiled P07-P12", models.from_compiled(assembled))
        checkpoint = BENCH_SIIT / name.lower().replace("_", "-") / "model.npz"
        if checkpoint.exists() and name not in INVALID_BENCH_SIIT:
            held_out[f"{name}:siit"] = entry("SIIT original programs", models.load_model(checkpoint))
    for spec in library.grid_specs():
        program, assembled, hl_model = pool.compiled(spec.name)
        truth = np.asarray([row["first_write"] for row in stage_ground_truth(program, assembled, spec.config)][1:])
        calibration[spec.name] = ("compiled calibration", analysis.detector_scores(pool.measure(spec.name, hl_model)), truth)
        for seed, model in analysis.models_of(root, spec.name, "siit").items():
            held_out[f"{spec.name}:siit:{seed}"] = ("SIIT grid", analysis.detector_scores(pool.measure(spec.name, model)), truth)
    return calibration, held_out


def pair_table(root, pairings, classes):
    return pd.DataFrame([row for pairing in pairings for model_class in classes
                         for row in analysis.pair_distances(root, pairing, model_class)])


def analyze(root, pilot):
    root = Path(root)
    tables = {"models": model_table(root)}
    twins = library.twin_pairings(pilot_only=pilot)
    trained = ("siit", "output-only")
    if pilot:
        specs = [spec for pairing in twins for spec in (pairing.a, pairing.b)]
        tables["exclusions"] = analysis.exclusions(root, specs, trained)
        tables["l3-pairs"] = pair_table(root, twins, analysis.REAL_CLASSES)
        tables["l3"] = analysis.level_three(tables["l3-pairs"])
        usable = tables["l3-pairs"][(tables["l3-pairs"].model_class == "siit") & (tables["l3-pairs"].metric == "cka")]
        trainable = int(((usable.models_a >= 2) & (usable.models_b >= 2)).sum())
        decision = dict(pairs=len(twins), pairs_with_two_siit_seeds_per_member=trainable, required=PILOT_GO_PAIRS,
                        decision="go" if trainable >= PILOT_GO_PAIRS else "redesign")
    else:
        grid = library.grid_specs()
        specs = list(library.all_specs().values())
        tables["exclusions"] = analysis.exclusions(root, specs, trained)
        classes = analysis.REAL_CLASSES + ("shuffled-label", "untrained", "untrained-compiled")
        tables["stages"] = analysis.stage_table(root, grid, classes)
        tables["l1"] = analysis.level_one(tables["stages"])
        tables["l2"] = analysis.level_two(tables["stages"])
        tables["l3-pairs"] = pair_table(root, twins + library.control_pairings(), analysis.REAL_CLASSES)
        tables["l3"] = analysis.level_three(tables["l3-pairs"])
        tables["verdicts"] = analysis.verdicts(tables["l1"], tables["l2"], tables["l3"])
        tables["detectors"] = analysis.harden_detectors(*detector_inputs(root))
        compressed = [f"{model_class}-compressed-{fraction}" for fraction in pool.COMPRESSION_FRACTIONS
                      for model_class in ("compiled", "siit")]
        if any((root / "models").glob("*/*-compressed-*")):
            tables["exclusions-compressed"] = analysis.exclusions(root, grid, compressed)
            tables["stages-compressed"] = analysis.stage_table(root, grid, compressed, nulls=False)
            tables["l2-compressed"] = analysis.level_two(tables["stages-compressed"], compressed)
            siit_compressed = [name for name in compressed if name.startswith("siit")]
            tables["l3-pairs-compressed"] = pair_table(root, twins, siit_compressed)
            tables["l3-compressed"] = analysis.level_three(tables["l3-pairs-compressed"])
        decision = None
    for name, table in tables.items():
        table.to_csv(root / f"{name}.csv", index=False)
    manifest = dict(status="completed", recorded_utc=datetime.now(UTC).isoformat(), pilot=pilot,
                    settings=dict(seeds=list(pool.SEEDS), steps=sorted(int(value) for value in tables["models"].steps.dropna().unique()), width=32, heads=4,
                                  probe_sequences=library.PROBE_SEQUENCES, probe_seed=library.PROBE_SEED,
                                  iia_threshold=pool.IIA_THRESHOLD, max_training_rounds=pool.MAX_ROUNDS, l1_threshold=analysis.L1_THRESHOLD,
                                  l2_threshold=analysis.L2_THRESHOLD, l3_ratio=analysis.L3_RATIO, alpha=analysis.ALPHA,
                                  permutation_draws=analysis.DRAWS, bootstraps=analysis.BOOTSTRAPS,
                                  **environment_metadata()),
                    models=len(tables["models"]), excluded=len(tables["exclusions"]),
                    twin_pairs=[pairing.name for pairing in twins], tables=sorted(tables))
    if decision:
        manifest["pilot_decision"] = decision
    write_json(root / "manifest.json", manifest)
    print(write_summary(root, tables, manifest))


def promote(root, destination):
    """Publish tables, manifest and summary. Checkpoints stay in the run folder."""
    root, destination = Path(root), Path(destination)
    if not (root / "manifest.json").exists():
        raise SystemExit("Analyze the run before promoting it.")
    if destination.exists():
        if not (destination / "manifest.json").exists():
            raise SystemExit(f"{destination} exists and is not an earlier promotion; refusing to replace it.")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in sorted(root.iterdir()):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    print(f"Saved {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train", "compress", "analyze", "promote"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="only the five pilot pairs")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--steps", type=int, default=pool.STEPS)
    parser.add_argument("--to", type=Path, help="destination for promote")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.root, args.pilot, args.workers, args.steps)
    elif args.command == "compress":
        compress(args.root, args.workers)
    elif args.command == "analyze":
        analyze(args.root, args.pilot)
    else:
        if args.to is None:
            parser.error("promote needs --to")
        promote(args.root, args.to)


if __name__ == "__main__":
    main()

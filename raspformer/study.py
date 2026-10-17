"""Reproduce the original studies and the research validation experiments."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import jax
import numpy as np
import pandas as pd
from tracr.rasp import rasp

from .alignment import measure_alignment, run_alignment_study, summarize_alignment
from .compiler import CompilerConfig, record_run, run_study, run_trace_study, write_json
from .geometry import (
    ProbeConfig, build_balanced_probes, build_matched_probes, candidate_pool,
    collect_activations, pca_profile, all_pair_cka, run_geometry_study,
)
from .programs import build_programs


SAMPLES = [[1, 2, 1], [-2, 0, 3, 1], [5], [0, 0, 0, 0], [5, 4, 3, 2, 1, 0]]
VARIANCE_FLOOR = 1e-12


def promote_reference(run_dir, study_root):
    """Publish a copy only after numerical completion, retaining the original run."""
    run_dir, study_root = Path(run_dir), Path(study_root)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("A reference must come from a completed run.")
    temporary = study_root / f".reference-{uuid4().hex[:8]}"
    shutil.copytree(run_dir, temporary)
    reference = study_root / "reference"
    previous = study_root / f".previous-{uuid4().hex[:8]}"
    if reference.exists():
        reference.rename(previous)
    try:
        temporary.rename(reference)
    except BaseException:
        if previous.exists():
            previous.rename(reference)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return reference


def _variance_sensitivity(stages):
    rows = []
    for floor in (0.0, 1e-14, 1e-12, 1e-10, 1e-8):
        for (_, _), group in stages.groupby(["program", "view"], sort=False):
            group = group.sort_values("step")
            actual = group.actual_total_variance.to_numpy() > floor
            expected = group.expected_total_variance.to_numpy() > floor
            previous_actual = np.r_[False, actual[:-1]]
            previous_expected = np.r_[False, expected[:-1]]
            for offset, row in enumerate(group.itertuples()):
                rows.append({
                    "program": row.program, "view": row.view, "stage": row.stage,
                    "step": row.step, "variance_floor": floor,
                    "actual_total_variance": row.actual_total_variance,
                    "expected_total_variance": row.expected_total_variance,
                    "actual_pr_raw": row.actual_pr, "expected_pr_raw": row.expected_pr,
                    "actual_pr": row.actual_pr if actual[offset] else 0.0,
                    "expected_pr": row.expected_pr if expected[offset] else 0.0,
                    "adjacent_cka": row.adjacent_cka if actual[offset] and previous_actual[offset] else np.nan,
                    "expected_adjacent_cka": row.expected_adjacent_cka if expected[offset] and previous_expected[offset] else np.nan,
                    "actual_active": bool(actual[offset]), "expected_active": bool(expected[offset]),
                })
    return pd.DataFrame(rows)


def run_comparison(programs, config, probes, output_root):
    settings = {
        "probe_config": asdict(probes), "variance_floor": VARIANCE_FLOOR,
        "ground_truth": "independent RASP values in compiler coordinates and first-write schedule",
        "interpretation": "compiler fidelity; distinct from blind recovery; positions are dependent within sequences",
    }
    frames = []
    with record_run(output_root, programs, config, **settings) as (run_dir, evidence):
        for design, sampler in (("matched", build_matched_probes), ("balanced", build_balanced_probes)):
            folder = run_dir / design
            folder.mkdir()
            evidence["stage"] = f"{design}:activations"
            probeset = sampler(programs, config, probes)
            probeset.audit.to_csv(folder / "class-counts.csv", index=False)
            data = collect_activations(programs, config, probeset)
            sensitivity = []
            for name, program in programs.items():
                evidence.update(stage=f"{design}:alignment", current_program=name)
                write_json(run_dir / "manifest.json", evidence)
                measured = measure_alignment(name, program, config, data)
                raw = measure_alignment(name, program, config, data, variance_floor=0.0)["stages"]
                measured["stages"] = measured["stages"].assign(
                    actual_pr_raw=raw.actual_pr, expected_pr_raw=raw.expected_pr,
                    adjacent_cka_raw=raw.adjacent_cka,
                    expected_adjacent_cka_raw=raw.expected_adjacent_cka,
                )
                for key, frame in measured.items():
                    file = folder / f"{key}.csv"
                    frame.to_csv(file, mode="a", header=not file.exists(), index=False)
                sensitivity.append(_variance_sensitivity(raw))
                table = data.observations[name]
                table.assign(tokens=table.tokens.map(json.dumps)).to_csv(folder / f"probes_{name}.csv", index=False)
                frames.append(summarize_alignment(measured["stages"], measured["lanes"]).assign(design=design))
                evidence["programs"].append({"program": name, "design": design, "observations": len(table)})
                print(f"comparison {design} / {name}", flush=True)
            pd.concat(sensitivity, ignore_index=True).to_csv(folder / "variance-sensitivity.csv", index=False)
            del data
        comparison = pd.concat(frames, ignore_index=True)
        comparison.to_csv(run_dir / "comparison.csv", index=False)
    return run_dir


def run_baseline(output):
    programs = build_programs()
    examples = build_programs(include_examples=True)
    config = CompilerConfig(rel_tol=1e-4, abs_tol=1e-4)
    probes = ProbeConfig()
    jobs = (
        ("programs", lambda folder: run_study(programs, CompilerConfig(), SAMPLES, folder)["run_dir"]),
        ("trace", lambda folder: run_trace_study(examples, config, SAMPLES, folder)["run_dir"]),
        ("geometry", lambda folder: run_geometry_study(examples, config, probes, folder)["run_dir"]),
        ("alignment", lambda folder: run_alignment_study(examples, config, probes, folder)["run_dir"]),
        ("comparison", lambda folder: run_comparison(programs, config, probes, folder)),
    )
    for name, job in jobs:
        print(f"Running {name}", flush=True)
        reference = promote_reference(job(output / name), output / name)
        print(f"Saved {reference}", flush=True)


def _accuracy(logits, labels):
    predicted = np.asarray(logits)[:, 1:].argmax(axis=-1)
    return float((predicted == labels).mean()), float((predicted == labels).all(axis=1).mean())


def run_learning(programs, config, probe_config, output, seeds, steps):
    from .compiler import compile_program
    from .models import forward, from_compiled, random_model, train_model, save_model, parameter_count

    candidates, families = candidate_pool(config, probe_config)
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(candidates))
    end_train, end_validation = int(len(order) * 0.6), int(len(order) * 0.8)
    splits = {"train": order[:end_train], "validation": order[end_train:end_validation], "test": order[end_validation:]}
    split_rows = [
        {"sequence_id": int(index), "split": split, "family": families[index], "tokens": json.dumps(candidates[index])}
        for split, indices in splits.items() for index in indices
    ]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(output / "sequence-splits.csv", index=False)
    summary, geometry = [], []
    evaluator = rasp.DefaultRASPEvaluator()
    for name, program in programs.items():
        compiled = compile_program(program, config)
        oracle = from_compiled(compiled)
        classes = np.asarray(compiled.output_encoder.decode(list(range(oracle.params["unembed"].shape[1]))))
        class_index = {int(value): index for index, value in enumerate(classes)}
        task_folder = output / name.lower().replace("_", "-")
        task_folder.mkdir(parents=True, exist_ok=True)
        write_json(task_folder / "encoding.json", {
            "input_tokens": [{"token": token, "id": index} for token, index in compiled.input_encoder.encoding_map.items()],
            "output_classes": classes.tolist(), "bos": config.bos, "pad": config.pad,
            "sequence_length": probe_config.length,
        })
        inputs = np.asarray([compiled.input_encoder.encode([config.bos, *tokens]) for tokens in candidates], dtype=np.int32)
        targets = np.asarray([[class_index[int(value)] for value in evaluator.evaluate(program, tokens)] for tokens in candidates], dtype=np.int32)
        train_ids, validation_ids, test_ids = (splits[key] for key in ("train", "validation", "test"))
        counts = np.bincount(targets[train_ids].ravel(), minlength=len(classes))
        majority = int(counts.argmax())
        test_labels = targets[test_ids]
        majority_token = float((test_labels == majority).mean())
        compiled_logits, compiled_stages = forward(oracle, inputs[test_ids])
        compiled_token, compiled_sequence = _accuracy(compiled_logits, test_labels)
        for seed in seeds:
            folder = output / name.lower().replace("_", "-") / f"seed-{seed}"
            folder.mkdir(parents=True)
            model = random_model(
                vocab_size=oracle.params["token_embed"].shape[0], max_length=config.max_seq_len + 1,
                output_classes=len(classes), seed=seed, width=32, layers=2, heads=2,
            )
            before_logits, _ = forward(model, inputs[test_ids])
            before_accuracy, _ = _accuracy(before_logits, test_labels)
            initial_parameters = np.concatenate([np.asarray(value).ravel() for value in jax.tree_util.tree_leaves(model.params)])
            model, history = train_model(
                model, inputs[train_ids], targets[train_ids], inputs[validation_ids], targets[validation_ids],
                steps=steps, seed=seed, batch_size=128, lr=0.003,
            )
            pd.DataFrame(history).to_csv(folder / "training.csv", index=False)
            save_model(model, folder / "model.npz")
            learned_logits, learned_stages = forward(model, inputs[test_ids])
            token_accuracy, sequence_accuracy = _accuracy(learned_logits, test_labels)
            final_parameters = np.concatenate([np.asarray(value).ravel() for value in jax.tree_util.tree_leaves(model.params)])
            summary.append({
                "program": name, "seed": seed, "origin": "random-init-trained", "steps": steps,
                "train_sequences": len(train_ids), "validation_sequences": len(validation_ids), "test_sequences": len(test_ids),
                "width": model.config.width, "layers": model.config.layers, "parameters": parameter_count(model),
                "initial_token_accuracy": before_accuracy, "token_accuracy": token_accuracy,
                "sequence_accuracy": sequence_accuracy, "majority_token_accuracy": majority_token,
                "compiled_token_accuracy": compiled_token, "compiled_sequence_accuracy": compiled_sequence,
                "parameter_change_l2": float(np.linalg.norm(final_parameters - initial_parameters)),
                "initial_validation_loss": history[0]["val_loss"], "final_validation_loss": history[-1]["val_loss"],
                "checkpoint": str(folder.relative_to(output) / "model.npz"),
            })
            predicted = np.asarray(learned_logits)[:, 1:].argmax(axis=-1)
            pd.DataFrame([
                {"sequence_id": int(sequence_id), "position": position, "expected": int(classes[expected]), "actual": int(classes[actual])}
                for sequence_id, expected_row, predicted_row in zip(test_ids, test_labels, predicted, strict=True)
                for position, (expected, actual) in enumerate(zip(expected_row, predicted_row, strict=True))
            ]).to_csv(folder / "predictions.csv", index=False)
            for step, learned in enumerate(learned_stages):
                learned = np.asarray(learned)[:, 1:].reshape(-1, model.config.width)
                learned_profile = pca_profile(learned)
                oracle_step = round(step * (len(compiled_stages) - 1) / (len(learned_stages) - 1))
                target = np.asarray(compiled_stages[oracle_step])[:, 1:].reshape(len(learned), -1)
                target_profile = pca_profile(target)
                geometry.append({
                    "program": name, "seed": seed, "learned_step": step, "compiled_step": oracle_step,
                    "pairing": "relative-stage-progress; not a variable correspondence",
                    "learned_total_variance": learned_profile["total_variance"],
                    "compiled_total_variance": target_profile["total_variance"],
                    "learned_variance_active": learned_profile["variance_active"],
                    "compiled_variance_active": target_profile["variance_active"],
                    "learned_pr": learned_profile["participation_ratio"],
                    "compiled_pr": target_profile["participation_ratio"],
                    "cka": all_pair_cka([learned, target])[0, 1],
                })
            pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
            pd.DataFrame(geometry).to_csv(output / "geometry.csv", index=False)
            print(f"learning {name} seed {seed}: token={token_accuracy:.3f}, sequence={sequence_accuracy:.3f}", flush=True)
        del compiled_logits, compiled_stages, learned_logits, learned_stages, oracle, model
        jax.clear_caches()
    return {"models": len(summary), "steps": steps, "seeds": list(seeds), "split_seed": 2026, "origin": "random-init-trained"}


def run_validation(output, seeds):
    from .research import run_diagnostics, run_families

    programs = build_programs()
    config, probes = CompilerConfig(rel_tol=1e-4, abs_tol=1e-4), ProbeConfig()
    settings = {
        "probe_config": asdict(probes), "seeds": list(seeds),
        "sections": ["diagnostics", "families"], "variance_floor": VARIANCE_FLOOR,
        "interpretation": "completed means executed; negative learning, recovery, and compression results remain valid measurements",
    }
    with record_run(output / "validation", programs, config, **settings) as (run_dir, evidence):
        evidence["stage"] = "diagnostics"
        write_json(run_dir / "manifest.json", evidence)
        evidence["diagnostics"] = run_diagnostics(programs, config, probes, run_dir, seeds=seeds)
        evidence["stage"] = "families"
        write_json(run_dir / "manifest.json", evidence)
        evidence["families"] = run_families(config, probes, run_dir / "complexity")
        evidence["programs"] = [{"program": name} for name in programs]
    reference = promote_reference(run_dir, output / "validation")
    print(f"Saved {reference}", flush=True)
    return reference


def run_learning_study(output, seeds, steps):
    programs = build_programs()
    config, probes = CompilerConfig(rel_tol=1e-4, abs_tol=1e-4), ProbeConfig()
    settings = {
        "probe_config": asdict(probes), "seeds": list(seeds), "training_steps": steps,
        "optimizer": "Adam", "learning_rate": 0.003, "batch_size": 128,
        "origin": "random-init-trained", "split_seed": 2026,
        "split_unit": "whole unique sequence", "fractions": [0.6, 0.2, 0.2],
        "interpretation": "bounded CPU training; completed does not mean algorithm learned",
    }
    with record_run(output / "learning", programs, config, **settings) as (run_dir, evidence):
        evidence["stage"] = "training"
        write_json(run_dir / "manifest.json", evidence)
        evidence["learning"] = run_learning(programs, config, probes, run_dir, seeds, steps)
        evidence["programs"] = [{"program": name} for name in programs]
    return promote_reference(run_dir, output / "learning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--training-steps", type=int, default=600)
    parser.add_argument("--sections", nargs="+", choices=("baseline", "validation", "learning"), default=["baseline", "validation", "learning"])
    args = parser.parse_args()
    if args.training_steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive training steps and distinct seeds.")
    output = args.output.resolve()
    if "baseline" in args.sections:
        run_baseline(output)
    if "validation" in args.sections:
        run_validation(output, tuple(args.seeds))
    if "learning" in args.sections:
        run_learning_study(output, tuple(args.seeds), args.training_steps)


if __name__ == "__main__":
    main()

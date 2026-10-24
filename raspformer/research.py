"""Executable controls for representation geometry and blind write detection."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from tracr.compiler import nodes
from tracr.rasp import rasp

from . import models
from .alignment import encode_values, rasp_with_patch
from .compiler import compile_program, extract_schedule, trace_craft_graph, write_json
from .geometry import (all_pair_cka, candidate_pool, output_classes, pca_profile,
                       sample_targets)


@dataclass(frozen=True)
class SequenceSplit:
    candidates: list
    calibration_ids: np.ndarray
    test_ids: np.ndarray

    @property
    def calibration(self):
        return [self.candidates[i] for i in self.calibration_ids]

    @property
    def test(self):
        return [self.candidates[i] for i in self.test_ids]


def split_sequences(config, probe_config, output):
    candidates, _ = candidate_pool(config, probe_config)
    if len(candidates) < 192:
        raise ValueError("Research controls require at least 192 unique sequences.")
    order = np.random.default_rng(42).permutation(len(candidates))
    split = SequenceSplit(candidates, order[:96], order[96:192])
    rows = []
    for label, ids in (("calibration", split.calibration_ids), ("test", split.test_ids)):
        for index in ids:
            for position in range(len(candidates[index])):
                rows.append(dict(split=label, candidate_id=int(index), position=position,
                                 probe_id=f"{index}:{position}", tokens=candidates[index]))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "sequence-splits.csv", index=False)
    return split


def stage_name(index):
    return "Embedding" if index == 0 else f"L{(index + 1) // 2}/{'Attn' if index % 2 else 'MLP'}"


def flat(residual):
    array = np.asarray(residual, dtype=np.float64)
    return array[:, 1:, :].reshape(-1, array.shape[-1])


def encode(assembled, sequences, config):
    return jnp.asarray([assembled.input_encoder.encode([config.bos, *seq])
                        for seq in sequences], dtype=jnp.int32)


def decode(assembled, logits):
    logits = np.asarray(logits)
    if hasattr(assembled.output_encoder, "encoding_map"):
        return np.asarray([assembled.output_encoder.decode(row.tolist())
                           for row in logits.argmax(axis=-1)])[:, 1:]
    return logits[..., 0][:, 1:]


def accuracy(actual, expected, config):
    close = np.isclose(actual, expected, atol=config.abs_tol, rtol=config.rel_tol)
    return dict(token_accuracy=float(close.mean()), sequence_accuracy=float(close.all(axis=1).mean()))


def finite(value):
    return float(value) if np.isfinite(value) else None


def pr(matrix, variance_floor=1e-12):
    return pca_profile(matrix, variance_floor=variance_floor)["participation_ratio"]


def cka(left, right, variance_floor=1e-12):
    return float(all_pair_cka([left, right], variance_floor=variance_floor)[0, 1])


def infer_writes(matrices, variance_floor=1e-12):
    """Infer transitions from ordered arrays alone, using fixed geometric thresholds."""
    arrays = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    if not arrays or any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Supply identically shaped observation-by-feature matrices.")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Representations must be finite.")
    rows = []
    for stage, (before, after) in enumerate(zip(arrays, arrays[1:]), 1):
        delta = after - before
        energy = float(np.mean(delta * delta))
        relative = energy / max(float(np.mean(before * before)), variance_floor)
        before_profile = pca_profile(before, variance_floor=variance_floor)
        after_profile = pca_profile(after, variance_floor=variance_floor)
        active_before = before_profile["total_variance"] > variance_floor
        active_after = after_profile["total_variance"] > variance_floor
        adjacent = cka(before, after, variance_floor) if active_before and active_after else np.nan
        change = 1 - adjacent if np.isfinite(adjacent) else float(active_before != active_after)
        pr_change = abs(after_profile["participation_ratio"] - before_profile["participation_ratio"])
        rows.append(dict(stage=stage, delta_energy=energy, relative_delta_energy=relative,
                         before_total_variance=before_profile["total_variance"],
                         after_total_variance=after_profile["total_variance"],
                         before_variance_active=active_before, after_variance_active=active_after,
                         variance_floor=variance_floor,
                         delta_pr=pr(delta, variance_floor) if energy > variance_floor else 0.0,
                         adjacent_cka=finite(adjacent), geometry_change=float(change),
                         predict_change=bool(relative > 1e-6),
                         predict_pr=bool(pr_change > 0.05), predict_cka=bool(change > 0.01)))
    return rows


def recovery_scores(predictions, truth, seed=42):
    truth = np.asarray(truth, dtype=bool)
    rng = np.random.default_rng(seed)
    def score(predicted):
        tp = int((predicted & truth).sum())
        precision = tp / max(1, int(predicted.sum()))
        recall = tp / max(1, int(truth.sum()))
        return precision, recall, 2 * precision * recall / max(1e-15, precision + recall)
    rows = []
    for detector in ("change", "pr", "cka"):
        predicted = np.asarray([row[f"predict_{detector}"] for row in predictions])
        precision, recall, f1 = score(predicted)
        null = np.asarray([score(rng.permutation(predicted))[2] for _ in range(1000)])
        rows.append(dict(detector=detector, precision=precision, recall=recall, f1=f1,
                         predicted_count=int(predicted.sum()), true_count=int(truth.sum()),
                         null_mean_f1=float(null.mean()), null_p_ge=float((1 + (null >= f1).sum()) / 1001),
                         oracle_prevalence=float(truth.mean()),
                         all_positive_f1=score(np.ones_like(truth, dtype=bool))[2]))
    return rows


def import_check(assembled, model, encoded, config):
    logits, residuals = models.forward(model, encoded)
    original = assembled.forward(assembled.params, encoded)
    expected = (original.transformer_output.input_embeddings,
                *original.transformer_output.residuals)
    error = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(residuals, expected, strict=True))
    original_output = np.asarray(original.unembedded_output)[:, 1:]
    if hasattr(assembled.output_encoder, "encoding_map"):
        original_output = np.asarray([assembled.output_encoder.decode(row.tolist()) for row in original_output])
    result = accuracy(decode(assembled, logits), original_output, config)
    return logits, residuals, dict(residual_max_abs_error=error, decoded_agreement=result["token_accuracy"])


def basis_controls(assembled, model, calibration, encoded, reference, expected, config, output, seeds):
    matrices = [flat(value) for value in reference]
    width = matrices[0].shape[1]
    reference_cka = all_pair_cka(matrices)
    reference_pr = np.asarray([pr(value) for value in matrices])
    rotations, transform_predictions = [], []
    for seed in seeds:
        basis, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, width)))
        rotated = [value @ basis for value in matrices]
        post_cka = all_pair_cka(rotated)
        logits, states = models.forward(models.change_basis(model, basis), encoded)
        agreement = accuracy(decode(assembled, logits), decode(assembled, models.forward(model, encoded)[0]), config)
        transform_predictions.append(dict(kind="rotation", seed=seed, fraction=1.0,
            token_accuracy=agreement["token_accuracy"], predictions=infer_writes([flat(x) for x in states])))
        for stage, (original, posthoc, actual) in enumerate(zip(matrices, rotated, states)):
            rotations.append(dict(seed=seed, stage=stage_name(stage),
                original_total_variance=pca_profile(original)["total_variance"],
                rotated_total_variance=pca_profile(posthoc)["total_variance"],
                pr_error=abs(pr(posthoc) - reference_pr[stage]),
                cka_max_error=float(np.nanmax(np.abs(post_cka[stage] - reference_cka[stage]))),
                executed_residual_max_error=float(np.max(np.abs(flat(actual) - posthoc))),
                canonical_lane_agreement=float(np.isclose(posthoc, original, rtol=config.rel_tol, atol=config.abs_tol).mean()),
                **agreement))
    pd.DataFrame(rotations).to_csv(output / "rotations.csv", index=False)
    scales = np.geomspace(0.1, 10, width)
    pd.DataFrame([dict(stage=stage_name(i), original_pr=pr(x), scaled_pr=pr(x * scales),
                       original_scaled_cka=finite(cka(x, x * scales)))
                  for i, x in enumerate(matrices)]).to_csv(output / "nonorthogonal.csv", index=False)
    _, singular, vt = np.linalg.svd(np.concatenate([flat(x) for x in calibration]), full_matrices=False)
    uncentered_basis = vt.T
    np.savez(output / "calibration-basis.npz", basis=uncentered_basis, singular_values=singular)
    compressions = []
    for fraction, kind in [(f, "pca") for f in (0.25, 0.5, 0.75, 1.0)] + [(0.5, "random")]:
        rank = max(1, int(np.ceil(width * fraction)))
        selected = uncentered_basis[:, :rank] if kind == "pca" else np.linalg.qr(np.random.default_rng(42).normal(size=(width, width)))[0][:, :rank]
        norms = np.linalg.norm(selected, axis=1)
        directions = selected[norms > 1e-12] / norms[norms > 1e-12, None]
        overlap = np.abs(directions @ directions.T)
        overlaps = overlap[np.triu_indices(len(directions), 1)]
        reduced = models.change_basis(model, selected)
        logits, states = models.forward(reduced, encoded)
        scores = accuracy(decode(assembled, logits), expected, config)
        transform_predictions.append(dict(kind=f"compression-{kind}", seed=42, fraction=fraction,
            token_accuracy=scores["token_accuracy"], predictions=infer_writes([flat(x) for x in states])))
        reference_agreement = accuracy(decode(assembled, logits), decode(assembled, models.forward(model, encoded)[0]), config)["token_accuracy"]
        if kind == "pca" and fraction in (0.5, 1.0):
            models.save_model(reduced, output / f"compressed-{fraction}.npz")
        for stage, (original, state) in enumerate(zip(matrices, states)):
            actual = flat(state)
            profile = pca_profile(actual)
            compressions.append(dict(kind=kind, fraction=fraction, rank=rank, state_width=reduced.config.width,
                parameters=models.parameter_count(reduced), original_width=width, original_parameters=models.parameter_count(model),
                nonzero_axis_count=int(len(directions)), axis_overlap_mean=float(overlaps.mean()) if overlaps.size else 0.0,
                axis_overlap_max=float(overlaps.max()) if overlaps.size else 0.0, reference_token_agreement=reference_agreement,
                residual_projection_max_error=float(np.max(np.abs(actual - original @ selected))), heads=reduced.config.heads,
                key_size=reduced.config.key_size, mlp_hidden_size=reduced.config.mlp_hidden_size,
                stage=stage_name(stage), reconstruction_mse=float(np.mean((original - actual @ selected.T)**2)),
                pr=profile["participation_ratio"], total_variance=profile["total_variance"],
                variance_active=profile["variance_active"], stage_cka=finite(cka(original, actual)), **scores))
    pd.DataFrame(compressions).to_csv(output / "compression.csv", index=False)
    write_json(output / "transformed-blind-predictions.json", transform_predictions)
    return dict(rotation_geometry_max_error=max(max(r["pr_error"], r["cka_max_error"]) for r in rotations),
                rotation_min_token_agreement=min(r["token_accuracy"] for r in rotations),
                transform_predictions=transform_predictions,
                compression_final=[r for r in compressions if r["stage"] == stage_name(len(reference)-1)])


def ridge_fit(x, y):
    x = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(x.shape[1]) * 1e-3
    penalty[-1, -1] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def ridge_predict(x, weights):
    return np.column_stack([x, np.ones(len(x))]) @ weights


def decode_score(predicted, target, categorical):
    error = float(np.sum((target - predicted)**2))
    variance = float(np.sum((target - target.mean(axis=0))**2))
    r2 = 1 - error / variance if variance > 1e-12 else None
    agree = np.mean(predicted.argmax(axis=1) == target.argmax(axis=1)) if categorical else np.mean(np.isclose(predicted, target, atol=1e-3))
    return dict(r2=r2, accuracy=float(agree))


def geometry_patches(program, assembled, model, config, split, calibration, states, encoded, predictions, output):
    chosen = max(predictions, key=lambda row: row["geometry_change"])["stage"]
    matrix = flat(calibration[chosen])
    profile = pca_profile(matrix)
    basis = profile["components"]
    np.savez(output / "patch-calibration-basis.npz", basis=basis, mean=profile["mean"], stage=chosen)
    traced = trace_craft_graph(program, config)
    _, variables, _ = extract_schedule(program, assembled, config, traced=traced)
    evaluator = rasp.DefaultRASPEvaluator()
    decoder_rows, candidates = [], []
    length = len(split.calibration[0])
    held_out_calibration_sequences = np.arange(len(split.calibration)) % 4 == 0
    fold = np.repeat(held_out_calibration_sequences, length)
    pcs = (matrix - profile["mean"]) @ basis[:, :3]
    test_pcs = (flat(states[chosen]) - profile["mean"]) @ basis[:, :3]
    for variable in variables:
        node = traced.graph.nodes[variable["variable"]]
        expression = node[nodes.EXPR]
        targets = []
        for sequences in (split.calibration, split.test):
            values = [value for seq in sequences for value in evaluator.evaluate(expression, seq)]
            targets.append(encode_values(expression, node[nodes.OUTPUT_BASIS], values))
        weights = ridge_fit(pcs[~fold], targets[0][~fold])
        validation = decode_score(ridge_predict(pcs[fold], weights), targets[0][fold], rasp.is_categorical(expression))
        weights = ridge_fit(pcs, targets[0])
        heldout = decode_score(ridge_predict(test_pcs, weights), targets[1], rasp.is_categorical(expression))
        decoder_rows.append(dict(variable=variable["variable"], stage=chosen,
            calibration_cv_r2=validation["r2"], calibration_cv_accuracy=validation["accuracy"],
            test_r2=heldout["r2"], test_accuracy=heldout["accuracy"]))
        if variable["first_write_half_layer_index"] == chosen - 1:
            candidates.append((validation["r2"] if validation["r2"] is not None else -1e9, variable, expression))
    pd.DataFrame(decoder_rows).to_csv(output / "variable-decoders.csv", index=False)
    if not candidates:
        return dict(selected_stage=chosen, status="no compiler variable first written at geometrically selected stage")
    _, variable, expression = max(candidates, key=lambda item: item[0])
    rng = np.random.default_rng(42)
    pairs = [(int(rng.integers(96)), int(rng.integers(96)), int(rng.integers(length))) for _ in range(16)]
    random_basis = np.linalg.qr(rng.normal(size=(matrix.shape[1], matrix.shape[1])))[0]
    baseline = decode(assembled, models.forward(model, encoded)[0])
    rows = []
    for pair_id, (recipient, donor, position) in enumerate(pairs):
        donor_values = evaluator.evaluate(expression, split.test[donor])
        counterfactual = np.asarray(rasp_with_patch(program, expression, split.test[recipient], position, donor_values[position]))
        delta = np.asarray(states[chosen])[donor, position+1] - np.asarray(states[chosen])[recipient, position+1]
        for kind, rank in [("pca", 1), ("pca", 3), ("random", 1), ("random", 3), ("self", 0), ("oracle", len(variable["residual_lane_indices"]))]:
            change = np.zeros_like(delta)
            if kind in ("pca", "random"):
                projection = (basis if kind == "pca" else random_basis)[:, :rank]
                change = delta @ projection @ projection.T
            elif kind == "oracle":
                lanes = variable["residual_lane_indices"]
                change[lanes] = delta[lanes]
            logits, _ = models.forward(model, encoded[recipient:recipient+1], patches=[
                dict(stage=chosen, batch=0, position=position+1, delta=change)])
            actual = decode(assembled, logits)[0]
            rows.append(dict(pair_id=pair_id, recipient_id=int(split.test_ids[recipient]), donor_id=int(split.test_ids[donor]),
                position=position, kind=kind, rank=rank, stage=stage_name(chosen), variable=variable["variable"],
                decoded_outputs=actual.tolist(), symbolic_counterfactual=counterfactual.tolist(),
                behavior_changed=bool(not np.allclose(actual, baseline[recipient], atol=config.abs_tol, rtol=config.rel_tol)),
                symbolic_counterfactual_defined=bool(all(value is not None for value in counterfactual)),
                full_counterfactual_match=bool(all(value is not None for value in counterfactual) and np.allclose(actual, counterfactual.astype(float), atol=config.abs_tol, rtol=config.rel_tol))))
    pd.DataFrame(rows).to_csv(output / "patches.csv", index=False)
    return dict(selected_stage=chosen, variable=variable["variable"],
                controls=pd.DataFrame(rows).groupby(["kind", "rank"])[["behavior_changed", "full_counterfactual_match"]].mean().reset_index().to_dict("records"))


def sharpening(program, assembled, model, config, probe_config, output, program_index):
    candidates, families = candidate_pool(config, probe_config)
    observations = sample_targets("P12_sorting", program, candidates, families,
        output_classes(config, probe_config.length)["P12_sorting"], probe_config,
        seed=probe_config.seed + 1009 * program_index)
    observations.to_csv(output / "balanced-probes.csv", index=False)
    traced = trace_craft_graph(program, config)
    _, variables, _ = extract_schedule(program, assembled, config, traced=traced)
    variable = next(row for row in variables if row["variable"] == program.label)
    first = variable["first_write_half_layer_index"] + 1
    node = traced.graph.nodes[program.label]
    expected = encode_values(program, node[nodes.OUTPUT_BASIS], observations.target_class.tolist())
    encoded = encode(assembled, observations.tokens, config)
    rows, summaries, failing, failing_lanes = [], [], {}, {}
    for scale in (1, 2, 4, 8):
        logits, states = models.forward(model, encoded, attention_scale=float(scale))
        decoded = decode(assembled, logits)
        positions = observations.position.to_numpy()
        actual_outputs = decoded[np.arange(len(observations)), positions]
        for stage in sorted(set((first, len(states)-1))):
            actual = np.asarray(states[stage])[np.arange(len(observations)), positions+1][:, variable["residual_lane_indices"]]
            close = np.isclose(actual, expected, rtol=config.rel_tol, atol=config.abs_tol)
            for i, observation in enumerate(observations.itertuples()):
                for lane, label in enumerate(variable["residual_lane_labels"]):
                    rows.append(dict(attention_scale=scale, probe_id=int(observation.probe_id), tokens=observation.tokens,
                        target_position=int(observation.position), stage=stage_name(stage), output_lane=label,
                        expected=float(expected[i,lane]), observed=float(actual[i,lane]), error=float(actual[i,lane]-expected[i,lane]), close=bool(close[i,lane])))
            failing[f"{scale}/{stage_name(stage)}"] = observations.probe_id[~close.all(axis=1)].tolist()
            for lane, label in enumerate(variable["residual_lane_labels"]):
                failing_lanes[f"{scale}/{stage_name(stage)}/{label}"] = observations.probe_id[~close[:,lane]].tolist()
        summaries.append(dict(attention_scale=scale, token_accuracy=float(np.isclose(actual_outputs, observations.target_class, atol=config.abs_tol, rtol=config.rel_tol).mean()),
                              max_abs_lane_error=float(np.max(np.abs(actual-expected))), mean_abs_lane_error=float(np.mean(np.abs(actual-expected)))))
    pd.DataFrame(rows).to_csv(output / "per-observation-errors.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "attention-sharpening.csv", index=False)
    baseline_lane_sets = [set(ids) for key, ids in failing_lanes.items() if key.startswith("1/")]
    same_rows = all(ids == baseline_lane_sets[0] for ids in baseline_lane_sets)
    write_json(output / "failing-probe-sets.json", dict(stage_union=failing, per_lane=failing_lanes, baseline_all_lanes_identical=same_rows))
    baseline_failed = set().union(*(set(ids) for key, ids in failing.items() if key.startswith("1/")))
    observations[observations.probe_id.isin(baseline_failed)].to_csv(output / "failing-input-patterns.csv", index=False)
    base_keys = [key for key in failing if key.startswith("1/")]
    shared = sorted(set.intersection(*(set(failing[key]) for key in base_keys)))
    return dict(sweep=summaries, shared_baseline_failing_probe_ids=shared, baseline_all_lanes_identical=same_rows,
                tightening_reduces_max_error=summaries[-1]["max_abs_lane_error"] < summaries[0]["max_abs_lane_error"])


def run_diagnostics(programs, config, probe_config, output: Path, seeds=(42, 43, 44)):
    output = Path(output)
    split = split_sequences(config, probe_config, output)
    summary = dict(sequence_count_per_split=96, split_seed=42, programs=[])
    evaluator = rasp.DefaultRASPEvaluator()
    for program_index, (name, program) in enumerate(programs.items()):
        directory = output / name.lower().replace("_", "-")
        directory.mkdir(parents=True, exist_ok=True)
        assembled = compile_program(program, config)
        model = models.from_compiled(assembled)
        encoded = encode(assembled, split.test, config)
        logits, states, imported = import_check(assembled, model, encoded, config)
        _, calibration = models.forward(model, encode(assembled, split.calibration, config))
        np.savez_compressed(directory / "activations.npz",
            calibration=np.stack(calibration).astype(np.float32), test=np.stack(states).astype(np.float32),
            stage_names=np.asarray(model.stage_names), calibration_ids=split.calibration_ids, test_ids=split.test_ids)
        expected = np.asarray([evaluator.evaluate(program, seq) for seq in split.test])
        predictions = infer_writes([flat(state) for state in states])
        patch_predictions = infer_writes([flat(state) for state in calibration])
        write_json(directory / "patch-stage-predictions.json", {
            "source_split": "calibration", "predictions": patch_predictions,
            "selected_stage": max(patch_predictions, key=lambda row: row["geometry_change"])["stage"],
        })
        frozen = directory / "blind-predictions.json"
        write_json(frozen, predictions)
        digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
        timestamp = datetime.now(UTC).isoformat()
        controls = basis_controls(assembled, model, calibration, encoded, states, expected, config, directory, seeds)
        transform_predictions = controls.pop("transform_predictions")
        transform_digest = hashlib.sha256((directory / "transformed-blind-predictions.json").read_bytes()).hexdigest()
        _, variables, _ = extract_schedule(program, assembled, config)
        writes = {v["first_write_half_layer_index"] + 1 for v in variables if v["first_write_half_layer_index"] is not None}
        truth = [stage in writes for stage in range(1, len(states))]
        scores = recovery_scores(predictions, truth)
        pd.DataFrame(scores).to_csv(directory / "write-recovery-scores.csv", index=False)
        group = "calibration" if int(name[1:3]) <= 6 else "heldout"
        transformed_scores = [
            {"kind": item["kind"], "fraction": item["fraction"], "seed": item["seed"],
             "token_accuracy": item["token_accuracy"], **score}
            for item in transform_predictions for score in recovery_scores(item["predictions"], truth)
        ]
        pd.DataFrame(transformed_scores).to_csv(directory / "transformed-write-recovery.csv", index=False)
        patches = geometry_patches(program, assembled, model, config, split, calibration, states, encoded, patch_predictions, directory)
        row = dict(program=name, import_check=imported, baseline=accuracy(decode(assembled, logits), expected, config),
                   recovery_group=group, recovery=scores, predictions_sha256=digest, predictions_frozen_at=timestamp,
                   transformed_recovery=transformed_scores, transformed_predictions_sha256=transform_digest,
                   patches=patches, **controls)
        if name == "P12_sorting":
            row["sharpening"] = sharpening(program, assembled, model, config, probe_config, directory, 11)
        summary["programs"].append(row)
        write_json(output / "summary.json", summary)
        print(f"{name}: executing rotation, compression, recovery and patches complete", flush=True)
    return summary


def family_program(depth, branches, vocab_size):
    branch_outputs = []
    for branch in range(branches):
        value = rasp.tokens
        for level in range(depth):
            offset = branch + level + 1
            sign = -1 if level % 2 else 1
            value = rasp.Map(lambda token, a=sign, b=offset: (a * token + b) % vocab_size,
                             value, simplify=False).named(f"branch_{branch}_step_{level}")
        branch_outputs.append(value)
    result = branch_outputs[0]
    for branch, value in enumerate(branch_outputs[1:], 1):
        result = rasp.SequenceMap(lambda left, right: (left + right) % vocab_size,
                                  result, value).named(f"merge_{branch}")
    return result


def run_families(config, probe_config, output: Path):
    output = Path(output)
    split = split_sequences(config, probe_config, output)
    rows, stages = [], []
    evaluator = rasp.DefaultRASPEvaluator()
    for depth in (1, 2, 3):
        for branches in (1, 2, 3):
            program = family_program(depth, branches, len(config.vocab))
            assembled = compile_program(program, config)
            metadata, variables, _ = extract_schedule(program, assembled, config)
            model = models.from_compiled(assembled)
            logits, states = models.forward(model, encode(assembled, split.test, config))
            expected = np.asarray([evaluator.evaluate(program, seq) for seq in split.test])
            row = dict(requested_depth=depth, requested_branches=branches, **metadata,
                       **accuracy(decode(assembled, logits), expected, config))
            rows.append(row)
            for index, state in enumerate(states):
                profile = pca_profile(flat(state))
                stages.append(dict(requested_depth=depth, requested_branches=branches,
                                   stage=stage_name(index), pr=profile["participation_ratio"],
                                   total_variance=profile["total_variance"], variance_active=profile["variance_active"]))
            write_json(output / f"schedule-d{depth}-b{branches}.json", variables)
    pd.DataFrame(rows).to_csv(output / "factorial-families.csv", index=False)
    pd.DataFrame(stages).to_csv(output / "family-geometry.csv", index=False)
    axes = np.asarray([[row["max_dependency_depth"], row["computed_variable_count"]] for row in rows])
    rank = int(np.linalg.matrix_rank(axes - axes.mean(axis=0)))
    result = dict(families=rows, realized_axes_rank=rank, realized_axes_noncollinear=rank == 2)
    write_json(output / "summary.json", result)
    return result

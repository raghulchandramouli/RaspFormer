"""Compare representation metrics with RASP values, writes, and A/B interventions."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from tracr.compiler import nodes
from tracr.rasp import rasp

from .compiler import (
    build_trace_model, check_trace_sample, extract_schedule, manual_craft_forward,
    record_run, trace_craft_graph, write_json,
)
from .geometry import all_pair_cka, build_balanced_probes, collect_activations, pca_profile

EXAMPLES = ("A_prev_token_class", "B_histogram_then_repeated_class")


def encode_values(expression, basis, values):
    """Use the compiler's basis order; undefined RASP values have zero encoding."""
    if rasp.is_categorical(expression):
        support = {direction.value for direction in basis}
        if any(value is not None and value not in support for value in values):
            raise ValueError(f"Value outside the inferred basis for {expression.label}.")
        return np.asarray([
            [float(value is not None and value == direction.value) for direction in basis]
            for value in values
        ])
    return np.asarray([[0.0 if value is None else float(value)] for value in values])


def measure_alignment(name, program, config, data):
    """Compare stage vectors with encoded variables written so far (not live variables)."""
    model, table = data.models[name], data.observations[name]
    traced = trace_craft_graph(program, config)
    summary, variables, _ = extract_schedule(program, model, config, traced=traced)
    evaluator = rasp.DefaultRASPEvaluator()
    encoded, value_rows = {}, []
    for variable in variables:
        label = variable["variable"]
        node = traced.graph.nodes[label]
        values = [
            evaluator.evaluate(node[nodes.EXPR], row.tokens)[row.position]
            for row in table.itertuples()
        ]
        encoded[label] = encode_values(node[nodes.EXPR], node[nodes.OUTPUT_BASIS], values)
        value_rows.extend(
            {"program": name, "probe_id": probe_id, "variable": label, "value": value}
            for probe_id, value in zip(table.probe_id, values, strict=True)
        )
    expected = np.zeros_like(next(iter(data.stages[name].values())))
    # Tracr embeds positions even when the RASP graph never reads indices.
    for prefix, values in (("tokens", table.token), ("indices", table.position)):
        for value in values.unique():
            label = f"{prefix}:{value}"
            expected[:, model.residual_labels.index(label)] = (values == value).to_numpy()
    if "one" in model.residual_labels:
        expected[:, model.residual_labels.index("one")] = 1
    stage_rows, lane_rows, written = [], [], []
    previous, previous_expected = {}, {}
    for step, (stage, actual) in enumerate(data.stages[name].items()):
        new_variables = []
        for variable in variables:
            first = variable["first_write_half_layer_index"]
            if step != (0 if first is None else first + 1):
                continue
            expected[:, variable["residual_lane_indices"]] = encoded[variable["variable"]]
            if first is not None:
                written.append(variable)
                new_variables.append(variable["variable"])
        for variable in written:
            for column, label in zip(variable["residual_lane_indices"], variable["residual_lane_labels"]):
                target, observed = expected[:, column], actual[:, column]
                correlation = np.nan
                if np.var(target) > 0 and np.var(observed) > 0:
                    correlation = float(np.corrcoef(target, observed)[0, 1])
                lane_rows.append({
                    "program": name, "stage": stage, "variable": variable["variable"],
                    "is_first_write": variable["variable"] in new_variables,
                    "lane": label, "max_abs_error": float(np.max(np.abs(target - observed))),
                    "fraction_close": float(np.isclose(observed, target, rtol=config.rel_tol, atol=config.abs_tol).mean()),
                    "pearson_r": correlation,
                })
        for view, columns in data.columns[name].items():
            observed, target = actual[:, columns], expected[:, columns]
            actual_pr = pca_profile(observed)["participation_ratio"]
            expected_pr = pca_profile(target)["participation_ratio"]
            cka = all_pair_cka([previous[view], observed])[0, 1] if step else np.nan
            expected_cka = all_pair_cka([previous_expected[view], target])[0, 1] if step else np.nan
            stage_rows.append({
                "program": name, "view": view, "stage": stage, "step": step,
                "observations": len(table), "written_variables": len(written),
                "new_variables": ";".join(new_variables), "new_variable_count": len(new_variables),
                "computed_variable_count": summary["computed_variable_count"],
                "max_dependency_depth": summary["max_dependency_depth"],
                "transformer_layers": summary["transformer_layers"],
                "actual_pr": actual_pr, "expected_pr": expected_pr,
                "pr_gap": actual_pr - expected_pr, "adjacent_cka": cka,
                "expected_adjacent_cka": expected_cka,
            })
            previous[view] = observed
            previous_expected[view] = target  # Advanced column indexing above makes a copy.
    return {
        "stages": pd.DataFrame(stage_rows), "lanes": pd.DataFrame(lane_rows),
        "variables": pd.DataFrame([{"program": name, **v} for v in variables]),
        "values": pd.DataFrame(value_rows),
    }


def summarize_alignment(stages, lanes):
    """Equal stage weights for PR/CKA; equal scalar-lane weights at first writes."""
    rows = []
    for (name, view), group in stages.groupby(["program", "view"], sort=False):
        transitions = group[group.step > 0]
        first_writes = lanes[(lanes.program == name) & lanes.is_first_write]
        observed = np.isfinite(transitions.adjacent_cka)
        expected = np.isfinite(transitions.expected_adjacent_cka)
        paired = transitions[observed & expected]
        rows.append({
            "program": name, "view": view,
            **group.iloc[0][[
                "observations", "computed_variable_count", "max_dependency_depth", "transformer_layers",
            ]].to_dict(),
            "pr_mae": transitions.pr_gap.abs().mean(),
            "pr_signed_mean": transitions.pr_gap.mean(),
            "pr_final_gap": group.iloc[-1].pr_gap,
            "transitions": len(transitions),
            "cka_mae": (paired.adjacent_cka - paired.expected_adjacent_cka).abs().mean(),
            "cka_observed_defined": int(observed.sum()),
            "cka_expected_defined": int(expected.sum()),
            "cka_paired_defined": len(paired),
            "cka_definition_mismatches": int((observed != expected).sum()),
            "lane_agreement": first_writes.fraction_close.mean(),
            "first_write_lanes": len(first_writes),
            "lane_max_abs_error": first_writes.max_abs_error.max(),
        })
    return pd.DataFrame(rows)


def rasp_with_patch(program, variable, tokens, position, value):
    """Reevaluate the RASP graph with one intermediate target value substituted."""
    class PatchedEvaluator(rasp.DefaultRASPEvaluator):
        def evaluate(self, expression, xs):
            values = super().evaluate(expression, xs)
            if expression.label == variable.label:
                values = list(values)
                values[position] = value
            return values

    return PatchedEvaluator().evaluate(program, tokens)


def intervene(name, program, config, table):
    """Patch a valid donor value at one target after its write; keep self-controls."""
    if name not in EXAMPLES:
        raise ValueError("Causal checks are limited to the agreed A/B examples.")
    config.validate_samples(table.tokens.tolist())
    traced = trace_craft_graph(program, config)
    trace = build_trace_model(program, config, traced=traced)
    variable = program if name == EXAMPLES[0] else program.inner
    basis = traced.graph.nodes[variable.label][nodes.OUTPUT_BASIS]
    _, schedule, _ = extract_schedule(program, trace.assembled, config, traced=traced)
    [allocation] = [row for row in schedule if row["variable"] == variable.label]
    half_layer = allocation["first_write_half_layer_index"]
    kind = "attn" if half_layer % 2 == 0 else "mlp"
    writer = f"transformer/layer_{half_layer // 2}/{kind}"
    if writer not in trace.module_names:
        raise ValueError(f"No active writer for {variable.label}.")
    columns = allocation["residual_lane_indices"]
    evaluator = rasp.DefaultRASPEvaluator()
    rows, baselines = [], {}
    for target in table.itertuples():
        tokens, position = target.tokens, target.position
        key = tuple(tokens)
        if key not in baselines:
            record, warnings = check_trace_sample(name, trace, tokens, config)
            if warnings:
                raise ValueError(f"{name}: baseline residual mismatch; intervention is not validated.")
            baselines[key] = {field: record[field] for field in ("rasp_output", "residual_max_abs_error")}
        baseline = baselines[key]
        original = evaluator.evaluate(variable, tokens)[position]
        candidates = table[(table.position == position) & (table.target_class != target.target_class)]
        if candidates.empty:
            raise ValueError(f"{name}: no outcome-changing donor at position {position}.")
        donor = candidates.iloc[0]
        replacement = evaluator.evaluate(variable, donor.tokens)[position]
        for condition, value in (("control", original), ("donor", replacement)):
            encoded = encode_values(variable, basis, [value])[0]

            def patch(module, residual):
                if module != writer:
                    return residual
                magnitudes = residual.magnitudes.copy()
                magnitudes[position + 1, columns] = encoded  # BOS is never patched.
                return trace.full_space.make_vector(magnitudes)

            final, _, _ = manual_craft_forward(trace, tokens, bos=config.bos, after_block=patch)
            scores = final.project(trace.output_space).magnitudes[1:]
            actual = trace.assembled.output_encoder.decode(scores.argmax(axis=1).tolist())
            expected = list(rasp_with_patch(program, variable, tokens, position, value))
            expected_scores = encode_values(program, trace.output_space.basis, expected)
            changed = [i for i, (a, b) in enumerate(zip(actual, baseline["rasp_output"])) if a != b]
            predicted_changed = [i for i, (a, b) in enumerate(zip(expected, baseline["rasp_output"])) if a != b]
            rows.append({
                "program": name, "probe_id": target.probe_id, "position": position,
                "tokens": json.dumps(tokens), "variable": variable.label, "writer": writer,
                "condition": condition, "donor_probe_id": int(donor.probe_id) if condition == "donor" else target.probe_id,
                "donor_tokens": json.dumps(donor.tokens if condition == "donor" else tokens),
                "original_value": original, "patched_value": value,
                "baseline_output": json.dumps(baseline["rasp_output"]),
                "expected_output": json.dumps(expected), "actual_output": json.dumps(actual),
                "predicted_changed_positions": json.dumps(predicted_changed),
                "changed_positions": json.dumps(changed), "output_matches": actual == expected,
                "scores_close": bool(np.allclose(scores, expected_scores, rtol=config.rel_tol, atol=config.abs_tol)),
                "max_abs_score_error": float(np.max(np.abs(scores - expected_scores))),
                "baseline_max_abs_residual_error": baseline["residual_max_abs_error"],
            })
    return pd.DataFrame(rows)


def run_alignment_study(programs, config, probe_config, output_root):
    """Reuse Step 3's sampler and activation collector; export descriptive checks."""
    settings = {
        "probe_config": asdict(probe_config),
        "ground_truth": "compiler-encoded RASP variables written so far; unwritten and scratch lanes zero; full view includes inputs",
        "complexity": ["computed_variable_count", "max_dependency_depth"],
        "intervention": "A output and B histogram; one target; first donor at same position with different output class; self-control",
        "scope": "observational alignment for supplied programs; causal checks only for A/B",
        "interpretation": "descriptive hypotheses; separate program probes; repeats retain weight; no live-variable count; constant-lane correlation and zero-variance CKA undefined",
    }
    with record_run(output_root, programs, config, **settings) as (run_dir, evidence):
        evidence["stage"] = "probes"
        write_json(run_dir / "manifest.json", evidence)
        probes = build_balanced_probes(programs, config, probe_config)
        evidence["stage"] = "activations"
        write_json(run_dir / "manifest.json", evidence)
        data = collect_activations(programs, config, probes)
        tables = {key: [] for key in ("stages", "lanes", "variables", "values", "interventions")}

        def save_table(key, frame):
            tables[key].append(frame)
            path = run_dir / f"{key}.csv"
            frame.to_csv(path, mode="a", header=not path.exists(), index=False)

        for name, program in programs.items():
            evidence.update(stage="alignment", current_program=name)
            write_json(run_dir / "manifest.json", evidence)
            measurements = measure_alignment(name, program, config, data)
            for key, frame in measurements.items():
                save_table(key, frame)
            data.observations[name].assign(tokens=data.observations[name].tokens.map(json.dumps)).to_csv(
                run_dir / f"probes_{name}.csv", index=False,
            )
            if name in EXAMPLES:
                evidence["stage"] = "interventions"
                write_json(run_dir / "manifest.json", evidence)
                save_table("interventions", intervene(name, program, config, data.observations[name]))
            evidence["programs"].append({"program": name, "observations": len(data.observations[name])})
            print(f"{name}: alignment recorded")
        tables = {key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for key, frames in tables.items()}
    return {**tables, "run_dir": run_dir, "evidence": evidence}

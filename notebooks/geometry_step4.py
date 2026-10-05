"""Ground-truth schedule, lane-correlation, and ablation checks for Step 4."""

from collections import defaultdict
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from tracr.compiler import (
    assemble,
    basis_inference,
    craft_graph_to_model,
    expr_to_craft_graph,
    nodes,
    rasp_to_graph,
)
from tracr.craft import bases, transformers
from tracr.rasp import rasp


NULL_DRAWS = 64
CV_FOLDS = 4
ABLATION_ROWS_PER_CLASS = 8


def _extract_schedule(program, model, vocab, max_seq_len, bos, mlp_exactness):
    """Re-run Tracr's graph analysis to get every variable's first-write stage."""
    traced = rasp_to_graph.extract_rasp_graph(program)
    graph, sources, sink = traced.graph, traced.sources, traced.sink
    basis_inference.infer_bases(graph, sink, vocab, max_seq_len)
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        graph,
        bos_dir=bases.BasisDirection(rasp.tokens.label, bos),
        mlp_exactness=mlp_exactness,
    )
    allocation = craft_graph_to_model._allocate_modules_to_layers(graph, sources)
    depth = craft_graph_to_model.compute_computational_depth(
        graph, [source[nodes.ID] for source in sources]
    )
    lane_index = {label: i for i, label in enumerate(model.residual_labels)}
    variables = []
    expr_by_name = {}

    for node_id, node in graph.nodes(data=True):
        expr = node[nodes.EXPR]
        if not isinstance(expr, rasp.SOp):
            continue
        block = node.get(nodes.MODEL_BLOCK)
        first = allocation.get(node_id)
        if isinstance(block, transformers.SeriesWithResiduals):
            first += 1
        if first is None and expr.label not in (rasp.tokens.label, rasp.indices.label):
            raise ValueError(f"No writer allocated for {expr.label}")
        labels = [str(direction) for direction in node[nodes.OUTPUT_BASIS]]
        variables.append({
            "variable": expr.label,
            "lane_labels": labels,
            "lane_indices": [lane_index[label] for label in labels],
            "first_write_half_layer_index": first,
            "first_write_after": (
                "input embedding" if first is None else
                f"after {'attention' if first % 2 == 0 else 'mlp'}{first // 2 + 1}"
            ),
            "computed": first is not None,
        })
        expr_by_name[expr.label] = expr

    craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)
    model_config, module_names = assemble._get_model_config_and_module_names(craft_model)
    if model_config.num_layers != model.model_config.num_layers:
        raise ValueError(f"Schedule/model depth mismatch: {model_config.num_layers}")
    output_space = bases.VectorSpaceWithBasis(sink[nodes.OUTPUT_BASIS])
    return {
        "variables": variables,
        "expr_by_name": expr_by_name,
        "craft_model": craft_model,
        "full_space": craft_model.residual_space,
        "output_space": output_space,
        "module_names": module_names,
        "sink": sink,
        "max_dependency_depth": max(depth.values()),
    }


def _categories(values):
    keys = [(type(value).__name__, repr(value)) for value in values]
    unique = sorted(set(keys))
    index = {key: i for i, key in enumerate(unique)}
    codes = np.asarray([index[key] for key in keys], dtype=int)
    return np.eye(len(unique), dtype=float)[codes], unique


def _group_folds(groups, n_splits=CV_FOLDS, seed=0):
    unique = np.unique(groups)
    if len(unique) < 2:
        return []
    n_splits = min(n_splits, len(unique))
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    fold_for_group = {group: i % n_splits for i, group in enumerate(unique)}
    folds = []
    for fold in range(n_splits):
        test = np.asarray([fold_for_group[group] == fold for group in groups])
        folds.append((~test, test))
    return folds


def _cv_r2(features, labels, groups, folds):
    if features.shape[1] == 0 or np.max(np.abs(features), initial=0) < 1e-12:
        return 0.0
    sse = denominator = 0.0
    for train, test in folds:
        if not train.any() or not test.any():
            continue
        design = np.column_stack([np.ones(train.sum()), features[train]])
        weights, *_ = np.linalg.lstsq(design, labels[train], rcond=None)
        predicted = np.column_stack([np.ones(test.sum()), features[test]]) @ weights
        baseline = labels[train].mean(axis=0, keepdims=True)
        sse += float(np.square(labels[test] - predicted).sum())
        denominator += float(np.square(labels[test] - baseline).sum())
    return 0.0 if denominator == 0 else 1.0 - sse / denominator


def _manual_batch(ctx, model, token_batches, ablate_after=None, ablate_lanes=None):
    """Batched manual Tracr forward pass with an optional residual intervention."""
    full_space = ctx["full_space"]
    output_space = ctx["output_space"]
    craft_model = ctx["craft_model"]
    module_names = ctx["module_names"]
    batch_size, seq_len = len(token_batches), len(token_batches[0]) + 1
    x = np.zeros((batch_size, seq_len, full_space.num_dims), dtype=np.float64)
    direction_index = full_space.index_by_direction
    for row, tokens in enumerate(token_batches):
        for position, token in enumerate([model.input_encoder.bos_token, *tokens]):
            x[row, position, direction_index[bases.BasisDirection(rasp.tokens.label, token)]] = 1.0
            if position:
                x[row, position, direction_index[bases.BasisDirection(rasp.indices.label, position - 1)]] = 1.0
    one = bases.BasisDirection("one")
    if one in full_space:
        x[:, :, direction_index[one]] = 1.0
    residual = full_space.make_vector(x)

    blocks = {}
    for name, block in zip(module_names, craft_model.blocks):
        _, layer_name, component = name.split("/")
        layer = int(layer_name.split("_")[-1])
        half = 2 * layer + (component == "mlp")
        blocks[half] = block

    for half in range(2 * model.model_config.num_layers):
        block = blocks.get(half)
        if block is not None:
            block_input = residual.project(block.residual_space)
            if isinstance(block, (transformers.AttentionHead, transformers.MultiAttentionHead)):
                heads = [block] if isinstance(block, transformers.AttentionHead) else list(block.heads())
                head_outputs = []
                for head in heads:
                    queries = block_input.project(head.w_qk.left_space).magnitudes
                    keys = block_input.project(head.w_qk.right_space).magnitudes
                    logits = (queries @ head.w_qk.matrix @ keys.transpose(0, 2, 1)) / np.sqrt(model.model_config.key_size)
                    if model.model_config.causal:
                        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))[None, :, :]
                        logits = np.where(causal_mask, logits, -1e30)
                    logits -= logits.max(axis=-1, keepdims=True)
                    weights = np.exp(logits)
                    weights /= weights.sum(axis=-1, keepdims=True)
                    values = head.w_ov_residual(block_input).magnitudes
                    head_outputs.append(head.residual_space.make_vector(weights @ values))
                delta = bases.VectorInBasis.sum(head_outputs)
            elif isinstance(block, transformers.MLP):
                delta = block.apply(block_input)
            else:
                raise TypeError(f"Unsupported block {type(block).__name__}")
            residual = residual + delta.project(full_space)

        if half == ablate_after and ablate_lanes:
            magnitudes = np.array(residual.magnitudes, copy=True)
            magnitudes[:, :, ablate_lanes] = 0.0
            residual = residual.copy_with_new_magnitudes(magnitudes)

    output = residual.project(output_space).magnitudes
    return output


def _decode_scores(model, scores):
    if model.output_encoder is None:
        return np.asarray(scores)
    if hasattr(model.output_encoder, "encoding_map"):
        indices = np.argmax(scores, axis=-1).reshape(-1).tolist()
        return np.asarray(model.output_encoder.decode(indices), dtype=object).reshape(scores.shape[:-1])
    return np.asarray(scores).squeeze(-1)


def run_step4(namespace):
    """Compute and save the three ground-truth comparisons for all programs."""
    started = perf_counter()
    programs = namespace["PROGRAMS"]
    models = namespace["MODELS"]
    activations = namespace["ACTIVATIONS"]
    observations = namespace["OBSERVATIONS"]
    metrics = namespace["METRICS"]
    run_dir = namespace["RUN_DIR"]
    figure_dir = namespace["FIGURE_DIR"]
    vocab = namespace["VOCAB"]
    max_seq_len = namespace["MAX_SEQ_LEN"]
    bos = namespace["BOS"]
    mlp_exactness = namespace["MLP_EXACTNESS"]
    evaluator = rasp.DefaultRASPEvaluator()

    contexts = {}
    summaries = {}
    for name, program in programs.items():
        ctx = _extract_schedule(program, models[name], vocab, max_seq_len, bos, mlp_exactness)
        contexts[name] = ctx
        variables = ctx["variables"]
        summaries[name] = {
            "program": name,
            "layers": models[name].model_config.num_layers,
            "computed_variables": sum(v["computed"] for v in variables),
            "max_dependency_depth": int(ctx["max_dependency_depth"]),
        }

    schedule_rows = []
    for name, ctx in contexts.items():
        for var in ctx["variables"]:
            schedule_rows.append({"program": name, **var})
    schedule = pd.DataFrame(schedule_rows)

    stage_rows = []
    cka_rows = []
    for name, model in models.items():
        variables = contexts[name]["variables"]
        for stage, X in activations[name].items():
            if stage == "Embedding":
                half = -1
            else:
                layer = int(stage.split("/")[0][1:])
                half = 2 * (layer - 1) + (stage.endswith("/MLP"))
            live = sum(v["computed"] and v["first_write_half_layer_index"] <= half for v in variables)
            row = metrics[(metrics.program == name) & (metrics.view == "computed") & (metrics.stage == stage)].iloc[0]
            stage_rows.append({
                "program": name, "stage": stage, "half_layer_index": half,
                "live_computed_variables": live,
                "participation_ratio": float(row.participation_ratio),
                "pc95": int(row.pc95),
                "max_dependency_depth": summaries[name]["max_dependency_depth"],
                "computed_variable_total": summaries[name]["computed_variables"],
            })

        cka = namespace["GEOMETRY"][name]["computed"]["cka"]
        stages = list(activations[name])
        for i in range(1, len(stages)):
            destination = stages[i]
            half = 2 * (int(destination.split("/")[0][1:]) - 1) + destination.endswith("/MLP")
            writes = [v["variable"] for v in variables if v["computed"] and v["first_write_half_layer_index"] == half]
            value = float(cka[i - 1, i]) if np.isfinite(cka[i - 1, i]) else np.nan
            cka_rows.append({
                "program": name, "from_stage": stages[i - 1], "to_stage": destination,
                "linear_cka": value, "variables_first_written": ", ".join(writes),
                "new_variable_count": len(writes),
                "cka_low_lt_0_95": bool(value < 0.95) if np.isfinite(value) else pd.NA,
                "prediction_mismatch": (
                    "undefined_zero_variance" if not np.isfinite(value) else
                    "high_CKA_despite_write" if writes and value >= 0.95 else
                    "low_CKA_without_write" if not writes and value < 0.95 else ""
                ),
            })

    stage_metrics = pd.DataFrame(stage_rows)
    cka_edges = pd.DataFrame(cka_rows)
    schedule.to_csv(run_dir / "ground_truth_variable_schedule.csv", index=False)
    stage_metrics.to_csv(run_dir / "pr_vs_live_variables.csv", index=False)
    cka_edges.to_csv(run_dir / "cka_boundary_checks.csv", index=False)

    # Check 1: computed-view PR against the exact number of variables written so far.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    points = axes[0].scatter(
        stage_metrics.live_computed_variables,
        stage_metrics.participation_ratio,
        c=stage_metrics.max_dependency_depth, cmap="viridis", alpha=0.8, s=36,
    )
    upper = max(1.0, float(stage_metrics[["live_computed_variables", "participation_ratio"]].to_numpy().max()))
    axes[0].plot([0, upper], [0, upper], "k--", lw=1, label="PR = live variable count")
    axes[0].set(xlabel="Ground-truth live computed variables", ylabel="Computed-view participation ratio",
                 title="PR versus Tracr variable count")
    axes[0].legend(fontsize=8)
    fig.colorbar(points, ax=axes[0], label="Max write depth (layers)")
    gap = stage_metrics.assign(abs_gap=lambda d: np.abs(d.participation_ratio - d.live_computed_variables))
    complexity = gap.groupby(["program", "max_dependency_depth", "computed_variable_total"], as_index=False).abs_gap.mean()
    axes[1].scatter(complexity.max_dependency_depth, complexity.abs_gap,
                    c=complexity.computed_variable_total, cmap="plasma", s=45)
    axes[1].set(xlabel="Ground-truth write depth (layers)", ylabel="Mean |PR − live variable count|",
                 title="PR/count gap versus program depth")
    fig.savefig(figure_dir / "step4_pr_vs_ground_truth.png", bbox_inches="tight")
    plt.show()

    # Check 2: adjacent computed-view CKA, annotated by first-write boundaries.
    grouped_cka = cka_edges[np.isfinite(pd.to_numeric(cka_edges.linear_cka, errors="coerce"))].copy()
    grouped_cka["has_write"] = grouped_cka.new_variable_count > 0
    cka_summary = grouped_cka.groupby(["program", "has_write"], as_index=False).agg(
        mean_cka=("linear_cka", "mean"), median_cka=("linear_cka", "median"), edges=("linear_cka", "size"),
        mismatches=("prediction_mismatch", lambda s: int(s.astype(str).str.len().gt(0).sum())),
    )
    cka_summary.to_csv(run_dir / "cka_write_vs_no_write_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for has_write, label, color in [(True, "New variable written", "#c44e52"), (False, "No new variable", "#4c72b0")]:
        subset = grouped_cka[grouped_cka.has_write == has_write]
        ax.scatter(np.full(len(subset), int(has_write)) + np.linspace(-0.08, 0.08, len(subset)),
                   subset.linear_cka, alpha=0.65, s=24, label=label, color=color)
    ax.axhline(0.95, color="gray", linestyle="--", label="High-CKA flag threshold")
    ax.set(xticks=[0, 1], xticklabels=["No write", "Write boundary"], ylim=(-0.03, 1.03),
           ylabel="Adjacent computed-view linear CKA", title="Does CKA mark known write boundaries?")
    ax.legend()
    fig.savefig(figure_dir / "step4_cka_write_boundaries.png", bbox_inches="tight")
    plt.show()

    # Check 3a: cross-validated variable-label decoding, compared with random directions.
    corr_rows = []
    rng = np.random.default_rng(20261005)
    null_cache = {}
    for name, ctx in contexts.items():
        table = observations[name]
        groups = table.candidate_id.to_numpy()
        folds = _group_folds(groups, seed=31415 + list(programs).index(name))
        for var in ctx["variables"]:
            if not var["computed"]:
                continue
            lanes = var["lane_indices"]
            expr = ctx["expr_by_name"][var["variable"]]
            values = []
            for tokens, position in zip(table.tokens, table.position):
                outputs = evaluator.evaluate(expr, tokens)
                values.append(outputs[int(position)])
            labels, label_categories = _categories(values)
            first = var["first_write_half_layer_index"]
            for stage_idx, (stage, full_X) in enumerate(activations[name].items()):
                if stage == "Embedding":
                    half = -1
                else:
                    layer = int(stage.split("/")[0][1:])
                    half = 2 * (layer - 1) + stage.endswith("/MLP")
                known = full_X[:, lanes]
                known_r2 = _cv_r2(known, labels, groups, folds)
                k = max(1, min(len(lanes), full_X.shape[1]))
                key = (name, var["variable"], stage, k)
                if key not in null_cache:
                    null_scores = []
                    for _ in range(NULL_DRAWS):
                        directions, _ = np.linalg.qr(rng.standard_normal((full_X.shape[1], k)))
                        null_scores.append(_cv_r2(full_X @ directions[:, :k], labels, groups, folds))
                    null_cache[key] = np.asarray(null_scores)
                null = null_cache[key]
                corr_rows.append({
                    "program": name, "variable": var["variable"], "stage": stage,
                    "half_layer_index": half, "first_write_half_layer_index": first,
                    "known_lane_cv_r2": known_r2,
                    "random_direction_median_r2": float(np.median(null)),
                    "random_direction_p95_r2": float(np.quantile(null, 0.95)),
                    "known_minus_null_p95": known_r2 - float(np.quantile(null, 0.95)),
                    "above_null_p95": bool(known_r2 > np.quantile(null, 0.95)),
                    "label_categories": len(label_categories), "observations": len(table),
                    "unique_input_sequences": int(table.candidate_id.nunique()),
                })
    lane_corr = pd.DataFrame(corr_rows)
    lane_corr.to_csv(run_dir / "known_lane_vs_random_null.csv", index=False)

    # Draw selected lane-correlation timelines to keep the report readable.
    focus = [name for name in ("A_prev_token_class", "B_histogram_then_repeated_class", "P11_check_palindrome") if name in programs]
    fig, axes = plt.subplots(len(focus), 1, figsize=(12, 3.6 * len(focus)), squeeze=False, constrained_layout=True)
    for ax, name in zip(axes[:, 0], focus):
        data = lane_corr[lane_corr.program == name]
        for variable, part in data.groupby("variable"):
            part = part.sort_values("half_layer_index")
            ax.plot(part.half_layer_index, part.known_lane_cv_r2, marker="o", label=variable)
            ax.plot(part.half_layer_index, part.random_direction_p95_r2, linestyle=":", alpha=0.75)
            ax.axvline(part.first_write_half_layer_index.iloc[0], color="gray", linewidth=0.8, alpha=0.25)
        ax.set(title=name, xlabel="Stage half-layer index (attention, then MLP)", ylabel="Grouped-CV label R²")
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(figure_dir / "step4_known_lane_correlation.png", bbox_inches="tight")
    plt.show()

    # Check 3b: remove each computed variable's lanes after every stage, then continue forward.
    ablation_rows = []
    for name, ctx in contexts.items():
        model = models[name]
        table = observations[name]
        chosen = []
        for _, group in table.groupby("target_class", sort=True):
            unique_targets = group.drop_duplicates(["candidate_id", "position"])
            chosen.append(unique_targets.sample(
                n=min(ABLATION_ROWS_PER_CLASS, len(unique_targets)),
                random_state=2026 + len(chosen),
            ))
        sampled = pd.concat(chosen, ignore_index=True)
        seq_by_id = {}
        for row in sampled.itertuples(index=False):
            seq_by_id[int(row.candidate_id)] = list(row.tokens)
        candidate_ids = list(seq_by_id)
        token_batches = [seq_by_id[candidate_id] for candidate_id in candidate_ids]
        batch_index = {candidate_id: i for i, candidate_id in enumerate(candidate_ids)}
        baseline = _manual_batch(ctx, model, token_batches)

        # Confirm the intervention baseline matches the compiled Transformer's residual output.
        encoded = np.asarray([model.input_encoder.encode([bos, *tokens]) for tokens in token_batches], dtype=np.int32)
        actual = model.forward(model.params, namespace["jnp"].asarray(encoded))
        actual_residual = np.asarray(actual.transformer_output.output)
        output_columns = [model.residual_labels.index(str(direction)) for direction in ctx["output_space"].basis]
        max_baseline_error = float(np.max(np.abs(baseline - actual_residual[:, :, output_columns])))
        if max_baseline_error > 1e-4:
            raise AssertionError(f"{name}: manual ablation baseline differs from model by {max_baseline_error}")

        expected_lookup = {}
        for row in sampled.itertuples(index=False):
            expected_lookup[(int(row.candidate_id), int(row.position))] = row.model_output
        for var in ctx["variables"]:
            if not var["computed"]:
                continue
            first = int(var["first_write_half_layer_index"])
            lanes = var["lane_indices"]
            for after in range(-1, 2 * model.model_config.num_layers):
                changed_scores = _manual_batch(ctx, model, token_batches, after, lanes)
                selected_base, selected_ablated, expected = [], [], []
                for row in sampled.itertuples(index=False):
                    batch = batch_index[int(row.candidate_id)]
                    pos = int(row.position) + 1
                    selected_base.append(baseline[batch, pos])
                    selected_ablated.append(changed_scores[batch, pos])
                    expected.append(expected_lookup[(int(row.candidate_id), int(row.position))])
                selected_base = np.asarray(selected_base)
                selected_ablated = np.asarray(selected_ablated)
                decoded = _decode_scores(model, selected_ablated)
                expected = np.asarray(expected, dtype=object)
                changed = np.asarray([a != b for a, b in zip(decoded, expected)], dtype=bool)
                base_decoded = _decode_scores(model, selected_base)
                baseline_wrong = np.asarray([a != b for a, b in zip(base_decoded, expected)], dtype=bool)
                ablation_rows.append({
                    "program": name, "variable": var["variable"],
                    "first_write_half_layer_index": first,
                    "first_write_after": var["first_write_after"],
                    "ablate_after_half_layer_index": after,
                    "ablate_after_stage": "embedding" if after == -1 else
                        f"L{after // 2 + 1}/{'Attn' if after % 2 == 0 else 'MLP'}",
                    "before_first_write": bool(after < first),
                    "mean_output_score_delta": float(np.linalg.norm(selected_ablated - selected_base, axis=1).mean()),
                    "prediction_changed_fraction": float(changed.mean()),
                    "baseline_error_fraction": float(baseline_wrong.mean()),
                    "sampled_targets": len(sampled),
                    "unique_sequences": len(candidate_ids),
                    "baseline_max_abs_error": max_baseline_error,
                })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(run_dir / "lane_ablation_by_depth.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    before = ablation[ablation.before_first_write]
    after = ablation[~ablation.before_first_write]
    axes[0].boxplot(
        [before.mean_output_score_delta, after.mean_output_score_delta],
        tick_labels=["Before first write", "At/after first write"], showfliers=False,
    )
    axes[0].set(ylabel="Mean output-score change after continuing forward",
                title="Ablation effect before versus after Tracr writes the variable")
    for name in focus:
        data = ablation[ablation.program == name]
        for variable, part in data.groupby("variable"):
            part = part.sort_values("ablate_after_half_layer_index")
            axes[1].plot(part.ablate_after_half_layer_index,
                         part.prediction_changed_fraction, marker="o", label=f"{name}: {variable}")
    axes[1].set(xlabel="Ablation stage half-layer index", ylabel="Fraction of targets changed",
                ylim=(-0.03, 1.03), title="Output changes after variable-lane ablation")
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(figure_dir / "step4_causal_ablation.png", bbox_inches="tight")
    plt.show()

    print(f"Step 4 complete: {len(schedule)} scheduled variables, {len(stage_metrics)} PR stages, "
          f"{len(cka_edges)} CKA boundaries, {len(lane_corr)} lane-label checks, "
          f"{len(ablation)} ablations; {perf_counter() - started:.1f}s")
    display(cka_edges[cka_edges.prediction_mismatch.astype(str).str.len() > 0].head(25))
    display(cka_summary)
    display(lane_corr.groupby("program").agg(
        variable_stage_checks=("above_null_p95", "size"),
        above_random_null=("above_null_p95", "sum"),
        median_known_minus_null_p95=("known_minus_null_p95", "median"),
    ).reset_index())
    display(ablation.groupby(["program", "before_first_write"], as_index=False).agg(
        mean_output_score_delta=("mean_output_score_delta", "mean"),
        changed_fraction=("prediction_changed_fraction", "mean"),
        comparisons=("prediction_changed_fraction", "size"),
    ))
    return {
        "schedule": schedule,
        "pr_vs_live_variables": stage_metrics,
        "cka_edges": cka_edges,
        "cka_summary": cka_summary,
        "lane_correlations": lane_corr,
        "ablations": ablation,
    }

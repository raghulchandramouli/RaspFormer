"""Geometry metrics and compiler ground truth for the calibration study.

Every metric takes observation-by-feature matrices and applies the absolute
total-variance floor used across the repository. A representation at or below
the floor has spectrum metrics of zero and undefined relational metrics.
"""

import numpy as np
from tracr.compiler import craft_graph_to_model, nodes
from tracr.rasp import rasp

from .compiler import extract_schedule, trace_craft_graph
from .geometry import DEFAULT_VARIANCE_FLOOR, all_pair_cka, feature_matrix, pca_profile

SPECTRUM_METRICS = ("participation_ratio", "spectral_entropy", "effective_rank",
                    "explained_variance_top1", "explained_variance_top3", "explained_variance_top5")
RELATIONAL_METRICS = ("cka", "procrustes", "svcca")
METRICS = SPECTRUM_METRICS + RELATIONAL_METRICS
GROUND_TRUTH = ("live_variables", "summed_cardinality", "dependency_depth", "first_write")


def spectrum_metrics(matrix, *, variance_floor=DEFAULT_VARIANCE_FLOOR):
    """Summaries of one representation's PCA spectrum."""
    profile = pca_profile(matrix, variance_floor=variance_floor)
    if not profile["variance_active"]:
        return dict.fromkeys(SPECTRUM_METRICS, 0.0)
    ratio = profile["explained_variance_ratio"]
    positive = ratio[ratio > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {"participation_ratio": profile["participation_ratio"],
            "spectral_entropy": entropy, "effective_rank": float(np.exp(entropy)),
            **{f"explained_variance_top{count}": float(ratio[:count].sum()) for count in (1, 3, 5)}}


def _centered(matrix, variance_floor):
    centered = feature_matrix(matrix)
    centered = centered - centered.mean(axis=0)
    variance = float(np.sum(centered * centered) / (len(centered) - 1))
    return centered if variance > variance_floor else None


def _procrustes(left, right):
    """Smallest Frobenius distance between unit-norm representations over rotations."""
    width = max(left.shape[1], right.shape[1])
    left, right = (np.pad(x, ((0, 0), (0, width - x.shape[1]))) / np.linalg.norm(x) for x in (left, right))
    alignment = np.linalg.svd(left.T @ right, compute_uv=False).sum()
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * alignment)))


def _svcca(left, right, coverage=0.99):
    """One minus the mean canonical correlation between the leading SVD subspaces."""
    subspaces = []
    for matrix in (left, right):
        scores, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        subspaces.append(scores[:, :int(np.searchsorted(energy, coverage) + 1)])
    correlations = np.linalg.svd(subspaces[0].T @ subspaces[1], compute_uv=False)
    return float(1.0 - np.clip(correlations, 0.0, 1.0).mean())


def dissimilarity(left, right, metric, *, variance_floor=DEFAULT_VARIANCE_FLOOR):
    """Distance between two representations of the same observations.

    Zero means identical up to rotation and isotropic scale. The result is NaN
    when either representation is at or below the variance floor.
    """
    if metric not in RELATIONAL_METRICS:
        raise ValueError(f"metric must be one of {RELATIONAL_METRICS}")
    a, b = _centered(left, variance_floor), _centered(right, variance_floor)
    if len(np.asarray(left)) != len(np.asarray(right)):
        raise ValueError("Representations must contain the same aligned observations.")
    if a is None or b is None:
        return float("nan")
    if metric == "cka":
        return float(1.0 - all_pair_cka([a, b], variance_floor=variance_floor)[0, 1])
    return _procrustes(a, b) if metric == "procrustes" else _svcca(a, b)


def stage_metrics(matrices, *, variance_floor=DEFAULT_VARIANCE_FLOOR):
    """One row per residual stage: its spectrum and its change from the previous stage."""
    rows = []
    for stage, matrix in enumerate(matrices):
        row = {"stage": stage, **spectrum_metrics(matrix, variance_floor=variance_floor)}
        for metric in RELATIONAL_METRICS:
            row[metric] = (dissimilarity(matrices[stage - 1], matrix, metric, variance_floor=variance_floor)
                           if stage else float("nan"))
        rows.append(row)
    return rows


def model_distance(left, right, metric, *, variance_floor=DEFAULT_VARIANCE_FLOOR):
    """Distance between two models from their matched stages after the embedding.

    Relational metrics compare the two representations of each stage directly.
    Spectrum metrics compare the scalar value each model has at that stage.
    """
    if len(left) != len(right):
        raise ValueError("Models must have the same number of stages.")
    if metric in RELATIONAL_METRICS:
        values = [dissimilarity(a, b, metric, variance_floor=variance_floor)
                  for a, b in zip(left[1:], right[1:])]
    elif metric in SPECTRUM_METRICS:
        values = [abs(spectrum_metrics(a, variance_floor=variance_floor)[metric]
                      - spectrum_metrics(b, variance_floor=variance_floor)[metric])
                  for a, b in zip(left[1:], right[1:])]
    else:
        raise ValueError(f"metric must be one of {METRICS}")
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def stage_ground_truth(program, assembled, config, *, stages=None):
    """Compiler ground truth at every residual stage of a compiled program.

    A variable is live from the stage that first writes it; Tracr never erases
    one. Its cardinality term is the rank its encoding can add: lanes minus one
    for a categorical variable, one for a numerical variable. ``stages`` pads
    the result for a model with more layers than the compiled program; padded
    stages repeat the final stage and contain no write.
    """
    traced = trace_craft_graph(program, config)
    _, variables, _ = extract_schedule(program, assembled, config, traced=traced)
    depth = craft_graph_to_model.compute_computational_depth(
        traced.graph, [source[nodes.ID] for source in traced.sources])
    computed = []
    for variable in variables:
        if variable["first_write_half_layer_index"] is None:
            continue
        expression = traced.graph.nodes[variable["variable"]][nodes.EXPR]
        lanes = len(variable["residual_lane_indices"])
        computed.append(dict(stage=variable["first_write_half_layer_index"] + 1,
                             cardinality=lanes - 1 if rasp.is_categorical(expression) else 1,
                             depth=depth[variable["variable"]]))
    compiled_stages = 2 * assembled.model_config.num_layers + 1
    stages = compiled_stages if stages is None else stages
    if stages < compiled_stages:
        raise ValueError("A model cannot have fewer stages than its compiled program.")
    rows = []
    for stage in range(stages):
        live = [variable for variable in computed if variable["stage"] <= stage]
        rows.append(dict(stage=stage, live_variables=len(live),
                         summed_cardinality=sum(variable["cardinality"] for variable in live),
                         dependency_depth=max((variable["depth"] for variable in live), default=0),
                         first_write=any(variable["stage"] == stage for variable in computed)))
    return rows

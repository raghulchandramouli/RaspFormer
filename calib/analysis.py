"""The three graded levels of the calibration protocol, and detector hardening.

Every function here reads measured tables or trained models and returns a
table. The definitions follow section 9 of ``protocol/calibration.md``.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from calib import library, pool
from raspformer import models
from raspformer.calibration import (GROUND_TRUTH, METRICS, RELATIONAL_METRICS, model_distance,
                                    stage_ground_truth, stage_metrics)
from raspformer.research import infer_writes, pr

REAL_CLASSES = ("compiled", "siit", "output-only")
TWIN_ACCURACY_TOLERANCE = 0.01
L1_THRESHOLD = 0.90
L2_THRESHOLD = 0.70
L3_RATIO = 2.0
ALPHA = 0.05
DRAWS = 10_000
BOOTSTRAPS = 2_000
DETECTORS = ("change", "pr", "cka")
THRESHOLDS = np.geomspace(1e-8, 1e2, 101)


# --- statistics ---------------------------------------------------------------

def auroc(real, null):
    """Direction-free probability that a real value and a null value are ordered consistently."""
    real, null = np.asarray(real, dtype=float), np.asarray(null, dtype=float)
    if not len(real) or not len(null):
        return float("nan")
    ranks = rankdata(np.concatenate([real, null]))
    area = (ranks[:len(real)].sum() - len(real) * (len(real) + 1) / 2) / (len(real) * len(null))
    return float(max(area, 1 - area))


def _correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    scale = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / scale) if scale > 0 else float("nan")


def spearman(x, y):
    return _correlation(rankdata(x), rankdata(y))


def rank_residuals(values, control):
    """Ranks of ``values`` with the linear effect of the ranks of ``control`` removed."""
    ranked, design = rankdata(values), np.column_stack([rankdata(control), np.ones(len(values))])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def permutation_p(x, y, rng, draws=DRAWS):
    """Two-sided p-value of the correlation between two residual vectors."""
    observed = abs(_correlation(x, y))
    if not np.isfinite(observed):
        return float("nan")
    exceed = sum(abs(_correlation(rng.permutation(x), y)) >= observed for _ in range(draws))
    return (1 + exceed) / (draws + 1)


def sign_flip_p(values, rng, draws=DRAWS):
    """One-sided p-value that the mean of ``values`` exceeds zero under random sign flips."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan")
    flipped = rng.choice([-1.0, 1.0], size=(draws, len(values))) * values
    return float((1 + (flipped.mean(axis=1) >= values.mean()).sum()) / (draws + 1))


def holm(pvalues):
    """Holm-adjusted p-values; NaN entries are left out of the family."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(pvalues[valid])]
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(order) - position) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


# --- models -------------------------------------------------------------------

def included(root, name, model_class, seed):
    """Whether a trained model enters the analysis, and the reason when it does not."""
    record = pool.load_metrics(root, name, model_class, seed)
    if record is None:
        return False, "not trained"
    if model_class == "siit" and record["mean_iia"] < pool.IIA_THRESHOLD:
        return False, f"mean interchange accuracy {record['mean_iia']:.3f} < {pool.IIA_THRESHOLD}"
    if model_class == "output-only":
        twin = pool.load_metrics(root, name, "siit", seed)
        if twin is None or abs(record["token_accuracy"] - twin["token_accuracy"]) > TWIN_ACCURACY_TOLERANCE:
            return False, "task accuracy not within 0.01 of the SIIT model"
    if "-compressed-" in model_class and record["token_accuracy"] < pool.FINE_TUNE_TARGET:
        return False, f"token accuracy {record['token_accuracy']:.3f} < {pool.FINE_TUNE_TARGET} after fine-tuning"
    return True, ""


def models_of(root, name, model_class):
    """Seed-indexed models of one class for one program."""
    _, _, hl_model = pool.compiled(name)
    if model_class == "compiled":
        return {0: hl_model}
    if model_class == "untrained":
        return {seed: pool.initial_model(name, seed) for seed in pool.SEEDS}
    if model_class == "untrained-compiled":
        spec = library.spec(name)
        return {seed: models.random_model(vocab_size=hl_model.params["token_embed"].shape[0],
                                          max_length=spec.config.max_seq_len + 1,
                                          output_classes=hl_model.params["unembed"].shape[1], seed=seed,
                                          width=hl_model.config.width, layers=hl_model.config.layers, heads=1)
                for seed in pool.SEEDS}
    seeds = pool.SEEDS[:1] if model_class.startswith("compiled-compressed") else pool.SEEDS
    return {seed: pool.load_model(root, name, model_class, seed)
            for seed in seeds if included(root, name, model_class, seed)[0]}


def exclusions(root, specs, classes):
    rows = []
    for spec in specs:
        for model_class in classes:
            seeds = pool.SEEDS[:1] if model_class.startswith("compiled-compressed") else pool.SEEDS
            for seed in seeds:
                keep, reason = included(root, spec.name, model_class, seed)
                if not keep:
                    rows.append(dict(spec=spec.name, model_class=model_class, seed=seed, reason=reason))
    return pd.DataFrame(rows, columns=["spec", "model_class", "seed", "reason"])


def padded(stages, count):
    """Repeat the final stage so a shallower model lines up with a deeper one."""
    return list(stages) + [stages[-1]] * (count - len(stages))


# --- per-stage table ----------------------------------------------------------

def stage_table(root, specs, classes, *, nulls=True):
    """Metric values and compiler ground truth for every model and residual stage."""
    rows = []
    for spec in specs:
        program, assembled, _ = pool.compiled(spec.name)
        truth = stage_ground_truth(program, assembled, spec.config, stages=2 * spec.layers + 1)
        for model_class in classes:
            for seed, model in models_of(root, spec.name, model_class).items():
                stages = pool.measure(spec.name, model)
                variants = [(model_class, stages)]
                if nulls and model_class in REAL_CLASSES:
                    rng = np.random.default_rng(seed)
                    variants.append((f"{model_class}+row-shuffle", [stage[rng.permutation(len(stage))] for stage in stages]))
                for label, matrices in variants:
                    for measured, expected in zip(stage_metrics(matrices), truth):
                        rows.append(dict(spec=spec.name, depth=spec.depth, variables=spec.variables,
                                         vocabulary=spec.vocabulary, model_class=label, seed=seed,
                                         **measured, **{key: expected[key] for key in GROUND_TRUTH}))
    return pd.DataFrame(rows)


# --- L1: above null -----------------------------------------------------------

def _nulls(model_class, metric):
    nulls = {"untrained": "untrained-compiled" if model_class.startswith("compiled") else "untrained",
             "shuffled-label": "shuffled-label"}
    if metric in RELATIONAL_METRICS:
        nulls["row-shuffle"] = f"{model_class}+row-shuffle"
    return nulls


def level_one(stages, classes=REAL_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    programs = sorted(stages.spec.unique())
    rows = []
    for model_class in classes:
        for metric in METRICS:
            for null_name, null_class in _nulls(model_class, metric).items():
                def values(label):
                    frame = stages[stages.model_class == label]
                    return {name: group[metric].dropna().to_numpy() for name, group in frame.groupby("spec")}
                real, null = values(model_class), values(null_class)
                shared = [name for name in programs if len(real.get(name, ())) and len(null.get(name, ()))]
                if not shared:
                    continue
                def score(names):
                    return auroc(np.concatenate([real[name] for name in names]),
                                 np.concatenate([null[name] for name in names]))
                resampled = [score(rng.choice(shared, size=len(shared))) for _ in range(BOOTSTRAPS)]
                rows.append(dict(model_class=model_class, metric=metric, null=null_name, programs=len(shared),
                                 auroc=score(shared), ci_low=float(np.quantile(resampled, 0.025)),
                                 ci_high=float(np.quantile(resampled, 0.975))))
    table = pd.DataFrame(rows)
    table["passed"] = table.auroc >= L1_THRESHOLD
    return table


# --- L2: calibrated -----------------------------------------------------------

def level_two(stages, classes=REAL_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model_class in classes:
        frame = stages[stages.model_class == model_class]
        if frame.empty:
            continue
        frame = frame.groupby(["spec", "stage"], as_index=False)[list(METRICS + GROUND_TRUTH)].mean()
        for metric in METRICS:
            usable = frame.dropna(subset=[metric])
            for quantity in GROUND_TRUTH:
                oriented = np.zeros(len(usable))
                for name in usable.spec.unique():
                    held_out = (usable.spec == name).to_numpy()
                    sign = np.sign(spearman(usable[metric][~held_out], usable[quantity][~held_out]))
                    oriented[held_out] = (sign if np.isfinite(sign) and sign else 1.0) * usable[metric][held_out]
                row = dict(model_class=model_class, metric=metric, quantity=quantity, rows=len(usable),
                           programs=usable.spec.nunique(), rho=spearman(oriented, usable[quantity]),
                           partial_rho=float("nan"), partial_p=float("nan"))
                if quantity != "summed_cardinality":
                    left = rank_residuals(oriented, usable.summed_cardinality)
                    right = rank_residuals(usable[quantity], usable.summed_cardinality)
                    row.update(partial_rho=_correlation(left, right), partial_p=permutation_p(left, right, rng))
                rows.append(row)
    table = pd.DataFrame(rows)
    table["partial_p_holm"] = np.nan
    for model_class in table.model_class.unique():
        mask = (table.model_class == model_class).to_numpy()
        table.loc[mask, "partial_p_holm"] = holm(table.partial_p[mask])
    table["passed"] = (table.quantity != "summed_cardinality") & (table.rho >= L2_THRESHOLD) & (table.partial_p_holm < ALPHA)
    table["tracks_cardinality"] = (table.quantity == "summed_cardinality") & (table.rho >= L2_THRESHOLD)
    return table


# --- L3: algorithm-specific ---------------------------------------------------

def pair_distances(root, pairing, model_class):
    """Between-algorithm and between-seed distance of one pairing, for every metric."""
    members = {member: {seed: pool.measure(spec.name, model)
                        for seed, model in models_of(root, spec.name, model_class).items()}
               for member, spec in (("a", pairing.a), ("b", pairing.b))}
    count = max((len(stages) for group in members.values() for stages in group.values()), default=0)
    labelled = [(member, seed, padded(stages, count)) for member, group in members.items() for seed, stages in group.items()]
    rows = []
    for metric in METRICS:
        distance = {(i, j): model_distance(labelled[i][2], labelled[j][2], metric)
                    for i in range(len(labelled)) for j in range(i + 1, len(labelled))}
        between = [value for (i, j), value in distance.items() if labelled[i][0] != labelled[j][0]]
        within = [value for (i, j), value in distance.items() if labelled[i][0] == labelled[j][0]]
        cross_seed = [value for (i, j), value in distance.items()
                      if labelled[i][0] != labelled[j][0] and labelled[i][1] != labelled[j][1]]
        correct = []
        for index, (member, _, _) in enumerate(labelled):
            others = {other: distance[tuple(sorted((index, other)))] for other in range(len(labelled)) if other != index}
            same = [value for other, value in others.items() if labelled[other][0] == member]
            different = [value for other, value in others.items() if labelled[other][0] != member]
            if same and different:
                correct.append(float(np.mean(same) < np.mean(different)))
        between_mean = float(np.mean(between)) if between else float("nan")
        within_mean = float(np.mean(within)) if within else float("nan")
        rows.append(dict(pairing=pairing.name, kind=pairing.kind, model_class=model_class, metric=metric,
                         models_a=len(members["a"]), models_b=len(members["b"]),
                         between=between_mean, within=within_mean,
                         ratio=between_mean / within_mean if within and within_mean > 0 else float("nan"),
                         ratio_cross_seed=float(np.mean(cross_seed)) / within_mean if cross_seed and within and within_mean > 0 else float("nan"),
                         identified=float(np.mean(correct)) if correct else float("nan"),
                         identification_trials=len(correct)))
    return rows


def level_three(pairs, seed=0):
    """Grade each metric on the twin pairs; control pairs are summarised beside them."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model_class, kind, metric), group in pairs.groupby(["model_class", "kind", "metric"], sort=False):
        usable = group[np.isfinite(group.ratio) & (group.ratio > 0)]
        trials = group.identification_trials.sum()
        rows.append(dict(model_class=model_class, kind=kind, metric=metric, pairs=len(usable),
                         median_ratio=float(usable.ratio.median()) if len(usable) else float("nan"),
                         median_ratio_cross_seed=float(usable.ratio_cross_seed.median()) if len(usable) else float("nan"),
                         mean_log_ratio=float(np.log(usable.ratio).mean()) if len(usable) else float("nan"),
                         p=sign_flip_p(np.log(usable.ratio), rng) if kind == "twin" else float("nan"),
                         identification_accuracy=float((group.identified * group.identification_trials).sum() / trials) if trials else float("nan"),
                         mean_between=float(group.between.mean())))
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for model_class in table.model_class.unique():
        mask = ((table.model_class == model_class) & (table.kind == "twin")).to_numpy()
        table.loc[mask, "p_holm"] = holm(table.p[mask])
    table["passed"] = (table.kind == "twin") & (table.p_holm < ALPHA) & (table.median_ratio >= L3_RATIO)
    return table


# --- verdict ------------------------------------------------------------------

def verdicts(l1, l2, l3, classes=REAL_CLASSES):
    rows = []
    for model_class in classes:
        for metric in METRICS:
            above = l1[(l1.model_class == model_class) & (l1.metric == metric)]
            calibrated = l2[(l2.model_class == model_class) & (l2.metric == metric)]
            specific = l3[(l3.model_class == model_class) & (l3.metric == metric) & (l3.kind == "twin") & (l3.pairs > 0)]
            if above.empty or calibrated.empty:
                continue
            one, two = bool(above.passed.all()), bool(calibrated.passed.any())
            three = bool(specific.passed.all()) if len(specific) else None
            if not one:
                verdict = "describes noise"
            elif two and three:
                verdict = "tracks the algorithm"
            elif two and three is None:
                verdict = "above null and calibrated; algorithm-specificity not gradable"
            elif two:
                verdict = "tracks representational structure, not the algorithm"
            elif three:
                verdict = "separates algorithms, not calibrated"
            else:
                verdict = "above null, not calibrated"
            rows.append(dict(model_class=model_class, metric=metric, L1=one, L2=two, L3=three,
                             tracks_cardinality=bool(calibrated.tracks_cardinality.any()), verdict=verdict))
    return pd.DataFrame(rows)


# --- detector hardening -------------------------------------------------------

def detector_scores(stages):
    """Raw score of each blind detector at every transition of one model."""
    rows = []
    for row, before, after in zip(infer_writes(stages), stages, stages[1:]):
        rows.append(dict(change=row["relative_delta_energy"], pr=abs(pr(after) - pr(before)), cka=row["geometry_change"]))
    return pd.DataFrame(rows)


def f1(predicted, truth):
    hits = int((predicted & truth).sum())
    precision, recall = hits / max(1, int(predicted.sum())), hits / max(1, int(truth.sum()))
    return 2 * precision * recall / max(1e-15, precision + recall)


def harden_detectors(calibration, held_out, seed=0):
    """Learn each threshold on calibration programs, then score the held-out pools once.

    Both arguments map a program label to ``(pool, scores, truth)`` where
    ``scores`` comes from ``detector_scores`` and ``truth`` flags the
    transitions that contain a first write.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for detector in DETECTORS:
        mean_f1 = [np.mean([f1(scores[detector].to_numpy() > threshold, truth)
                            for _, scores, truth in calibration.values()]) for threshold in THRESHOLDS]
        threshold = float(THRESHOLDS[int(np.argmax(mean_f1))])
        for pool_name in sorted({entry[0] for entry in held_out.values()}):
            entries = [entry for entry in held_out.values() if entry[0] == pool_name]
            scores = np.asarray([f1(frame[detector].to_numpy() > threshold, truth) for _, frame, truth in entries])
            baseline = np.asarray([f1(np.ones_like(truth), truth) for _, _, truth in entries])
            resampled = [scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(BOOTSTRAPS)]
            rows.append(dict(detector=detector, threshold=threshold, calibration_f1=float(max(mean_f1)),
                             pool=pool_name, models=len(entries), f1=float(scores.mean()),
                             ci_low=float(np.quantile(resampled, 0.025)), ci_high=float(np.quantile(resampled, 0.975)),
                             always_write_f1=float(baseline.mean()), beats_always_write=bool(scores.mean() > baseline.mean())))
    return pd.DataFrame(rows)

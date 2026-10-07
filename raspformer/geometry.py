"""Sample targets, collect residuals, measure geometry, and export the study."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tracr.rasp import rasp

from .compiler import (
    CompilerConfig,
    compile_program,
    record_run,
    write_json,
)


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 42
    length: int = 6
    samples_per_class: int = 64
    random_candidates: int = 2048
    structured_candidates: int = 128

    def __post_init__(self):
        for name in ("seed", "random_candidates", "structured_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("length", "samples_per_class"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass
class ProbeSet:
    candidates: list[list[int]]
    families: list[str]
    observations: dict[str, pd.DataFrame]
    audit: pd.DataFrame
    expected_classes: dict[str, list[int]]


@dataclass
class ActivationData:
    models: dict
    stages: dict[str, dict[str, np.ndarray]]
    columns: dict[str, dict[str, np.ndarray]]
    observations: dict[str, pd.DataFrame]


@dataclass
class GeometryAnalysis:
    profiles: dict
    metrics: pd.DataFrame


def candidate_pool(config: CompilerConfig, probes: ProbeConfig):
    """Deduplicate sequences; preserve the first family and insertion order."""
    if not 2 <= probes.length <= min(config.max_seq_len, len(config.vocab)):
        raise ValueError("Probe length must be 2..min(max_seq_len, vocabulary size).")
    vocab = sorted(config.vocab)
    length = probes.length
    rng = np.random.default_rng(probes.seed)
    alphabet = np.asarray(vocab)
    candidates = {}

    def add(tokens, family):
        candidates.setdefault(tuple(int(value) for value in tokens), family)

    for a in vocab:
        add([a] * length, "constant")
        for b in vocab:
            add(
                [a if position % 2 == 0 else b for position in range(length)],
                "alternating",
            )
            if a != b:
                for count in range(1, length):
                    base = [a] * count + [b] * (length - count)
                    for shift in range(length):
                        add(np.roll(base, shift), "count_controlled")
    for _ in range(probes.random_candidates):
        add(rng.choice(alphabet, size=length), "random")
    for _ in range(probes.structured_candidates):
        values = rng.choice(alphabet, size=length)
        add(np.sort(values), "increasing")
        add(np.sort(values)[::-1], "decreasing")
        add(rng.choice(alphabet, size=length, replace=False), "unique")
        half = rng.choice(alphabet, size=(length + 1) // 2)
        mirror = half[::-1] if length % 2 == 0 else half[-2::-1]
        add(np.concatenate([half, mirror]), "palindrome")
    return [list(tokens) for tokens in candidates], list(candidates.values())


def output_classes(config: CompilerConfig, length: int) -> dict[str, list[int]]:
    """Analytic output support for the built-in programs at one probe length."""
    vocab = sorted(config.vocab)
    sums = sorted({a + b for a in vocab for b in vocab})
    return {
        "P01_absolute": sorted({abs(value) for value in vocab}),
        "P02_index_parity": sorted({position % 2 for position in range(length)}),
        "P03_increment_by_index": sorted(
            {value + position for value in vocab for position in range(length)}
        ),
        "P04_first_element": vocab,
        "P05_histogram": list(range(1, length + 1)),
        "P06_count_greater_than": list(range(length)),
        "P07_sum_with_next": sums,
        "P08_pairwise_sum": sums,
        "P09_check_increasing": [0, 1],
        "P10_rotate_left": vocab,
        "P11_check_palindrome": [0, 1],
        "P12_sorting": vocab,
        "A_prev_token_class": vocab,
        "B_histogram_then_repeated_class": [0, 1],
    }


def sample_targets(name, program, candidates, families, classes, config, *, seed):
    """Balance output classes, then stratify each class across reachable positions."""
    buckets = {label: {} for label in classes}
    evaluator = rasp.DefaultRASPEvaluator()
    for candidate_id, tokens in enumerate(candidates):
        for position, value in enumerate(evaluator.evaluate(program, tokens)):
            if value not in buckets:
                raise ValueError(f"{name}: unexpected output class {value}.")
            buckets[value].setdefault(position, []).append(candidate_id)
    missing = [label for label, positions in buckets.items() if not positions]
    if missing:
        raise ValueError(f"{name}: candidate pool misses output classes {missing}.")
    rng = np.random.default_rng(seed)
    records = []
    for label, positions in buckets.items():
        available = rng.permutation(sorted(positions))
        quotient, remainder = divmod(config.samples_per_class, len(available))
        for index, position in enumerate(available):
            quota = quotient + (index < remainder)
            choices = positions[position]
            selected = rng.choice(choices, size=quota, replace=len(choices) < quota)
            for candidate_id in selected:
                tokens = candidates[candidate_id]
                records.append(
                    {
                        "candidate_id": int(candidate_id),
                        "position": int(position),
                        "target_class": int(label),
                        "token": tokens[position],
                        "family": families[candidate_id],
                        "tokens": tokens,
                    }
                )
    table = pd.DataFrame([records[index] for index in rng.permutation(len(records))])
    table.insert(0, "probe_id", range(len(table)))
    return table


def build_balanced_probes(
    programs, compiler_config: CompilerConfig, config: ProbeConfig
) -> ProbeSet:
    candidates, families = candidate_pool(compiler_config, config)
    supports = output_classes(compiler_config, config.length)
    observations, audit = {}, []
    for index, (name, program) in enumerate(programs.items()):
        if name not in supports:
            raise KeyError(
                f"Define analytic output classes for {name} in output_classes()."
            )
        table = sample_targets(
            name,
            program,
            candidates,
            families,
            supports[name],
            config,
            seed=config.seed + 1009 * index,
        )
        observations[name] = table
        for label, group in table.groupby("target_class"):
            audit.append(
                {
                    "program": name,
                    "output_class": label,
                    "observations": len(group),
                    "unique_sequences": group.candidate_id.nunique(),
                    "unique_targets": len(
                        group[["candidate_id", "position"]].drop_duplicates()
                    ),
                }
            )
        print(
            f"{name}: {len(supports[name])} classes × {config.samples_per_class} targets"
        )
    return ProbeSet(
        candidates,
        families,
        observations,
        pd.DataFrame(audit),
        {name: supports[name] for name in programs},
    )


def target_rows(array, positions) -> np.ndarray:
    """One target per sequence; positions include the BOS offset."""
    return np.asarray(array)[np.arange(len(positions)), positions, :].astype(np.float64)


def collect_activations(
    programs, config: CompilerConfig, probes: ProbeSet
) -> ActivationData:
    models, activations, columns, observations = {}, {}, {}, {}
    for name, program in programs.items():
        started = perf_counter()
        model = compile_program(program, config)
        table = probes.observations[name].copy()
        encoded = jnp.asarray(
            [model.input_encoder.encode([config.bos, *tokens]) for tokens in table.tokens],
            dtype=jnp.int32,
        )
        result = jax.jit(model.forward)(model.params, encoded)
        jax.block_until_ready(result)
        positions = table.position.to_numpy() + 1  # Skip BOS when selecting targets.
        transformer = result.transformer_output
        stages = {"Embedding": target_rows(transformer.input_embeddings, positions)}
        for step, residual in enumerate(transformer.residuals):
            kind = "Attn" if step % 2 == 0 else "MLP"
            stages[f"L{step // 2 + 1}/{kind}"] = target_rows(residual, positions)
        raw = np.asarray(result.unembedded_output)[np.arange(len(table)), positions]
        decoded = (
            model.output_encoder.decode(raw.tolist())
            if model.output_encoder
            else raw.tolist()
        )
        if not np.allclose(
            decoded, table.target_class, rtol=config.rel_tol, atol=config.abs_tol
        ):
            raise ValueError(f"{name}: compiled target outputs disagree with RASP.")
        table["model_output"] = decoded
        input_names = {rasp.tokens.label, rasp.indices.label, "one"}
        computed = [
            index
            for index, label in enumerate(model.residual_labels)
            if label.split(":", 1)[0] not in input_names
        ]
        models[name], activations[name], observations[name] = model, stages, table
        columns[name] = {
            "full": np.arange(len(model.residual_labels)),
            "computed": np.asarray(computed, dtype=int),
        }
        print(
            f"{name}: {len(stages)} stages; width={len(model.residual_labels)}; {perf_counter() - started:.1f}s"
        )
    return ActivationData(models, activations, columns, observations)


def representation(
    data: ActivationData, program: str, view: str, stage: str
) -> np.ndarray:
    return data.stages[program][stage][:, data.columns[program][view]]


def feature_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not matrix.shape[1]:
        raise ValueError(
            "Supply a matrix with at least two observations and one feature."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Representation contains nonfinite values.")
    return matrix


def pca_profile(values) -> dict:
    matrix = feature_matrix(values)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / (len(matrix) - 1))
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    total = float(eigenvalues.sum())
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(ratio)
    return {
        "mean": mean,
        "components": vectors[:, ::-1],
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": ratio,
        "total_variance": total,
        "participation_ratio": float(total**2 / (eigenvalues @ eigenvalues))
        if total > 0
        else 0.0,
        "pc95": min(len(eigenvalues), int(np.searchsorted(cumulative, 0.95) + 1))
        if total > 0
        else 0,
        "pc99": min(len(eigenvalues), int(np.searchsorted(cumulative, 0.99) + 1))
        if total > 0
        else 0,
    }


def all_pair_cka(matrices) -> np.ndarray:
    centered, norms = [], []
    for values in matrices:
        matrix = feature_matrix(values)
        if centered and len(matrix) != len(centered[0]):
            raise ValueError("CKA matrices must contain the same aligned observations.")
        matrix = matrix - matrix.mean(axis=0)
        centered.append(matrix)
        norms.append(np.linalg.norm(matrix.T @ matrix, "fro"))
    scores = np.full((len(centered), len(centered)), np.nan)
    for i in range(len(centered)):
        for j in range(i, len(centered)):
            denominator = norms[i] * norms[j]
            if denominator > 0:
                numerator = np.linalg.norm(centered[i].T @ centered[j], "fro") ** 2
                scores[i, j] = scores[j, i] = np.clip(numerator / denominator, 0.0, 1.0)
    return scores


def analyze_geometry(data: ActivationData) -> GeometryAnalysis:
    profiles, rows = {}, []
    for name, stages in data.stages.items():
        profiles[name] = {}
        for view, columns in data.columns[name].items():
            matrices = [representation(data, name, view, stage) for stage in stages]
            pca = {stage: pca_profile(matrix) for stage, matrix in zip(stages, matrices)}
            profiles[name][view] = {"pca": pca, "cka": all_pair_cka(matrices)}
            for step, (stage, profile) in enumerate(pca.items()):
                rows.append({
                    "program": name,
                    "view": view,
                    "stage": stage,
                    "step": step,
                    "observations": len(data.observations[name]),
                    "features": len(columns),
                    "total_variance": profile["total_variance"],
                    "participation_ratio": profile["participation_ratio"],
                    "pr_over_width": profile["participation_ratio"] / len(columns),
                    "pc95": profile["pc95"],
                    "pc99": profile["pc99"],
                })
        print(f"{name}: PCA and all stage-pair CKA complete")
    return GeometryAnalysis(profiles, pd.DataFrame(rows))


def lane_variances(
    data: ActivationData, program: str, view: str = "computed", top: int = 25
):
    if type(top) is not int or top < 1:
        raise ValueError("top must be a positive integer.")
    stages = list(data.stages[program])
    variances = np.stack(
        [
            representation(data, program, view, stage).var(axis=0, ddof=1)
            for stage in stages
        ]
    )
    selected = np.argsort(variances.max(axis=0))[::-1][:top]
    columns = data.columns[program][view]
    labels = [
        data.models[program].residual_labels[columns[index]] for index in selected
    ]
    return variances[:, selected].T, stages, labels


def pca_coordinates(matrix, profile) -> np.ndarray:
    """Use two display coordinates, padding PC2 only for one-feature views."""
    points = (matrix - profile["mean"]) @ profile["components"][:, :2]
    return np.pad(points, ((0, 0), (0, max(0, 2 - points.shape[1]))))


def trajectory_coordinates(
    data: ActivationData, program: str, probe_id: int = 0, view: str = "full"
):
    if type(probe_id) is not int or not 0 <= probe_id < len(data.observations[program]):
        raise ValueError("Unknown probe_id.")
    stages = list(data.stages[program])
    matrices = [representation(data, program, view, stage) for stage in stages]
    common = pca_profile(np.concatenate(matrices, axis=0))
    coordinates = np.stack(
        [pca_coordinates(matrix[[probe_id]], common)[0] for matrix in matrices]
    )
    return coordinates, stages, common, data.observations[program].iloc[probe_id]


def export_geometry(
    run_dir: Path, probes: ProbeSet, data: ActivationData, analysis: GeometryAnalysis
) -> None:
    analysis.metrics.to_csv(run_dir / "layer_metrics.csv", index=False)
    probes.audit.to_csv(run_dir / "balance_audit.csv", index=False)
    for name, table in data.observations.items():
        table.assign(tokens=table.tokens.map(json.dumps)).to_csv(
            run_dir / f"probes_{name}.csv", index=False
        )
    spectra = {}
    for name, views in analysis.profiles.items():
        for view, profile in views.items():
            stages = list(profile["pca"])
            cka = pd.DataFrame(profile["cka"], index=stages, columns=stages)
            cka.to_csv(run_dir / f"cka_{name}_{view}_all_stages.csv")
            full_layers = [
                stage
                for stage in stages
                if stage == "Embedding" or stage.endswith("/MLP")
            ]
            cka.loc[full_layers, full_layers].to_csv(
                run_dir / f"cka_{name}_{view}_full_layers.csv"
            )
            for stage, pca in profile["pca"].items():
                prefix = f"{name}__{view}__{stage.replace('/', '_')}"
                for field in (
                    "eigenvalues",
                    "explained_variance_ratio",
                    "components",
                    "mean",
                ):
                    spectra[f"{prefix}__{field}"] = pca[field]
    np.savez_compressed(run_dir / "pca_spectra.npz", **spectra)


def run_geometry_study(
    programs, compiler_config: CompilerConfig, config: ProbeConfig, output_root: Path
):
    """Run a descriptive study; plots are made afterward by the notebook."""
    settings = {
        "probe_config": asdict(config),
        "sampling": "separate program probes; equal observations per reachable output class; positions stratified; repeats retain weight",
        "observation": "one target token per probe; BOS excluded; no PAD; same row order within each program",
        "centering": "global feature centering; no whitening or standardization",
        "zero_variance": "PCA spectrum and PR recorded as zero; CKA undefined",
    }
    with record_run(output_root, programs, compiler_config, **settings) as (run_dir, evidence):
        evidence["stage"] = "probes"
        write_json(run_dir / "manifest.json", evidence)
        probes = build_balanced_probes(programs, compiler_config, config)
        evidence.update(stage="activations", candidate_sequences=len(probes.candidates))
        write_json(run_dir / "manifest.json", evidence)
        data = collect_activations(programs, compiler_config, probes)
        evidence["stage"] = "geometry"
        write_json(run_dir / "manifest.json", evidence)
        analysis = analyze_geometry(data)
        evidence["stage"] = "export"
        write_json(run_dir / "manifest.json", evidence)
        export_geometry(run_dir, probes, data, analysis)
        evidence["programs"] = [
            {
                "program": name,
                "layers": model.model_config.num_layers,
                "residual_labels": model.residual_labels,
                "stages": list(data.stages[name]),
                "observations": len(data.observations[name]),
                "output_classes": probes.expected_classes[name],
                "view_columns": {
                    view: columns.tolist() for view, columns in data.columns[name].items()
                },
            }
            for name, model in data.models.items()
        ]
    return {
        "probes": probes, "activations": data, "analysis": analysis,
        "evidence": evidence, "run_dir": run_dir,
    }

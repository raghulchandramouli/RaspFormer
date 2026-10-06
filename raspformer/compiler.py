"""Compilation, sampled correctness checks, writer schedules, and manual traces.

Tracr's private allocation and assembly helpers are isolated here because
there is no public scheduling API. Every study records the installed revision.
"""

import hashlib
import json
import logging
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distribution, version
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
from uuid import uuid4

import jax
import numpy as np
from tracr.compiler import (
    assemble,
    basis_inference,
    compiling,
    craft_graph_to_model,
    craft_model_to_transformer,
    expr_to_craft_graph,
    nodes,
    rasp_to_graph,
)
from tracr.craft import bases, transformers
from tracr.rasp import rasp


@dataclass(frozen=True)
class CompilerConfig:
    vocab: tuple[int, ...] = tuple(range(-2, 6))
    max_seq_len: int = 6
    causal: bool = False
    bos: str = "BOS"
    pad: str = "PAD"
    mlp_exactness: int = 100
    rel_tol: float = 1e-6
    abs_tol: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "vocab", tuple(self.vocab))
        if not self.vocab or any(type(token) is not int for token in self.vocab):
            raise ValueError("Supply a nonempty integer vocabulary.")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("Vocabulary tokens must be unique.")
        for name in ("max_seq_len", "mlp_exactness"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.causal) is not bool:
            raise ValueError("causal must be a boolean.")
        if not all(isinstance(token, str) and token for token in (self.bos, self.pad)):
            raise ValueError("BOS and PAD must be nonempty strings.")
        if self.bos == self.pad:
            raise ValueError("BOS and PAD must be distinct.")
        if any(not isfinite(tol) or tol < 0 for tol in (self.rel_tol, self.abs_tol)):
            raise ValueError("Tolerances must be finite and nonnegative.")

    def validate_samples(self, samples):
        if not samples:
            raise ValueError("Supply at least one validation sequence.")
        for tokens in samples:
            if not 1 <= len(tokens) <= self.max_seq_len:
                raise ValueError(
                    f"Sequence length must be 1..{self.max_seq_len}: {tokens}"
                )
            if any(
                type(token) is not int or token not in self.vocab for token in tokens
            ):
                raise ValueError(f"Out-of-vocabulary tokens: {tokens}")


def trace_craft_graph(program, config):
    traced = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        traced.graph,
        traced.sink,
        set(config.vocab),
        config.max_seq_len,
    )
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        traced.graph,
        bos_dir=bases.BasisDirection(rasp.tokens.label, config.bos),
        mlp_exactness=config.mlp_exactness,
    )
    return traced


def variable_schedule(node, first_write, model):
    expression = node[nodes.EXPR]
    block = node.get(nodes.MODEL_BLOCK)
    if first_write is None and expression.label not in {
        rasp.tokens.label,
        rasp.indices.label,
    }:
        raise ValueError(f"No writer allocated for {expression.label}.")
    if isinstance(block, transformers.SeriesWithResiduals):
        # A SelectorWidth pair starts with attention, but its MLP writes
        # the sequence variable's output basis.
        if first_write is None:
            raise ValueError(f"No composite allocation for {expression.label}.")
        first_write += 1
    if (
        first_write is not None
        and not 0 <= first_write < 2 * model.model_config.num_layers
    ):
        raise ValueError(f"Writer for {expression.label} lies outside the model.")

    labels = [str(direction) for direction in node[nodes.OUTPUT_BASIS]]
    lane_index = {label: index for index, label in enumerate(model.residual_labels)}
    missing = set(labels) - lane_index.keys()
    if missing:
        raise ValueError(f"Output lanes for {expression.label} are missing: {missing}")
    stage = "input embedding"
    if first_write is not None:
        kind = "attention" if first_write % 2 == 0 else "mlp"
        stage = f"after {kind}{first_write // 2 + 1}"
    return {
        "variable": expression.label,
        "residual_lane_indices": [lane_index[label] for label in labels],
        "residual_lane_labels": labels,
        "first_write_half_layer_index": first_write,
        "first_write_after": stage,
        "compiled_block": (
            "attention+mlp"
            if isinstance(block, transformers.SeriesWithResiduals)
            else type(block).__name__
            if block is not None
            else "input"
        ),
    }


def extract_schedule(program, model, config):
    traced = trace_craft_graph(program, config)
    graph, sources = traced.graph, traced.sources
    allocation = craft_graph_to_model._allocate_modules_to_layers(graph, sources)
    if len(set(model.residual_labels)) != len(model.residual_labels):
        raise ValueError("The model has duplicate residual labels.")
    variables, selectors = [], []
    for node_id, node in graph.nodes(data=True):
        expression = node[nodes.EXPR]
        if isinstance(expression, rasp.Select):
            selectors.append({"selector": expression.label, "node_id": node_id})
        if isinstance(expression, rasp.SOp):
            variables.append(variable_schedule(node, allocation.get(node_id), model))
    depth = craft_graph_to_model.compute_computational_depth(
        graph,
        [source[nodes.ID] for source in sources],
    )
    summary = {
        "transformer_layers": model.model_config.num_layers,
        "residual_width": len(model.residual_labels),
        "sequence_variable_count": len(variables),
        "computed_variable_count": sum(
            row["first_write_half_layer_index"] is not None for row in variables
        ),
        "selector_count": len(selectors),
        "max_dependency_depth": max(depth.values()),
    }
    return summary, variables, selectors


def validate_outputs(name, program, model, samples, config):
    config.validate_samples(samples)
    evaluator = rasp.DefaultRASPEvaluator()
    checks = []
    for sample in samples:
        tokens = list(sample)
        expected = list(evaluator.evaluate(program, tokens))
        actual = list(model.apply([config.bos, *tokens]).decoded[1:])
        matches = len(expected) == len(actual) and all(
            left == right
            or (
                isinstance(left, Real)
                and isinstance(right, Real)
                and isclose(left, right, rel_tol=config.rel_tol, abs_tol=config.abs_tol)
            )
            for left, right in zip(expected, actual)
        )
        if not matches:
            raise ValueError(f"{name}: input={tokens}, RASP={expected}, model={actual}")
        checks.append(
            {
                "program": name,
                "tokens": tokens,
                "expected": expected,
                "actual": actual,
            }
        )
    return checks


def lane_text(indices):
    if not indices:
        return "—"
    if indices == list(range(indices[0], indices[-1] + 1)):
        return str(indices[0]) if len(indices) == 1 else f"{indices[0]}–{indices[-1]}"
    return ", ".join(map(str, indices))


def summary_table(summaries):
    lines = [
        "| Program | Layers | Residual width | Computed variables | Checks passed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['program']} | {row['transformer_layers']} | {row['residual_width']} | "
        f"{row['computed_variable_count']} | {row['checks_passed']} |"
        for row in summaries
    )
    return "\n".join(lines)


def schedule_table(variables, program_name):
    rows = sorted(
        (row for row in variables if row["program"] == program_name),
        key=lambda row: (
            -1
            if row["first_write_half_layer_index"] is None
            else row["first_write_half_layer_index"],
            row["variable"],
        ),
    )
    if not rows:
        raise KeyError(f"No schedule recorded for {program_name}.")
    lines = ["| Variable | Lanes | First writer |", "| --- | ---: | --- |"]
    lines.extend(
        f"| {row['variable']} | {lane_text(row['residual_lane_indices'])} | "
        f"{row['first_write_after']} |"
        for row in rows
    )
    return "\n".join(lines)


def save_artifacts(evidence, run_dir):
    lines = [
        "# Tracr compiler-derived schedules",
        "",
        (
            f"Status: {evidence['status']}; "
            f"{len(evidence['programs'])}/{len(evidence['planned_programs'])} programs passed; "
            f"{len(evidence['checks'])} sampled sequences matched RASP."
        ),
        "",
        "Compiler allocations do not measure activation onset.",
        "",
        summary_table(evidence["programs"]),
    ]
    for row in evidence["programs"]:
        name = row["program"]
        lines.extend(
            ["", f"## {name}", "", schedule_table(evidence["variables"], name)]
        )
    for failure in evidence["failures"]:
        lines.extend(
            ["", f"Failed: {failure['program']}", "", f"    {failure['error']}"]
        )
    for filename, text in (
        (
            "step1_ground_truth_schedule.json",
            json.dumps(evidence, indent=2, allow_nan=False) + "\n",
        ),
        ("step1_ground_truth_schedule.md", "\n".join(lines) + "\n"),
    ):
        path = run_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def compile_and_check(name, program, config, samples):
    model = compile_program(program, config)
    summary, variables, selectors = extract_schedule(program, model, config)
    checks = validate_outputs(name, program, model, samples, config)
    return model, summary, variables, selectors, checks


def run_study(programs, config, samples, output_root):
    if not programs:
        raise ValueError("Supply at least one program.")
    samples = tuple(tuple(tokens) for tokens in samples)
    config.validate_samples(samples)
    settings = {**asdict(config), **environment_metadata(), "samples": samples}
    run_dir = new_run_directory(output_root)
    evidence = {
        "schema_version": 1,
        "status": "running",
        "planned_programs": list(programs),
        "settings": settings,
        "programs": [],
        "variables": [],
        "selectors": [],
        "checks": [],
        "failures": [],
    }
    models = {}
    save_artifacts(evidence, run_dir)
    logger = logging.getLogger(f"raspformer.programs.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "tracr.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    try:
        for name, program in programs.items():
            logger.info("START %s", name)
            try:
                model, summary, variables, selectors, checks = compile_and_check(
                    name,
                    program,
                    config,
                    samples,
                )
            except Exception as error:
                evidence["status"] = "failed"
                evidence["failures"].append(
                    {
                        "program": name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                save_artifacts(evidence, run_dir)
                logger.exception("FAILED %s", name)
                raise
            models[name] = model
            evidence["programs"].append(
                {"program": name, **summary, "checks_passed": len(checks)}
            )
            evidence["variables"].extend({"program": name, **row} for row in variables)
            evidence["selectors"].extend({"program": name, **row} for row in selectors)
            evidence["checks"].extend(checks)
            save_artifacts(evidence, run_dir)
            logger.info("PASS %s: %d/%d checks", name, len(checks), len(samples))
            print(
                f"{name}: {summary['transformer_layers']} layers; {len(checks)}/{len(samples)} checks"
            )
        evidence["status"] = "completed"
        save_artifacts(evidence, run_dir)
    finally:
        logger.removeHandler(handler)
        handler.close()
    return {"models": models, "evidence": evidence, "run_dir": run_dir}


def compile_program(program, config: CompilerConfig):
    """Compile one expression with explicit shared settings."""
    return compiling.compile_rasp_to_model(
        program,
        vocab=set(config.vocab),
        max_seq_len=config.max_seq_len,
        causal=config.causal,
        compiler_bos=config.bos,
        compiler_pad=config.pad,
        mlp_exactness=config.mlp_exactness,
    )


def new_run_directory(output_root: Path, label: str = "") -> Path:
    """Never overwrite an earlier run, even with identical settings."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"{label + '_' if label else ''}{stamp}_{uuid4().hex[:8]}"
    path = Path(output_root).resolve() / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_metadata() -> dict:
    direct_url = distribution("tracr").read_text("direct_url.json")
    source = json.loads(direct_url) if direct_url else {}
    return {
        "tracr_version": version("tracr"),
        "tracr_commit": source.get("vcs_info", {}).get("commit_id"),
        "python_version": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in (
                "tracr", "jax", "jaxlib", "numpy", "dm-haiku", "networkx",
                "pandas", "matplotlib", "ipywidgets",
            )
        },
        "devices": [str(device) for device in jax.devices()],
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


@dataclass
class TraceModel:
    program: rasp.SOp
    craft: transformers.SeriesWithResiduals
    assembled: assemble.AssembledTransformerModel
    full_space: bases.VectorSpaceWithBasis
    output_space: bases.VectorSpaceWithBasis
    module_names: list[str]


def build_trace_model(program: rasp.SOp, config: CompilerConfig) -> TraceModel:
    """Keep the Craft graph and the assembled model from the same compilation."""
    traced = trace_craft_graph(program, config)
    graph, sources, sink = traced.graph, traced.sources, traced.sink
    output_basis = list(graph.nodes[sink[nodes.ID]][nodes.OUTPUT_BASIS])
    craft = craft_graph_to_model.craft_graph_to_model(graph, sources)
    model = craft_model_to_transformer.craft_model_to_transformer(
        craft_model=craft,
        graph=graph,
        sink=sink,
        max_seq_len=config.max_seq_len,
        compiler_bos=config.bos,
        compiler_pad=config.pad,
        causal=config.causal,
    )
    _, module_names = assemble._get_model_config_and_module_names(craft)
    if len(module_names) != len(craft.blocks):
        raise ValueError("Craft blocks and assembled module names are not aligned.")
    tokens = set(graph.nodes[rasp.tokens.label][nodes.VALUE_SET]) | {
        config.bos,
        config.pad,
    }
    token_space = bases.VectorSpaceWithBasis.from_values(rasp.tokens.label, tokens)
    index_space = bases.VectorSpaceWithBasis.from_values(
        rasp.indices.label, range(config.max_seq_len)
    )
    output_space = bases.VectorSpaceWithBasis(output_basis)
    full_space = bases.join_vector_spaces(
        craft.residual_space, token_space, index_space, output_space
    )
    if [str(direction) for direction in full_space.basis] != model.residual_labels:
        raise ValueError("Manual residual basis differs from the assembled model.")
    return TraceModel(program, craft, model, full_space, output_space, module_names)


def manual_craft_forward(trace: TraceModel, tokens: list[int], *, bos: str):
    """Trace active Craft blocks with the assembled model's attention scaling."""
    sequence = [bos, *tokens]
    full_space = trace.full_space
    x = np.zeros((len(sequence), full_space.num_dims), dtype=np.float64)
    for position, token in enumerate(sequence):
        direction = bases.BasisDirection(rasp.tokens.label, token)
        x[position, full_space.index_by_direction[direction]] = 1.0
        if position:
            direction = bases.BasisDirection(rasp.indices.label, position - 1)
            x[position, full_space.index_by_direction[direction]] = 1.0
    one = bases.BasisDirection("one")
    if one in full_space:
        x[:, full_space.index_by_direction[one]] = 1.0

    residual = full_space.make_vector(x)
    block_trace = []
    for module_name, block in zip(trace.module_names, trace.craft.blocks):
        block_input = residual.project(block.residual_space)
        if isinstance(
            block, (transformers.AttentionHead, transformers.MultiAttentionHead)
        ):
            heads = (
                [block]
                if isinstance(block, transformers.AttentionHead)
                else list(block.heads())
            )
            outputs = []
            for head in heads:
                queries = block_input.project(head.w_qk.left_space).magnitudes
                keys = block_input.project(head.w_qk.right_space).magnitudes
                logits = (queries @ head.w_qk.matrix @ keys.T) / np.sqrt(
                    trace.assembled.model_config.key_size
                )
                if trace.assembled.model_config.causal:
                    logits = np.where(
                        np.tril(np.ones(logits.shape, dtype=bool)), logits, -1e30
                    )
                weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
                weights /= weights.sum(axis=-1, keepdims=True)
                values = head.w_ov_residual(block_input).magnitudes
                outputs.append(head.residual_space.make_vector(weights @ values))
            delta = bases.VectorInBasis.sum(outputs)
        elif isinstance(block, transformers.MLP):
            delta = block.apply(block_input)
        else:
            raise TypeError(f"Unsupported block: {type(block).__name__}")
        residual = residual + delta.project(full_space)
        block_trace.append(
            {
                "module": module_name,
                "residual": residual.magnitudes.copy(),
                "output_scores": residual.project(trace.output_space).magnitudes.copy(),
            }
        )
    return residual, x, block_trace


def check_trace_sample(
    name: str, trace: TraceModel, tokens: list[int], config: CompilerConfig
):
    """Check embeddings and output scores strictly; record residual discrepancies."""
    final, inputs, steps = manual_craft_forward(trace, tokens, bos=config.bos)
    actual = trace.assembled.apply([config.bos, *tokens])
    np.testing.assert_allclose(
        inputs,
        np.asarray(actual.input_embeddings)[0],
        rtol=config.rel_tol,
        atol=config.abs_tol,
        err_msg=f"{name}, input embeddings, input={tokens}",
    )
    comparisons = []
    for step in steps:
        _, layer_name, kind = step["module"].split("/")
        half_layer = 2 * int(layer_name.split("_")[-1]) + (kind == "mlp")
        comparisons.append(
            (
                step["module"],
                step["residual"],
                np.asarray(actual.residuals[half_layer])[0],
            )
        )
    actual_final = np.asarray(actual.transformer_output)[0]
    comparisons.append(("final_residual", final.magnitudes, actual_final))
    warnings, max_residual_error = [], 0.0
    for module, expected_residual, actual_residual in comparisons:
        error = float(np.max(np.abs(expected_residual - actual_residual)))
        max_residual_error = max(max_residual_error, error)
        if not np.allclose(
            expected_residual, actual_residual, rtol=config.rel_tol, atol=config.abs_tol
        ):
            warnings.append(
                {
                    "program": name,
                    "module": module,
                    "tokens": tokens,
                    "max_abs_residual_error": error,
                }
            )
    columns = [
        trace.assembled.residual_labels.index(str(direction))
        for direction in trace.output_space.basis
    ]
    manual_scores = final.project(trace.output_space).magnitudes
    model_scores = actual_final[:, columns]
    np.testing.assert_allclose(
        manual_scores,
        model_scores,
        rtol=config.rel_tol,
        atol=config.abs_tol,
        err_msg=f"{name}, output scores, input={tokens}",
    )
    # Reuse the same strict decoded-output check used by the schedule notebook.
    [check] = validate_outputs(name, trace.program, trace.assembled, [tokens], config)
    record = {
        "program": name,
        "tokens": tokens,
        "max_abs_score_error": float(np.max(np.abs(manual_scores - model_scores))),
        "residual_max_abs_error": max_residual_error,
        "rasp_output": check["expected"],
        "model_decoded": check["actual"],
        "manual_scores": manual_scores.tolist(),
        "model_scores": model_scores.tolist(),
        "block_trace": [
            {
                "module": step["module"],
                "residual": step["residual"].tolist(),
                "output_scores": step["output_scores"].tolist(),
            }
            for step in steps
        ],
    }
    return record, warnings


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_trace_study(programs, config: CompilerConfig, samples, output_root: Path):
    """Trace all supplied programs, retaining partial evidence on strict failure."""
    if not programs:
        raise ValueError("Supply at least one program.")
    samples = [list(tokens) for tokens in samples]
    config.validate_samples(samples)
    run_dir = new_run_directory(output_root)
    evidence = {
        "status": "running",
        "planned_programs": list(programs),
        "settings": {**asdict(config), **environment_metadata(), "samples": samples},
        "programs": [],
        "traces": [],
        "residual_warnings": [],
        "failures": [],
    }
    models = {}
    write_json(run_dir / "manifest.json", evidence)
    for name, program in programs.items():
        try:
            trace = build_trace_model(program, config)
            records, warnings = [], []
            for tokens in samples:
                record, sample_warnings = check_trace_sample(
                    name, trace, tokens, config
                )
                records.append(record)
                warnings.extend(sample_warnings)
            models[name] = trace
            evidence["traces"].extend(records)
            evidence["residual_warnings"].extend(warnings)
            evidence["programs"].append(
                {
                    "program": name,
                    "inputs_checked": len(records),
                    "max_abs_score_error": max(
                        row["max_abs_score_error"] for row in records
                    ),
                    "residual_max_abs_error": max(
                        row["residual_max_abs_error"] for row in records
                    ),
                    "residual_mismatches": len(warnings),
                    "status": "PASS_WITH_RESIDUAL_WARN" if warnings else "PASS",
                }
            )
            write_json(run_dir / "manifest.json", evidence)
            print(
                f"{name}: {len(records)} output checks; {len(warnings)} residual warnings"
            )
        except Exception as error:
            evidence["status"] = "failed"
            evidence["failures"].append(
                {
                    "program": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            write_json(run_dir / "manifest.json", evidence)
            raise
    evidence["status"] = "completed"
    write_json(run_dir / "manifest.json", evidence)
    write_json(run_dir / "step2_manual_trace.json", evidence["traces"])
    return {"models": models, "evidence": evidence, "run_dir": run_dir}

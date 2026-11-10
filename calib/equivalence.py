"""Prove each twin pair is one function compiled to two different schedules."""

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
from tracr.rasp import rasp

from calib.pairs import DROPPED, PAIR_CONFIG, candidate_pairs, twin_pairs
from raspformer import models
from raspformer.compiler import compile_program, extract_schedule, write_json

BATCH = 16_384
MAX_HEADS = 4
MAX_LAYERS = 6


@dataclass(frozen=True)
class Member:
    program: rasp.SOp
    model: models.Model
    token_codes: np.ndarray
    bos_code: int
    class_values: np.ndarray
    signature: tuple


def schedule_signature(program, assembled, config) -> tuple:
    """Layer count plus the sorted (first-write stage, lane count) of each computed variable."""
    summary, variables, _ = extract_schedule(program, assembled, config)
    writes = sorted(
        (row["first_write_half_layer_index"], len(row["residual_lane_indices"]))
        for row in variables
        if row["first_write_half_layer_index"] is not None
    )
    return (summary["transformer_layers"], tuple(writes))


def compile_member(program, config) -> Member:
    assembled = compile_program(program, config)
    if not hasattr(assembled.output_encoder, "encoding_map"):
        raise ValueError(f"{program.label}: output must be categorical.")
    classes = assembled.output_encoder.encoding_map
    class_values = np.empty(len(classes))
    for value, index in classes.items():
        class_values[index] = value
    codes = assembled.input_encoder.encoding_map
    token_codes = np.zeros(max(config.vocab) - min(config.vocab) + 1, dtype=np.int32)
    for token in config.vocab:
        token_codes[token - min(config.vocab)] = codes[token]
    return Member(
        program=program,
        model=models.from_compiled(assembled),
        token_codes=token_codes,
        bos_code=codes[config.bos],
        class_values=class_values,
        signature=schedule_signature(program, assembled, config),
    )


@lru_cache(maxsize=2)
def all_sequences(config) -> np.ndarray:
    """Every full-length sequence over the vocabulary, one row each."""
    return np.asarray(list(itertools.product(config.vocab, repeat=config.max_seq_len)))


def compiled_outputs(member, sequences, config) -> np.ndarray:
    """Decoded output values, [sequence, position], from batched forward passes."""
    codes = member.token_codes[sequences - min(config.vocab)]
    bos = np.full((len(codes), 1), member.bos_code, dtype=np.int32)
    encoded = np.concatenate([bos, codes], axis=1)
    predict = jax.jit(lambda batch: models.forward(member.model, batch)[0].argmax(-1))
    classes = np.concatenate([
        np.asarray(predict(encoded[start:start + BATCH]))
        for start in range(0, len(encoded), BATCH)
    ])
    return member.class_values[classes[:, 1:]]


def rasp_agreement(member, sequences, outputs, config) -> float:
    """Fraction of sequences where the compiled output equals the RASP evaluator's."""
    evaluator = rasp.DefaultRASPEvaluator()
    agree = 0
    for tokens, actual in zip(sequences.tolist(), outputs):
        expected = evaluator.evaluate(member.program, tokens)
        agree += None not in expected and bool(
            np.allclose(expected, actual, rtol=config.rel_tol, atol=config.abs_tol)
        )
    return agree / len(sequences)


def check_pair(pair, config=PAIR_CONFIG, sample=2000, seed=0) -> dict:
    """Check one pair exhaustively at full length; ``valid`` requires every criterion."""
    sequences = all_sequences(config)
    a, b = (compile_member(make(), config) for make in (pair.make_a, pair.make_b))
    out_a, out_b = (compiled_outputs(member, sequences, config) for member in (a, b))
    mismatches = int((out_a != out_b).any(axis=1).sum())
    rows = np.random.default_rng(seed).choice(len(sequences), size=sample, replace=False)
    agreement_a, agreement_b = (
        rasp_agreement(member, sequences[rows], outputs[rows], config)
        for member, outputs in ((a, out_a), (b, out_b))
    )
    same_classes = sorted(a.class_values.tolist()) == sorted(b.class_values.tolist())
    result = {
        "name": pair.name,
        "a_label": pair.a_label,
        "b_label": pair.b_label,
        "pilot": pair.pilot,
        "inputs_checked": len(sequences),
        "compiled_outputs_equal": mismatches == 0,
        "mismatches": mismatches,
        "output_classes_equal": same_classes,
        "rasp_agreement_a": agreement_a,
        "rasp_agreement_b": agreement_b,
        "schedules_differ": a.signature != b.signature,
        "layers_a": a.model.config.layers,
        "layers_b": b.model.config.layers,
        "heads_a": a.model.config.heads,
        "heads_b": b.model.config.heads,
        "computed_variables_a": len(a.signature[1]),
        "computed_variables_b": len(b.signature[1]),
        "residual_width_a": a.model.config.width,
        "residual_width_b": b.model.config.width,
        "output_values": [
            int(value) if value.is_integer() else value
            for value in np.union1d(out_a, out_b).tolist()
        ],
    }
    result["valid"] = bool(
        result["compiled_outputs_equal"]
        and same_classes
        and agreement_a == agreement_b == 1.0
        and result["schedules_differ"]
        and max(result["heads_a"], result["heads_b"]) <= MAX_HEADS
        and max(result["layers_a"], result["layers_b"]) <= MAX_LAYERS
    )
    return result


def status(row) -> str:
    if row["name"] in DROPPED:
        return f"dropped: {DROPPED[row['name']]}"
    return "valid" if row["valid"] else "INVALID"


def format_table(rows) -> str:
    header = ("pair", "algorithm A", "algorithm B", "layers", "vars", "width", "heads",
              "mismatch", "rasp", "status")
    body = [
        (
            row["name"], row["a_label"], row["b_label"],
            f"{row['layers_a']}/{row['layers_b']}",
            f"{row['computed_variables_a']}/{row['computed_variables_b']}",
            f"{row['residual_width_a']}/{row['residual_width_b']}",
            f"{row['heads_a']}/{row['heads_b']}",
            str(row["mismatches"]),
            f"{row['rasp_agreement_a']:.3f}/{row['rasp_agreement_b']:.3f}",
            status(row),
        )
        for row in rows
    ]
    widths = [max(len(line[i]) for line in (header, *body)) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
        for line in (header, *body)
    )


def write_results(rows, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pairs.json", rows)
    with (output / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "output_values": " ".join(map(str, row["output_values"]))})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    rows = []
    for pair in candidate_pairs():
        row = check_pair(pair)
        row["status"] = status(row)
        rows.append(row)
        print(f"{pair.name}: {row['status']}", flush=True)
    write_results(rows, args.output)
    print(format_table(rows))
    by_name = {row["name"]: row for row in rows}
    invalid = [pair.name for pair in twin_pairs() if not by_name[pair.name]["valid"]]
    stale = [name for name in DROPPED if by_name[name]["valid"]]
    print(f"{len(twin_pairs())} library pairs, {len(DROPPED)} dropped, "
          f"{time.perf_counter() - start:.0f}s")
    if invalid:
        print(f"Invalid library pairs: {', '.join(invalid)}", file=sys.stderr)
    if stale:
        print(f"Dropped pairs that now pass: {', '.join(stale)}", file=sys.stderr)
    return 1 if invalid or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())

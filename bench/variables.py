"""Ground-truth program variables for a compiled case: labels, classes, values.

A variable is one computed RASP expression in the compiled graph. Its value at
every sequence position comes from independent RASP evaluation, never from the
model. ``None`` (an Aggregate with no selected key) is class ``-1`` and is
excluded from probe and intervention targets.
"""

from dataclasses import dataclass

import numpy as np
from tracr.compiler import nodes
from tracr.rasp import rasp

from bench.phase1 import shim_config
from raspformer.compiler import extract_schedule, trace_craft_graph


@dataclass(frozen=True)
class Variable:
    label: str
    expression: rasp.SOp
    classes: tuple            # sorted value set, class id = index
    first_write: int          # half-layer index; stage = first_write + 1
    lanes: tuple              # residual lane indices in the compiled model

    @property
    def first_stage(self) -> int:
        return self.first_write + 1

    def class_id(self, value) -> int:
        return -1 if value is None else self.classes.index(value)


def case_variables(case, program, assembled, *, include_constant=False) -> list[Variable]:
    """Computed variables in first-write order; constants are skipped by default."""
    config = shim_config(case)
    traced = trace_craft_graph(program, config)
    _, schedule, _ = extract_schedule(program, assembled, config, traced=traced)
    variables = []
    for row in schedule:
        if row["first_write_half_layer_index"] is None:
            continue
        node = traced.graph.nodes[row["variable"]]
        expression = node[nodes.EXPR]
        if not rasp.is_categorical(expression):
            raise ValueError(f"{row['variable']} is numerical; only categorical variables are supported.")
        classes = tuple(sorted(node[nodes.VALUE_SET], key=lambda v: (str(type(v)), v)))
        if len(classes) < 2 and not include_constant:
            continue
        variables.append(Variable(row["variable"], expression, classes,
                                  row["first_write_half_layer_index"],
                                  tuple(row["residual_lane_indices"])))
    return sorted(variables, key=lambda v: (v.first_write, v.label))


def variable_values(variable: Variable, sequences) -> np.ndarray:
    """[sequence, position] class ids by RASP evaluation; -1 where undefined."""
    evaluator = rasp.DefaultRASPEvaluator()
    rows = []
    for tokens in sequences:
        values = evaluator.evaluate(variable.expression, list(tokens))
        rows.append([variable.class_id(v) for v in values])
    return np.asarray(rows, dtype=np.int64)


def output_ids(assembled, outputs) -> np.ndarray:
    """Map decoded output values (one sequence) to class ids; -1 for None."""
    encoding = assembled.output_encoder.encoding_map
    # A substituted numerical intermediate can push the output outside the
    # compiled value set (sorting keys); such positions are undefined.
    return np.asarray([-1 if v is None or v not in encoding else encoding[v] for v in outputs], dtype=np.int64)


def position_determined(values: np.ndarray) -> bool:
    """True when every sequence has the same value at each position (e.g. length
    or an index arithmetic at fixed sequence length); such a variable cannot be
    intervened on and is readable from the position embedding alone."""
    return bool((values == values[:1]).all())

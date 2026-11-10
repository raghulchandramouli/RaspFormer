"""Programs of the calibration study: the grid, same-function pairs and control pairs."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
from tracr.rasp import rasp

from calib.pairs import PAIR_CONFIG, twin_pairs
from raspformer.compiler import CompilerConfig, compile_program
from raspformer.research import family_program

GRID_DEPTHS = (3, 4, 5)
GRID_VARIABLES = (5, 6, 7)
GRID_VOCABULARIES = (4, 8, 12)
CONTROL_VOCABULARY = 8
SEQUENCE_LENGTH = 6
PROBE_SEQUENCES = 256
PROBE_SEED = 2026


@dataclass(frozen=True)
class Spec:
    """One program and the configuration its models share."""
    name: str
    kind: str
    make: Callable[[], rasp.SOp]
    config: CompilerConfig
    layers: int
    depth: int | None = None
    variables: int | None = None
    vocabulary: int | None = None


@dataclass(frozen=True)
class Pairing:
    """Two specs compared as one unit: ``twin`` shares a function, ``control`` shares a structure."""
    name: str
    kind: str
    a: Spec
    b: Spec


def grid_config(vocabulary):
    """Tracr needs the value width to fit the key size, which ``max_seq_len`` sets."""
    return CompilerConfig(vocab=tuple(range(vocabulary)), max_seq_len=10)


@cache
def compiled_layers(make, config):
    return compile_program(make(), config).model_config.num_layers


def _family_spec(kind, depth, variables, vocabulary, variant):
    suffix = "" if variant == 0 else f"-variant{variant}"
    return Spec(name=f"{kind}-d{depth}-v{variables}-k{vocabulary}{suffix}", kind=kind,
                make=lambda: family_program(depth, variables, vocabulary, length=SEQUENCE_LENGTH, variant=variant),
                config=grid_config(vocabulary), layers=depth - 1,
                depth=depth, variables=variables, vocabulary=vocabulary)


def grid_specs():
    return [_family_spec("grid", depth, variables, vocabulary, 0)
            for vocabulary in GRID_VOCABULARIES for depth in GRID_DEPTHS for variables in GRID_VARIABLES]


def control_pairings():
    """Same tree, different constants: a different function on an identical structure."""
    return [Pairing(f"control-d{depth}-v{variables}", "control",
                    _family_spec("grid", depth, variables, CONTROL_VOCABULARY, 0),
                    _family_spec("control", depth, variables, CONTROL_VOCABULARY, 1))
            for depth in GRID_DEPTHS for variables in GRID_VARIABLES]


@cache
def _twin_pairing(pair):
    """Both members train at the deeper member's layer count, so a pair shares one architecture."""
    layers = max(compiled_layers(pair.make_a, PAIR_CONFIG), compiled_layers(pair.make_b, PAIR_CONFIG))
    a, b = (Spec(name=f"pair-{pair.name}-{member}", kind="pair", make=make, config=PAIR_CONFIG, layers=layers)
            for member, make in (("a", pair.make_a), ("b", pair.make_b)))
    return Pairing(pair.name, "twin", a, b)


def twin_pairings(*, pilot_only=False):
    return [_twin_pairing(pair) for pair in twin_pairs() if pair.pilot or not pilot_only]


def all_specs(*, pilot_only=False):
    """Every distinct program, keyed by name."""
    specs = {} if pilot_only else {spec.name: spec for spec in grid_specs()}
    pairings = twin_pairings(pilot_only=pilot_only) + ([] if pilot_only else control_pairings())
    for pairing in pairings:
        specs.setdefault(pairing.a.name, pairing.a)
        specs.setdefault(pairing.b.name, pairing.b)
    return specs


@cache
def spec(name):
    """Look one program up by name without building the rest of the library."""
    if name.startswith("pair-"):
        pair_name, member = name[len("pair-"):].rsplit("-", 1)
        pairing = _twin_pairing(next(pair for pair in twin_pairs() if pair.name == pair_name))
        return pairing.a if member == "a" else pairing.b
    candidates = grid_specs() + [pairing.b for pairing in control_pairings()]
    return next(candidate for candidate in candidates if candidate.name == name)


def probe_tokens(config):
    """The held-out sequences every model of a configuration is measured on."""
    rng = np.random.default_rng(PROBE_SEED)
    return rng.choice(np.asarray(config.vocab), size=(PROBE_SEQUENCES, SEQUENCE_LENGTH))


def training_tokens(config, seed, count):
    """Uniform sequences with every probe sequence removed."""
    held_out = {tuple(row) for row in probe_tokens(config).tolist()}
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        batch = rng.choice(np.asarray(config.vocab), size=(count, SEQUENCE_LENGTH))
        rows.extend(row for row in batch.tolist() if tuple(row) not in held_out)
    return np.asarray(rows[:count])

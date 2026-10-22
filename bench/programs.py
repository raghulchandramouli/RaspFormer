"""Six InterpBench programs, ported verbatim from circuits-benchmark.

Each case mirrors ``circuits_benchmark/benchmark/cases/case_<id>.py`` at
FlyingPumba/circuits-benchmark. InterpBench compiles with Tracr using
``max_seq_len - 1`` (its max_seq_len counts BOS), BOS "BOS", PAD "PAD", and
``causal=supports_causal_masking()``. Those settings are reproduced here so the
compiled model matches the trained ``ll_model.pth`` for the same case id.

The trained models, their configs, and ground-truth circuit edges live in
``bench/interpbench/<id>/`` and come from huggingface.co/cybershiptrooper/InterpBench.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence
import itertools
import random
import string

from tracr.compiler import compiling
from tracr.rasp import rasp

BOS = "BOS"
PAD = "PAD"
MLP_EXACTNESS = 100  # Tracr default; InterpBench does not override it.


# --- Shared helpers (circuits_benchmark/benchmark/common_programs.py) ------


def make_length() -> rasp.SOp:
    all_true_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    return rasp.SelectorWidth(all_true_selector).named("length")


def make_hist() -> rasp.SOp:
    same_tok = rasp.Select(rasp.tokens, rasp.tokens,
                           rasp.Comparison.EQ).named("same_tok")
    return rasp.SelectorWidth(same_tok).named("hist")


def shift_by(offset: int, sop: rasp.SOp, default=None) -> rasp.SOp:
    select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,
                                       lambda k, q: q == k + offset)
    out = rasp.Aggregate(select_off_by_offset, sop, default=default)
    return out.named(f"shift_by({offset})")


def make_reverse(sop: rasp.SOp) -> rasp.SOp:
    opp_idx = (make_length() - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")
    reverse_selector = rasp.Select(rasp.indices, opp_idx,
                                   rasp.Comparison.EQ).named("reverse_selector")
    return rasp.Aggregate(reverse_selector, sop).named("reverse")


def make_unique_token_extractor(sop: rasp.SOp) -> rasp.SOp:
    tokens_with_indices = rasp.SequenceMap(lambda x, i: (x, i), sop, rasp.indices)
    occurrences_selector = rasp.Select(tokens_with_indices,
                                       tokens_with_indices,
                                       lambda k, q: k[0] == q[0] and k[1] <= q[1])
    occurrences_per_row = rasp.SelectorWidth(occurrences_selector)
    first_occurrences_indices = rasp.SequenceMap(
        lambda count, i: i if count == 1 else -1, occurrences_per_row, rasp.indices)
    tokens_selector = rasp.Select(first_occurrences_indices, rasp.indices,
                                  rasp.Comparison.EQ).named("unique_tokens_selector")
    return rasp.Aggregate(tokens_selector, sop, default=None).named("unique_tokens")


# --- Case programs (circuits_benchmark/benchmark/cases/case_<id>.py) -------


def make_case_2() -> rasp.SOp:
    """Reverse the input sequence."""
    return make_reverse(rasp.tokens)


def make_case_7() -> rasp.SOp:
    """Returns the number of times each token occurs in the input."""
    return make_hist()


def make_case_13() -> rasp.SOp:
    """Analyzes the trend (increasing, decreasing, constant) of numeric tokens.

    This is the program at circuits-benchmark commit 30c3c43, the version the
    published model was trained on (weights uploaded 2024-06-12). Upstream
    commit bceb41a "Fix Case 13" (2024-09-09) later replaced it with a single
    next-token comparison, but the Hugging Face weights were not retrained.
    Against the current program the model scores 12% token accuracy; against
    this one it matches. Note the second comparison labels "increasing" when
    the next token is smaller, and the merge keeps the previous-token branch
    only where the next-token branch says "constant".
    """
    prev_token = shift_by(1, rasp.tokens)
    next_token = shift_by(-1, rasp.tokens)
    first_part = rasp.SequenceMap(
        lambda x, y: "increasing" if y > x else ("decreasing" if y < x else "constant"),
        prev_token, rasp.tokens)
    second_part = rasp.SequenceMap(
        lambda x, y: "increasing" if y < x else ("decreasing" if y > x else "constant"),
        rasp.tokens, next_token)
    return rasp.SequenceMap(lambda x, y: x if y == "constant" else y, first_part, second_part)


def make_case_124() -> rasp.SOp:
    """Check if all elements in a list are equal."""
    unequal_selector = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x != y)
    checks = rasp.SelectorWidth(unequal_selector)
    zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices),
                                rasp.Comparison.EQ)
    return rasp.Map(lambda x: 1 if x > 0 else 0, rasp.SelectorWidth(zero_selector))


def make_case_21() -> rasp.SOp:
    """Extract unique tokens from a string."""
    return make_unique_token_extractor(rasp.tokens)


def make_case_44() -> rasp.SOp:
    """Replaces each element with the number of elements greater than it in the sequence."""
    greater_than_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.GT).named("greater_than_selector")
    return rasp.SelectorWidth(greater_than_selector).named("count_greater_than")


# --- Case registry ---------------------------------------------------------


def _letters(count: int) -> tuple:
    return tuple(string.ascii_lowercase[:count])


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    description: str
    vocab: tuple
    max_seq_len: int          # InterpBench convention: counts BOS.
    min_seq_len: int          # From benchmark_cases_metadata.csv; also counts BOS.
    causal: bool
    build: Callable[[], rasp.SOp]
    raspformer_match: str | None = None   # Same algorithm in raspformer/programs.py.
    sampler: str = "uniform"              # InterpBench cases may override sample_data.
    notes: str = ""

    @property
    def compile_seq_len(self) -> int:
        """Value passed to Tracr: max tokens without BOS."""
        return self.max_seq_len - 1

    @property
    def artifact_dir(self) -> str:
        return f"bench/interpbench/{self.case_id}"


CASES: dict[str, BenchCase] = {
    "2": BenchCase(
        case_id="2",
        description="Reverse the input sequence.",
        vocab=_letters(26),
        max_seq_len=10, min_seq_len=4, causal=False,
        build=make_case_2,
        notes="length -> opposite index -> aggregate. Four trained layers.",
    ),
    "7": BenchCase(
        case_id="7",
        description="Returns the number of times each token occurs in the input.",
        vocab=_letters(3),
        max_seq_len=10, min_seq_len=4, causal=False,
        build=make_case_7,
        raspformer_match="P05_histogram",
    ),
    "13": BenchCase(
        case_id="13",
        description="Analyzes the trend (increasing, decreasing, constant) of numeric tokens.",
        vocab=(0, 1, 2),
        max_seq_len=10, min_seq_len=4, causal=False,
        build=make_case_13,
        notes="Pre-fix upstream program (commit 30c3c43) matching the published weights; "
              "two shifts merge through three SequenceMaps. See make_case_13.",
    ),
    "124": BenchCase(
        case_id="124",
        description="Check if all elements in a list are equal.",
        vocab=tuple(range(0, 11)),
        max_seq_len=10, min_seq_len=4, causal=False,
        build=make_case_124,
        sampler="balanced_all_equal",
        notes="Predicate over the whole sequence, like P09/P11. InterpBench samples "
              "half all-equal and half unequal sequences; uniform sampling almost "
              "never produces a positive.",
    ),
    "21": BenchCase(
        case_id="21",
        description="Extract unique tokens from a string",
        vocab=_letters(3),
        max_seq_len=10, min_seq_len=4, causal=True,
        build=make_case_21,
        notes="Causal attention. Tuple-valued intermediate; four trained layers.",
    ),
    "44": BenchCase(
        case_id="44",
        description="Replaces each element with the number of elements greater than it in the sequence",
        vocab=tuple(range(0, 11)),
        max_seq_len=10, min_seq_len=4, causal=False,
        build=make_case_44,
        raspformer_match="P06_count_greater_than",
    ),
}


def compile_case(case: BenchCase, program: rasp.SOp | None = None):
    """Compile exactly as InterpBench's TracrBenchmarkCase.get_tracr_output does.

    Pass ``program`` to compile a specific expression object; RASP auto-labels
    are per object, so schedule extraction must use the compiled instance.
    """
    return compiling.compile_rasp_to_model(
        case.build() if program is None else program,
        vocab=set(case.vocab),
        max_seq_len=case.compile_seq_len,
        causal=case.causal,
        compiler_bos=BOS,
        compiler_pad=PAD,
        mlp_exactness=MLP_EXACTNESS,
    )


def evaluate(case: BenchCase, tokens: Sequence) -> list:
    """Independent RASP evaluation (no compiler) for one token sequence without BOS.

    Positions where RASP yields ``None`` (an Aggregate with no selected key)
    are undefined; the compiled model emits an arbitrary class there.
    """
    return list(rasp.evaluate(case.build(), list(tokens)))


def reference_outputs(compiled_model, tokens: Sequence) -> list:
    """Labels the InterpBench model was trained on: the compiled model's argmax.

    InterpBench builds training targets from the compiled Tracr model, not from
    ``rasp.evaluate`` (circuits_benchmark TracrDataset.get_encoded_dataset),
    so this is the ground truth for scoring a trained ``ll_model.pth``.
    The BOS position is excluded; InterpBench forces its label to class zero.
    """
    return list(compiled_model.apply([BOS] + list(tokens)).decoded[1:])


def sample_inputs(case: BenchCase, count: int, *, seed: int = 0,
                  variable_length: bool = False) -> list[list]:
    """Random token sequences without BOS, matching InterpBench's sampler.

    InterpBench defaults to fixed length ``max_seq_len - 1``; with
    ``variable_length`` it draws lengths from ``min_seq_len - 1`` upward.
    """
    rng = random.Random(seed)
    vals = sorted(case.vocab, key=str)
    low = case.min_seq_len - 1 if variable_length else case.compile_seq_len
    lengths = [rng.randint(low, case.compile_seq_len) for _ in range(count)]
    if case.sampler == "balanced_all_equal":
        # Case124.sample_data: first half all-equal, second half with repeats of
        # distinct elements, so both predicate outcomes are represented.
        rows = []
        for i, length in enumerate(lengths):
            if i < count // 2:
                rows.append([rng.choice(vals)] * length)
            else:
                picks = rng.sample(vals, k=min(length, len(vals)))
                rows.append((picks * (length // len(picks) + 1))[:length])
        return rows
    if case.sampler != "uniform":
        raise ValueError(f"Unknown sampler {case.sampler!r}.")
    return [[rng.choice(vals) for _ in range(length)] for length in lengths]


def all_inputs(case: BenchCase, length: int | None = None):
    """Every sequence of one length. Case 2 has 26**9 sequences; do not exhaust it."""
    length = case.compile_seq_len if length is None else length
    return itertools.product(sorted(case.vocab, key=str), repeat=length)

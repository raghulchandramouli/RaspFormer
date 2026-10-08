"""Define the sequence algorithms. Each factory returns a fresh RASP expression."""

from tracr.compiler import lib
from tracr.rasp import rasp

PROGRAM_DESCRIPTIONS = {
    "P01_absolute": "Absolute value of each token.",
    "P02_index_parity": "Zero-based position modulo two.",
    "P03_increment_by_index": "Token plus its zero-based position.",
    "P04_first_element": "First token repeated at every position.",
    "P05_histogram": "Occurrences of each token in the full sequence.",
    "P06_count_greater_than": "Number of tokens strictly greater than the current token.",
    "P07_sum_with_next": "Token plus successor; the final token adds itself.",
    "P08_pairwise_sum": "Token plus predecessor; the first wraps to the final token.",
    "P09_check_increasing": "One everywhere iff the sequence is nondecreasing.",
    "P10_rotate_left": "Cyclic left rotation by one position.",
    "P11_check_palindrome": "One everywhere iff the sequence is a palindrome.",
    "P12_sorting": "Ascending sort, including duplicate tokens.",
    "A_prev_token_class": "Previous token; the first position repeats itself.",
    "B_histogram_then_repeated_class": "One iff the token occurs more than once.",
}


def make_absolute() -> rasp.SOp:
    """Absolute value of each token."""
    return rasp.Map(abs, rasp.tokens).named("absolute")


def make_index_parity() -> rasp.SOp:
    """Zero-based position modulo two."""
    return rasp.SequenceMap(
        lambda token, index: index % 2, rasp.tokens, rasp.indices
    ).named("index_parity")


def make_increment_by_index() -> rasp.SOp:
    """Token plus its zero-based position."""
    return rasp.SequenceMap(
        lambda token, index: token + index, rasp.tokens, rasp.indices
    ).named("increment_by_index")


def make_first_element() -> rasp.SOp:
    """First token repeated at every position."""
    first_position = rasp.Select(
        rasp.indices, rasp.indices, lambda key, query: key == 0
    ).named("first_position")

    return rasp.Aggregate(first_position, rasp.tokens).named("first_element")


def make_histogram() -> rasp.SOp:
    """Occurrences of each token in the full sequence."""
    return lib.make_hist().named("histogram")


def make_count_greater_than() -> rasp.SOp:
    """Number of tokens strictly greater than the current token."""
    greater_than_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.GT
    ).named("greater_than_selector")

    return rasp.SelectorWidth(greater_than_selector).named("count_greater_than")


def make_sum_with_next() -> rasp.SOp:
    """Token plus successor; the final token adds itself."""
    next_clamped_index = rasp.SequenceMap(
        lambda index, length: min(index + 1, length - 1),
        rasp.indices,
        lib.make_length(),
    ).named("next_clamped_index")

    next_selector = rasp.Select(
        rasp.indices, next_clamped_index, rasp.Comparison.EQ
    ).named("next_selector")

    next_value = rasp.Aggregate(next_selector, rasp.tokens).named("next_value")

    return rasp.SequenceMap(
        lambda current, following: current + following,
        rasp.tokens,
        next_value,
    ).named("sum_with_next")


def make_pairwise_sum() -> rasp.SOp:
    """Token plus predecessor; the first wraps to the final token."""
    previous_cyclic_index = rasp.SequenceMap(
        lambda index, length: (index - 1) % length if length else 0,
        rasp.indices,
        lib.make_length(),
    ).named("previous_cyclic_index")

    previous_selector = rasp.Select(
        rasp.indices, previous_cyclic_index, rasp.Comparison.EQ
    ).named("previous_selector")

    previous_value = rasp.Aggregate(previous_selector, rasp.tokens).named("previous_value")

    return rasp.SequenceMap(
        lambda current, previous: current + previous,
        rasp.tokens,
        previous_value,
    ).named("pairwise_sum")


def make_check_increasing() -> rasp.SOp:
    """One everywhere iff the sequence is nondecreasing."""
    locally_increasing = rasp.SequenceMap(
        lambda current, previous: True if previous is None else previous <= current,
        rasp.tokens,
        lib.shift_by(1, rasp.tokens),
    ).named("locally_increasing")

    decrease_selector = rasp.Select(
        locally_increasing,
        rasp.indices,
        lambda is_ok, query: is_ok is False,
    ).named("decrease_selector")

    decrease_count = rasp.SelectorWidth(decrease_selector).named("decrease_count")

    return rasp.Map(lambda count: int(count == 0), decrease_count).named("check_increasing")


def make_rotate_left() -> rasp.SOp:
    """Cyclic left rotation by one position."""
    next_index = rasp.SequenceMap(
        lambda index, length: 0 if length == 0 else (index + 1) % length,
        rasp.indices,
        lib.make_length(),
    ).named("next_index")

    rotate_selector = rasp.Select(
        rasp.indices, next_index, rasp.Comparison.EQ
    ).named("rotate_selector")

    return rasp.Aggregate(rotate_selector, rasp.tokens).named("rotate_left")


def make_check_palindrome() -> rasp.SOp:
    """One everywhere iff the sequence is a palindrome."""
    mirror_matches = rasp.SequenceMap(
        lambda current, mirror: current == mirror,
        rasp.tokens,
        lib.make_reverse(rasp.tokens),
    ).named("mirror_matches")

    mismatch_selector = rasp.Select(
        mirror_matches,
        rasp.indices,
        lambda matches, query: not matches,
    ).named("mismatch_selector")

    mismatch_count = rasp.SelectorWidth(mismatch_selector).named("mismatch_count")

    return rasp.Map(lambda count: int(count == 0), mismatch_count).named("check_palindrome")


def make_sorting(max_seq_len: int) -> rasp.SOp:
    """Ascending sort, including duplicate tokens."""
    return lib.make_sort(
        rasp.tokens,
        rasp.tokens,
        max_seq_len=max_seq_len,
        min_key=1.0,
    ).named("sorting")


def make_previous_token_class() -> rasp.SOp:
    """Previous token, with the first position selecting itself."""
    previous_or_self = rasp.Select(
        rasp.indices,
        rasp.indices,
        lambda key, query: key == query - 1 or (query == 0 and key == 0),
    ).named("previous_or_self")
    return rasp.Aggregate(previous_or_self, rasp.tokens).named("prev_token_class")


def make_repeated_token_class() -> rasp.SOp:
    """One iff a token occurs more than once in the sequence."""
    histogram = lib.make_hist().named("histogram")
    return rasp.Map(lambda count: int(count > 1), histogram).named("repeated_class")


def build_programs(
    max_seq_len: int = 6, *, include_examples: bool = False
) -> dict[str, rasp.SOp]:
    """Build the original twelve programs, optionally adding the focused A/B examples."""
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("max_seq_len must be a positive integer.")
    programs = {
        "P01_absolute": make_absolute(),
        "P02_index_parity": make_index_parity(),
        "P03_increment_by_index": make_increment_by_index(),
        "P04_first_element": make_first_element(),
        "P05_histogram": make_histogram(),
        "P06_count_greater_than": make_count_greater_than(),
        "P07_sum_with_next": make_sum_with_next(),
        "P08_pairwise_sum": make_pairwise_sum(),
        "P09_check_increasing": make_check_increasing(),
        "P10_rotate_left": make_rotate_left(),
        "P11_check_palindrome": make_check_palindrome(),
        "P12_sorting": make_sorting(max_seq_len),
    }
    if include_examples:
        programs.update(
            {
                "A_prev_token_class": make_previous_token_class(),
                "B_histogram_then_repeated_class": make_repeated_token_class(),
            }
        )
    return programs

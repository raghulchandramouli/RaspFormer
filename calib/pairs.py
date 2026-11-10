"""Twin RASP programs: one function, two algorithms, correct at the full length of six.

Every member is checked by ``calib.equivalence``. Programs may hard-code the
sequence length, because the study only uses full-length inputs.
"""

from collections.abc import Callable
from dataclasses import dataclass

from tracr.rasp import rasp

from raspformer.compiler import CompilerConfig

PAIR_CONFIG = CompilerConfig()
LENGTH = PAIR_CONFIG.max_seq_len
LAST = LENGTH - 1


@dataclass(frozen=True)
class TwinPair:
    name: str
    description: str
    a_label: str
    b_label: str
    make_a: Callable[[], rasp.SOp]
    make_b: Callable[[], rasp.SOp]
    pilot: bool = False


def fetch(keys, queries, predicate, name, source=rasp.tokens) -> rasp.SOp:
    """Copy ``source`` from the position the selector picks for each query."""
    selector = rasp.Select(keys, queries, predicate).named(f"{name}_selector")
    return rasp.Aggregate(selector, source).named(name)


def count(keys, queries, predicate, name) -> rasp.SOp:
    """Number of positions the selector picks for each query."""
    selector = rasp.Select(keys, queries, predicate).named(f"{name}_selector")
    return rasp.SelectorWidth(selector).named(name)


def sequence_length() -> rasp.SOp:
    return count(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE, "length")


def token_index_pair() -> rasp.SOp:
    return rasp.SequenceMap(
        lambda token, index: (token, index), rasp.tokens, rasp.indices
    ).named("token_index_pair")


def token_and_is_first() -> rasp.SOp:
    return rasp.SequenceMap(
        lambda token, index: (token, index == 0), rasp.tokens, rasp.indices
    ).named("token_and_is_first")


def tiebroken_key() -> rasp.SOp:
    """Token plus a position fraction, so equal tokens order by position."""
    return rasp.SequenceMap(
        lambda token, index: token + index / LENGTH, rasp.tokens, rasp.indices
    ).named("tiebroken_key")


def is_position_zero(key, query):
    return key == 0


def is_last_position(key, query):
    return key == LAST


def is_next_position(key, query):
    return key == (query + 1) % LENGTH


def is_next_or_last(key, query):
    return key == min(query + 1, LAST)


def is_previous_or_first(key, query):
    return key == max(query - 1, 0)


def is_mirror_position(key, query):
    return key == LAST - query


def same_token_earlier(key, query):
    return key[0] == query[0] and key[1] < query[1]


def same_token_later(key, query):
    return key[0] == query[0] and key[1] > query[1]


def make_sort_tiebroken_key() -> rasp.SOp:
    """Rank every position by one key that already breaks ties."""
    key = tiebroken_key()
    rank = count(key, key, rasp.Comparison.LT, "rank")
    return fetch(rank, rasp.indices, rasp.Comparison.EQ, "sorted")


def make_sort_rank_parts() -> rasp.SOp:
    """Rank as smaller tokens plus earlier copies of the same token."""
    pair = token_index_pair()
    smaller = count(rasp.tokens, rasp.tokens, rasp.Comparison.LT, "smaller_count")
    earlier = count(pair, pair, same_token_earlier, "earlier_copies")
    rank = rasp.SequenceMap(
        lambda smaller, earlier: smaller + earlier, smaller, earlier
    ).named("rank")
    return fetch(rank, rasp.indices, rasp.Comparison.EQ, "sorted")


def make_reverse_mirrored_index() -> rasp.SOp:
    """Attend straight to the mirrored position."""
    return fetch(rasp.indices, rasp.indices, is_mirror_position, "reversed")


def make_reverse_length_minus_index() -> rasp.SOp:
    """Measure the length, then attend to length minus index minus one."""
    opposite = rasp.SequenceMap(
        lambda length, index: length - index - 1, sequence_length(), rasp.indices
    ).named("opposite_index")
    return fetch(rasp.indices, opposite, rasp.Comparison.EQ, "reversed")


def make_palindrome_reverse_and_compare() -> rasp.SOp:
    """Reverse the tokens, compare in place, and average the disagreements."""
    differs = rasp.SequenceMap(
        lambda token, mirrored: int(token != mirrored),
        rasp.tokens,
        make_reverse_mirrored_index(),
    ).named("differs")
    differs_value = rasp.numerical(rasp.Map(lambda flag: flag, differs)).named("differs_value")
    everything = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE).named("everything")
    mismatch_share = rasp.numerical(
        rasp.Aggregate(everything, differs_value, default=0)
    ).named("mismatch_share")
    return rasp.Map(lambda share: int(share == 0), mismatch_share).named("palindrome")


def make_palindrome_mismatch_count() -> rasp.SOp:
    """Count mirror mismatches with selectors; the reversed tokens are never written."""
    pair = token_index_pair()
    mirror_mismatch = count(
        pair,
        pair,
        lambda key, query: key[1] == LAST - query[1] and key[0] != query[0],
        "mirror_mismatch",
    )
    mismatches = count(
        mirror_mismatch, mirror_mismatch, lambda key, query: key == 1, "mismatch_count"
    )
    return rasp.Map(lambda total: int(total == 0), mismatches).named("palindrome")


def make_histogram_selector_width() -> rasp.SOp:
    """Width of the same-token selector."""
    return count(rasp.tokens, rasp.tokens, rasp.Comparison.EQ, "histogram")


def make_histogram_aggregate_count() -> rasp.SOp:
    """Average an anchor flag over same-token positions; the mean is one over the group size."""
    is_first = rasp.numerical(
        rasp.Map(lambda index: int(index == 0), rasp.indices)
    ).named("is_first")
    same_or_first = rasp.Select(
        token_and_is_first(), rasp.tokens, lambda key, query: key[0] == query or key[1]
    ).named("same_or_first")
    first_share = rasp.numerical(
        rasp.Aggregate(same_or_first, is_first, default=0)
    ).named("first_share")
    group_size = rasp.Map(
        lambda share: round(1 / share) if share else None, first_share
    ).named("group_size")
    first_token = fetch(rasp.indices, rasp.indices, is_position_zero, "first_token")
    first_is_extra = rasp.SequenceMap(
        lambda token, first: int(token != first), rasp.tokens, first_token
    ).named("first_is_extra")
    return rasp.SequenceMap(
        lambda size, extra: size - extra, group_size, first_is_extra
    ).named("histogram")


def make_count_greater_strict() -> rasp.SOp:
    """Width of the strictly-greater selector."""
    return count(rasp.tokens, rasp.tokens, rasp.Comparison.GT, "count_greater")


def make_count_greater_at_least_minus_equal() -> rasp.SOp:
    """Tokens at least as large, minus tokens that are equal."""
    at_least = count(rasp.tokens, rasp.tokens, rasp.Comparison.GEQ, "at_least_count")
    equal = count(rasp.tokens, rasp.tokens, rasp.Comparison.EQ, "equal_count")
    return rasp.SequenceMap(
        lambda at_least, equal: max(at_least - equal, 0), at_least, equal
    ).named("count_greater")


def make_first_element_position_selector() -> rasp.SOp:
    """Attend to index zero."""
    return fetch(rasp.indices, rasp.indices, is_position_zero, "first_element")


def make_first_element_flag_selector() -> rasp.SOp:
    """Write an is-first flag, then attend to the flagged position."""
    is_first = rasp.Map(lambda index: int(index == 0), rasp.indices).named("is_first")
    # Queries are indices because Tracr sizes every head from its widest query or
    # key space, and a two-valued query leaves no room for the eight token lanes.
    return fetch(is_first, rasp.indices, lambda key, query: key == 1, "first_element")


def make_last_element_fixed_index() -> rasp.SOp:
    """Attend to the hard-coded final index."""
    return fetch(rasp.indices, rasp.indices, is_last_position, "last_element")


def make_last_element_first_of_reversed() -> rasp.SOp:
    """Reverse, then read the first position."""
    return fetch(
        rasp.indices,
        rasp.indices,
        is_position_zero,
        "last_element",
        source=make_reverse_mirrored_index(),
    )


def make_rotate_left_cyclic_selector() -> rasp.SOp:
    """Attend to the next index, wrapping at the end."""
    return fetch(rasp.indices, rasp.indices, is_next_position, "rotated")


def make_rotate_left_clamp_and_repair() -> rasp.SOp:
    """Fetch the clamped successor and the first token; an MLP wraps the final position."""
    following = fetch(rasp.indices, rasp.indices, is_next_or_last, "clamped_following")
    first_token = fetch(rasp.indices, rasp.indices, is_position_zero, "first_token")
    wrap = rasp.SequenceMap(
        lambda index, first: (index == LAST, first), rasp.indices, first_token
    ).named("wrap_candidate")
    return rasp.SequenceMap(
        lambda wrap, following: wrap[1] if wrap[0] else following, wrap, following
    ).named("rotated")


def make_rotate_two_single_hop() -> rasp.SOp:
    """Attend two positions ahead in one step."""
    return fetch(
        rasp.indices,
        rasp.indices,
        lambda key, query: key == (query + 2) % LENGTH,
        "rotated_twice",
    )


def make_rotate_two_double_hop() -> rasp.SOp:
    """Rotate by one, then rotate the result by one again."""
    once = fetch(rasp.indices, rasp.indices, is_next_position, "rotated_once")
    return fetch(rasp.indices, rasp.indices, is_next_position, "rotated_twice", source=once)


def make_previous_token_position_selector() -> rasp.SOp:
    """Attend to the previous position; the first attends to itself."""
    return fetch(rasp.indices, rasp.indices, is_previous_or_first, "previous_token")


def make_previous_token_reverse_shift_reverse() -> rasp.SOp:
    """Reverse, take the clamped successor, and reverse back."""
    shifted = fetch(
        rasp.indices,
        rasp.indices,
        is_next_or_last,
        "reversed_successor",
        source=make_reverse_mirrored_index(),
    )
    return fetch(rasp.indices, rasp.indices, is_mirror_position, "previous_token", source=shifted)


def make_repeated_histogram_threshold() -> rasp.SOp:
    """Count copies of the token, then test for more than one."""
    return rasp.Map(lambda copies: int(copies > 1), make_histogram_selector_width()).named("repeated")


def make_repeated_earlier_or_later_copy() -> rasp.SOp:
    """Count copies before and after this position separately."""
    pair = token_index_pair()
    earlier = count(pair, pair, same_token_earlier, "earlier_copies")
    later = count(pair, pair, same_token_later, "later_copies")
    return rasp.SequenceMap(
        lambda earlier, later: int(earlier + later > 0), earlier, later
    ).named("repeated")


def make_equals_first_fetch_then_compare() -> rasp.SOp:
    """Copy the first token everywhere, then compare."""
    first_token = fetch(rasp.indices, rasp.indices, is_position_zero, "first_token")
    return rasp.SequenceMap(
        lambda token, first: int(token == first), rasp.tokens, first_token
    ).named("equals_first")


def make_equals_first_match_selector() -> rasp.SOp:
    """Select the first position only when it holds the same token; never copy it."""
    matches = count(
        token_and_is_first(),
        rasp.tokens,
        lambda key, query: key[1] and key[0] == query,
        "first_match_count",
    )
    return rasp.Map(lambda matches: int(matches > 0), matches).named("equals_first")


def make_nondecreasing_adjacent_comparison() -> rasp.SOp:
    """Compare each token with its predecessor and count the descents."""
    previous = fetch(rasp.indices, rasp.indices, is_previous_or_first, "previous_token")
    descends = rasp.SequenceMap(
        lambda token, previous: int(previous > token), rasp.tokens, previous
    ).named("descends")
    descents = count(descends, descends, lambda key, query: key == 1, "descent_count")
    return rasp.Map(lambda total: int(total == 0), descents).named("nondecreasing")


def make_nondecreasing_inversion_count() -> rasp.SOp:
    """Count, per position, every earlier token that is larger."""
    pair = token_index_pair()
    inversions = count(
        pair,
        pair,
        lambda key, query: key[1] < query[1] and key[0] > query[0],
        "inversion_count",
    )
    inverted = count(inversions, inversions, lambda key, query: key > 0, "inverted_positions")
    return rasp.Map(lambda total: int(total == 0), inverted).named("nondecreasing")


def make_sum_with_next_clamped_selector() -> rasp.SOp:
    """Attend to the successor, clamped so the final token reads itself."""
    following = fetch(rasp.indices, rasp.indices, is_next_or_last, "following_token")
    return rasp.SequenceMap(
        lambda token, following: token + following, rasp.tokens, following
    ).named("sum_with_next")


def make_sum_with_next_wrap_then_repair() -> rasp.SOp:
    """Attend cyclically, then repair the final position in an MLP."""
    wrapped = fetch(rasp.indices, rasp.indices, is_next_position, "wrapped_following")
    following = rasp.SequenceMap(
        lambda pair, wrapped: pair[0] if pair[1] == LAST else wrapped,
        token_index_pair(),
        wrapped,
    ).named("following_token")
    return rasp.SequenceMap(
        lambda token, following: token + following, rasp.tokens, following
    ).named("sum_with_next")


def make_maximum_no_greater_token() -> rasp.SOp:
    """Attend to every token that nothing exceeds; ties hold the same value."""
    greater = count(rasp.tokens, rasp.tokens, rasp.Comparison.GT, "greater_count")
    return fetch(greater, greater, lambda key, query: key == 0, "maximum")


def make_maximum_last_of_sorted() -> rasp.SOp:
    """Sort, then read the final position."""
    return fetch(
        rasp.indices, rasp.indices, is_last_position, "maximum", source=make_sort_tiebroken_key()
    )


def make_minimum_no_smaller_token() -> rasp.SOp:
    """Attend to every token that nothing undercuts; ties hold the same value."""
    smaller = count(rasp.tokens, rasp.tokens, rasp.Comparison.LT, "smaller_count")
    return fetch(smaller, smaller, lambda key, query: key == 0, "minimum")


def make_minimum_first_in_tiebroken_order() -> rasp.SOp:
    """Rank by the tiebroken key, then attend to the unique rank-zero position."""
    key = tiebroken_key()
    rank = count(key, key, rasp.Comparison.LT, "rank")
    return fetch(rank, rank, lambda key, query: key == 0, "minimum")


def make_swap_adjacent_xor_selector() -> rasp.SOp:
    """Attend to the index with its lowest bit flipped."""
    return fetch(rasp.indices, rasp.indices, lambda key, query: key == query ^ 1, "swapped")


def make_swap_adjacent_fetch_both_then_choose() -> rasp.SOp:
    """Fetch both neighbours; an MLP keeps the successor at even positions."""
    following = fetch(rasp.indices, rasp.indices, is_next_or_last, "following_token")
    previous = fetch(rasp.indices, rasp.indices, is_previous_or_first, "previous_token")
    choice = rasp.SequenceMap(
        lambda index, following: (index % 2, following), rasp.indices, following
    ).named("parity_and_following")
    return rasp.SequenceMap(
        lambda choice, previous: choice[1] if choice[0] == 0 else previous, choice, previous
    ).named("swapped")


def make_has_duplicate_repeated_positions() -> rasp.SOp:
    """Count positions whose token occurs more than once."""
    histogram = make_histogram_selector_width()
    repeated = count(histogram, histogram, lambda key, query: key > 1, "repeated_positions")
    return rasp.Map(lambda total: int(total > 0), repeated).named("has_duplicate")


def make_has_duplicate_distinct_count() -> rasp.SOp:
    """Count first occurrences; fewer than the length means a duplicate."""
    return rasp.Map(
        lambda distinct: int(distinct < LENGTH), make_distinct_count_first_occurrences()
    ).named("has_duplicate")


def make_distinct_count_first_occurrences() -> rasp.SOp:
    """Count positions with no earlier copy of their token."""
    pair = token_index_pair()
    earlier = count(pair, pair, same_token_earlier, "earlier_copies")
    return count(earlier, earlier, lambda key, query: key == 0, "distinct_count")


def make_distinct_count_last_occurrences() -> rasp.SOp:
    """Count positions with no later copy of their token."""
    pair = token_index_pair()
    later = count(pair, pair, same_token_later, "later_copies")
    return count(later, later, lambda key, query: key == 0, "distinct_count")


def make_absolute_direct() -> rasp.SOp:
    """One lookup from token to magnitude."""
    return rasp.Map(abs, rasp.tokens).named("absolute")


def make_absolute_rectified_parts() -> rasp.SOp:
    """Positive part and negative part, then the larger."""
    positive = rasp.Map(lambda token: max(token, 0), rasp.tokens).named("positive_part")
    negative = rasp.Map(lambda token: max(-token, 0), rasp.tokens).named("negative_part")
    return rasp.SequenceMap(max, positive, negative).named("absolute")


def make_mod_four_direct() -> rasp.SOp:
    """One lookup from token to residue."""
    return rasp.Map(lambda token: token % 4, rasp.tokens).named("mod_four")


def make_mod_four_two_bits() -> rasp.SOp:
    """Low bit and high bit separately, then recombine."""
    low = rasp.Map(lambda token: token % 2, rasp.tokens).named("low_bit")
    high = rasp.Map(lambda token: (token // 2) % 2, rasp.tokens).named("high_bit")
    return rasp.SequenceMap(lambda low, high: low + 2 * high, low, high).named("mod_four")


def make_sign_direct() -> rasp.SOp:
    """One lookup from token to sign."""
    return rasp.Map(lambda token: (token > 0) - (token < 0), rasp.tokens).named("sign")


def make_sign_two_flags() -> rasp.SOp:
    """Positive flag minus negative flag."""
    positive = rasp.Map(lambda token: int(token > 0), rasp.tokens).named("is_positive")
    negative = rasp.Map(lambda token: int(token < 0), rasp.tokens).named("is_negative")
    return rasp.SequenceMap(
        lambda positive, negative: positive - negative, positive, negative
    ).named("sign")


def make_clamp_direct() -> rasp.SOp:
    """One lookup that clamps the token to zero through three."""
    return rasp.Map(lambda token: min(max(token, 0), 3), rasp.tokens).named("clamped")


def make_clamp_floor_then_cap() -> rasp.SOp:
    """Floor at zero, then cap at three, as two lookups."""
    floored = rasp.Map(lambda token: max(token, 0), rasp.tokens).named("floored")
    # Tracr folds a Map of a Map into one lookup unless told otherwise.
    return rasp.Map(lambda token: min(token, 3), floored, simplify=False).named("clamped")


def make_increment_by_index_direct() -> rasp.SOp:
    """One lookup over token and index."""
    return rasp.SequenceMap(
        lambda token, index: token + index, rasp.tokens, rasp.indices
    ).named("increment_by_index")


def make_increment_by_index_partial_sums() -> rasp.SOp:
    """Add at most two first, then the rest of the index."""
    partial = rasp.SequenceMap(
        lambda token, index: token + min(index, 2), rasp.tokens, rasp.indices
    ).named("partial_sum")
    return rasp.SequenceMap(
        lambda partial, index: partial + max(index - 2, 0), partial, rasp.indices
    ).named("increment_by_index")


def make_parity_sum_direct() -> rasp.SOp:
    """One lookup for the parity of token plus index."""
    return rasp.SequenceMap(
        lambda token, index: (token + index) % 2, rasp.tokens, rasp.indices
    ).named("parity_sum")


def make_parity_sum_xor_of_bits() -> rasp.SOp:
    """Token parity and index parity separately, then exclusive or."""
    token_parity = rasp.Map(lambda token: token % 2, rasp.tokens).named("token_parity")
    index_parity = rasp.Map(lambda index: index % 2, rasp.indices).named("index_parity")
    return rasp.SequenceMap(
        lambda token_bit, index_bit: token_bit ^ index_bit, token_parity, index_parity
    ).named("parity_sum")


def make_exceeds_index_direct() -> rasp.SOp:
    """One lookup comparing token with index."""
    return rasp.SequenceMap(
        lambda token, index: int(token > index), rasp.tokens, rasp.indices
    ).named("exceeds_index")


def make_exceeds_index_rectified_token() -> rasp.SOp:
    """Floor the token at zero first; indices are never negative."""
    floored = rasp.Map(lambda token: max(token, 0), rasp.tokens).named("floored")
    return rasp.SequenceMap(
        lambda floored, index: int(floored > index), floored, rasp.indices
    ).named("exceeds_index")


def make_index_distance_direct() -> rasp.SOp:
    """One lookup for the distance between token and index."""
    return rasp.SequenceMap(
        lambda token, index: abs(token - index), rasp.tokens, rasp.indices
    ).named("index_distance")


def make_index_distance_signed_gap() -> rasp.SOp:
    """Signed gap first, then its magnitude."""
    gap = rasp.SequenceMap(
        lambda token, index: token - index, rasp.tokens, rasp.indices
    ).named("signed_gap")
    return rasp.Map(abs, gap).named("index_distance")


CANDIDATES = (
    TwinPair("sort", "Ascending sort, including duplicate tokens.",
             "tiebroken-key", "rank-parts",
             make_sort_tiebroken_key, make_sort_rank_parts, pilot=True),
    TwinPair("reverse", "Tokens in reverse order.",
             "mirrored-index", "length-minus-index",
             make_reverse_mirrored_index, make_reverse_length_minus_index, pilot=True),
    TwinPair("palindrome", "One everywhere iff the sequence is a palindrome.",
             "reverse-and-compare", "mismatch-count",
             make_palindrome_reverse_and_compare, make_palindrome_mismatch_count, pilot=True),
    TwinPair("histogram", "Occurrences of each token in the sequence.",
             "selector-width", "aggregate-count",
             make_histogram_selector_width, make_histogram_aggregate_count, pilot=True),
    TwinPair("count-greater-than", "Number of tokens strictly greater than the current token.",
             "strict-selector", "at-least-minus-equal",
             make_count_greater_strict, make_count_greater_at_least_minus_equal, pilot=True),
    TwinPair("first-element", "First token repeated at every position.",
             "position-selector", "flag-selector",
             make_first_element_position_selector, make_first_element_flag_selector),
    TwinPair("last-element", "Final token repeated at every position.",
             "fixed-index", "first-of-reversed",
             make_last_element_fixed_index, make_last_element_first_of_reversed),
    TwinPair("rotate-left", "Cyclic left rotation by one position.",
             "cyclic-selector", "clamp-and-repair",
             make_rotate_left_cyclic_selector, make_rotate_left_clamp_and_repair),
    TwinPair("rotate-two", "Cyclic left rotation by two positions.",
             "single-hop", "double-hop",
             make_rotate_two_single_hop, make_rotate_two_double_hop),
    TwinPair("previous-token", "Previous token; the first position repeats itself.",
             "position-selector", "reverse-shift-reverse",
             make_previous_token_position_selector, make_previous_token_reverse_shift_reverse),
    TwinPair("token-repeated", "One iff the token occurs more than once.",
             "histogram-threshold", "earlier-or-later-copy",
             make_repeated_histogram_threshold, make_repeated_earlier_or_later_copy),
    TwinPair("equals-first", "One iff the token equals the first token.",
             "fetch-then-compare", "match-selector",
             make_equals_first_fetch_then_compare, make_equals_first_match_selector),
    TwinPair("check-nondecreasing", "One everywhere iff the sequence is nondecreasing.",
             "adjacent-comparison", "inversion-count",
             make_nondecreasing_adjacent_comparison, make_nondecreasing_inversion_count),
    TwinPair("sum-with-next", "Token plus successor; the final token adds itself.",
             "clamped-selector", "wrap-then-repair",
             make_sum_with_next_clamped_selector, make_sum_with_next_wrap_then_repair),
    TwinPair("maximum", "Largest token repeated at every position.",
             "no-greater-token", "last-of-sorted",
             make_maximum_no_greater_token, make_maximum_last_of_sorted),
    TwinPair("minimum", "Smallest token repeated at every position.",
             "no-smaller-token", "first-in-tiebroken-order",
             make_minimum_no_smaller_token, make_minimum_first_in_tiebroken_order),
    TwinPair("swap-adjacent", "Swap positions pairwise: 0 with 1, 2 with 3, 4 with 5.",
             "xor-selector", "fetch-both-then-choose",
             make_swap_adjacent_xor_selector, make_swap_adjacent_fetch_both_then_choose),
    TwinPair("has-duplicate", "One everywhere iff any token occurs more than once.",
             "repeated-positions", "distinct-count",
             make_has_duplicate_repeated_positions, make_has_duplicate_distinct_count),
    TwinPair("distinct-count", "Number of distinct tokens in the sequence.",
             "first-occurrences", "last-occurrences",
             make_distinct_count_first_occurrences, make_distinct_count_last_occurrences),
    TwinPair("absolute", "Absolute value of each token.",
             "direct", "rectified-parts",
             make_absolute_direct, make_absolute_rectified_parts),
    TwinPair("mod-four", "Token modulo four.",
             "direct", "two-bits",
             make_mod_four_direct, make_mod_four_two_bits),
    TwinPair("sign", "Sign of each token: minus one, zero or one.",
             "direct", "two-flags",
             make_sign_direct, make_sign_two_flags),
    TwinPair("clamp", "Token clamped to the range zero through three.",
             "direct", "floor-then-cap",
             make_clamp_direct, make_clamp_floor_then_cap),
    TwinPair("increment-by-index", "Token plus its zero-based position.",
             "direct", "partial-sums",
             make_increment_by_index_direct, make_increment_by_index_partial_sums),
    TwinPair("parity-sum", "Parity of token plus position.",
             "direct", "xor-of-bits",
             make_parity_sum_direct, make_parity_sum_xor_of_bits),
    TwinPair("exceeds-index", "One iff the token is greater than its position.",
             "direct", "rectified-token",
             make_exceeds_index_direct, make_exceeds_index_rectified_token),
    TwinPair("index-distance", "Absolute difference between token and position.",
             "direct", "signed-gap",
             make_index_distance_direct, make_index_distance_signed_gap),
)

DROPPED: dict[str, str] = {
    "distinct-count": (
        "identical schedule signature: both members are 3 layers writing "
        "(stage 1, 48 lanes), (stage 3, 7 lanes), (stage 5, 7 lanes); "
        "the selectors differ only in the direction of one comparison"
    ),
}


def candidate_pairs() -> list[TwinPair]:
    """Every pair that was written, including the dropped ones."""
    return list(CANDIDATES)


def twin_pairs() -> list[TwinPair]:
    """Every pair that passes ``calib.equivalence.check_pair``."""
    return [pair for pair in CANDIDATES if pair.name not in DROPPED]

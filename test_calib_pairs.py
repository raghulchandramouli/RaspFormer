"""Twin-pair library: membership rules and the exhaustive equivalence check."""

import unittest

import numpy as np
from tracr.rasp import rasp

from calib.equivalence import check_pair
from calib.pairs import DROPPED, PAIR_CONFIG, TwinPair, candidate_pairs, twin_pairs

PILOT_NAMES = {"sort", "reverse", "palindrome", "histogram", "count-greater-than"}


def expressions(program):
    """Every distinct expression object reachable from the program."""
    seen, stack = {}, [program]
    while stack:
        expression = stack.pop()
        if id(expression) not in seen:
            seen[id(expression)] = expression
            stack.extend(expression.children)
    return list(seen.values())


def relabelled_pair():
    """Both members route through a shifted copy of the token: a relabelling, not an algorithm."""
    def shifted_by(offset):
        def make():
            shifted = rasp.Map(lambda token: (token + offset) % 8, rasp.tokens).named("shifted")
            return rasp.Map(
                lambda value: (value - offset) % 4, shifted, simplify=False
            ).named("mod_four")
        return make

    return TwinPair("relabelled", "Token modulo four.", "shift-two", "shift-three",
                    shifted_by(2), shifted_by(3))


def positive_share_scaled():
    """Mean of a flag times six: Tracr's inferred value set lacks 2/3 and 5/6, so it misrounds."""
    positive = rasp.numerical(
        rasp.Map(lambda token: int(token > 0), rasp.tokens)
    ).named("is_positive")
    everything = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE).named("everything")
    share = rasp.numerical(rasp.Aggregate(everything, positive, default=0)).named("positive_share")
    return rasp.Map(lambda share: round(share * 6), share).named("positive_count")


def positive_selector_width():
    positive = rasp.Map(lambda token: int(token > 0), rasp.tokens).named("is_positive")
    selector = rasp.Select(positive, rasp.indices, lambda key, query: key == 1).named("positives")
    return rasp.SelectorWidth(selector).named("positive_count")


class TwinPairTests(unittest.TestCase):
    def test_library_has_at_least_twenty_pairs_with_unique_names(self):
        names = [pair.name for pair in candidate_pairs()]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(twin_pairs()), 20)
        for name in names:
            self.assertRegex(name, r"^[a-z0-9]+(-[a-z0-9]+)*$")

    def test_pilot_pairs_are_the_five_protocol_candidates(self):
        self.assertEqual({pair.name for pair in twin_pairs() if pair.pilot}, PILOT_NAMES)

    def test_dropped_pairs_are_logged_candidates_outside_the_library(self):
        self.assertLessEqual(set(DROPPED), {pair.name for pair in candidate_pairs()})
        self.assertFalse(set(DROPPED) & {pair.name for pair in twin_pairs()})
        self.assertTrue(all(DROPPED.values()))

    def test_members_agree_under_the_rasp_evaluator(self):
        rng = np.random.default_rng(0)
        sequences = rng.choice(PAIR_CONFIG.vocab, size=(50, PAIR_CONFIG.max_seq_len)).tolist()
        evaluator = rasp.DefaultRASPEvaluator()
        for pair in candidate_pairs():
            a, b = pair.make_a(), pair.make_b()
            for tokens in sequences:
                with self.subTest(pair=pair.name, tokens=tokens):
                    expected = evaluator.evaluate(a, tokens)
                    self.assertNotIn(None, expected)
                    self.assertEqual(expected, evaluator.evaluate(b, tokens))

    def test_names_are_unique_within_a_program_and_stable_across_builds(self):
        for pair in candidate_pairs():
            for make in (pair.make_a, pair.make_b):
                with self.subTest(pair=pair.name, member=make.__name__):
                    names = [expression.name for expression in expressions(make())]
                    self.assertEqual(len(names), len(set(names)))
                    rebuilt = [expression.name for expression in expressions(make())]
                    self.assertEqual(sorted(names), sorted(rebuilt))

    def test_cheap_pair_passes_the_exhaustive_check(self):
        pair = next(pair for pair in twin_pairs() if pair.name == "absolute")
        result = check_pair(pair)
        self.assertTrue(result["valid"])
        self.assertEqual(result["inputs_checked"], 262_144)
        self.assertEqual(result["mismatches"], 0)
        self.assertEqual(result["rasp_agreement_a"], 1.0)
        self.assertEqual(result["rasp_agreement_b"], 1.0)

    def test_relabelled_pair_is_reported_with_identical_schedules(self):
        result = check_pair(relabelled_pair())
        self.assertTrue(result["compiled_outputs_equal"])
        self.assertFalse(result["schedules_differ"])
        self.assertFalse(result["valid"])

    def test_different_functions_are_counted_as_mismatches(self):
        positive_part = lambda: rasp.Map(lambda token: max(token, 0), rasp.tokens).named("absolute")
        absolute = lambda: rasp.Map(abs, rasp.tokens).named("absolute")
        result = check_pair(TwinPair("wrong", "Not one function.", "absolute", "positive-part",
                                     absolute, positive_part))
        self.assertEqual(result["mismatches"], 8**6 - 6**6)
        self.assertFalse(result["compiled_outputs_equal"])
        self.assertFalse(result["valid"])

    def test_compiled_model_that_departs_from_rasp_is_invalid(self):
        result = check_pair(TwinPair("diverges", "Number of positive tokens.", "scaled-share",
                                     "selector-width", positive_share_scaled,
                                     positive_selector_width))
        self.assertLess(result["rasp_agreement_a"], 1.0)
        self.assertEqual(result["rasp_agreement_b"], 1.0)
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()

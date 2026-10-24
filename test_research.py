"""Behavioral controls for blind recovery and sequence separation."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from raspformer.compiler import CompilerConfig
from raspformer.geometry import ProbeConfig
from raspformer.research import (family_program, infer_writes, recovery_scores,
                                split_sequences)
from tracr.rasp import rasp


class ResearchTests(unittest.TestCase):
    def test_blind_detector_sees_write_and_ignores_unchanged_stage(self):
        rng = np.random.default_rng(9)
        first = np.column_stack([rng.normal(size=80), np.zeros(80)])
        second = first.copy()
        second[:, 1] = rng.normal(size=80)
        rows = infer_writes([first, first.copy(), second, second.copy()])
        for detector in ("change", "pr", "cka"):
            self.assertEqual([r[f"predict_{detector}"] for r in rows], [False, True, False])
        self.assertGreater(rows[1]["delta_energy"], 0)

    def test_recovery_interface_has_no_truth_dependency(self):
        self.assertEqual(list(inspect.signature(infer_writes).parameters), ["matrices", "variance_floor"])
        with patch("raspformer.research.extract_schedule", side_effect=AssertionError("truth accessed")), patch("raspformer.research.trace_craft_graph", side_effect=AssertionError("truth accessed")):
            self.assertFalse(infer_writes([np.ones((4, 2)), np.ones((4, 2))])[0]["predict_change"])

    def test_inactive_geometry_does_not_create_cka_event(self):
        rows = infer_writes([np.ones((10, 3)), np.ones((10, 3)) + 1e-9])
        self.assertIsNone(rows[0]["adjacent_cka"])
        self.assertFalse(rows[0]["predict_cka"])

    def test_variance_floor_changes_detector_activity(self):
        first = np.column_stack([np.arange(10), np.zeros(10)]) * 1e-4
        second = first.copy()
        second[:, 1] = np.arange(10)[::-1] * 1e-4
        active = infer_writes([first, second], variance_floor=1e-12)[0]
        inactive = infer_writes([first, second], variance_floor=1)[0]
        self.assertIsNotNone(active["adjacent_cka"])
        self.assertIsNone(inactive["adjacent_cka"])
        self.assertGreater(active["delta_pr"], 0)
        self.assertEqual(inactive["delta_pr"], 0)

    def test_whole_sequences_are_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            a = split_sequences(CompilerConfig(), ProbeConfig(), Path(directory))
            b = split_sequences(CompilerConfig(), ProbeConfig(), Path(directory))
            self.assertEqual(len(a.calibration), 96)
            self.assertEqual(len(a.test), 96)
            self.assertFalse(set(map(tuple, a.calibration)) & set(map(tuple, a.test)))
            np.testing.assert_array_equal(a.test_ids, b.test_ids)

    def test_perfect_recovery_and_null_are_distinct(self):
        predictions = [{f"predict_{kind}": i == 2 for kind in ("change", "pr", "cka")} for i in range(8)]
        for result in recovery_scores(predictions, np.arange(8) == 2):
            self.assertEqual(result["f1"], 1)
            self.assertLess(result["null_mean_f1"], 0.25)
            self.assertAlmostEqual(result["all_positive_f1"], 2 / 9)

    def test_family_output_uses_bounded_support(self):
        evaluator = rasp.DefaultRASPEvaluator()
        for depth in (1, 2, 3):
            for branches in (1, 2, 3):
                values = evaluator.evaluate(family_program(depth, branches, 8), [-2, -1, 0, 1, 4, 5])
                self.assertTrue(all(0 <= value < 8 for value in values))


if __name__ == "__main__":
    unittest.main()

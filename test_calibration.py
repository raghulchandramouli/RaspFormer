"""Invariances of the calibration metrics and correctness of the compiler ground truth."""

import unittest

import numpy as np

from raspformer.calibration import (METRICS, RELATIONAL_METRICS, SPECTRUM_METRICS, dissimilarity,
                                    model_distance, spectrum_metrics, stage_ground_truth, stage_metrics)
from raspformer.compiler import CompilerConfig, compile_program
from raspformer.programs import build_programs
from raspformer.research import family_program


def one_hot(classes, repeats):
    return np.repeat(np.eye(classes), repeats, axis=0)


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.left = rng.normal(size=(200, 6)) * np.geomspace(1, 0.05, 6)
        self.right = self.left @ rng.normal(size=(6, 6)) + 0.3 * rng.normal(size=(200, 6))
        self.rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0]

    def test_participation_ratio_counts_values_of_a_balanced_one_hot_variable(self):
        for classes in (2, 5, 12):
            self.assertAlmostEqual(spectrum_metrics(one_hot(classes, 20))["participation_ratio"], classes - 1)

    def test_effective_rank_of_equal_directions_is_their_count(self):
        metrics = spectrum_metrics(one_hot(5, 20))
        self.assertAlmostEqual(metrics["effective_rank"], 4)
        self.assertAlmostEqual(metrics["explained_variance_top1"], 0.25)
        self.assertAlmostEqual(metrics["explained_variance_top5"], 1)

    def test_spectrum_metrics_ignore_rotation_and_scale(self):
        reference = spectrum_metrics(self.left)
        for transformed in (self.left @ self.rotation, self.left * 1e3, self.left * 1e-3):
            for name, value in spectrum_metrics(transformed).items():
                self.assertAlmostEqual(value, reference[name], places=8, msg=name)

    def test_relational_metrics_ignore_rotation_and_scale_of_either_side(self):
        for metric in RELATIONAL_METRICS:
            reference = dissimilarity(self.left, self.right, metric)
            self.assertGreater(reference, 0.01, metric)
            self.assertAlmostEqual(dissimilarity(self.left @ self.rotation, self.right * 50, metric), reference, places=8, msg=metric)
            self.assertAlmostEqual(dissimilarity(self.left, self.left @ self.rotation * 0.2, metric), 0, places=6, msg=metric)

    def test_scale_invariance_stops_at_the_absolute_variance_floor(self):
        shrunk = self.left * 1e-8
        self.assertEqual(spectrum_metrics(shrunk)["participation_ratio"], 0)
        self.assertTrue(np.isnan(dissimilarity(shrunk, self.right, "cka")))
        self.assertGreater(spectrum_metrics(shrunk, variance_floor=0)["participation_ratio"], 1)

    def test_relational_metrics_compare_representations_of_different_width(self):
        narrow = self.left[:, :3]
        for metric in RELATIONAL_METRICS:
            self.assertTrue(0 < dissimilarity(narrow, self.right, metric) < 2, metric)

    def test_stage_rows_report_change_from_the_previous_stage(self):
        rows = stage_metrics([self.left, self.left @ self.rotation, self.right])
        self.assertTrue(all(np.isnan(rows[0][metric]) for metric in RELATIONAL_METRICS))
        for metric in RELATIONAL_METRICS:
            self.assertAlmostEqual(rows[1][metric], 0, places=6)
            self.assertGreater(rows[2][metric], 0.01)

    def test_model_distance_is_zero_for_rotated_copies_and_positive_otherwise(self):
        model = [self.left, self.right, self.left + self.right]
        rotated = [stage @ self.rotation for stage in model]
        other = [self.left, self.left, self.right[::-1]]
        for metric in METRICS:
            self.assertAlmostEqual(model_distance(model, rotated, metric), 0, places=6, msg=metric)
            self.assertGreater(model_distance(model, other, metric), 0, metric)
        self.assertEqual(set(METRICS), set(SPECTRUM_METRICS) | set(RELATIONAL_METRICS))


class GroundTruthTests(unittest.TestCase):
    def test_family_program_stages_follow_its_tree(self):
        config = CompilerConfig()
        program = family_program(3, 5, 8)
        rows = stage_ground_truth(program, compile_program(program, config), config)
        self.assertEqual([row["live_variables"] for row in rows], [0, 2, 4, 4, 5])
        self.assertEqual([row["summed_cardinality"] for row in rows], [0, 14, 28, 28, 35])
        self.assertEqual([row["dependency_depth"] for row in rows], [0, 1, 2, 2, 3])
        self.assertEqual([row["first_write"] for row in rows], [False, True, True, False, True])

    def test_single_map_program_writes_once(self):
        config = CompilerConfig()
        program = build_programs()["P02_index_parity"]
        rows = stage_ground_truth(program, compile_program(program, config), config)
        self.assertEqual([row["live_variables"] for row in rows], [0, 0, 1])
        self.assertEqual(rows[-1]["summed_cardinality"], 1)

    def test_padding_repeats_the_final_stage_without_a_write(self):
        config = CompilerConfig()
        program = build_programs()["P02_index_parity"]
        rows = stage_ground_truth(program, compile_program(program, config), config, stages=5)
        self.assertEqual([row["live_variables"] for row in rows], [0, 0, 1, 1, 1])
        self.assertEqual([row["first_write"] for row in rows], [False, False, True, False, False])
        with self.assertRaises(ValueError):
            stage_ground_truth(program, compile_program(program, config), config, stages=2)


if __name__ == "__main__":
    unittest.main()

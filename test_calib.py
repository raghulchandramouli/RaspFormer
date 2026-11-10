"""Statistics, program library and training pool of the calibration study."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from tracr.rasp import rasp

from bench import siit
from calib import analysis, library, pool
from raspformer import models
from raspformer.programs import build_programs
from raspformer.calibration import GROUND_TRUTH, METRICS
from raspformer.compiler import compile_program, extract_schedule


def stage_frame(seed=0):
    """Eight programs whose cardinality per variable differs, so it is separable from the variable count."""
    rng = np.random.default_rng(seed)
    rows = []
    for program in range(8):
        for stage in range(6):
            live = min(stage, 2 + program % 3)
            truth = dict(live_variables=live, summed_cardinality=live * (3 + 2 * (program % 4)),
                         dependency_depth=min(stage, 3), first_write=stage in (1, 2))
            row = dict(spec=f"p{program}", model_class="siit", seed=0, stage=stage, **truth,
                       **{metric: rng.normal() for metric in METRICS})
            row["participation_ratio"] = truth["summed_cardinality"]
            row["effective_rank"] = live + 0.01 * rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def pair_frame(ratio):
    return pd.DataFrame([dict(pairing=f"pair{index}", kind="twin", model_class="siit", metric=metric,
                              models_a=3, models_b=3, between=ratio * (1 + 0.05 * (index % 5)), within=1.0,
                              ratio=ratio * (1 + 0.05 * (index % 5)), ratio_cross_seed=ratio, identified=1.0, identification_trials=6)
                         for index in range(20) for metric in METRICS])


class StatisticsTests(unittest.TestCase):
    def test_auroc_ignores_direction_and_is_half_for_identical_samples(self):
        low, high = np.arange(10), np.arange(10) + 100
        self.assertEqual(analysis.auroc(low, high), 1)
        self.assertEqual(analysis.auroc(high, low), 1)
        self.assertEqual(analysis.auroc(low, low), 0.5)

    def test_holm_scales_the_smallest_p_by_the_family_size(self):
        adjusted = analysis.holm([0.01, 0.04, 0.03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_sign_flip_needs_a_consistent_direction(self):
        rng = np.random.default_rng(0)
        self.assertLess(analysis.sign_flip_p(np.full(20, 0.7), rng), 0.001)
        self.assertGreater(analysis.sign_flip_p(np.tile([0.7, -0.7], 10), rng), 0.4)
        self.assertGreater(analysis.sign_flip_p(np.full(5, 0.7), rng), 0.02)

    def test_a_metric_that_only_counts_values_does_not_pass_l2(self):
        table = analysis.level_two(stage_frame(), classes=("siit",))
        counts_values = table[table.metric == "participation_ratio"]
        self.assertTrue(counts_values.tracks_cardinality.any())
        self.assertFalse(counts_values.passed.any())

    def test_a_metric_that_tracks_live_variables_beyond_cardinality_passes_l2(self):
        table = analysis.level_two(stage_frame(), classes=("siit",))
        row = table[(table.metric == "effective_rank") & (table.quantity == "live_variables")].iloc[0]
        self.assertGreater(row.rho, 0.9)
        self.assertTrue(row.passed)
        self.assertFalse(table[table.metric == "cka"].passed.any())

    def test_l3_passes_only_when_algorithms_are_further_apart_than_seeds(self):
        self.assertTrue(analysis.level_three(pair_frame(3.0)).passed.all())
        self.assertFalse(analysis.level_three(pair_frame(1.2)).passed.any())
        self.assertFalse(analysis.level_three(pair_frame(0.5)).passed.any())

    def test_verdict_follows_the_three_levels(self):
        def tables(above, calibrated, ratio):
            l1 = pd.DataFrame([dict(model_class="siit", metric=metric, passed=above) for metric in METRICS])
            l2 = pd.DataFrame([dict(model_class="siit", metric=metric, passed=calibrated, tracks_cardinality=True) for metric in METRICS])
            return analysis.verdicts(l1, l2, analysis.level_three(pair_frame(ratio)), classes=("siit",))
        self.assertEqual(set(tables(False, True, 3.0).verdict), {"describes noise"})
        self.assertEqual(set(tables(True, True, 3.0).verdict), {"tracks the algorithm"})
        self.assertEqual(set(tables(True, True, 1.0).verdict), {"tracks representational structure, not the algorithm"})
        self.assertEqual(set(tables(True, False, 1.0).verdict), {"above null, not calibrated"})

    def test_l3_is_not_gradable_when_no_pair_has_a_ratio(self):
        pairs = pair_frame(3.0).assign(ratio=np.nan, ratio_cross_seed=np.nan)
        l1 = pd.DataFrame([dict(model_class="siit", metric=metric, passed=True) for metric in METRICS])
        l2 = pd.DataFrame([dict(model_class="siit", metric=metric, passed=True, tracks_cardinality=False) for metric in METRICS])
        table = analysis.verdicts(l1, l2, analysis.level_three(pairs), classes=("siit",))
        self.assertTrue(table.L3.isna().all())
        self.assertEqual(set(table.verdict), {"above null and calibrated; algorithm-specificity not gradable"})

    def test_hardened_threshold_is_learned_on_calibration_programs_only(self):
        truth = np.array([True, False, True, False])
        clean = pd.DataFrame(dict(change=[1.0, 1e-4, 1.0, 1e-4], pr=[1.0, 1e-4, 1.0, 1e-4], cka=[1.0, 1e-4, 1.0, 1e-4]))
        busy = pd.DataFrame(dict(change=[1.0, 0.5, 1.0, 0.5], pr=[1.0, 0.5, 1.0, 0.5], cka=[1.0, 0.5, 1.0, 0.5]))
        table = analysis.harden_detectors({"c": ("calibration", clean, truth)},
                                          {"clean": ("held-out clean", clean, truth), "busy": ("held-out busy", busy, truth)})
        scores = table.set_index(["detector", "pool"]).f1
        self.assertEqual(scores["cka", "held-out clean"], 1)
        self.assertAlmostEqual(scores["cka", "held-out busy"], 2 / 3)
        self.assertFalse(table[table["pool"] == "held-out busy"].beats_always_write.any())


class LibraryTests(unittest.TestCase):
    def test_grid_crosses_depth_variables_and_vocabulary(self):
        specs = library.grid_specs()
        self.assertEqual(len(specs), 27)
        self.assertEqual(len({spec.name for spec in specs}), 27)
        design = np.array([[spec.depth, spec.variables, spec.vocabulary] for spec in specs], dtype=float)
        np.testing.assert_allclose(np.corrcoef(design.T), np.eye(3), atol=1e-12)

    def test_grid_program_compiles_to_its_declared_shape(self):
        spec = library.spec("grid-d4-v6-k12")
        program = spec.make()
        assembled = compile_program(program, spec.config)
        summary, _, _ = extract_schedule(program, assembled, spec.config)
        self.assertEqual((summary["max_dependency_depth"], summary["computed_variable_count"]), (4, 6))
        self.assertEqual(assembled.model_config.num_layers, spec.layers)

    def test_control_pair_shares_structure_and_differs_in_function(self):
        pairing = library.control_pairings()[0]
        self.assertEqual((pairing.a.depth, pairing.a.variables, pairing.a.layers),
                         (pairing.b.depth, pairing.b.variables, pairing.b.layers))
        tokens = [0, 1, 2, 3, 4, 5]
        evaluator = rasp.DefaultRASPEvaluator()
        self.assertNotEqual(evaluator.evaluate(pairing.a.make(), tokens), evaluator.evaluate(pairing.b.make(), tokens))

    def test_twin_members_share_one_architecture(self):
        for pairing in library.twin_pairings(pilot_only=True):
            self.assertEqual(pairing.a.layers, pairing.b.layers)
            self.assertEqual(pairing.a.config, pairing.b.config)

    def test_training_sequences_never_contain_a_probe_sequence(self):
        config = library.grid_config(4)
        probes = {tuple(row) for row in library.probe_tokens(config).tolist()}
        training = library.training_tokens(config, 0, 2000)
        self.assertEqual(len(training), 2000)
        self.assertFalse(probes & {tuple(row) for row in training.tolist()})
        np.testing.assert_array_equal(library.probe_tokens(config), library.probe_tokens(config))


class PoolTests(unittest.TestCase):
    def test_a_finished_task_is_not_retrained(self):
        with tempfile.TemporaryDirectory() as root:
            first = pool.train_task(root, "grid-d3-v5-k4", "output-only", 0, steps=3)
            checkpoint = pool.task_folder(root, "grid-d3-v5-k4", "output-only", 0) / "model.npz"
            written = checkpoint.stat().st_mtime_ns
            second = pool.train_task(root, "grid-d3-v5-k4", "output-only", 0, steps=3)
            self.assertEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
            self.assertEqual(checkpoint.stat().st_mtime_ns, written)
            stages = pool.measure("grid-d3-v5-k4", pool.load_model(root, "grid-d3-v5-k4", "output-only", 0))
            self.assertEqual(len(stages), 2 * library.spec("grid-d3-v5-k4").layers + 1)
            self.assertEqual(stages[0].shape, (library.PROBE_SEQUENCES * library.SEQUENCE_LENGTH, 32))
            self.assertEqual(analysis.included(root, "grid-d3-v5-k4", "siit", 0), (False, "not trained"))


class TrainerTests(unittest.TestCase):
    def test_extra_layers_train_outside_the_circuit(self):
        with tempfile.TemporaryDirectory() as root:
            metrics = siit.train_program("P02_index_parity", build_programs()["P02_index_parity"], root, steps=2,
                                         train_count=128, test_count=32, layers=2)
            model = models.load_model(Path(root) / "p02-index-parity" / "model.npz")
            self.assertEqual(model.config.layers, 2)
            self.assertEqual(list(metrics["iia"]), ["blocks.0.hook_mlp_out"])
            self.assertIn("blocks.1.hook_mlp_out", metrics["siia"])
            with self.assertRaises(ValueError):
                siit.train_program("P11_check_palindrome", build_programs()["P11_check_palindrome"], root, steps=1, layers=1)

    def test_zero_weights_train_on_the_output_loss_alone(self):
        spec = library.spec("grid-d3-v5-k4")
        _, _, hl_model = pool.compiled(spec.name)
        train = pool.encode(spec.name, library.training_tokens(spec.config, 0, 128))
        nodes = siit.active_nodes(hl_model, train)
        _, history = siit.train_siit(hl_model, pool.initial_model(spec.name, 0), train, nodes, steps=2, log_every=1,
                                     verbose=False, iit_weight=0.0, strict_weight=0.0)
        self.assertEqual({(row["iit"], row["strict"]) for row in history}, {(0.0, 0.0)})
        _, history = siit.train_siit(hl_model, pool.initial_model(spec.name, 0), train, nodes, steps=2, log_every=1, verbose=False)
        self.assertTrue(all(row["iit"] > 0 and row["strict"] > 0 for row in history))


if __name__ == "__main__":
    unittest.main()

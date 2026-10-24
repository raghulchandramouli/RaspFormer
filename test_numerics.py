"""Numerical policy, alignment coverage, and run provenance checks."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from raspformer.alignment import measure_alignment, summarize_alignment
from raspformer.compiler import CompilerConfig, environment_metadata, record_run
from raspformer.geometry import (
    ProbeConfig, all_pair_cka, build_balanced_probes, build_matched_probes,
    collect_activations, pca_profile,
)
from raspformer.programs import build_programs


class NumericalChecks(unittest.TestCase):
    def test_near_constant_noise_keeps_spectrum_without_claiming_dimension(self):
        noise = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * 1e-8
        values = np.ones_like(noise) + noise
        profile = pca_profile(values)
        self.assertGreater(profile["total_variance"], 0)
        self.assertLess(profile["total_variance"], 1e-12)
        np.testing.assert_allclose(profile["eigenvalues"], [2e-16 / 3] * 2, rtol=1e-7, atol=0)
        np.testing.assert_allclose(profile["explained_variance_ratio"], [.5, .5], rtol=1e-7)
        self.assertEqual(profile["participation_ratio"], 0)
        self.assertEqual((profile["pc95"], profile["pc99"]), (0, 0))
        self.assertFalse(profile["variance_active"])
        self.assertEqual(profile["variance_reason"], "at_or_below_variance_floor")
        self.assertEqual(profile["numerical_policy"]["variance_floor"], 1e-12)
        scores = all_pair_cka([values, 1e8 * noise])
        self.assertTrue(np.isnan(scores[0]).all())
        self.assertAlmostEqual(scores[1, 1], 1)
        self.assertAlmostEqual(pca_profile(values, variance_floor=0)["participation_ratio"], 2)
        np.testing.assert_allclose(all_pair_cka([values], variance_floor=0), [[1]])

    def test_variance_floor_is_validated_and_uses_strict_sample_variance(self):
        values = [[-1.0], [1.0]]
        for floor in (-1, np.inf, np.nan, True, "1e-12", None):
            for function, argument in ((pca_profile, values), (all_pair_cka, [values])):
                with self.subTest(function=function.__name__, floor=floor):
                    with self.assertRaisesRegex(ValueError, "variance_floor"):
                        function(argument, variance_floor=floor)
        profile = pca_profile(values, variance_floor=2.0)
        self.assertEqual(profile["total_variance"], 2)
        self.assertFalse(profile["variance_active"])
        self.assertEqual(profile["pc95"], 0)
        self.assertTrue(np.isnan(all_pair_cka([values], variance_floor=2.0)[0, 0]))
        self.assertEqual(pca_profile(values, variance_floor=1.99)["participation_ratio"], 1)
        np.testing.assert_allclose(all_pair_cka([values], variance_floor=1.99), [[1]])

    def test_rotation_preserves_usable_participation_ratio_and_cka(self):
        values = np.array([[2, 0], [-2, 0], [0, 1], [0, -1]], dtype=float)
        rotation = np.array([[.6, -.8], [.8, .6]])
        rotated = values @ rotation
        for matrix in (values, rotated, 3 * rotated + 10):
            profile = pca_profile(matrix)
            self.assertTrue(profile["variance_active"])
            self.assertEqual(profile["variance_reason"], "above_variance_floor")
            self.assertAlmostEqual(profile["participation_ratio"], 25 / 17)
            self.assertEqual((profile["pc95"], profile["pc99"]), (2, 2))
        np.testing.assert_allclose(all_pair_cka([values, rotated, 3 * rotated + 10]), 1, atol=1e-14)

    def test_alignment_records_actual_and_expected_variance(self):
        name = "A_prev_token_class"
        programs = {name: build_programs(include_examples=True)[name]}
        config = CompilerConfig(rel_tol=1e-4, abs_tol=1e-4)
        probes = build_balanced_probes(
            programs, config,
            ProbeConfig(length=3, samples_per_class=2, random_candidates=0, structured_candidates=0),
        )
        data = collect_activations(programs, config, probes)
        columns = data.columns[name]["computed"]
        data.stages[name]["Embedding"][:, columns] = np.arange(len(data.observations[name]))[:, None] * 1e-10
        stages = measure_alignment(name, programs[name], config, data)["stages"]
        computed = stages[stages.view == "computed"].set_index("stage")
        embedding = computed.loc["Embedding"]
        self.assertGreater(embedding.actual_total_variance, 0)
        self.assertLess(embedding.actual_total_variance, embedding.variance_floor)
        self.assertEqual(embedding.expected_total_variance, 0)
        self.assertFalse(embedding.actual_variance_active)
        self.assertFalse(embedding.expected_variance_active)
        self.assertEqual((embedding.actual_pr, embedding.expected_pr), (0, 0))
        self.assertTrue(np.isnan(computed.loc["L1/Attn", "adjacent_cka"]))
        written = computed.loc["L1/Attn"]
        self.assertTrue(written.actual_variance_active)
        self.assertTrue(written.expected_variance_active)
        np.testing.assert_allclose(written.actual_total_variance, written.expected_total_variance, rtol=1e-6)
        raw = measure_alignment(name, programs[name], config, data, variance_floor=0)["stages"]
        raw = raw[raw.view == "computed"].set_index("stage")
        self.assertAlmostEqual(raw.loc["Embedding", "actual_pr"], 1)
        self.assertTrue(raw.loc["Embedding", "actual_variance_active"])
        self.assertFalse(raw.loc["Embedding", "expected_variance_active"])
        self.assertTrue(np.isfinite(raw.loc["L1/Attn", "adjacent_cka"]))
        self.assertEqual(raw.loc["Embedding", "variance_floor"], 0)
        self.assertEqual(raw.loc["Embedding", "variance_definition"], "sum of feature sample variances, ddof=1")
        self.assertEqual(raw.loc["Embedding", "variance_active_if"], "total_variance > variance_floor")
        for row in stages.itertuples():
            actual = data.stages[name][row.stage][:, data.columns[name][row.view]]
            self.assertAlmostEqual(row.actual_total_variance, actual.var(axis=0, ddof=1).sum())

    def test_matched_sampling_preserves_target_identity_across_programs(self):
        all_programs = build_programs()
        programs = {name: all_programs[name] for name in ("P01_absolute", "P05_histogram")}
        probes = build_matched_probes(
            programs, CompilerConfig(),
            ProbeConfig(length=3, random_candidates=4, structured_candidates=2),
        )
        absolute, histogram = probes.observations.values()
        self.assertEqual(len(absolute), len(probes.candidates) * 3)
        pd.testing.assert_frame_equal(
            absolute.drop(columns="target_class"), histogram.drop(columns="target_class"),
        )
        self.assertEqual(absolute[["candidate_id", "position"]].duplicated().sum(), 0)
        for row in absolute.itertuples():
            self.assertEqual(row.target_class, abs(row.tokens[row.position]))
        for row in histogram.itertuples():
            self.assertEqual(row.target_class, row.tokens.count(row.tokens[row.position]))
        self.assertEqual(probes.audit.observations.sum(), 2 * len(absolute))

    def test_alignment_summary_preserves_missing_cka_and_reports_coverage(self):
        stages = pd.DataFrame([
            {
                "program": name, "view": "computed", "step": step,
                "observations": 4, "computed_variable_count": 2,
                "max_dependency_depth": 1, "transformer_layers": 2,
                "pr_gap": gap, "adjacent_cka": actual,
                "expected_adjacent_cka": expected,
            }
            for name in ("partly_defined", "inactive")
            for step, (gap, actual, expected) in enumerate([
                (0, np.nan, np.nan), (.3, .9, .8), (-.1, np.nan, .7), (.2, .4, np.nan),
            ])
        ])
        stages.loc[stages.program == "inactive", ["adjacent_cka", "expected_adjacent_cka"]] = np.nan
        lanes = pd.DataFrame([
            {"program": "partly_defined", "is_first_write": True, "fraction_close": 1., "max_abs_error": 0.},
            {"program": "partly_defined", "is_first_write": True, "fraction_close": .5, "max_abs_error": .2},
        ])
        summary = summarize_alignment(stages, lanes).set_index("program")
        row = summary.loc["partly_defined"]
        self.assertAlmostEqual(row.cka_mae, .1)
        self.assertEqual((row.cka_observed_defined, row.cka_expected_defined, row.cka_paired_defined), (2, 2, 1))
        self.assertEqual(row.cka_definition_mismatches, 2)
        self.assertAlmostEqual(row.cka_paired_coverage, 1 / 3)
        self.assertAlmostEqual(row.pr_mae, .2)
        self.assertEqual(row.lane_agreement, .75)
        self.assertTrue(np.isnan(summary.loc["inactive", "cka_mae"]))
        self.assertEqual(summary.loc["inactive", "cka_paired_coverage"], 0)


class ProvenanceChecks(unittest.TestCase):
    def test_metadata_tracks_source_commit_and_ignores_notebook_outputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "raspformer"
            source_root.mkdir()
            compiler = source_root / "compiler.py"
            compiler.write_text("source = True\n")
            (root / "notebooks").mkdir()
            notebook_path = root / "notebooks" / "study.ipynb"
            notebook = {"cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": [], "execution_count": None}]}
            notebook_path.write_text(json.dumps(notebook))
            for arguments in (
                ["init", "--quiet"], ["add", "."],
                ["-c", "user.name=Numerical Test", "-c", "user.email=numerics@example.invalid", "commit", "--quiet", "-m", "source"],
            ):
                subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            with patch("raspformer.compiler.__file__", str(compiler)):
                clean = environment_metadata()
                self.assertEqual(clean["git_commit"], commit)
                self.assertFalse(clean["git_dirty"])
                notebook["cells"][0].update(outputs=[{"output_type": "stream", "text": ["output\n"]}], execution_count=5)
                notebook_path.write_text(json.dumps(notebook))
                executed = environment_metadata()
                self.assertTrue(executed["git_dirty"])
                self.assertEqual(clean["notebook_source_sha256"], executed["notebook_source_sha256"])
                notebook["cells"][0]["source"] = ["x = 2\n"]
                notebook_path.write_text(json.dumps(notebook))
                edited = environment_metadata()
                self.assertNotEqual(clean["notebook_source_sha256"], edited["notebook_source_sha256"])
                self.assertEqual(list(edited["notebook_source_sha256"]), ["notebooks/study.ipynb"])

    def test_run_directories_use_utc_and_preserve_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with record_run(root, {"example": None}, CompilerConfig()) as (run_dir, evidence):
                self.assertEqual(run_dir.parent, root.resolve() / "runs")
                self.assertRegex(run_dir.name, r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
                self.assertEqual(evidence["status"], "running")
            saved = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertIn("git_commit", saved["settings"])
            self.assertIn("notebooks/03_geometry.ipynb", saved["settings"]["notebook_source_sha256"])


if __name__ == "__main__":
    unittest.main()

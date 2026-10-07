"""Run with: python -m unittest -v test_studies (uses only temporary run directories)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from tracr.rasp import rasp

from raspformer.compiler import CompilerConfig, check_trace_sample, run_study, run_trace_study
from raspformer.geometry import ProbeConfig, all_pair_cka, pca_profile, run_geometry_study
from raspformer.programs import build_programs

RESULTS = Path(__file__).resolve().parent / "results"
SAMPLES = [[1, 2, 1], [-2, 0, 3, 1], [5], [0, 0, 0, 0], [5, 4, 3, 2, 1, 0]]


class StudyChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def test_program_meanings_and_boundaries(self):
        evaluator = rasp.DefaultRASPEvaluator()
        programs = build_programs(include_examples=True)
        for x in SAMPLES + [[-2, -1, 0, 1, 2, 5], [2, 1], [-2, -2, 5, 5]]:
            n = len(x)
            expected = [
                list(map(abs, x)), [i % 2 for i in range(n)],
                [v + i for i, v in enumerate(x)], [x[0]] * n,
                [x.count(v) for v in x], [sum(w > v for w in x) for v in x],
                [v + x[min(i + 1, n - 1)] for i, v in enumerate(x)],
                [v + x[i - 1] for i, v in enumerate(x)],
                [int(x == sorted(x))] * n, x[1:] + x[:1],
                [int(x == x[::-1])] * n, sorted(x),
                x[:1] + x[:-1], [int(x.count(v) > 1) for v in x],
            ]
            for (name, program), values in zip(programs.items(), expected, strict=True):
                with self.subTest(program=name, tokens=x):
                    self.assertEqual(list(evaluator.evaluate(program, x)), values)
        for sample in ([], [6], [True], [0] * 7):
            with self.assertRaises(ValueError):
                CompilerConfig().validate_samples([sample])

    def test_program_study_matches_reference(self):
        reference = json.loads((RESULTS / "programs/reference/step1_ground_truth_schedule.json").read_text())
        result = run_study(build_programs(), CompilerConfig(), SAMPLES, self.output)
        evidence = json.loads((result["run_dir"] / "manifest.json").read_text())
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["programs"], reference["programs"])
        self.assertEqual(evidence["checks"], reference["checks"])
        self.assertEqual(len(evidence["variables"]), len(reference["variables"]))
        self.assertTrue((result["run_dir"] / "schedule.md").is_file())
        # Expression IDs change between fresh builds; compare semantic lane/writer allocations.
        for new, old in zip(evidence["variables"], reference["variables"], strict=True):
            self.assertEqual(new["program"], old["program"])
            self.assertEqual(new["compiled_block"], old["compiled_block"])
            self.assertEqual(new["first_write_after"], old["first_write_after"])
            self.assertEqual(len(new["residual_lane_indices"]), len(old["residual_lane_indices"]))

    def test_trace_study_matches_reference(self):
        reference = json.loads((RESULTS / "trace/reference/step2_manual_trace.json").read_text())
        config = CompilerConfig(rel_tol=1e-4, abs_tol=1e-4)
        result = run_trace_study(build_programs(include_examples=True), config, SAMPLES, self.output)
        evidence = result["evidence"]
        saved = json.loads((result["run_dir"] / "traces.json").read_text())
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["residual_warnings"], [])
        self.assertEqual(len(saved), 70)
        self.assertNotIn("traces", json.loads((result["run_dir"] / "manifest.json").read_text()))
        for new, old in zip(saved, reference, strict=True):
            with self.subTest(program=new["program"], tokens=new["tokens"]):
                for field in ("program", "tokens", "rasp_output", "model_decoded"):
                    self.assertEqual(new[field], old[field])
                np.testing.assert_allclose(new["manual_scores"], old["manual_scores"], atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(new["model_scores"], old["model_scores"], atol=1e-4, rtol=1e-4)
                self.assertEqual(len(new["block_trace"]), len(old["block_trace"]))

    def test_geometry_matches_reference(self):
        reference = RESULTS / "geometry/reference"
        result = run_geometry_study(
            build_programs(include_examples=True),
            CompilerConfig(rel_tol=1e-4, abs_tol=1e-4), ProbeConfig(), self.output,
        )
        self.assertEqual(result["evidence"]["status"], "completed")
        self.assertEqual(len(result["probes"].candidates), 3383)
        for old_path in reference.glob("*.csv"):
            with self.subTest(artifact=old_path.name):
                pd.testing.assert_frame_equal(
                    pd.read_csv(result["run_dir"] / old_path.name),
                    pd.read_csv(old_path), check_exact=False, atol=1e-6, rtol=1e-6,
                )
        # Eigenvectors in degenerate eigenspaces are not unique; compare spectra.
        with np.load(reference / "pca_spectra.npz") as old, np.load(result["run_dir"] / "pca_spectra.npz") as new:
            self.assertEqual(set(old.files), set(new.files))
            for name in old.files:
                if name.endswith(("__eigenvalues", "__explained_variance_ratio")):
                    np.testing.assert_allclose(new[name], old[name], atol=1e-6, rtol=1e-6)

    def test_geometry_known_values(self):
        values = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        profile = pca_profile(values)
        self.assertAlmostEqual(profile["participation_ratio"], 2)
        self.assertEqual(profile["pc95"], 2)
        scores = all_pair_cka([values, 3 * values + 10, np.zeros_like(values)])
        np.testing.assert_allclose(scores[:2, :2], 1)
        self.assertTrue(np.isnan(scores[2]).all())
        self.assertEqual(pca_profile(np.ones((4, 2)))["participation_ratio"], 0)
        with self.assertRaises(ValueError):
            all_pair_cka([values, values[:2]])
        with self.assertRaises(ValueError):
            pca_profile([[np.nan], [0]])

    def test_failed_trace_preserves_completed_sample(self):
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("deliberate regression check")
            return check_trace_sample(*args)

        programs = {"P01_absolute": build_programs()["P01_absolute"]}
        with patch("raspformer.compiler.check_trace_sample", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                run_trace_study(programs, CompilerConfig(), SAMPLES, self.output)
        [manifest] = self.output.glob("*/manifest.json")
        evidence = json.loads(manifest.read_text())
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["failures"][0]["program"], "P01_absolute")
        self.assertEqual(len(json.loads((manifest.parent / "traces.json").read_text())), 1)
        # A subsequent successful run must not overwrite the failed run.
        result = run_study(programs, CompilerConfig(), SAMPLES, self.output)
        self.assertNotEqual(result["run_dir"], manifest.parent)
        self.assertEqual(json.loads(manifest.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()

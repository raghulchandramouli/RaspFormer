"""Behavioral checks for imported, transformed, intervened, and learned models."""

import tempfile
from pathlib import Path
import unittest

import jax
import numpy as np

from raspformer.compiler import CompilerConfig, compile_program
from raspformer.models import (change_basis, forward, from_compiled, load_model,
                              parameter_count, random_model, save_model, train_model)
from raspformer.programs import build_programs, make_first_element


class ExecutableModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler_config = CompilerConfig(vocab=(-1, 0, 1), max_seq_len=4)
        cls.compiled = compile_program(make_first_element(), cls.compiler_config)
        cls.model = from_compiled(cls.compiled)
        cls.sequences = [["BOS", -1, 0, 1, 0], ["BOS", 1, 1, 0, -1]]
        cls.inputs = np.array([cls.compiled.input_encoder.encode(x) for x in cls.sequences])

    def test_import_matches_full_tracr_trace_and_scores(self):
        for program in build_programs(max_seq_len=4).values():
            compiled = compile_program(program, self.compiler_config)
            model = from_compiled(compiled)
            inputs = np.array([compiled.input_encoder.encode(x) for x in self.sequences])
            scores, residuals = forward(model, inputs)
            for b, sequence in enumerate(self.sequences):
                reference = compiled.apply(sequence)
                expected = [reference.input_embeddings, *reference.residuals]
                for actual, original in zip(residuals, expected, strict=True):
                    np.testing.assert_allclose(actual[b], original[0], atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(scores[b], reference.transformer_output[0] @ model.params["unembed"],
                                           atol=2e-6, rtol=2e-6)
                np.testing.assert_array_equal(scores[b].argmax(-1), reference.unembedded[0])

    def test_rotation_and_compression_execute_in_new_state(self):
        model = random_model(5, 5, 3, seed=7, width=8, layers=2, heads=2)
        inputs = np.array([[0, 1, 2, 3, 4], [0, 2, 2, 1, 3]])
        basis = np.linalg.qr(np.random.default_rng(42).normal(size=(8, 8)))[0]
        expected, trace = forward(model, inputs)
        rotated = change_basis(model, basis)
        actual, rotated_trace = forward(rotated, inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for before, after in zip(trace, rotated_trace, strict=True):
            np.testing.assert_allclose(after, before @ basis, atol=2e-6, rtol=2e-6)
        compressed = change_basis(model, basis[:, :3])
        scores, compressed_trace = forward(compressed, inputs)
        self.assertEqual(compressed.config.width, 3)
        self.assertTrue(all(x.shape == (2, 5, 3) for x in compressed_trace))
        self.assertEqual(scores.shape, expected.shape)
        self.assertLess(parameter_count(compressed), parameter_count(model))
        self.assertFalse(np.allclose(scores, expected))

    def test_attention_scaling_and_after_stage_patch(self):
        baseline, trace = forward(self.model, self.inputs)
        softened, _ = forward(self.model, self.inputs, attention_scale=0.)
        self.assertGreater(float(np.max(np.abs(baseline - softened))), .1)
        delta = np.asarray(self.model.params["unembed"][:, 0])
        changed, patched = forward(self.model, self.inputs, patches=[{
            "stage": len(trace) - 1, "batch": 0, "position": 2, "delta": delta}])
        np.testing.assert_allclose(patched[-1][0, 2], trace[-1][0, 2] + delta)
        np.testing.assert_allclose(changed[0, 2], baseline[0, 2] + delta @ self.model.params["unembed"])
        np.testing.assert_array_equal(changed[1], baseline[1])

    def test_training_updates_transformer_and_checkpoint_roundtrip(self):
        model = random_model(5, 5, 2, seed=4, width=8, layers=1, heads=2)
        rng = np.random.default_rng(99)
        inputs = np.column_stack([np.zeros(160, dtype=int), rng.integers(1, 5, size=(160, 4))])
        labels = inputs[:, 1:] % 2
        trained, history = train_model(model, inputs[:128], labels[:128], inputs[128:], labels[128:],
                                       steps=100, seed=12, batch_size=32)
        self.assertLess(history[-1]["val_loss"], history[0]["val_loss"] / 5)
        self.assertGreater(history[-1]["val_accuracy"], .95)
        self.assertFalse(np.array_equal(model.params["layers"][0]["q"]["w"],
                                        trained.params["layers"][0]["q"]["w"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_model(trained, path)
            loaded = load_model(path)
        self.assertEqual(loaded.config, trained.config)
        for original, restored in zip(jax.tree_util.tree_leaves(trained.params),
                                      jax.tree_util.tree_leaves(loaded.params), strict=True):
            np.testing.assert_array_equal(original, restored)
        np.testing.assert_array_equal(forward(loaded, inputs[128:])[0], forward(trained, inputs[128:])[0])

    def test_unsupported_architecture_and_bad_basis_fail(self):
        original = self.compiled.model_config.layer_norm
        try:
            self.compiled.model_config.layer_norm = True
            with self.assertRaises(ValueError):
                from_compiled(self.compiled)
        finally:
            self.compiled.model_config.layer_norm = original
        with self.assertRaises(ValueError):
            change_basis(self.model, np.ones((self.model.config.width, 2)))


if __name__ == "__main__":
    unittest.main()

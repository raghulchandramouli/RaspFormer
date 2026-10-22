"""Checks for the InterpBench subset in bench/: programs, artifacts, and shapes."""

import json
import pickle
from pathlib import Path
import unittest

from bench import CASES, compile_case, evaluate, reference_outputs, sample_inputs
from bench.fetch import main as fetch_main
from bench.programs import BOS
from bench.load import load_interpbench, read_state_dict
from raspformer.models import change_basis, forward
import numpy as np
import jax.numpy as jnp
from raspformer.programs import build_programs

ROOT = Path(__file__).resolve().parent / "bench" / "interpbench"
CASE_IDS = sorted(CASES, key=int)


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _ConfigUnpickler(pickle.Unpickler):
    """Read ll_model_cfg.pkl without torch: unknown classes become inert stubs."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (_Stub,), {})


def trained_config(case_id: str) -> dict:
    with open(ROOT / case_id / "ll_model_cfg.pkl", "rb") as handle:
        return _ConfigUnpickler(handle).load()


class BenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = {case_id: compile_case(case) for case_id, case in CASES.items()}

    def test_six_cases_registered(self):
        self.assertEqual(CASE_IDS, ["2", "7", "13", "21", "44", "124"])
        for case_id, case in CASES.items():
            self.assertEqual(case.case_id, case_id)
            self.assertEqual(case.compile_seq_len, case.max_seq_len - 1)

    def test_artifacts_verified(self):
        self.assertEqual(fetch_main(["--verify"]), 0)

    def test_manifest_lists_every_case(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(set(manifest["cases"]), set(CASES))
        for case_id in CASES:
            files = manifest["cases"][case_id]["files"]
            self.assertEqual(set(files),
                             {"ll_model.pth", "ll_model_cfg.pkl", "edges.pkl", "meta.json"})

    def test_compiled_spaces_match_trained_config(self):
        for case_id in CASE_IDS:
            with self.subTest(case=case_id):
                model, cfg, case = self.compiled[case_id], trained_config(case_id), CASES[case_id]
                self.assertEqual(len(model.input_encoder.encoding_map), cfg["d_vocab"])
                self.assertEqual(cfg["d_vocab"], len(case.vocab) + 2)
                self.assertEqual(len(model.output_encoder.encoding_map), cfg["d_vocab_out"])
                self.assertEqual(cfg["n_ctx"], case.max_seq_len)
                self.assertEqual(cfg["attention_dir"] == "causal", case.causal)
                self.assertGreaterEqual(cfg["n_layers"], model.model_config.num_layers)

    def test_balanced_sampler_covers_both_predicate_values(self):
        case = CASES["124"]
        outputs = {evaluate(case, tokens)[0] for tokens in sample_inputs(case, 8, seed=3)}
        self.assertEqual(outputs, {0, 1})

    def test_compiled_matches_rasp_where_defined(self):
        for case_id in CASE_IDS:
            case = CASES[case_id]
            for tokens in sample_inputs(case, 8, seed=7):
                expected = evaluate(case, tokens)
                decoded = reference_outputs(self.compiled[case_id], tokens)
                self.assertEqual(len(expected), len(decoded))
                for want, got in zip(expected, decoded):
                    if want is not None:
                        self.assertEqual(got, want, (case_id, tokens, expected, decoded))

    def test_circuit_edges_load(self):
        for case_id in CASE_IDS:
            with self.subTest(case=case_id):
                with open(ROOT / case_id / "edges.pkl", "rb") as handle:
                    edges = pickle.load(handle)
                self.assertIsInstance(edges, list)
                self.assertTrue(edges)
                for edge in edges:
                    self.assertIsInstance(edge, tuple)
                    self.assertEqual(len(edge), 2)
                    self.assertTrue(all(isinstance(end, str) for end in edge))
                nodes = {end for edge in edges for end in edge}
                self.assertIn("hook_embed", nodes)
                layers = trained_config(case_id)["n_layers"]
                for node in nodes:
                    if node.startswith("blocks."):
                        self.assertLess(int(node.split(".")[1]), layers)

    def test_checkpoint_reader_matches_config_shapes(self):
        for case_id in CASE_IDS:
            with self.subTest(case=case_id):
                state = read_state_dict(ROOT / case_id / "ll_model.pth")
                cfg = trained_config(case_id)
                self.assertEqual(state["embed.W_E"].shape, (cfg["d_vocab"], cfg["d_model"]))
                self.assertEqual(state["unembed.W_U"].shape, (cfg["d_model"], cfg["d_vocab_out"]))
                self.assertEqual(state["blocks.0.attn.W_Q"].shape,
                                 (cfg["n_heads"], cfg["d_model"], cfg["d_head"]))
                self.assertTrue(all(np.isfinite(v).all() for k, v in state.items()
                                    if not k.endswith("IGNORE")))  # IGNORE is -inf by design

    def test_trained_models_reproduce_compiled_outputs(self):
        for case_id in CASE_IDS:
            with self.subTest(case=case_id):
                case, assembled = CASES[case_id], self.compiled[case_id]
                model = load_interpbench(case.artifact_dir)
                self.assertEqual(model.config.activation, "gelu")
                self.assertEqual(model.config.causal, case.causal)
                encoded = jnp.asarray([assembled.input_encoder.encode([BOS, *s])
                                       for s in sample_inputs(case, 64, seed=5)], dtype=jnp.int32)
                expected = np.asarray(assembled.forward(assembled.params, encoded).unembedded_output)[:, 1:]
                logits, states = forward(model, encoded)
                self.assertEqual(len(states), 2 * model.config.layers + 1)
                self.assertGreaterEqual(float((np.asarray(logits).argmax(-1)[:, 1:] == expected).mean()), 0.98)

    def test_rotation_keeps_trained_model_outputs(self):
        case, assembled = CASES["7"], self.compiled["7"]
        model = load_interpbench(case.artifact_dir)
        basis, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(model.config.width,) * 2))
        encoded = jnp.asarray([assembled.input_encoder.encode([BOS, *s])
                               for s in sample_inputs(case, 32, seed=9)], dtype=jnp.int32)
        original = np.asarray(forward(model, encoded)[0])
        rotated = np.asarray(forward(change_basis(model, basis), encoded)[0])
        # Float32 error scales with logit magnitude (case 44 logits reach 4e6).
        self.assertTrue((original.argmax(-1) == rotated.argmax(-1)).all())
        self.assertLess(np.abs(original - rotated).max(), 1e-4 * np.abs(original).max())

    def test_variables_and_values(self):
        from bench.variables import case_variables, variable_values
        case = CASES["13"]
        program = case.build()
        assembled = compile_case(case, program)
        variables = case_variables(case, program, assembled)
        self.assertEqual(len(variables), 5)
        self.assertTrue(all(v.first_stage >= 1 for v in variables))
        output = [v for v in variables if v.label.startswith("sequence_map_3")][0]
        values = variable_values(output, [[0, 1, 2, 2, 1, 0, 0, 0, 0]])
        self.assertEqual(values.shape, (1, 9))
        # Shifted neighbours are missing at the ends, so only the ends may be None.
        self.assertTrue((values[:, 1:-1] >= 0).all())

    def test_probe_recovers_compiled_variable(self):
        from bench.probes import fit_logistic, predict
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=300)
        x = np.eye(3)[y] * 5.0 + rng.normal(scale=0.1, size=(300, 3))
        probe = fit_logistic(x, y, 3, steps=200)
        self.assertGreaterEqual(float((predict(probe, x) == y).mean()), 0.99)

    def test_das_finds_compiled_lanes(self):
        from bench.das import (counterfactual_targets, interchange_accuracy, intervention_inputs,
                               sample_pairs, train_subspace)
        from bench.variables import case_variables, variable_values
        from raspformer.models import from_compiled
        case = CASES["7"]
        program = case.build()
        assembled = compile_case(case, program)
        model = from_compiled(assembled)
        [variable] = case_variables(case, program, assembled)
        sequences = sample_inputs(case, 64, seed=2)
        values = variable_values(variable, sequences)
        pairs = sample_pairs(values, np.random.default_rng(0), 64)
        targets, ok = counterfactual_targets(program, assembled, variable, sequences, pairs)
        self.assertTrue(ok.all())
        encoded = jnp.asarray([assembled.input_encoder.encode([BOS, *s]) for s in sequences], dtype=jnp.int32)
        stage = variable.first_stage
        base, diff = intervention_inputs(model, encoded, pairs, stage)
        q = train_subspace(model, stage, len(variable.classes) - 1, base, diff, targets, steps=150)
        self.assertGreaterEqual(interchange_accuracy(model, stage, q, base, diff, targets), 0.95)

    def test_eap_universe_and_truth(self):
        from bench.eap import edge_universe, ground_truth, rank_metrics
        model = load_interpbench(CASES["44"].artifact_dir)
        universe = edge_universe(model.config)
        self.assertEqual(len(universe), len(set(universe)))
        truth = ground_truth(CASES["44"].artifact_dir, model.config)
        self.assertEqual(len(truth["scorable"]), 4)
        self.assertEqual(len(truth["out_of_universe"]), 0)
        metrics = rank_metrics([3, 2, 1, 0], [True, True, False, False])
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 1.0)

    def test_decompiler_recovers_compiled_histogram(self):
        from bench.decompile import full_circuit, match_variables, recover, test_counterfactuals, test_outputs
        from bench.variables import case_variables, variable_values
        from raspformer.models import from_compiled
        case = CASES["7"]
        program = case.build()
        assembled = compile_case(case, program)
        model = from_compiled(assembled)
        sequences = sample_inputs(case, 96, seed=4)
        encoded = np.asarray([assembled.input_encoder.encode([BOS, *s]) for s in sequences])
        rec, classes = recover(model, full_circuit(model.config), encoded[:64])
        [variable] = case_variables(case, program, assembled)
        self.assertEqual(rec.kinds[rec.circuit.nodes[0]], "select")
        outputs = test_outputs(rec, model, encoded[64:])
        self.assertGreaterEqual(outputs["token_agreement"], 0.95)
        [match] = match_variables(classes, {variable.label: variable_values(variable, sequences[:64])})
        self.assertEqual(match["status"], "exact")
        counterfactuals = test_counterfactuals(rec, model, encoded[64:], pairs=32)
        self.assertTrue(all(r["agreement"] is None or r["agreement"] >= 0.9 for r in counterfactuals))

    def test_attended_keys_ignores_softmax_dust(self):
        from bench.decompile import attended_keys
        row = np.array([1.0, 1e-9, 1e-9, 1e-9])
        self.assertEqual(attended_keys(row).tolist(), [0])
        row = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(attended_keys(row).tolist(), [0, 1, 2, 3])
        row = np.array([0.6, 0.0, 0.2, 0.2])
        self.assertEqual(attended_keys(row).tolist(), [0, 2, 3])

    def test_overlap_with_raspformer_programs(self):
        names = set(build_programs())
        for case in CASES.values():
            if case.raspformer_match:
                self.assertIn(case.raspformer_match, names)


if __name__ == "__main__":
    unittest.main()

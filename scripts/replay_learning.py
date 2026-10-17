"""Replay saved checkpoints against their exact held-out predictions and targets."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tracr.rasp import rasp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raspformer.compiler import CompilerConfig, compile_program, write_json
from raspformer.geometry import pca_profile
from raspformer.models import forward, from_compiled, load_model
from raspformer.programs import build_programs


def main():
    reference = ROOT / "results/learning/reference"
    summary = pd.read_csv(reference / "summary.csv")
    geometry = pd.read_csv(reference / "geometry.csv")
    splits = pd.read_csv(reference / "sequence-splits.csv")
    test = splits[splits.split == "test"]
    sequences = [json.loads(value) for value in test.tokens]
    evaluator = rasp.DefaultRASPEvaluator()
    checked = []
    for name, program in build_programs().items():
        assembled = compile_program(program, CompilerConfig())
        inputs = np.asarray([assembled.input_encoder.encode(["BOS", *seq]) for seq in sequences], dtype=np.int32)
        target = np.asarray([evaluator.evaluate(program, seq) for seq in sequences])
        oracle = from_compiled(assembled)
        _, states = forward(oracle, inputs)
        for row in summary[summary.program == name].itertuples():
            path = reference / row.checkpoint
            encoding = json.loads((path.parent.parent / "encoding.json").read_text())
            classes = np.asarray(encoding["output_classes"])
            logits, _ = forward(load_model(path), inputs)
            actual = classes[np.asarray(logits)[:, 1:].argmax(axis=-1)]
            stored = pd.read_csv(path.parent / "predictions.csv")
            np.testing.assert_array_equal(actual.ravel(), stored.actual)
            np.testing.assert_array_equal(target.ravel(), stored.expected)
            np.testing.assert_array_equal(np.repeat(test.sequence_id.to_numpy(), 6), stored.sequence_id)
            if not np.isclose((actual == target).mean(), row.token_accuracy):
                raise AssertionError(f"Incorrect token accuracy for {name}, seed {row.seed}.")
            if not np.isclose((actual == target).all(axis=1).mean(), row.sequence_accuracy):
                raise AssertionError(f"Incorrect sequence accuracy for {name}, seed {row.seed}.")
            checked.append({"program": name, "seed": row.seed, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for index in geometry.index[geometry.program == name]:
            step = int(geometry.loc[index, "compiled_step"])
            state = np.asarray(states[step])[:, 1:]
            profile = pca_profile(state.reshape(-1, state.shape[-1]))
            geometry.loc[index, "compiled_total_variance"] = profile["total_variance"]
            geometry.loc[index, "compiled_variance_active"] = profile["variance_active"]
    geometry["learned_variance_active"] = geometry.learned_total_variance > 1e-12
    geometry.to_csv(reference / "geometry.csv", index=False)
    write_json(reference / "replay.json", {"models_verified": len(checked), "heldout_sequences": len(test), "checkpoints": checked})
    print(f"Verified {len(checked)} checkpoints against saved predictions and independent RASP targets.")


if __name__ == "__main__":
    main()

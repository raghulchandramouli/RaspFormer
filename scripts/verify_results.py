"""Check canonical evidence, frozen predictions, and checkpoint replay hashes."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raspformer.research import flat, infer_writes, select_patch_stage


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--test-log", type=Path)
    parser.add_argument("--notebook-summary", type=Path)
    args = parser.parse_args()
    reference = args.results.resolve() / "validation/reference"
    statuses = {}
    for name in ("programs", "trace", "geometry", "alignment", "comparison", "validation", "learning"):
        status = read_json(args.results / name / "reference/manifest.json")["status"]
        if status != "completed":
            raise AssertionError(f"Incomplete reference for {name}.")
        statuses[name] = status
    manifest = read_json(reference / "manifest.json")
    summary = read_json(reference / "summary.json")
    if summary != manifest["diagnostics"]:
        raise AssertionError("Validation summary disagrees with the completed manifest.")
    for item in summary["programs"]:
        folder = reference / item["program"].lower().replace("_", "-")
        for filename, key in (("blind-predictions.json", "predictions_sha256"),
                              ("transformed-blind-predictions.json", "transformed_predictions_sha256")):
            if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != item[key]:
                raise AssertionError(f"Frozen predictions changed for {item['program']}.")
        expected_rows = 3 * (len(manifest["settings"]["seeds"]) + 5)
        if len(item["transformed_recovery"]) != expected_rows:
            raise AssertionError("Missing transformed recovery variants.")
        stage = read_json(folder / "patch-stage-predictions.json")
        with np.load(folder / "activations.npz", allow_pickle=False) as data:
            predictions = infer_writes([flat(state) for state in data["calibration"]])
            if set(data["calibration_ids"]) & set(data["test_ids"]):
                raise AssertionError("Calibration and test sequences overlap.")
        if stage["source_split"] != "calibration" or predictions != stage["predictions"]:
            raise AssertionError("Patch stage did not use calibration activations.")
        selector = stage.get("selector", "cka")
        if selector != "cka":
            raise AssertionError(f"Patch stage selector is {selector!r}; the canonical selector is adjacent-CKA change.")
        selected_stage = select_patch_stage(predictions, selector)
        if stage["selected_stage"] != selected_stage:
            raise AssertionError("Patch stage is not the calibration adjacent-CKA-change argmax.")
        if stage["selected_stage"] != item["patches"]["selected_stage"]:
            raise AssertionError("Patch stage changed after calibration selection.")
    learning = args.results / "learning/reference"
    replay = read_json(learning / "replay.json")
    checked_count = len(replay["checkpoints"])
    learning_manifest = read_json(learning / "manifest.json")
    expected_count = len(learning_manifest["planned_programs"]) * len(learning_manifest["settings"]["seeds"])
    if checked_count != replay["models_verified"] or checked_count != expected_count:
        raise AssertionError("Checkpoint replay coverage is incomplete.")
    for item in replay["checkpoints"]:
        path = learning / item["program"].lower().replace("_", "-") / f"seed-{item['seed']}" / "model.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["checkpoint_sha256"]:
            raise AssertionError("A checkpoint changed after replay.")
    verification = {
        "checked_at": datetime.now(UTC).isoformat(), "canonical_manifests": statuses,
        "programs_verified": len(summary["programs"]), "checkpoint_hashes_verified": checked_count,
        "patch_stage_selection": "recomputed from calibration activations only",
    }
    if args.test_log:
        log = args.test_log.read_text()
        match = re.search(r"Ran (\d+) tests in ([0-9.]+)s", log)
        if not match or "\nOK\n" not in log or "\nFAILED" in log:
            raise AssertionError("The supplied test log does not record a passing suite.")
        verification["tests"] = {"count": int(match[1]), "seconds": float(match[2]), "result": "passed"}
        shutil.copyfile(args.test_log, reference / "verification-tests.log")
    if args.notebook_summary:
        notebooks = read_json(args.notebook_summary)
        if len(notebooks) != 5 or any(row["status"] != "passed" for row in notebooks):
            raise AssertionError("Notebook execution is incomplete.")
        verification["notebooks"] = notebooks
        shutil.copyfile(args.notebook_summary, reference / "notebook-execution.json")
    (reference / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Verified seven canonical studies, {len(summary['programs'])} research programs, and {checked_count} checkpoints.")


if __name__ == "__main__":
    main()

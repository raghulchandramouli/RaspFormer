"""Readable summary of a calibration run, written only from its saved tables."""

from pathlib import Path

import pandas as pd

from calib import analysis


def _table(frame, columns=None):
    frame = frame if columns is None else frame[columns]
    return frame.to_markdown(index=False, floatfmt=".3g") + "\n"


def _training(models):
    grouped = models.groupby("model_class").agg(
        models=("seed", "size"), token_accuracy=("token_accuracy", "mean"),
        lowest_accuracy=("token_accuracy", "min")).reset_index()
    if "mean_iia" in models:
        grouped = grouped.merge(models.groupby("model_class").mean_iia.mean().reset_index(), how="left")
    return grouped


def _pilot(tables, manifest):
    decision = manifest["pilot_decision"]
    pairs = tables["l3-pairs"]
    siit = pairs[pairs.model_class == "siit"].pivot(index="pairing", columns="metric", values="ratio").reset_index()
    return [
        "# Calibration pilot\n",
        f"Decision: **{decision['decision']}**. {decision['pairs_with_two_siit_seeds_per_member']} of {decision['pairs']} pilot pairs "
        f"have at least two SIIT seeds per member at mean interchange accuracy ≥ {analysis.IIA_THRESHOLD}; the rule needs {decision['required']}. "
        "The pilot decides only whether the pairs train and the pipeline runs. Five pairs cannot support the L3 test, so the ratios below are descriptive.\n",
        "## Training\n", _table(_training(tables["models"])),
        "## Excluded models\n", _table(tables["exclusions"]) if len(tables["exclusions"]) else "None.\n",
        "## Between-algorithm over between-seed distance, SIIT models\n", _table(siit),
        "## Summary over the five pairs\n",
        _table(tables["l3"], ["model_class", "metric", "pairs", "median_ratio", "identification_accuracy", "mean_between"]),
    ]


def _full(tables, manifest):
    verdicts, l1, l2, l3 = tables["verdicts"], tables["l1"], tables["l2"], tables["l3"]
    headline = verdicts[verdicts.model_class == "siit"]
    text = [
        "# Calibration study\n",
        "Each metric is graded on each model class at three levels. L1 asks whether it separates real models from null models. "
        "L2 asks whether it tracks a ground-truth quantity of the program beyond the number of values its variables take. "
        "L3 asks whether two models that compute the same function by different algorithms sit further apart than two seeds of the same algorithm. "
        "The headline is the verdict on SIIT models, whose circuits are fixed by training.\n",
        "## Verdict on SIIT models\n", _table(headline, ["metric", "L1", "L2", "L3", "tracks_cardinality", "verdict"]),
        "## Verdicts on every model class\n", _table(verdicts),
        "## Training\n", _table(_training(tables["models"])),
        f"{manifest['excluded']} trained models are excluded by the protocol's inclusion rules; they are listed in `exclusions.csv`.\n",
        "## L1: separation from null models\n",
        "AUROC is direction-free, with a bootstrap interval over programs. A metric passes when every applicable null reaches "
        f"{analysis.L1_THRESHOLD}.\n", _table(l1),
        "## L2: calibration against ground truth\n",
        f"ρ is the leave-program-out Spearman correlation. A metric passes when ρ ≥ {analysis.L2_THRESHOLD} and the Holm-corrected partial "
        "correlation controlling for summed cardinality is significant, for at least one quantity other than summed cardinality.\n",
        _table(l2, ["model_class", "metric", "quantity", "rho", "partial_rho", "partial_p_holm", "passed", "tracks_cardinality"]),
        "## L3: same function, different algorithm\n",
        f"A metric passes on the twin pairs when the Holm-corrected sign-flip p is below {analysis.ALPHA} and the median ratio of "
        f"between-algorithm to between-seed distance is at least {analysis.L3_RATIO}. Control pairs have different functions on an identical "
        "structure and carry no pass rule. Identification accuracy is how often a model is nearer to the other seeds of its own algorithm; chance is 0.5. "
        "Compiled models have no seeds, so only their between-algorithm distance is listed.\n",
        _table(l3, ["model_class", "kind", "metric", "pairs", "median_ratio", "p_holm", "identification_accuracy", "mean_between", "passed"]),
        "## Detector thresholds learned on calibration programs\n",
        "Each threshold maximises mean F1 on compiled P01–P06 and the compiled grid, and is then scored once on the held-out pools.\n",
        _table(tables["detectors"]),
    ]
    if "l2-compressed" in tables:
        l2c, l3c = tables["l2-compressed"], tables["l3-compressed"]
        passed = l2c.groupby(["model_class", "metric"]).passed.any().reset_index()
        text += [
            "## Superposition with behaviour preserved\n",
            "Models are compressed to a fraction of their residual width and fine-tuned back to 0.99 token accuracy. "
            f"{len(tables['exclusions-compressed'])} compressed models did not recover and are excluded.\n",
            "### L2 after compression\n", _table(passed.pivot(index="metric", columns="model_class", values="passed").reset_index()),
            "### L3 after compression\n",
            _table(l3c, ["model_class", "metric", "pairs", "median_ratio", "p_holm", "identification_accuracy", "passed"]),
        ]
    return text


def write_summary(root, tables, manifest):
    text = _pilot(tables, manifest) if manifest["pilot"] else _full(tables, manifest)
    path = Path(root) / "summary.md"
    path.write_text("\n".join(text))
    return path

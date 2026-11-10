"""Train and load the calibration models.

A task is one (program, model class, seed). Its folder holds ``model.npz`` and
``metrics.json``; a task whose metrics exist is never retrained, so an
interrupted pool run resumes where it stopped.
"""

import hashlib
import itertools
import json
import multiprocessing
import os
from functools import cache
from pathlib import Path

import numpy as np

from bench import siit
from calib import library
from raspformer import models
from raspformer.compiler import compile_program, write_json
from raspformer.research import flat

TRAINED_CLASSES = ("siit", "output-only", "shuffled-label")
SEEDS = (0, 1, 2)
STEPS = 6000
MAX_ROUNDS = 4
IIA_THRESHOLD = 0.95
TRAIN_SEQUENCES = 4096
TEST_SEQUENCES = 512
COMPRESSION_FRACTIONS = (0.75, 0.5, 0.25)
FINE_TUNE_STEPS = 2000
FINE_TUNE_CHUNK = 250
FINE_TUNE_TARGET = 0.99


@cache
def compiled(name):
    """One program instance, its compiled form and its executable high-level model.

    Tracr labels are unique per instance, so the schedule can only be read
    from the instance that was compiled.
    """
    spec = library.spec(name)
    program = spec.make()
    assembled = compile_program(program, spec.config)
    return program, assembled, models.from_compiled(assembled)


def encode(name, tokens):
    spec = library.spec(name)
    _, assembled, _ = compiled(name)
    return np.asarray([assembled.input_encoder.encode([spec.config.bos, *row]) for row in np.asarray(tokens).tolist()],
                      dtype=np.int32)


def task_folder(root, name, model_class, seed):
    return Path(root) / "models" / name / model_class / f"seed-{seed}"


def initial_model(name, seed):
    spec = library.spec(name)
    _, _, hl_model = compiled(name)
    return models.random_model(vocab_size=hl_model.params["token_embed"].shape[0],
                               max_length=spec.config.max_seq_len + 1,
                               output_classes=hl_model.params["unembed"].shape[1],
                               seed=seed, width=siit.WIDTH, layers=spec.layers, heads=siit.HEADS)


@cache
def rare_outcome_strata(name):
    """Inputs of each answer, for a whole-sequence yes/no program whose rarer answer is under a tenth of inputs.

    Uniform sampling almost never shows such a program its rare answer, and a
    model that always gives the common answer then passes every accuracy and
    interchange check. Other programs return None and train on uniform inputs.
    """
    spec = library.spec(name)
    if spec.kind != "pair":
        return None
    _, _, hl_model = compiled(name)
    tokens = np.asarray(list(itertools.product(spec.config.vocab, repeat=library.SEQUENCE_LENGTH)))
    encoded = encode(name, tokens)
    labels = np.concatenate([np.asarray(models.forward(hl_model, encoded[start:start + 16384])[0])[:, 1:].argmax(-1)
                             for start in range(0, len(encoded), 16384)])
    answers = np.unique(labels)
    if len(answers) != 2 or not (labels == labels[:, :1]).all():
        return None
    strata = [tokens[labels[:, 0] == answer] for answer in answers]
    return strata if min(map(len, strata)) < library.RARE_OUTCOME_SHARE * len(tokens) else None


def _data(name, seed):
    tokens = library.training_tokens(library.spec(name).config, seed, TRAIN_SEQUENCES + TEST_SEQUENCES,
                                     strata=rare_outcome_strata(name))
    encoded = encode(name, tokens)
    return encoded[:TRAIN_SEQUENCES], encoded[TRAIN_SEQUENCES:]


def train_task(root, name, model_class, seed, steps=STEPS):
    folder = task_folder(root, name, model_class, seed)
    if (folder / "metrics.json").exists():
        return json.loads((folder / "metrics.json").read_text())
    _, _, hl_model = compiled(name)
    train, test = _data(name, seed)
    circuit_nodes = siit.active_nodes(hl_model, train[:256])
    model = initial_model(name, seed)
    if model_class == "shuffled-label":
        labels = siit.hl_labels(hl_model, train)
        shuffled = labels[np.random.default_rng(seed).permutation(len(labels))]
        trained, history = models.train_model(model, train, shuffled, test, siit.hl_labels(hl_model, test),
                                              steps=steps, seed=seed, batch_size=64, lr=1e-3)
        history = [row for row in history if "val_accuracy" in row][-6:]
    elif model_class == "output-only":
        trained, history = siit.train_siit(hl_model, model, train, circuit_nodes, steps=steps, seed=seed,
                                           log_every=max(1, steps // 6), verbose=False, iit_weight=0.0, strict_weight=0.0)
    elif model_class == "siit":
        trained, history, rounds = model, [], 0
        while rounds < MAX_ROUNDS and (not rounds or siit.evaluate_siit(hl_model, trained, test, circuit_nodes)["mean_iia"] < IIA_THRESHOLD):
            trained, added = siit.train_siit(hl_model, trained, train, circuit_nodes, steps=steps, seed=seed + 1000 * rounds,
                                             log_every=max(1, steps // 6), verbose=False)
            history += [dict(row, step=row["step"] + steps * rounds) for row in added]
            rounds += 1
        steps *= rounds
    else:
        raise ValueError(f"model_class must be one of {TRAINED_CLASSES}")
    metrics = siit.evaluate_siit(hl_model, trained, test, circuit_nodes)
    folder.mkdir(parents=True, exist_ok=True)
    models.save_model(trained, folder / "model.npz")
    record = dict(spec=name, model_class=model_class, seed=seed, steps=steps, layers=trained.config.layers,
                  compiled_layers=hl_model.config.layers, width=trained.config.width, heads=trained.config.heads,
                  circuit_nodes=circuit_nodes, history=history, **metrics,
                  checkpoint_sha256=hashlib.sha256((folder / "model.npz").read_bytes()).hexdigest())
    write_json(folder / "metrics.json", record)
    return record


def compression_basis(model, encoded, fraction):
    """Leading uncentered principal directions of every residual stage on training inputs."""
    _, residuals = models.forward(model, encoded)
    _, _, directions = np.linalg.svd(np.concatenate([flat(stage) for stage in residuals]), full_matrices=False)
    return directions.T[:, :max(1, int(np.ceil(model.config.width * fraction)))]


def compress_task(root, name, model_class, seed, fraction):
    """Compress one model's residual stream, then fine-tune every parameter on the output loss."""
    folder = task_folder(root, name, f"{model_class}-compressed-{fraction}", seed)
    if (folder / "metrics.json").exists():
        return json.loads((folder / "metrics.json").read_text())
    _, _, hl_model = compiled(name)
    source = hl_model if model_class == "compiled" else load_model(root, name, model_class, seed)
    train, test = _data(name, seed)
    labels, test_labels = siit.hl_labels(hl_model, train), siit.hl_labels(hl_model, test)
    model = models.change_basis(source, compression_basis(source, train[:256], fraction))
    start = float((np.asarray(models.forward(model, test)[0])[:, 1:].argmax(-1) == test_labels).mean())
    accuracy, steps = start, 0
    while accuracy < FINE_TUNE_TARGET and steps < FINE_TUNE_STEPS:
        model, history = models.train_model(model, train, labels, test, test_labels, steps=FINE_TUNE_CHUNK,
                                            seed=seed + steps, batch_size=64, lr=1e-3)
        accuracy, steps = history[-1]["val_accuracy"], steps + FINE_TUNE_CHUNK
    folder.mkdir(parents=True, exist_ok=True)
    models.save_model(model, folder / "model.npz")
    record = dict(spec=name, model_class=model_class, seed=seed, fraction=fraction, width=model.config.width,
                  original_width=source.config.width, accuracy_after_compression=start,
                  token_accuracy=float(accuracy), fine_tune_steps=steps,
                  checkpoint_sha256=hashlib.sha256((folder / "model.npz").read_bytes()).hexdigest())
    write_json(folder / "metrics.json", record)
    return record


def _run(job):
    function, arguments = job
    try:
        return dict(ok=True, **function(*arguments))
    except Exception as error:
        return dict(ok=False, spec=arguments[1], model_class=arguments[2], seed=arguments[3],
                    error=f"{type(error).__name__}: {error}")


def run_jobs(jobs, workers):
    """Run jobs in fresh single-threaded processes; yield each record as it finishes."""
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["OMP_NUM_THREADS"] = "1"
    with multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        yield from pool.imap_unordered(_run, jobs)


def training_jobs(root, specs, steps=STEPS, classes=TRAINED_CLASSES):
    """Shuffled-label nulls are only needed for the grid, where L1 is run."""
    return [(train_task, (str(root), spec.name, model_class, seed, steps))
            for spec in specs for model_class in classes for seed in SEEDS
            if model_class != "shuffled-label" or spec.kind == "grid"]


def compression_jobs(root, specs):
    """Compiled models have no seeds, so each is compressed once under seed zero."""
    return [(compress_task, (str(root), spec.name, model_class, seed, fraction))
            for spec in specs for fraction in COMPRESSION_FRACTIONS
            for model_class, seeds in (("compiled", SEEDS[:1]), ("siit", SEEDS)) for seed in seeds
            if model_class == "siit" or spec.kind == "grid"]


def load_metrics(root, name, model_class, seed):
    path = task_folder(root, name, model_class, seed) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_model(root, name, model_class, seed):
    return models.load_model(task_folder(root, name, model_class, seed) / "model.npz")


def measure(name, model):
    """Residual stages of a model on its configuration's probe sequences, BOS removed."""
    _, residuals = models.forward(model, encode(name, library.probe_tokens(library.spec(name).config)))
    return [flat(stage) for stage in residuals]

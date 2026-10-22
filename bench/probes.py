"""Linear probes: multinomial logistic regression from a residual stage to a variable.

Features are standardised with calibration statistics (trained models have
residual scales up to 1e6). Weights are fit with Adam under an L2 penalty on
calibration observations and scored on held-out observations. A variable
counts as readable at a stage when held-out accuracy reaches ``READABLE``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

READABLE = 0.95


def fit_logistic(x, y, classes, *, l2=1e-3, steps=400, lr=0.05, seed=0):
    """Return (weights, bias, mean, std) for standardised multinomial regression."""
    x = np.asarray(x, dtype=np.float64)
    mean, std = x.mean(0), x.std(0) + 1e-8
    xs = jnp.asarray((x - mean) / std, dtype=jnp.float32)
    ys = jnp.asarray(y, dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    params = {"w": 0.01 * jax.random.normal(key, (x.shape[1], classes), dtype=jnp.float32),
              "b": jnp.zeros(classes, dtype=jnp.float32)}

    def loss(p):
        logits = xs @ p["w"] + p["b"]
        nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), ys[:, None], axis=1).mean()
        return nll + l2 * jnp.sum(p["w"] ** 2)

    @jax.jit
    def step(p, m, v, t):
        g = jax.grad(loss)(p)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(
            lambda q, a, b: q - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8),
            p, m, v)
        return p, m, v

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for t in range(1, steps + 1):
        params, m, v = step(params, m, v, t)
    return np.asarray(params["w"]), np.asarray(params["b"]), mean, std


def predict(probe, x):
    w, b, mean, std = probe
    return (((np.asarray(x) - mean) / std) @ w + b).argmax(-1)


def flatten(states, values):
    """Pool positions 1.. of every sequence; drop undefined (-1) targets."""
    x = np.asarray(states, dtype=np.float64)[:, 1:, :].reshape(-1, states.shape[-1])
    y = np.asarray(values).reshape(-1)
    keep = y >= 0
    return x[keep], y[keep]


def probe_table(case_id, kind, stage_names, calibration_states, test_states,
                variables, calibration_values, test_values):
    """One row per (variable, stage) with train/test accuracy and majority baseline."""
    rows = []
    for variable in variables:
        y_train_all, y_test_all = calibration_values[variable.label], test_values[variable.label]
        for stage, name in enumerate(stage_names):
            x_train, y_train = flatten(calibration_states[stage], y_train_all)
            x_test, y_test = flatten(test_states[stage], y_test_all)
            majority = np.bincount(y_train, minlength=len(variable.classes)).argmax()
            probe = fit_logistic(x_train, y_train, len(variable.classes))
            rows.append(dict(case=case_id, model=kind, variable=variable.label, stage=stage,
                             stage_name=name, first_write_stage=variable.first_stage,
                             classes=len(variable.classes), observations=len(y_train),
                             train_accuracy=float((predict(probe, x_train) == y_train).mean()),
                             test_accuracy=float((predict(probe, x_test) == y_test).mean()),
                             majority_accuracy=float((y_test == majority).mean())))
    frame = pd.DataFrame(rows)
    frame["readable"] = frame.test_accuracy >= READABLE
    return frame


def readability_map(frame):
    """Per variable: first readable stage versus first-write stage."""
    rows = []
    for (case, kind, variable), g in frame.groupby(["case", "model", "variable"], sort=False):
        g = g.sort_values("stage")
        readable = g[g.readable]
        first = int(readable.stage.iloc[0]) if len(readable) else None
        rows.append(dict(case=case, model=kind, variable=variable,
                         first_write_stage=int(g.first_write_stage.iloc[0]),
                         first_readable_stage=first,
                         readable_before_write=bool(first is not None and first < g.first_write_stage.iloc[0]),
                         readable_at_write=bool(g[g.stage == g.first_write_stage.iloc[0]].readable.iloc[0]),
                         max_test_accuracy=float(g.test_accuracy.max()),
                         stages_readable="".join("1" if r else "0" for r in g.readable)))
    return pd.DataFrame(rows)

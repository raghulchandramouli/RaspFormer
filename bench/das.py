"""Distributed Alignment Search: find the subspace that behaves like a variable.

For a variable v and a residual stage s, learn an orthonormal basis Q (width
by k). For a base input b and a source input s at position p, replace the
Q-component of the base residual at (s, p) with the source's, run the rest of
the model, and compare the full output with the program's own counterfactual
(RASP re-evaluated with v at p set to the source value). The fraction of pairs
where they agree is interchange intervention accuracy (IIA).

Pairs are sampled so the source value differs from the base value. Output
positions where the program's counterfactual is undefined (None) are masked
out of the loss and of IIA; a pair is invalid only if every position is
undefined.
"""

import jax
import jax.numpy as jnp
import numpy as np

from tracr.rasp import rasp

from bench.variables import output_ids, variable_values
from raspformer.alignment import rasp_with_patch
from raspformer.models import forward


def sample_pairs(values, rng, count, *, max_tries_factor=50):
    """(base, source, position) with both values defined and different."""
    n, length = values.shape
    pairs, tries = [], 0
    while len(pairs) < count and tries < max_tries_factor * count:
        tries += 1
        b, s, p = rng.integers(n), rng.integers(n), rng.integers(length)
        if b == s or values[b, p] < 0 or values[s, p] < 0 or values[b, p] == values[s, p]:
            continue
        pairs.append((int(b), int(s), int(p)))
    return pairs


def counterfactual_targets(program, assembled, variable, sequences, pairs):
    """Program output ids after substituting the source value at the position."""
    evaluator = rasp.DefaultRASPEvaluator()
    targets, valid = [], []
    for b, s, p in pairs:
        source_value = evaluator.evaluate(variable.expression, list(sequences[s]))[p]
        outputs = rasp_with_patch(program, variable.expression, list(sequences[b]), p, source_value)
        ids = output_ids(assembled, list(outputs))
        targets.append(ids)
        valid.append(bool((ids >= 0).any()))
    return np.asarray(targets), np.asarray(valid)


def _position_mask(pairs, length):
    mask = np.zeros((len(pairs), length), dtype=np.float32)
    for i, (_, _, p) in enumerate(pairs):
        mask[i, p + 1] = 1.0  # residual index: BOS occupies 0
    return mask


def intervention_inputs(model, encoded, pairs, stage):
    """Base inputs, and the source-minus-base residual at the stage and position."""
    base = encoded[np.asarray([b for b, _, _ in pairs])]
    source = encoded[np.asarray([s for _, s, _ in pairs])]
    _, base_states = forward(model, base)
    _, source_states = forward(model, source)
    diff = (source_states[stage] - base_states[stage]) * _position_mask(pairs, encoded.shape[1])[:, :, None]
    return base, jnp.asarray(diff)


def train_subspace(model, stage, k, base, diff, targets, *, steps=300, lr=0.01, seed=0):
    """Adam on an unconstrained width-by-k matrix; QR keeps the basis orthonormal."""
    targets = jnp.asarray(targets, dtype=jnp.int32)
    width = model.config.width
    w = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (width, k), dtype=jnp.float32)

    defined = (targets >= 0).astype(jnp.float32)
    safe_targets = jnp.where(targets >= 0, targets, 0)

    def loss(w):
        q, _ = jnp.linalg.qr(w)
        delta = (diff @ q) @ q.T
        logits, _ = forward(model, base, stage_deltas={stage: delta})
        logp = jax.nn.log_softmax(logits[:, 1:], axis=-1)
        nll = -jnp.take_along_axis(logp, safe_targets[..., None], axis=-1)[..., 0]
        return (nll * defined).sum() / jnp.maximum(defined.sum(), 1.0)

    @jax.jit
    def step(w, m, v, t):
        g = jax.grad(loss)(w)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        return w, m, v

    m, v = jnp.zeros_like(w), jnp.zeros_like(w)
    for t in range(1, steps + 1):
        w, m, v = step(w, m, v, t)
    q, _ = jnp.linalg.qr(w)
    return np.asarray(q)


def interchange_accuracy(model, stage, q, base, diff, targets, subset=None):
    """Fraction of pairs whose full patched output equals the counterfactual.

    ``subset`` restricts the mean to a boolean mask over pairs, e.g. the pairs
    whose counterfactual differs from the unpatched output.
    """
    q = jnp.asarray(q, dtype=jnp.float32)
    delta = (diff @ q) @ q.T
    logits, _ = forward(model, base, stage_deltas={stage: delta})
    predicted = np.asarray(logits).argmax(-1)[:, 1:]
    targets = np.asarray(targets)
    agree = ((predicted == targets) | (targets < 0)).all(axis=1)   # undefined positions do not count
    if subset is not None:
        subset = np.asarray(subset, dtype=bool)
        return float(agree[subset].mean()) if subset.any() else float("nan")
    return float(agree.mean())


def random_subspace(width, k, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, k)))
    return q[:, :k]

"""Circuit-discovery methods scored on the same edge frame as EAP.

Every method takes ``(model, clean, corrupt, labels)`` and returns a frame with
columns ``source, target, score`` over exactly ``edge_universe(model.config)``.
``patch`` and ``acdc`` use ``circuit_forward`` slot deltas, which add a source
difference to one target's input only, so they are exact for the non-linear
model. ``score_frame`` ranks any such frame against InterpBench's circuit edges.
"""

import re
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd

from bench.decompile import circuit_forward
from bench.eap import attribute, edge_universe, node_names, rank_metrics

ACDC_TAUS = (1e-4, 1e-3, 1e-2, 1e-1)
IG_STEPS = 5

_QKV = re.compile(r"blocks\.(\d+)\.hook_([qkv])_input\[(\d+)\]")
_MLP = re.compile(r"blocks\.(\d+)\.hook_mlp_in")
_RESID = re.compile(r"blocks\.(\d+)\.hook_resid_post")


def target_slot(target):
    """The (node, slot) key ``circuit_forward`` uses for an EAP target hook."""
    if m := _QKV.fullmatch(target):
        return f"blocks.{m[1]}.attn.hook_result[{m[3]}]", m[2]
    if m := _MLP.fullmatch(target):
        return f"blocks.{m[1]}.hook_mlp_out", "in"
    if _RESID.fullmatch(target):
        return "hook_resid_post", "out"
    raise ValueError(f"unknown EAP target {target!r}")


def _logit_difference(logits, labels):
    """Correct logit minus the mean of the others, as in ``bench.eap.attribute``."""
    logits = logits[:, 1:]
    correct = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    others = (logits.sum(-1) - correct) / max(1, logits.shape[-1] - 1)
    return (correct - others).mean()


def _kl_from(clean_logits, logits):
    """Mean KL(clean || patched) over non-BOS positions."""
    clean_lp = jax.nn.log_softmax(clean_logits[:, 1:], axis=-1)
    lp = jax.nn.log_softmax(logits[:, 1:], axis=-1)
    return (jnp.exp(clean_lp) * (clean_lp - lp)).sum(-1).mean()


class _Context:
    """Clean run, source differences, and a jitted forward over per-target deltas."""

    def __init__(self, model, clean, corrupt, labels):
        self.model, self.clean = model, clean
        self.labels = jnp.asarray(labels, dtype=jnp.int32)
        self.edges = edge_universe(model.config)
        _, self.targets = node_names(model.config)
        self.zeros = {t: jnp.zeros((*clean.shape, model.config.width), dtype=jnp.float32)
                      for t in self.targets}
        _, self.clean_out, _ = circuit_forward(model, clean)
        _, corrupt_out, _ = circuit_forward(model, corrupt)
        self.diff = {s: corrupt_out[s] - self.clean_out[s] for s in self.clean_out}

        def logits(deltas, patches=None):
            slots = {target_slot(t): d for t, d in deltas.items()}
            return circuit_forward(model, clean, node_patches=patches, slot_deltas=slots)[0]

        self.logits = jax.jit(logits)
        self.metric = lambda deltas, patches=None: _logit_difference(logits(deltas, patches), self.labels)
        self.metric_jit = jax.jit(self.metric)
        self.grad = jax.jit(jax.grad(self.metric))
        self.clean_logits = self.logits(self.zeros)

    def kl(self, deltas):
        return _kl_from(self.clean_logits, self.logits(deltas))

    def frame(self, scores):
        return pd.DataFrame([dict(source=s, target=t, score=float(v))
                             for (s, t), v in zip(self.edges, scores)])


def eap(model, clean, corrupt, labels):
    return attribute(model, clean, corrupt, labels)


def patch(model, clean, corrupt, labels):
    """Exact change in the metric from patching each edge alone."""
    ctx = _Context(model, clean, corrupt, labels)
    base = ctx.metric_jit(ctx.zeros)
    scores = [ctx.metric_jit({**ctx.zeros, t: ctx.diff[s]}) - base for s, t in ctx.edges]
    return ctx.frame(scores)


def acdc(model, clean, corrupt, labels, taus=ACDC_TAUS, metric="kl"):
    """Greedy ACDC pruning at each tau, latest targets first.

    ``score`` is the largest tau at which the edge survives (0 if it is pruned
    at every tau). ``kept@<tau>`` columns hold the binary circuit per tau.
    ``metric="kl"`` prunes on KL(clean || patched); ``"logit"`` prunes on the
    change in the logit-difference metric relative to its clean value, which
    stays informative where softmax saturates (these models' logits reach 1e6,
    so most single-edge KLs are exactly zero).
    """
    ctx = _Context(model, clean, corrupt, labels)
    rank = {t: i for i, t in enumerate(ctx.targets)}
    order = sorted(ctx.edges, key=lambda e: rank[e[1]], reverse=True)
    base = float(ctx.metric_jit(ctx.zeros))

    def distance(deltas):
        if metric == "kl":
            return float(ctx.kl(deltas))
        return abs(float(ctx.metric_jit(deltas)) - base) / max(abs(base), 1e-12)

    kept = {}
    for tau in taus:
        pruned = dict(ctx.zeros)
        for s, t in order:
            trial = {**pruned, t: pruned[t] + ctx.diff[s]}
            if distance(trial) < tau:
                pruned = trial
            else:
                kept[(s, t, tau)] = True
    frame = ctx.frame([max((tau for tau in taus if kept.get((s, t, tau))), default=0.0)
                       for s, t in ctx.edges])
    for tau in taus:
        frame[f"kept@{tau:g}"] = [kept.get((s, t, tau), False) for s, t in ctx.edges]
    return frame


def ig(model, clean, corrupt, labels, steps=IG_STEPS, interpolate="input"):
    """EAP with the gradient averaged over ``steps`` points from clean to corrupt.

    ``interpolate="input"`` interpolates the embeddings, so every source
    activation moves along its own forward pass, as in Hanna et al. (2024).
    ``"delta"`` instead adds alpha times the summed source difference to every
    target input of the clean run; later layers then see the upstream change
    twice, once through the residual stream and once through the delta.
    """
    ctx = _Context(model, clean, corrupt, labels)
    total = {t: sum(ctx.diff[s] for s, u in ctx.edges if u == t) for t in ctx.targets}
    everywhere = jnp.ones(clean.shape, dtype=bool)
    grads = []
    for k in range(1, steps + 1):
        alpha = k / steps
        if interpolate == "input":
            patches = {n: (ctx.clean_out[n] + alpha * ctx.diff[n], everywhere)
                       for n in ("hook_embed", "hook_pos_embed")}
            grads.append(ctx.grad(ctx.zeros, patches))
        else:
            grads.append(ctx.grad({t: alpha * total[t] for t in ctx.targets}))
    mean_grad = {t: sum(g[t] for g in grads) / steps for t in ctx.targets}
    return ctx.frame([jnp.sum(ctx.diff[s] * mean_grad[t]) / clean.shape[0] for s, t in ctx.edges])


METHODS = {"eap": eap, "patch": patch, "acdc": acdc, "acdc-logit": partial(acdc, metric="logit"),
           "ig": ig, "ig-delta": partial(ig, interpolate="delta")}


def score_frame(frame, truth_edges):
    """Edge- and node-level rank metrics, ranking by absolute score as ``evaluate_eap`` does."""
    truth_edges = set(truth_edges)
    frame = frame.assign(abs_score=frame.score.abs(),
                         true=[(s, t) in truth_edges for s, t in zip(frame.source, frame.target)])
    node = frame.groupby("source").agg(abs_score=("abs_score", "max"), true=("true", "any")).reset_index()
    return dict(edge=rank_metrics(frame.abs_score, frame["true"]),
                node=rank_metrics(node.abs_score, node["true"]))


def score_circuit(frame, kept, truth_edges):
    """Precision, recall and F1 of the binary circuit ``frame[kept]``, by edge and by node."""
    truth_edges = set(truth_edges)
    frame = frame.assign(true=[(s, t) in truth_edges for s, t in zip(frame.source, frame.target)])
    node = frame.groupby("source").agg(kept=(kept, "any"), true=("true", "any")).reset_index()
    out = {}
    for level, table, flag in (("edge", frame, kept), ("node", node, "kept")):
        tp = int((table[flag] & table["true"]).sum())
        precision = tp / max(1, int(table[flag].sum()))
        recall = tp / max(1, int(table["true"].sum()))
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        out[level] = dict(size=int(table[flag].sum()), precision=precision, recall=recall, f1=f1)
    return out

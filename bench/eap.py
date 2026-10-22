"""Edge attribution patching (EAP) scored against InterpBench's circuit edges.

Nodes follow TransformerLens hook names. Sources are ``hook_embed``,
``hook_pos_embed``, ``blocks.l.attn.hook_result[h]`` (one head's output through
its slice of W_O, no bias) and ``blocks.l.hook_mlp_out``. Targets are
``blocks.l.hook_{q,k,v}_input[h]``, ``blocks.l.hook_mlp_in`` and the final
``blocks.L-1.hook_resid_post``. In a residual network the input of a target is
the sum of all earlier source outputs, so the edge score is

    score(s -> t) = sum over batch and positions of
                    (source_corrupt - source_clean) . d metric / d input_t

evaluated on the clean run (Syed, Rager and Conmy, 2023). The metric is the
logit difference of the model's own clean predictions against the mean of
the other classes. Edges are ranked by
absolute score. Ground-truth edges that are internal to a node (q_input ->
hook_q, hook_q -> hook_result, mlp_in -> mlp_out) are not scorable and are
excluded; edges from internal nodes (e.g. hook_k -> v_input) are reported as
out of universe.
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from raspformer.models import ACTIVATIONS


def node_names(config):
    layers, heads = config.layers, config.heads
    sources = ["hook_embed", "hook_pos_embed"]
    targets = []
    for l in range(layers):
        for kind in ("q", "k", "v"):
            targets += [f"blocks.{l}.hook_{kind}_input[{h}]" for h in range(heads)]
        targets.append(f"blocks.{l}.hook_mlp_in")
        sources += [f"blocks.{l}.attn.hook_result[{h}]" for h in range(heads)]
        sources.append(f"blocks.{l}.hook_mlp_out")
    targets.append(f"blocks.{layers - 1}.hook_resid_post")
    return sources, targets


def _order(name, layers):
    """Position in the residual stream: embeddings 0, then per layer attn/mlp."""
    if name in ("hook_embed", "hook_pos_embed"):
        return 0
    l = int(name.split(".")[1])
    if "hook_result" in name or "_input" in name:
        return 2 * l + 1
    if "mlp" in name:
        return 2 * l + 2
    return 2 * layers + 1  # resid_post


def edge_universe(config):
    sources, targets = node_names(config)
    return [(s, t) for s in sources for t in targets
            if _order(s, config.layers) < _order(t, config.layers)]


def eap_forward(model, inputs, deltas):
    """Logits and every source output, with additive deltas on target inputs."""
    cfg, p = model.config, model.params
    act = ACTIVATIONS[cfg.activation]
    batch, length = inputs.shape
    sources = {}
    tok = p["token_embed"][inputs]
    pos = jnp.broadcast_to(p["pos_embed"][:length], tok.shape)
    sources["hook_embed"], sources["hook_pos_embed"] = tok, pos
    x = tok + pos
    mask = jnp.ones((length, length), dtype=bool)
    if cfg.causal:
        mask = jnp.tril(mask)
    for l, layer in enumerate(p["layers"]):
        head_outputs = []
        for h in range(cfg.heads):
            sl = slice(h * cfg.key_size, (h + 1) * cfg.key_size)
            q = (x + deltas[f"blocks.{l}.hook_q_input[{h}]"]) @ layer["q"]["w"][:, sl] + layer["q"]["b"][sl]
            k = (x + deltas[f"blocks.{l}.hook_k_input[{h}]"]) @ layer["k"]["w"][:, sl] + layer["k"]["b"][sl]
            v = (x + deltas[f"blocks.{l}.hook_v_input[{h}]"]) @ layer["v"]["w"][:, sl] + layer["v"]["b"][sl]
            scores = jnp.einsum("btd,bsd->bts", q, k) / np.sqrt(cfg.key_size)
            weights = jax.nn.softmax(jnp.where(mask, scores, -1e30), axis=-1)
            z = jnp.einsum("bts,bsd->btd", weights, v)
            out = z @ layer["o"]["w"][sl, :]
            sources[f"blocks.{l}.attn.hook_result[{h}]"] = out
            head_outputs.append(out)
        x = x + sum(head_outputs) + layer["o"]["b"]
        mlp_in = x + deltas[f"blocks.{l}.hook_mlp_in"]
        mlp_out = act(mlp_in @ layer["mlp_in"]["w"] + layer["mlp_in"]["b"]) @ layer["mlp_out"]["w"] + layer["mlp_out"]["b"]
        sources[f"blocks.{l}.hook_mlp_out"] = mlp_out
        x = x + mlp_out
    final = x + deltas[f"blocks.{cfg.layers - 1}.hook_resid_post"]
    logits = final @ p["unembed"]
    if "unembed_bias" in p:
        logits = logits + p["unembed_bias"]
    return logits, sources


def attribute(model, clean, corrupt, labels):
    """Signed EAP score for every edge in the universe, averaged over the batch."""
    _, targets = node_names(model.config)
    zeros = {t: jnp.zeros((*clean.shape, model.config.width), dtype=jnp.float32) for t in targets}
    labels = jnp.asarray(labels, dtype=jnp.int32)

    def metric(deltas):
        # Logit difference (correct minus mean of the others) stays informative
        # where cross-entropy saturates: trained logits reach 1e6 on case 44.
        logits, _ = eap_forward(model, clean, deltas)
        logits = logits[:, 1:]
        correct = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        classes = logits.shape[-1]
        others = (logits.sum(-1) - correct) / max(1, classes - 1)
        return (correct - others).mean()

    grads = jax.grad(metric)(zeros)
    _, clean_sources = eap_forward(model, clean, zeros)
    _, corrupt_sources = eap_forward(model, corrupt, zeros)
    rows = []
    for s, t in edge_universe(model.config):
        diff = corrupt_sources[s] - clean_sources[s]
        score = jnp.sum(diff * grads[t]) / clean.shape[0]
        rows.append(dict(source=s, target=t, score=float(score)))
    return pd.DataFrame(rows)


def ground_truth(case_dir, config):
    """Truth edges split into scorable (in universe), internal, and out of universe."""
    with open(Path(case_dir) / "edges.pkl", "rb") as handle:
        edges = [tuple(e) for e in pickle.load(handle)]
    # InterpBench names the output node after the high-level model's last block
    # (blocks.0.hook_resid_post for a one-layer program); map it to this model's.
    final = f"blocks.{config.layers - 1}.hook_resid_post"
    edges = [(s, final if t.endswith("hook_resid_post") else t) for s, t in edges]
    universe = set(edge_universe(config))
    sources, targets = node_names(config)
    internal = [e for e in edges if (("_input" in e[0] and ".attn.hook_" in e[1])
                                     or (".attn.hook_" in e[0] and "hook_result" in e[1])
                                     or ("hook_mlp_in" in e[0] and "hook_mlp_out" in e[1]))]
    scorable = [e for e in edges if e in universe]
    other = [e for e in edges if e not in universe and e not in internal]
    return dict(scorable=scorable, internal=internal, out_of_universe=other)


def rank_metrics(scores, truth):
    """AUROC, average precision, and precision/recall at k = number of true edges."""
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=bool)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(auroc=None, average_precision=None, precision_at_k=None, recall_at_k=None, k=n_pos)
    order = np.argsort(-scores, kind="stable")
    ranked = truth[order]
    hits = np.cumsum(ranked)
    precision = hits / np.arange(1, len(ranked) + 1)
    ap = float((precision * ranked).sum() / n_pos)
    pos_scores, neg_scores = scores[truth], scores[~truth]
    auroc = float((pos_scores[:, None] > neg_scores[None, :]).mean()
                  + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).mean())
    return dict(auroc=auroc, average_precision=ap, precision_at_k=float(precision[n_pos - 1]),
                recall_at_k=float(hits[n_pos - 1] / n_pos), k=n_pos)


def evaluate_eap(model, case_dir, clean, corrupt, labels):
    frame = attribute(model, clean, corrupt, labels)
    truth = ground_truth(case_dir, model.config)
    true_edges = set(truth["scorable"])
    frame["abs_score"] = frame.score.abs()
    frame["true"] = [(s, t) in true_edges for s, t in zip(frame.source, frame.target)]
    edge = rank_metrics(frame.abs_score, frame["true"])
    # Node level: promote to sources; a source is true if any true edge leaves it.
    node = frame.groupby("source").agg(abs_score=("abs_score", "max"), true=("true", "any")).reset_index()
    node_metrics = rank_metrics(node.abs_score, node["true"])
    return frame, node, dict(edge=edge, node=node_metrics,
                             truth_scorable=len(truth["scorable"]), truth_internal=len(truth["internal"]),
                             truth_out_of_universe=len(truth["out_of_universe"]),
                             out_of_universe_edges=[list(e) for e in truth["out_of_universe"]],
                             universe_size=len(frame))

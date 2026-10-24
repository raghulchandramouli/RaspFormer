"""Blind program recovery from a transformer and its circuit edges.

Nothing here reads the program. The inputs are the executable model, the
circuit edges (which heads and MLPs matter and who feeds whom), and sampled
token sequences. The output is an executable table program:

1. Every circuit node's output is discretised. Fine clusters of its output
   vectors are merged into causal classes: two clusters merge when replacing
   one by the other at that node never changes the model's answer. The class
   id per (sequence, position) is the node's recovered variable.
2. An MLP node becomes a lookup table from its parents' classes to its class.
3. A head becomes a selector table over (query parents, key parents) read off
   its binarised attention pattern, plus an output table from the multiset of
   value-parent classes at attended keys to its class.
4. The final residual read is a table from the classes of the circuit nodes
   feeding the output to the predicted token.

Tables are fitted on calibration sequences and evaluated on held-out ones;
an input combination never seen is reported as undefined rather than guessed.
"""

import collections
import itertools
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from bench.eap import _order
from raspformer.models import ACTIVATIONS

TOKENS, INDICES = "tokens", "indices"


# --- Forward pass exposing circuit nodes --------------------------------------

def circuit_forward(model, inputs, node_patches=None, slot_deltas=None):
    """Logits, node outputs, and attention patterns; node outputs may be replaced.

    ``node_patches`` maps a source node name to a [batch, position, width]
    replacement and a [batch, position] boolean mask: where the mask is set the
    node's output is replaced before it enters the residual stream.
    ``slot_deltas`` maps (node, slot) with slot in q/k/v/in to a
    [batch, position, width] array added only to that node's slot input.
    """
    cfg, p = model.config, model.params
    act = ACTIVATIONS[cfg.activation]
    node_patches = node_patches or {}
    slot_deltas = slot_deltas or {}
    batch, length = inputs.shape

    def slot_input(x, name, slot):
        return x + slot_deltas[(name, slot)] if (name, slot) in slot_deltas else x

    def emit(name, value):
        if name in node_patches:
            replacement, mask = node_patches[name]
            value = jnp.where(mask[..., None], replacement, value)
        outputs[name] = value
        return value

    outputs, patterns = {}, {}
    x = emit("hook_embed", p["token_embed"][inputs]) + emit(
        "hook_pos_embed", jnp.broadcast_to(p["pos_embed"][:length], (batch, length, cfg.width)))
    mask = jnp.ones((length, length), dtype=bool)
    if cfg.causal:
        mask = jnp.tril(mask)
    for l, layer in enumerate(p["layers"]):
        head_sum = 0
        for h in range(cfg.heads):
            sl = slice(h * cfg.key_size, (h + 1) * cfg.key_size)
            name = f"blocks.{l}.attn.hook_result[{h}]"
            q = slot_input(x, name, "q") @ layer["q"]["w"][:, sl] + layer["q"]["b"][sl]
            k = slot_input(x, name, "k") @ layer["k"]["w"][:, sl] + layer["k"]["b"][sl]
            v = slot_input(x, name, "v") @ layer["v"]["w"][:, sl] + layer["v"]["b"][sl]
            scores = jnp.einsum("btd,bsd->bts", q, k) / np.sqrt(cfg.key_size)
            weights = jax.nn.softmax(jnp.where(mask, scores, -1e30), axis=-1)
            patterns[f"blocks.{l}.attn.hook_result[{h}]"] = weights
            head_sum = head_sum + emit(f"blocks.{l}.attn.hook_result[{h}]",
                                       jnp.einsum("bts,bsd->btd", weights, v) @ layer["o"]["w"][sl, :])
        x = x + head_sum + layer["o"]["b"]
        mlp_name = f"blocks.{l}.hook_mlp_out"
        mlp_in = slot_input(x, mlp_name, "in")
        x = x + emit(mlp_name, act(mlp_in @ layer["mlp_in"]["w"] + layer["mlp_in"]["b"]) @ layer["mlp_out"]["w"] + layer["mlp_out"]["b"])
    logits = slot_input(x, "hook_resid_post", "out") @ p["unembed"]
    if "unembed_bias" in p:
        logits = logits + p["unembed_bias"]
    return logits, outputs, patterns


# --- Circuit structure from edges ---------------------------------------------

@dataclass
class Circuit:
    nodes: list                       # circuit heads and MLPs in residual order
    parents: dict                     # node -> {"q": [...], "k": [...], "v": [...]} or {"in": [...]}
    output_parents: list              # sources feeding the final residual read

    @staticmethod
    def from_edges(path, config):
        with open(Path(path) / "edges.pkl", "rb") as handle:
            edges = [tuple(e) for e in pickle.load(handle)]
        parents = collections.defaultdict(lambda: collections.defaultdict(list))
        output_parents = []
        for src, dst in edges:
            if not (src.startswith("hook_") or "hook_result" in src or src.endswith("hook_mlp_out")):
                continue  # internal edges (q_input -> hook_q, ...) carry no structure
            m = re.fullmatch(r"blocks\.(\d+)\.hook_([qkv])_input\[(\d+)\]", dst)
            if m:
                parents[f"blocks.{m.group(1)}.attn.hook_result[{m.group(3)}]"][m.group(2)].append(src)
                continue
            m = re.fullmatch(r"blocks\.(\d+)\.hook_mlp_in", dst)
            if m:
                parents[f"blocks.{m.group(1)}.hook_mlp_out"]["in"].append(src)
                continue
            if dst.endswith("hook_resid_post"):
                output_parents.append(src)
        order = {}
        for l in range(config.layers):
            for h in range(config.heads):
                order[f"blocks.{l}.attn.hook_result[{h}]"] = 2 * l + 1
            order[f"blocks.{l}.hook_mlp_out"] = 2 * l + 2
        nodes = sorted({n for e in edges for n in e if n in order}, key=order.get)
        return Circuit(nodes, {n: dict(parents[n]) for n in nodes}, sorted(set(output_parents), key=order.get))


# --- Discretisation by causal quotient ----------------------------------------

def fine_clusters(vectors, threshold=None, max_clusters=64):
    """Average-linkage clusters of standardised vectors; labels start at 0.

    By default the dendrogram is cut into at most ``max_clusters`` clusters,
    which over-splits every variable seen so far while staying cheap; the
    causal signature step then merges what the model cannot distinguish.
    """
    x = np.asarray(vectors, dtype=np.float64)
    if len(x) < 2:
        return np.zeros(len(x), dtype=int)
    std = x.std(0)
    keep = std > 1e-6 * max(std.max(), 1e-12)
    if not keep.any():
        return np.zeros(len(x), dtype=int)
    z = (x[:, keep] - x[:, keep].mean(0)) / std[keep]
    z = z / np.sqrt(keep.sum())          # unit expected norm, so a distance threshold is relative
    tree = linkage(z, method="average")
    if threshold is None:
        labels = fcluster(tree, t=min(max_clusters, len(x)), criterion="maxclust") - 1
    else:
        labels = fcluster(tree, t=threshold, criterion="distance") - 1
    return labels


def _balanced_contexts(labels, count, rng):
    """Indices into a flat [batch*(length-1)] array, sampled so that every class
    of ``labels`` is equally represented; rare classes are otherwise invisible."""
    labels = np.asarray(labels).reshape(-1)
    counts = np.bincount(np.maximum(labels, 0))
    weights = 1.0 / np.maximum(counts[np.maximum(labels, 0)], 1)
    weights = weights / weights.sum()
    return rng.choice(len(labels), size=min(count, len(labels)), replace=False, p=weights)



def causal_classes(model, inputs, node, outputs, labels, *, contexts=96, tolerance=0.15, min_size=2, seed=0):
    """Merge clusters whose outputs are interchangeable for the model's answer.

    One shared set of random (sequence, position) contexts is drawn. For every
    fine cluster, its centroid replaces the node output at those contexts and
    the predicted tokens form the cluster's signature. Clusters with identical
    signatures are one causal class: the downstream circuit cannot tell them
    apart anywhere in the sample. Returns [batch, positions] class ids, the
    centroids, and the cluster-to-class map.
    """
    batch, length, width = outputs.shape
    flat = outputs[:, 1:, :].reshape(-1, width)
    labels = np.asarray(labels)
    # Fine clusters smaller than ``min_size`` are outliers whose centroids sit
    # off-distribution; fold them into the nearest larger cluster first.
    sizes = np.bincount(labels)
    large = np.flatnonzero(sizes >= min_size)
    if len(large) and len(large) < len(sizes):
        big_centroids = np.stack([flat[labels == c].mean(0) for c in large])
        for c in np.flatnonzero(sizes < min_size):
            members = labels == c
            nearest = large[np.argmin(((flat[members].mean(0) - big_centroids) ** 2).sum(-1))]
            labels[members] = nearest
        remap = {c: i for i, c in enumerate(sorted(set(labels)))}
        labels = np.asarray([remap[c] for c in labels])
    k = labels.max() + 1
    centroids = np.stack([flat[labels == c].mean(0) for c in range(k)])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat), size=min(contexts, len(flat)), replace=False)
    rows, cols = idx // (length - 1), idx % (length - 1) + 1
    mask = np.zeros((len(rows), length), dtype=bool)
    mask[np.arange(len(rows)), cols] = True
    context_inputs, mask = inputs[rows], jnp.asarray(mask)
    signatures = []
    for c in range(k):
        replacement = jnp.asarray(np.broadcast_to(centroids[c], (len(rows), length, width)))
        patched, _, _ = circuit_forward(model, context_inputs, {node: (replacement, mask)})
        signatures.append(np.asarray(patched).argmax(-1)[np.arange(len(rows)), cols])
    signatures = np.stack(signatures)            # [cluster, context]: output at the patched position
    if k == 1:
        classes = np.zeros(1, dtype=int)
    else:
        # Distance = share of contexts where the output at the patched position
        # differs between the two insertions; clusters within ``tolerance``
        # (average linkage) are one class.
        hamming = (signatures[:, None, :] != signatures[None, :, :]).mean(-1)
        condensed = hamming[np.triu_indices(k, 1)]
        classes = fcluster(linkage(condensed, method="average"), t=tolerance, criterion="distance") - 1
    cluster_to_class = {c: int(classes[c]) for c in range(k)}
    merged = np.asarray([cluster_to_class[c] for c in labels])
    return merged.reshape(batch, length - 1), centroids, cluster_to_class, labels


# --- Recovered table program --------------------------------------------------

BOS_CLASS = -2      # class of any node at the BOS position; -1 means undefined
EMBEDS = ("hook_embed", "hook_pos_embed")


def _majority(counter):
    value, count = counter.most_common(1)[0]
    return value, count / sum(counter.values())


@dataclass
class NodeVariable:
    centroids: np.ndarray        # fine-cluster centroids used to label new outputs
    cluster_to_class: dict
    classes: int

    def assign(self, vectors):
        """Class id of new output vectors by nearest fine centroid."""
        x = np.asarray(vectors, dtype=np.float64)
        nearest = np.argmin(((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1), axis=1)
        return np.asarray([self.cluster_to_class[int(c)] for c in nearest])


@dataclass
class Recovered:
    """An executable table program.

    Heads are either ``select`` ops (selector table over query and key parent
    features, output table over the attended count and attended value
    features) or ``bag`` ops when the attention pattern is not crisp (output
    table over the query features and the multiset of key features over the
    whole sequence). The op kind is recorded per head.
    """
    circuit: Circuit
    parents: dict = field(default_factory=dict)        # node -> {"q"/"k"/"v"/"in": [...]}
    output_parents: list = field(default_factory=list)
    variables: dict = field(default_factory=dict)      # node -> NodeVariable
    kinds: dict = field(default_factory=dict)          # node -> "mlp" | "select" | "bag"
    mlp_tables: dict = field(default_factory=dict)     # node -> {parent tuple: (class, purity)}
    selectors: dict = field(default_factory=dict)      # head -> {(q tuple, k tuple): (bool, purity)}
    head_tables: dict = field(default_factory=dict)    # head -> {(n attended, value tuple): (class, purity)}
    aggregate: dict = field(default_factory=dict)      # head -> "count" (key = n) or "values" (key = n, values)
    bag_tables: dict = field(default_factory=dict)     # head -> {(q tuple, key multiset): (class, purity)}
    output_table: dict = field(default_factory=dict)   # {parent tuple: (token id, purity)}
    crispness: dict = field(default_factory=dict)      # head -> attention crispness in [0, 1]
    effects: dict = field(default_factory=dict)        # (parent, node, slot) -> class-change fraction

    def run(self, token_ids, patch=None):
        """Execute on one encoded sequence (BOS first); ``patch`` = (node, position,
        class) overrides one node. Returns per-position token ids without BOS
        (-1 where undefined) and the per-node class trace."""
        token_ids = np.asarray(token_ids)
        length = len(token_ids)
        classes = {}

        def feat(node, pos):
            if node == "hook_embed":
                return int(token_ids[pos])
            if node == "hook_pos_embed":
                return int(pos)
            return classes[node][pos]

        for node in self.circuit.nodes:
            parents = self.parents[node]
            values = [BOS_CLASS] + [-1] * (length - 1)
            kind = self.kinds[node]
            for pos in range(1, length):
                if kind == "mlp":
                    key = tuple(feat(n, pos) for n in parents.get("in", []))
                    entry = self.mlp_tables[node].get(key) if -1 not in key else None
                elif kind == "bag":
                    qkey = tuple(feat(n, pos) for n in parents.get("q", []))
                    bag = tuple(sorted(tuple(feat(n, kp) for n in parents.get("k", [])) for kp in range(length)))
                    entry = self.bag_tables[node].get((qkey, bag)) if -1 not in qkey and all(-1 not in b for b in bag) else None
                else:
                    qkey = tuple(feat(n, pos) for n in parents.get("q", []))
                    if -1 in qkey:
                        continue
                    attended = []
                    for key_pos in range(length):
                        kkey = tuple(feat(n, key_pos) for n in parents.get("k", []))
                        sel = self.selectors[node].get((qkey, kkey)) if -1 not in kkey else None
                        if sel is None:
                            attended = None
                            break
                        if sel[0]:
                            attended.append(key_pos)
                    if attended is None:
                        continue
                    vals = tuple(sorted(feat(n, kp) for kp in attended for n in parents.get("v", [])))
                    if self.aggregate.get(node, "values") == "count":
                        vals = ()
                    entry = self.head_tables[node].get((len(attended), vals)) if -1 not in vals else None
                values[pos] = entry[0] if entry else -1
            if patch is not None and patch[0] == node:
                values[patch[1]] = patch[2]
            classes[node] = values
        output = [-1] * length
        for pos in range(1, length):
            key = tuple(feat(n, pos) for n in self.output_parents)
            entry = self.output_table.get(key) if -1 not in key else None
            output[pos] = entry[0] if entry else -1
        return np.asarray(output[1:]), classes


# --- Attention reading ----------------------------------------------------------

ATTENTION_FLOOR = 0.01


def attended_keys(row, threshold=0.5, floor=ATTENTION_FLOOR):
    """Keys whose weight is at least ``threshold`` of the largest non-BOS weight
    and above an absolute ``floor`` (softmax leaves tiny positive weights, not
    zeros); BOS is attended when its weight is at least ``threshold`` of the
    overall maximum."""
    rest = row[1:]
    top = rest.max()
    attended = list(np.flatnonzero((rest >= threshold * top) & (rest >= floor)) + 1) if top >= floor else []
    if row[0] >= threshold * row.max():
        attended = [0] + attended
    return np.asarray(attended, dtype=int)


def crispness(pattern):
    """Mean over queries of the gap between attended and unattended non-BOS
    weights relative to the largest; 1 for hard attention, 0 for uniform."""
    gaps = []
    for b in range(pattern.shape[0]):
        for pos in range(1, pattern.shape[1]):
            rest = pattern[b, pos, 1:]
            top = rest.max()
            if top < ATTENTION_FLOOR:
                continue
            on = rest[rest >= 0.5 * top]
            off = rest[rest < 0.5 * top]
            gaps.append((on.min() - (off.max() if len(off) else 0.0)) / top)
    return float(np.mean(gaps)) if gaps else 0.0


# --- Table fitting --------------------------------------------------------------

def _fit_mlp(feat, node, parents, rows, length):
    table = collections.defaultdict(collections.Counter)
    for b in rows:
        for pos in range(1, length):
            key = tuple(feat(n, b, pos) for n in parents)
            if -1 not in key:
                table[key][feat(node, b, pos)] += 1
    return table


def _fit_select(feat, node, parents, patterns, rows, length, threshold, aggregate="values"):
    selector = collections.defaultdict(collections.Counter)
    table = collections.defaultdict(collections.Counter)
    for b in rows:
        for pos in range(1, length):
            qkey = tuple(feat(n, b, pos) for n in parents.get("q", []))
            if -1 in qkey:
                continue
            attended = attended_keys(patterns[b, pos], threshold)
            for key_pos in range(length):
                kkey = tuple(feat(n, b, key_pos) for n in parents.get("k", []))
                if -1 not in kkey:
                    selector[(qkey, kkey)][bool(key_pos in attended)] += 1
            vals = tuple(sorted(feat(n, b, kp) for kp in attended for n in parents.get("v", [])))
            if aggregate == "count":
                vals = ()
            if -1 not in vals:
                table[(len(attended), vals)][feat(node, b, pos)] += 1
    return selector, table


def _fit_bag(feat, node, parents, rows, length):
    table = collections.defaultdict(collections.Counter)
    for b in rows:
        bag = tuple(sorted(tuple(feat(n, b, kp) for n in parents.get("k", [])) for kp in range(length)))
        if any(-1 in item for item in bag):
            continue
        for pos in range(1, length):
            qkey = tuple(feat(n, b, pos) for n in parents.get("q", []))
            if -1 not in qkey:
                table[(qkey, bag)][feat(node, b, pos)] += 1
    return table


def _execute_select(feat, node, parents, selector, table, rows, length, aggregate="values"):
    """Held-out execution of a select head; returns (correct, total)."""
    correct = total = 0
    for b in rows:
        for pos in range(1, length):
            total += 1
            qkey = tuple(feat(n, b, pos) for n in parents.get("q", []))
            if -1 in qkey:
                continue
            attended = []
            for key_pos in range(length):
                kkey = tuple(feat(n, b, key_pos) for n in parents.get("k", []))
                if -1 in kkey or (qkey, kkey) not in selector:
                    attended = None
                    break
                if selector[(qkey, kkey)]:
                    attended.append(key_pos)
            if attended is None:
                continue
            vals = tuple(sorted(feat(n, b, kp) for kp in attended for n in parents.get("v", [])))
            if aggregate == "count":
                vals = ()
            correct += int(table.get((len(attended), vals)) == feat(node, b, pos))
    return correct, total


def _heldout_select(feat, node, parents, patterns, folds, length, threshold, aggregate="values"):
    correct = total = 0
    for train_rows, test_rows in folds:
        selector, table = _fit_select(feat, node, parents, patterns, train_rows, length, threshold, aggregate)
        c, t = _execute_select(feat, node, parents, {k: v.most_common(1)[0][0] for k, v in selector.items()},
                               {k: v.most_common(1)[0][0] for k, v in table.items()}, test_rows, length, aggregate)
        correct += c
        total += t
    return correct / total if total else 0.0


def _heldout_table(table_fit, folds):
    """Fit on one half, score on the other, both ways; unseen keys are errors."""
    total = correct = 0
    for train_rows, test_rows in folds:
        table = {k: v.most_common(1)[0][0] for k, v in table_fit(train_rows).items()}
        for key, counter in table_fit(test_rows).items():
            total += sum(counter.values())
            correct += counter.get(table.get(key), 0) if key in table else 0
    return correct / total if total else 0.0


# --- Parents by intervention ------------------------------------------------------

def causal_parents(model, inputs, circuit, classes, variables, *, contexts=48,
                   min_change=0.1, relative=0.25, seed=0):
    """Which earlier nodes feed each node's slots, decided by intervention.

    For a candidate edge P -> N.slot, P's contribution to N's slot input is
    swapped for the centroid of a different class of P at random contexts
    (query positions for q and in, key positions for k and v). The edge is
    kept when N's class changes in at least ``min_change`` of the contexts.
    Returns {node: {slot: [parents]}}.
    """
    inputs = np.asarray(inputs)
    batch, length = inputs.shape
    rng = np.random.default_rng(seed)
    _, outputs, _ = circuit_forward(model, jnp.asarray(inputs))
    outputs = {k: np.asarray(v) for k, v in outputs.items()}
    width = model.config.width
    token_embed = np.asarray(model.params["token_embed"])
    pos_embed = np.asarray(model.params["pos_embed"])[:length]

    def class_centroids(p):
        if p == "hook_embed":
            return token_embed, inputs                      # class = token id
        if p == "hook_pos_embed":
            return pos_embed, np.broadcast_to(np.arange(length), (batch, length))
        var = variables[p]
        cents = np.stack([var.centroids[[c for c, cl in var.cluster_to_class.items() if cl == k]].mean(0)
                          for k in range(var.classes)])
        return cents, classes[p]

    result, effects = {}, {}
    for node in circuit.nodes:
        slots = ("in",) if node.endswith("hook_mlp_out") else ("q", "k", "v")
        result[node] = {slot: [] for slot in slots}
        var = variables[node]
        base_class = classes[node]
        node_order = _order(node, model.config.layers)
        earlier = list(EMBEDS) + [n for n in circuit.nodes if _order(n, model.config.layers) < node_order]
        for parent in earlier:
            cents, pclasses = class_centroids(parent)
            if len(cents) < 2:
                continue
            for slot in slots:
                idx = _balanced_contexts(base_class[:, 1:], contexts, rng)
                rows, cols = idx // (length - 1), idx % (length - 1) + 1
                delta = np.zeros((len(rows), length, width), dtype=np.float32)
                for i, (b, pos) in enumerate(zip(rows, cols)):
                    current = pclasses[b, pos]
                    choices = [c for c in range(len(cents)) if c != current]
                    new = rng.choice(choices)
                    delta[i, pos] = cents[new] - outputs[parent][b, pos]
                _, patched, _ = circuit_forward(model, jnp.asarray(inputs[rows]), slot_deltas={(node, slot): jnp.asarray(delta)})
                new_classes = var.assign(np.asarray(patched[node])[:, 1:, :].reshape(-1, width)).reshape(len(rows), length - 1)
                if slot in ("in", "q"):
                    changed = new_classes[np.arange(len(rows)), cols - 1] != base_class[rows, cols]
                else:
                    changed = (new_classes != base_class[rows][:, 1:]).any(axis=1)
                effects[(parent, node, slot)] = float(changed.mean())
            if parent in EMBEDS:
                # Embeddings may reach the node through nodes outside the circuit
                # (inert heads that copy position, for instance). Swap the
                # embedding itself and let the change propagate along every path.
                idx = _balanced_contexts(base_class[:, 1:], contexts, rng)
                rows, cols = idx // (length - 1), idx % (length - 1) + 1
                replacement = outputs[parent][rows].copy()
                mask = np.zeros((len(rows), length), dtype=bool)
                for i, (b, pos) in enumerate(zip(rows, cols)):
                    new = rng.choice([c for c in range(len(cents)) if c != pclasses[b, pos]])
                    replacement[i, pos] = cents[new]
                    mask[i, pos] = True
                _, patched, _ = circuit_forward(model, jnp.asarray(inputs[rows]), {parent: (jnp.asarray(replacement), jnp.asarray(mask))})
                new_classes = var.assign(np.asarray(patched[node])[:, 1:, :].reshape(-1, width)).reshape(len(rows), length - 1)
                effects[(parent, node, "any")] = float((new_classes != base_class[rows][:, 1:]).any(axis=1).mean())
        for slot in slots:
            scores = {par: effects[(par, node, slot)] for par in earlier if (par, node, slot) in effects}
            top = max(scores.values(), default=0.0)
            result[node][slot] = [par for par, f in scores.items() if f >= max(min_change, relative * top)]
        for parent in EMBEDS:
            total = effects.get((parent, node, "any"), 0.0)
            direct = max(effects.get((parent, node, slot), 0.0) for slot in slots)
            if total >= min_change and direct < min_change:
                # Reaches the node only indirectly: attach it to every slot that
                # has no direct parent of its own, so the dependence is kept.
                for slot in slots:
                    if parent not in result[node][slot]:
                        result[node][slot].append(parent)
    return result, effects


# --- Table minimisation -------------------------------------------------------------

def _compatible(rec, node, c1, c2):
    """True when no fitted table distinguishes classes c1 and c2 of ``node`` on
    any pair of otherwise identical keys, and at least one such pair exists."""
    evidence = False

    def check(table, keys_with):
        nonlocal evidence
        for key, entry in table.items():
            for alt in keys_with(key):
                if alt in table:
                    evidence = True
                    if table[alt][0] != entry[0]:
                        return False
        return True

    def tuple_subs(key, slots):
        """Alternatives of a flat key with c1 -> c2 at the node's slots."""
        out = []
        for i in slots:
            if key[i] == c1:
                out.append(key[:i] + (c2,) + key[i + 1:])
        return out

    for other in rec.circuit.nodes:
        parents = rec.parents[other]
        kind = rec.kinds[other]
        if kind == "mlp":
            slots = [i for i, p in enumerate(parents.get("in", [])) if p == node]
            if slots and not check(rec.mlp_tables[other], lambda key: tuple_subs(key, slots)):
                return False
        elif kind == "select":
            qs = [i for i, p in enumerate(parents.get("q", [])) if p == node]
            ks = [i for i, p in enumerate(parents.get("k", [])) if p == node]
            vs = node in parents.get("v", [])
            if qs or ks:
                def alts(key):
                    q, k = key
                    return [(a, k) for a in tuple_subs(q, qs)] + [(q, a) for a in tuple_subs(k, ks)]
                if not check(rec.selectors[other], alts):
                    return False
            if vs:
                def alts_v(key):
                    n, vals = key
                    out = []
                    for i, v in enumerate(vals):
                        if v == c1:
                            out.append((n, tuple(sorted(vals[:i] + (c2,) + vals[i + 1:]))))
                    return out
                if not check(rec.head_tables[other], alts_v):
                    return False
        else:
            qs = [i for i, p in enumerate(parents.get("q", [])) if p == node]
            ks = [i for i, p in enumerate(parents.get("k", [])) if p == node]
            def alts_b(key):
                q, bag = key
                out = [(a, bag) for a in tuple_subs(q, qs)]
                for j, item in enumerate(bag):
                    for a in tuple_subs(item, ks):
                        out.append((q, tuple(sorted(bag[:j] + (a,) + bag[j + 1:]))))
                return out
            if (qs or ks) and not check(rec.bag_tables[other], alts_b):
                return False
    slots = [i for i, p in enumerate(rec.output_parents) if p == node]
    if slots and not check(rec.output_table, lambda key: tuple_subs(key, slots)):
        return False
    return evidence


def minimise_classes(rec, classes):
    """Merge classes of each node that no downstream table distinguishes.
    Returns a {node: {old class: new class}} map; tables must be refitted."""
    maps = {}
    for node in rec.circuit.nodes:
        var = rec.variables[node]
        groups = []
        for c in range(var.classes):
            for g in groups:
                if all(_compatible(rec, node, c, m) for m in g):
                    g.append(c)
                    break
            else:
                groups.append([c])
        maps[node] = {c: i for i, g in enumerate(groups) for c in g}
    return maps


# --- Recovery driver -------------------------------------------------------------------

def _fit_all(model, circuit, inputs, classes, variables, parents, predicted, *, attention_threshold, patterns):
    """Fit every table given fixed classes and parents; choose select vs bag per head."""
    inputs = np.asarray(inputs)
    batch, length = inputs.shape
    half = batch // 2
    folds = [(range(0, half), range(half, batch)), (range(half, batch), range(0, half))]
    rows = range(batch)

    def feat(node, b, pos):
        if node == "hook_embed":
            return int(inputs[b, pos])
        if node == "hook_pos_embed":
            return int(pos)
        return int(classes[node][b, pos])

    rec = Recovered(circuit, parents=parents, variables=variables)
    for node in circuit.nodes:
        ps = parents[node]
        if node.endswith("hook_mlp_out"):
            rec.kinds[node] = "mlp"
            rec.mlp_tables[node] = {k: _majority(v) for k, v in _fit_mlp(feat, node, ps.get("in", []), rows, length).items()}
            continue
        rec.crispness[node] = crispness(patterns[node])
        aggregates = {}
        if ps.get("k") or ps.get("q"):
            for aggregate in ("count", "values"):
                aggregates[aggregate] = _heldout_select(feat, node, ps, patterns[node], folds, length, attention_threshold, aggregate)
        aggregate, select_score = max(aggregates.items(), key=lambda kv: kv[1]) if aggregates else ("values", 0.0)
        bag_parents = {"q": ps.get("q", []), "k": sorted(set(ps.get("k", [])) | set(ps.get("v", [])))}
        bag_score = _heldout_table(lambda r: _fit_bag(feat, node, bag_parents, r, length), folds) if bag_parents["k"] else 0.0
        if select_score >= bag_score:
            rec.kinds[node] = "select"
            rec.aggregate[node] = aggregate
            selector, table = _fit_select(feat, node, ps, patterns[node], rows, length, attention_threshold, aggregate)
            rec.selectors[node] = {k: _majority(v) for k, v in selector.items()}
            rec.head_tables[node] = {k: _majority(v) for k, v in table.items()}
        else:
            rec.kinds[node] = "bag"
            rec.parents[node] = bag_parents
            rec.bag_tables[node] = {k: _majority(v) for k, v in _fit_bag(feat, node, bag_parents, rows, length).items()}
    classes["__output__"] = predicted
    rec.output_table = {k: _majority(v) for k, v in _fit_mlp(feat, "__output__", rec.output_parents, rows, length).items()}
    del classes["__output__"]
    return rec


def recover(model, circuit, inputs, *, max_clusters=128, tolerance=0.15,
            attention_threshold=0.5, minimise=True, seed=0):
    """Fit a Recovered program from the model's behaviour on ``inputs``.

    The circuit supplies only the set of nodes. Wiring comes from
    interventions (``causal_parents``), classes from causal signatures, and
    classes are then minimised against the fitted tables and refitted.
    """
    inputs = np.asarray(inputs)
    logits, outputs, patterns = circuit_forward(model, jnp.asarray(inputs))
    outputs = {k: np.asarray(v) for k, v in outputs.items()}
    patterns = {k: np.asarray(v) for k, v in patterns.items()}
    predicted = np.asarray(logits).argmax(-1)
    batch, length = inputs.shape
    variables, classes = {}, {}
    for node in circuit.nodes:
        out = outputs[node]
        fine = fine_clusters(out[:, 1:, :].reshape(-1, out.shape[-1]), max_clusters=max_clusters)
        merged, centroids, cluster_to_class, _ = causal_classes(
            model, jnp.asarray(inputs), node, out, fine, tolerance=tolerance, seed=seed)
        variables[node] = NodeVariable(centroids, cluster_to_class, max(cluster_to_class.values()) + 1)
        classes[node] = np.concatenate([np.full((batch, 1), BOS_CLASS), merged], axis=1)
    parents, effects = causal_parents(model, inputs, circuit, classes, variables, seed=seed)
    # Output parents: nodes whose direct contribution to the final residual matters.
    output_parents, output_effects = _output_parents(model, inputs, circuit, classes, variables, predicted, seed=seed)
    effects.update(output_effects)
    rec = _fit_all(model, circuit, inputs, classes, variables, parents, predicted,
                   attention_threshold=attention_threshold, patterns=patterns)
    rec.output_parents = output_parents
    rec = _fit_all(model, circuit, inputs, classes, variables, rec.parents, predicted,
                   attention_threshold=attention_threshold, patterns=patterns)
    rec.output_parents = output_parents
    rec.output_table = _refit_output(rec, inputs, classes, predicted)
    if minimise:
        for _ in range(2):
            maps = minimise_classes(rec, classes)
            if all(len(set(m.values())) == len(m) for m in maps.values()):
                break
            for node, m in maps.items():
                var = rec.variables[node]
                var.cluster_to_class = {c: m[k] for c, k in var.cluster_to_class.items()}
                var.classes = len(set(m.values()))
                cl = classes[node]
                classes[node] = np.where(cl >= 0, np.vectorize(lambda c: m.get(c, c))(np.maximum(cl, 0)), cl)
            rec = _fit_all(model, circuit, inputs, classes, variables, rec.parents, predicted,
                           attention_threshold=attention_threshold, patterns=patterns)
            rec.output_parents = output_parents
            rec.output_table = _refit_output(rec, inputs, classes, predicted)
    rec.effects = effects
    return rec, classes


def _refit_output(rec, inputs, classes, predicted):
    batch, length = inputs.shape

    def feat(node, b, pos):
        if node == "hook_embed":
            return int(inputs[b, pos])
        if node == "hook_pos_embed":
            return int(pos)
        return int(classes[node][b, pos])

    classes["__output__"] = predicted
    table = {k: _majority(v) for k, v in _fit_mlp(feat, "__output__", rec.output_parents, range(batch), length).items()}
    del classes["__output__"]
    return table


def _output_parents(model, inputs, circuit, classes, variables, predicted, *, contexts=48, min_change=0.1, relative=0.25, seed=0):
    """Circuit nodes (and embeddings) whose output intervention changes the token."""
    inputs = np.asarray(inputs)
    batch, length = inputs.shape
    width = model.config.width
    rng = np.random.default_rng(seed)
    _, outputs, _ = circuit_forward(model, jnp.asarray(inputs))
    outputs = {k: np.asarray(v) for k, v in outputs.items()}
    parents, effects = [], {}
    for node in list(EMBEDS) + circuit.nodes:
        if node == "hook_embed":
            cents, pclasses = np.asarray(model.params["token_embed"]), inputs
        elif node == "hook_pos_embed":
            cents, pclasses = np.asarray(model.params["pos_embed"])[:length], np.broadcast_to(np.arange(length), (batch, length))
        else:
            var = variables[node]
            cents = np.stack([var.centroids[[c for c, cl in var.cluster_to_class.items() if cl == k]].mean(0) for k in range(var.classes)])
            pclasses = classes[node]
        if len(cents) < 2:
            continue
        idx = _balanced_contexts(predicted[:, 1:], contexts, rng)
        rows, cols = idx // (length - 1), idx % (length - 1) + 1
        delta = np.zeros((len(rows), length, width), dtype=np.float32)
        for i, (b, pos) in enumerate(zip(rows, cols)):
            new = rng.choice([c for c in range(len(cents)) if c != pclasses[b, pos]])
            delta[i, pos] = cents[new] - outputs[node][b, pos]
        patched, _, _ = circuit_forward(model, jnp.asarray(inputs[rows]), slot_deltas={("hook_resid_post", "out"): jnp.asarray(delta)})
        changed = np.asarray(patched).argmax(-1)[np.arange(len(rows)), cols] != predicted[rows, cols]
        effects[(node, "hook_resid_post", "out")] = float(changed.mean())
    top = max(effects.values(), default=0.0)
    parents = [node for node in list(EMBEDS) + circuit.nodes
               if effects.get((node, "hook_resid_post", "out"), 0.0) >= max(min_change, relative * top)]
    return parents, effects


def full_circuit(config):
    """Every head and MLP with every earlier node as a parent (compiled models)."""
    nodes, parents, earlier = [], {}, ["hook_embed", "hook_pos_embed"]
    for l in range(config.layers):
        heads = [f"blocks.{l}.attn.hook_result[{h}]" for h in range(config.heads)]
        for head in heads:
            parents[head] = {"q": list(earlier), "k": list(earlier), "v": list(earlier)}
        nodes += heads
        earlier = earlier + heads
        mlp = f"blocks.{l}.hook_mlp_out"
        parents[mlp] = {"in": list(earlier)}
        nodes.append(mlp)
        earlier = earlier + [mlp]
    return Circuit(nodes, parents, list(earlier))


# --- The three tests ------------------------------------------------------------

def test_outputs(rec, model, inputs):
    """Agreement between the recovered program and the model on new inputs."""
    inputs = np.asarray(inputs)
    predicted = np.asarray(circuit_forward(model, jnp.asarray(inputs))[0]).argmax(-1)[:, 1:]
    agree, defined = [], []
    for b in range(len(inputs)):
        out, _ = rec.run(inputs[b])
        defined.append(out >= 0)
        agree.append((out == predicted[b]))
    agree, defined = np.asarray(agree), np.asarray(defined)
    return dict(token_agreement=float((agree & defined).mean()),
                token_agreement_where_defined=float(agree[defined].mean()) if defined.any() else None,
                coverage=float(defined.mean()),
                sequence_agreement=float((agree & defined).all(1).mean()))


def test_counterfactuals(rec, model, inputs, *, pairs=128, seed=0):
    """Swap one circuit node's output between two inputs at one position, in the
    model (activation patch) and in the recovered program (class patch), and
    compare the resulting outputs. Reported per node."""
    inputs = np.asarray(inputs)
    rng = np.random.default_rng(seed)
    logits, outputs, _ = circuit_forward(model, jnp.asarray(inputs))
    outputs = {k: np.asarray(v) for k, v in outputs.items()}
    base_predicted = np.asarray(logits).argmax(-1)[:, 1:]
    batch, length = inputs.shape
    rows = []
    for node in rec.circuit.nodes:
        var = rec.variables[node]
        node_classes = var.assign(outputs[node][:, 1:, :].reshape(-1, outputs[node].shape[-1])).reshape(batch, length - 1)
        agree = changed_model = changed_program = defined = 0
        for _ in range(pairs):
            b, s, pos = rng.integers(batch), rng.integers(batch), rng.integers(1, length)
            if b == s or node_classes[b, pos - 1] == node_classes[s, pos - 1]:
                continue
            mask = np.zeros((1, length), dtype=bool)
            mask[0, pos] = True
            replacement = np.broadcast_to(outputs[node][s], (1, length, outputs[node].shape[-1]))
            patched, _, _ = circuit_forward(model, jnp.asarray(inputs[b:b + 1]),
                                            {node: (jnp.asarray(replacement), jnp.asarray(mask))})
            model_out = np.asarray(patched).argmax(-1)[0, 1:]
            program_out, _ = rec.run(inputs[b], patch=(node, pos, int(node_classes[s, pos - 1])))
            if (program_out < 0).any():
                continue
            defined += 1
            agree += int((program_out == model_out).all())
            changed_model += int((model_out != base_predicted[b]).any())
            changed_program += int((program_out != base_predicted[b]).any())
        rows.append(dict(node=node, pairs=defined, agreement=agree / defined if defined else None,
                         model_changed=changed_model / defined if defined else None,
                         program_changed=changed_program / defined if defined else None))
    return rows


def test_ablation(model, circuit, calibration, test):
    """Mean-ablate every head and MLP outside the circuit; report agreement."""
    _, cal_outputs, _ = circuit_forward(model, jnp.asarray(calibration))
    base = np.asarray(circuit_forward(model, jnp.asarray(test))[0]).argmax(-1)[:, 1:]
    all_nodes = full_circuit(model.config).nodes
    outside = [n for n in all_nodes if n not in circuit.nodes]
    length = np.asarray(test).shape[1]
    patches = {}
    for node in outside:
        mean = np.asarray(cal_outputs[node]).mean(axis=(0, 1))
        patches[node] = (jnp.asarray(np.broadcast_to(mean, (len(test), length, mean.shape[-1]))),
                         jnp.ones((len(test), length), dtype=bool))
    ablated = np.asarray(circuit_forward(model, jnp.asarray(test), patches)[0]).argmax(-1)[:, 1:]
    return dict(outside_nodes=len(outside), circuit_nodes=len(circuit.nodes),
                token_agreement=float((ablated == base).mean()),
                sequence_agreement=float((ablated == base).all(1).mean()))


def match_variables(classes, true_values, threshold=0.95):
    """For each true variable, the best node by mutual determinism.

    ``classes``: node -> [batch, length] class ids (BOS column -1).
    ``true_values``: label -> [batch, length-1] class ids (-1 undefined).
    """
    rows = []
    for label, truth in true_values.items():
        best = None
        for node, cls in classes.items():
            c, t = cls[:, 1:].reshape(-1), np.asarray(truth).reshape(-1)
            keep = (c >= 0) & (t >= 0)
            c, t = c[keep], t[keep]
            if len(c) == 0:
                continue
            c2t = collections.defaultdict(collections.Counter)
            t2c = collections.defaultdict(collections.Counter)
            for a, b in zip(c, t):
                c2t[a][b] += 1
                t2c[b][a] += 1
            det = sum(max(v.values()) for v in c2t.values()) / len(c)
            inv = sum(max(v.values()) for v in t2c.values()) / len(c)
            if best is None or det + inv > best[1] + best[2]:
                best = (node, det, inv)
        node, det, inv = best if best else (None, 0.0, 0.0)
        status = "exact" if det >= threshold and inv >= threshold else "refined" if det >= threshold else "missing"
        rows.append(dict(variable=label, node=node, class_to_value=det, value_to_class=inv, status=status))
    return rows

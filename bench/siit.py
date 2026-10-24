"""Strict Interchange Intervention Training (SIIT) for the RaspFormer programs.

The high-level (HL) model is the compiled Tracr transformer. The low-level
(LL) model is a randomly initialised transformer with the same depth and
four heads per layer. Correspondence is identity on names: HL head (l, h)
maps to LL head (l, h) and HL MLP l to LL MLP l; the remaining LL heads are
outside the circuit. Three losses are minimised (Gupta et al., 2024):

* behaviour: LL output must match the HL output on the base input;
* interchange: swapping one circuit node's activations from a source input
  into the base, in both models, must give the same output as the HL model;
* strictness: swapping any non-circuit LL node must leave the output alone.

Ground-truth edges between circuit nodes are read off the compiled weights:
a node reads a parent when its projection has non-zero weight on lanes that
parent writes. They are saved in InterpBench's ``edges.pkl`` format so the
Phase 1 to 3 tooling runs unchanged.
"""

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bench.decompile import circuit_forward
from raspformer import models
from raspformer.compiler import CompilerConfig, compile_program, extract_schedule, write_json

HEADS = 4
WIDTH = 32


def active_nodes(hl_model, inputs):
    """HL heads and MLPs whose output varies over the inputs (Tracr pads others)."""
    _, outputs, _ = circuit_forward(hl_model, jnp.asarray(inputs))
    nodes = []
    for l in range(hl_model.config.layers):
        for h in range(hl_model.config.heads):
            nodes.append(f"blocks.{l}.attn.hook_result[{h}]")
        nodes.append(f"blocks.{l}.hook_mlp_out")
    return [n for n in nodes if float(np.asarray(outputs[n]).std()) > 1e-9]


def compiled_edges(assembled, hl_model, program, config, circuit_nodes):
    """Edges in InterpBench format from lane overlap in the compiled weights."""
    _, variables, _ = extract_schedule(program, assembled, config)
    labels = list(assembled.residual_labels)
    writer_lanes = {}
    for v in variables:
        stage = v["first_write_half_layer_index"]
        if stage is None:
            name = "hook_embed" if v["variable"] == "tokens" else "hook_pos_embed"
        else:
            l, kind = stage // 2, stage % 2
            name = f"blocks.{l}.hook_mlp_out" if kind else f"blocks.{l}.attn.hook_result[0]"
            if v["compiled_block"] == "attention+mlp":
                # SelectorWidth: attention writes a helper lane, MLP writes the variable.
                name = f"blocks.{l}.hook_mlp_out"
        for lane in v["residual_lane_indices"]:
            writer_lanes.setdefault(lane, set()).add(name)
    # Attention helper lanes of SelectorWidth are written by the head itself.
    p = hl_model.params
    edges = set()
    for l, layer in enumerate(p["layers"]):
        head = f"blocks.{l}.attn.hook_result[0]"
        out_lanes = np.flatnonzero(np.abs(np.asarray(layer["o"]["w"])).sum(0) > 1e-9)
        for lane in out_lanes:
            writer_lanes.setdefault(lane, set()).add(head)
        for slot, key in (("q", "q"), ("k", "k"), ("v", "v")):
            read = np.flatnonzero(np.abs(np.asarray(layer[key]["w"])).sum(1) > 1e-9)
            for lane in read:
                for src in writer_lanes.get(lane, ()):
                    if src != head and (src.startswith("hook_") or src in circuit_nodes):
                        edges.add((src, f"blocks.{l}.hook_{slot}_input[0]"))
        mlp = f"blocks.{l}.hook_mlp_out"
        read = np.flatnonzero(np.abs(np.asarray(layer["mlp_in"]["w"])).sum(1) > 1e-9)
        for lane in read:
            for src in writer_lanes.get(lane, ()):
                if src != mlp and (src.startswith("hook_") or src in circuit_nodes):
                    edges.add((src, f"blocks.{l}.hook_mlp_in"))
        mlp_out_lanes = np.flatnonzero(np.abs(np.asarray(layer["mlp_out"]["w"])).sum(0) > 1e-9)
        for lane in mlp_out_lanes:
            writer_lanes.setdefault(lane, set()).add(mlp)
    unembed_lanes = np.flatnonzero(np.abs(np.asarray(p["unembed"])).sum(1) > 1e-9)
    final = f"blocks.{hl_model.config.layers - 1}.hook_resid_post"
    for lane in unembed_lanes:
        for src in writer_lanes.get(lane, ()):
            if src.startswith("hook_") or src in circuit_nodes:
                edges.add((src, final))
    # Internal node edges, as InterpBench lists them.
    for node in circuit_nodes:
        if "hook_result" in node:
            l = node.split(".")[1]
            for s in "qkv":
                edges.add((f"blocks.{l}.hook_{s}_input[0]", f"blocks.{l}.attn.hook_{s}[0]"))
                edges.add((f"blocks.{l}.attn.hook_{s}[0]", node))
        else:
            l = node.split(".")[1]
            edges.add((f"blocks.{l}.hook_mlp_in", node))
    return sorted(edges)


def hl_labels(hl_model, inputs, patches=None):
    logits, _, _ = circuit_forward(hl_model, jnp.asarray(inputs), patches)
    return np.asarray(logits).argmax(-1)[:, 1:]


def train_siit(hl_model, ll_model, train_inputs, circuit_nodes, *, steps=1500, batch=64,
               lr=1e-3, iit_weight=1.0, strict_weight=0.4, seed=0, log_every=100):
    """Return the trained LL model and a training history."""
    all_nodes = []
    for l in range(ll_model.config.layers):
        all_nodes += [f"blocks.{l}.attn.hook_result[{h}]" for h in range(ll_model.config.heads)]
        all_nodes.append(f"blocks.{l}.hook_mlp_out")
    outside = [n for n in all_nodes if n not in circuit_nodes]
    cfg = ll_model.config
    length = train_inputs.shape[1]

    def ce(logits, labels):
        logp = jax.nn.log_softmax(logits[:, 1:], axis=-1)
        return -jnp.take_along_axis(logp, labels[..., None], axis=-1).mean()

    def loss(params, base, source, base_labels, iit_labels):
        model = models.Model(params, cfg)
        full = jnp.ones((base.shape[0], length), dtype=bool)
        base_logits, _, _ = circuit_forward(model, base)
        _, source_out, _ = circuit_forward(model, source)
        behaviour = ce(base_logits, base_labels)
        iit = 0.0
        for i, node in enumerate(circuit_nodes):
            patched, _, _ = circuit_forward(model, base, {node: (source_out[node], full)})
            iit = iit + ce(patched, iit_labels[i])
        strict = 0.0
        for node in outside:
            patched, _, _ = circuit_forward(model, base, {node: (source_out[node], full)})
            strict = strict + ce(patched, base_labels)
        iit = iit / max(1, len(circuit_nodes))
        strict = strict / max(1, len(outside))
        return behaviour + iit_weight * iit + strict_weight * strict, (behaviour, iit, strict)

    @jax.jit
    def step(params, m, v, t, base, source, base_labels, iit_labels):
        (value, parts), g = jax.value_and_grad(loss, has_aux=True)(params, base, source, base_labels, iit_labels)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        params = jax.tree_util.tree_map(
            lambda q, a, b: q - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), params, m, v)
        return params, m, v, value, parts

    rng = np.random.default_rng(seed)
    params = ll_model.params
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    full = np.ones((batch, length), dtype=bool)
    history = []
    _, hl_outputs_all, _ = circuit_forward(hl_model, jnp.asarray(train_inputs))
    hl_outputs_all = {k: np.asarray(val) for k, val in hl_outputs_all.items()}
    labels_all = hl_labels(hl_model, train_inputs)
    for t in range(1, steps + 1):
        b_idx = rng.integers(0, len(train_inputs), size=batch)
        s_idx = rng.integers(0, len(train_inputs), size=batch)
        base, source = train_inputs[b_idx], train_inputs[s_idx]
        iit_labels = np.stack([hl_labels(hl_model, base, {node: (jnp.asarray(hl_outputs_all[node][s_idx]), jnp.asarray(full))})
                               for node in circuit_nodes]) if circuit_nodes else np.zeros((0, batch, length - 1), dtype=np.int32)
        params, m, v, value, parts = step(params, m, v, t, jnp.asarray(base), jnp.asarray(source),
                                          jnp.asarray(labels_all[b_idx]), jnp.asarray(iit_labels))
        if t % log_every == 0 or t == steps:
            history.append(dict(step=t, loss=float(value), behaviour=float(parts[0]),
                                iit=float(parts[1]), strict=float(parts[2])))
            print(f"  step {t}: loss {float(value):.4f} behaviour {float(parts[0]):.4f} iit {float(parts[1]):.4f} strict {float(parts[2]):.4f}", flush=True)
    return models.Model(params, cfg), history


def evaluate_siit(hl_model, ll_model, inputs, circuit_nodes, *, seed=1):
    """Behaviour accuracy, IIA per circuit node, SIIA per outside node."""
    rng = np.random.default_rng(seed)
    source = inputs[rng.permutation(len(inputs))]
    full = jnp.ones(inputs.shape, dtype=bool)
    labels = hl_labels(hl_model, inputs)
    ll_logits, ll_source_out, _ = circuit_forward(ll_model, jnp.asarray(source))
    _, hl_source_out, _ = circuit_forward(hl_model, jnp.asarray(source))
    base_pred = np.asarray(circuit_forward(ll_model, jnp.asarray(inputs))[0]).argmax(-1)[:, 1:]
    result = dict(token_accuracy=float((base_pred == labels).mean()),
                  sequence_accuracy=float((base_pred == labels).all(1).mean()), iia={}, siia={})
    for node in circuit_nodes:
        hl_patched = hl_labels(hl_model, inputs, {node: (hl_source_out[node], full)})
        ll_patched = np.asarray(circuit_forward(ll_model, jnp.asarray(inputs), {node: (ll_source_out[node], full)})[0]).argmax(-1)[:, 1:]
        result["iia"][node] = float((ll_patched == hl_patched).all(1).mean())
    all_nodes = []
    for l in range(ll_model.config.layers):
        all_nodes += [f"blocks.{l}.attn.hook_result[{h}]" for h in range(ll_model.config.heads)]
        all_nodes.append(f"blocks.{l}.hook_mlp_out")
    for node in all_nodes:
        if node in circuit_nodes:
            continue
        ll_patched = np.asarray(circuit_forward(ll_model, jnp.asarray(inputs), {node: (ll_source_out[node], full)})[0]).argmax(-1)[:, 1:]
        result["siia"][node] = float((ll_patched == labels).all(1).mean())
    result["mean_iia"] = float(np.mean(list(result["iia"].values()))) if result["iia"] else None
    result["mean_siia"] = float(np.mean(list(result["siia"].values()))) if result["siia"] else None
    return result


def sample_sequences(config, count, rng, length=None):
    length = config.max_seq_len if length is None else length
    return np.asarray([[int(rng.choice(config.vocab)) for _ in range(length)] for _ in range(count)])


def train_program(name, program, output, *, config=None, steps=1500, seed=0, train_count=4096, test_count=512,
                  train_length=None):
    """``config.max_seq_len`` sizes the compiled reference and the LL position
    embeddings; ``train_length`` (default the same) is the token count of every
    training and held-out sequence, so a smaller value leaves longer positions
    untrained for length-generalization tests."""
    config = config or CompilerConfig()
    train_length = config.max_seq_len if train_length is None else train_length
    assembled = compile_program(program, config)
    hl_model = models.from_compiled(assembled)
    rng = np.random.default_rng(seed)
    seqs = sample_sequences(config, train_count + test_count, rng, train_length)
    encoded = np.asarray([assembled.input_encoder.encode([config.bos, *s]) for s in seqs], dtype=np.int32)
    train, test = encoded[:train_count], encoded[train_count:]
    circuit_nodes = active_nodes(hl_model, train[:256])
    ll_model = models.random_model(vocab_size=hl_model.params["token_embed"].shape[0], max_length=config.max_seq_len + 1,
                                   output_classes=hl_model.params["unembed"].shape[1], seed=seed, width=WIDTH,
                                   layers=hl_model.config.layers, heads=HEADS)
    print(f"{name}: HL layers {hl_model.config.layers}, circuit nodes {circuit_nodes}", flush=True)
    trained, history = train_siit(hl_model, ll_model, train, circuit_nodes, steps=steps, seed=seed)
    metrics = evaluate_siit(hl_model, trained, test, circuit_nodes)
    folder = Path(output) / name.lower().replace("_", "-")
    folder.mkdir(parents=True, exist_ok=True)
    models.save_model(trained, folder / "model.npz")
    edges = compiled_edges(assembled, hl_model, program, config, circuit_nodes)
    with open(folder / "edges.pkl", "wb") as handle:
        pickle.dump(edges, handle)
    write_json(folder / "siit.json", dict(program=name, circuit_nodes=circuit_nodes, heads=HEADS, width=WIDTH,
                                          layers=hl_model.config.layers, steps=steps, seed=seed,
                                          max_seq_len=config.max_seq_len, train_length=train_length,
                                          train_sequences=train_count, test_sequences=test_count,
                                          history=history, metrics=metrics, edges=len(edges)))
    np.savez_compressed(folder / "sequences.npz", train=train, test=test, tokens=seqs)
    print(f"{name}: accuracy {metrics['token_accuracy']:.3f}, mean IIA {metrics['mean_iia']}, mean SIIA {metrics['mean_siia']}", flush=True)
    return metrics

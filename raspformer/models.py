"""Executable residual transformers with explicit parameters and basis changes."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    width: int
    layers: int
    heads: int
    key_size: int
    mlp_hidden_size: int
    causal: bool = False
    pad_token: int | None = None
    activation: str = "relu"      # "relu" (Tracr) or "gelu" (exact, TransformerLens "gelu")


ACTIVATIONS = {"relu": jax.nn.relu,
               "gelu": lambda x: jax.nn.gelu(x, approximate=False)}


@dataclass(frozen=True)
class Model:
    """params may carry an optional "unembed_bias" [classes] for imported models."""
    params: dict
    config: ModelConfig

    @property
    def stage_names(self):
        return ("Embedding",) + tuple(
            f"L{i + 1}/{kind}" for i in range(self.config.layers)
            for kind in ("Attn", "MLP")
        )


def from_compiled(assembled) -> Model:
    """Import Tracr weights and its exact pre-argmax output projection."""
    cfg = assembled.model_config
    if cfg.layer_norm or cfg.dropout_rate != 0:
        raise ValueError("Layer normalization and dropout are unsupported.")
    if cfg.activation_function is not jax.nn.relu:
        raise ValueError("Only ReLU compiled transformers are supported.")
    width = assembled.params["token_embed"]["embeddings"].shape[1]
    metadata = {}

    @hk.without_apply_rng
    @hk.transform
    def projection():
        compiled = assembled.get_compiled_model()
        metadata["pad_token"] = compiled.pad_token
        result = compiled.unembed(jnp.eye(width), use_unembed_argmax=False)
        return result.reshape(width, -1)

    unembed = projection.apply(assembled.params)
    layers = []
    for i in range(cfg.num_layers):
        prefix = f"transformer/layer_{i}"
        paths = {"q": "attn/query", "k": "attn/key", "v": "attn/value",
                 "o": "attn/linear", "mlp_in": "mlp/linear_1", "mlp_out": "mlp/linear_2"}
        layers.append({name: {key: jnp.asarray(value) for key, value in
                               assembled.params[f"{prefix}/{path}"].items()}
                       for name, path in paths.items()})
    params = {"token_embed": jnp.asarray(assembled.params["token_embed"]["embeddings"]),
              "pos_embed": jnp.asarray(assembled.params["pos_embed"]["embeddings"]),
              "layers": tuple(layers), "unembed": unembed}
    return Model(params, ModelConfig(width, cfg.num_layers, cfg.num_heads,
                                    cfg.key_size, cfg.mlp_hidden_size,
                                    cfg.causal, metadata["pad_token"]))


def forward(model, encoded_inputs, attention_scale=1.0, patches=None, stage_deltas=None):
    """Return [batch, position, class] scores and all residual stages.

    Inputs include BOS at position zero. Each patch is a mapping containing
    stage (zero is embedding), batch, position, and an additive width-vector
    delta. Patches take effect before the next stage and remain in the trace.
    ``stage_deltas`` maps a stage to a [batch, position, width] array added to
    that stage's residual in one shot; it is differentiable and jit-friendly.
    Attention scaling multiplies QK logits before masking and softmax.
    """
    inputs = jnp.asarray(encoded_inputs)
    cfg, params = model.config, model.params
    if inputs.ndim != 2 or not jnp.issubdtype(inputs.dtype, jnp.integer):
        raise ValueError("Inputs must be a two-dimensional integer array.")
    if inputs.shape[1] > params["pos_embed"].shape[0]:
        raise ValueError("Sequence exceeds the positional embedding length.")
    patches = () if patches is None else patches
    for patch in patches:
        if not 0 <= patch["stage"] <= 2 * cfg.layers:
            raise ValueError("Patch stage is outside the model.")
        if not (0 <= patch["batch"] < inputs.shape[0] and
                0 <= patch["position"] < inputs.shape[1]):
            raise ValueError("Patch location is outside the input batch.")
        if jnp.shape(patch["delta"]) != (cfg.width,):
            raise ValueError("Patch delta must have the executing residual width.")

    stage_deltas = {} if stage_deltas is None else dict(stage_deltas)
    for stage, delta in stage_deltas.items():
        if not 0 <= stage <= 2 * cfg.layers:
            raise ValueError("Delta stage is outside the model.")
        if jnp.shape(delta) != (*inputs.shape, cfg.width):
            raise ValueError("Stage delta must have shape [batch, position, width].")

    def patched(x, stage):
        for patch in patches:
            if patch["stage"] == stage:
                x = x.at[patch["batch"], patch["position"]].add(patch["delta"])
        if stage in stage_deltas:
            x = x + stage_deltas[stage]
        return x

    def affine(x, weights):
        return x @ weights["w"] + weights["b"]

    if cfg.activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {cfg.activation!r}.")
    activation = ACTIVATIONS[cfg.activation]
    x = params["token_embed"][inputs] + params["pos_embed"][:inputs.shape[1]]
    x = patched(x, 0)
    residuals = [x]
    mask = jnp.ones((inputs.shape[0], 1, inputs.shape[1], inputs.shape[1]), dtype=bool)
    if cfg.pad_token is not None:
        mask = mask & (inputs != cfg.pad_token)[:, None, None, :]
    if cfg.causal:
        mask = mask & jnp.tril(jnp.ones((inputs.shape[1], inputs.shape[1]), dtype=bool))
    for i, layer in enumerate(params["layers"]):
        q, k, v = (affine(x, layer[name]).reshape(*x.shape[:2], cfg.heads, cfg.key_size)
                   for name in ("q", "k", "v"))
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(cfg.key_size)
        weights = jax.nn.softmax(jnp.where(mask, scores * attention_scale, -1e30), axis=-1)
        values = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(*x.shape[:2], -1)
        x = patched(x + affine(values, layer["o"]), 2 * i + 1)
        residuals.append(x)
        x = patched(x + affine(activation(affine(x, layer["mlp_in"])), layer["mlp_out"]),
                    2 * i + 2)
        residuals.append(x)
    logits = x @ params["unembed"]
    if "unembed_bias" in params:
        logits = logits + params["unembed_bias"]
    return logits, tuple(residuals)


def change_basis(model, basis):
    """Execute in an orthonormal [old_width, rank] basis, including reduced ranks."""
    basis = np.asarray(basis)
    if (basis.ndim != 2 or basis.shape[0] != model.config.width or
            not 1 <= basis.shape[1] <= basis.shape[0]):
        raise ValueError("Basis must have shape [residual_width, rank].")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-5, rtol=1e-5):
        raise ValueError("Basis columns must be orthonormal.")
    b = jnp.asarray(basis, dtype=model.params["token_embed"].dtype)
    p = model.params
    layers = []
    for layer in p["layers"]:
        transformed = {}
        for name, weights in layer.items():
            if name in ("q", "k", "v", "mlp_in"):
                transformed[name] = {"w": b.T @ weights["w"], "b": weights["b"]}
            else:
                transformed[name] = {"w": weights["w"] @ b, "b": weights["b"] @ b}
        layers.append(transformed)
    params = {"token_embed": p["token_embed"] @ b,
              "pos_embed": p["pos_embed"] @ b,
              "layers": tuple(layers), "unembed": b.T @ p["unembed"]}
    if "unembed_bias" in p:
        params["unembed_bias"] = p["unembed_bias"]
    return Model(params, replace(model.config, width=b.shape[1]))


def random_model(vocab_size, max_length, output_classes, seed, width=32, layers=2, heads=2):
    """Create a ReLU transformer; max_length includes the BOS position."""
    if min(vocab_size, max_length, output_classes, width, layers, heads) < 1 or width % heads:
        raise ValueError("Dimensions must be positive and width divisible by heads.")
    rng = np.random.default_rng(seed)

    def weight(n, m, scale=1.):
        return jnp.asarray(rng.normal(size=(n, m)) * scale / np.sqrt(n), dtype=jnp.float32)

    def affine(n, m, scale=1.):
        return {"w": weight(n, m, scale), "b": jnp.zeros(m)}

    blocks = tuple({"q": affine(width, width), "k": affine(width, width),
                    "v": affine(width, width), "o": affine(width, width, 1 / np.sqrt(layers)),
                    "mlp_in": affine(width, 2 * width),
                    "mlp_out": affine(2 * width, width, 1 / np.sqrt(layers))}
                   for _ in range(layers))
    return Model({"token_embed": weight(vocab_size, width),
                  "pos_embed": weight(max_length, width, .1),
                  "layers": blocks, "unembed": weight(width, output_classes)},
                 ModelConfig(width, layers, heads, width // heads, 2 * width))


def train_model(model, train_inputs, train_labels, val_inputs, val_labels,
                steps, seed, batch_size=128, lr=0.003):
    """Train every parameter with Adam; categorical labels omit BOS.

    History records every minibatch loss. Validation is evaluated initially,
    every 50 steps, and on the final step. The final checkpoint is returned.
    """
    arrays = [np.asarray(x) for x in (train_inputs, train_labels, val_inputs, val_labels)]
    for inputs, labels in (arrays[:2], arrays[2:]):
        if (inputs.ndim != 2 or labels.shape != (len(inputs), inputs.shape[1] - 1)
                or len(inputs) == 0 or not np.issubdtype(labels.dtype, np.integer)
                or labels.min() < 0 or labels.max() >= model.params["unembed"].shape[1]):
            raise ValueError("Nonempty inputs require integer class labels excluding BOS.")
    if steps < 0 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid optimizer settings.")
    tx, ty, vx, vy = map(jnp.asarray, arrays)
    cfg = model.config

    def loss(params, x, y):
        logits = forward(Model(params, cfg), x)[0][:, 1:]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.take_along_axis(log_probs, y[..., None], axis=-1).mean()

    @jax.jit
    def evaluate(params):
        logits = forward(Model(params, cfg), vx)[0][:, 1:]
        return loss(params, vx, vy), jnp.mean(logits.argmax(-1) == vy)

    @jax.jit
    def update(params, first, second, x, y, step):
        value, grads = jax.value_and_grad(loss)(params, x, y)
        first = jax.tree_util.tree_map(lambda m, g: .9 * m + .1 * g, first, grads)
        second = jax.tree_util.tree_map(lambda v, g: .999 * v + .001 * g * g, second, grads)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - .9 ** step)) /
            (jnp.sqrt(v / (1 - .999 ** step)) + 1e-8), params, first, second)
        return params, first, second, value

    params = model.params
    first = jax.tree_util.tree_map(jnp.zeros_like, params)
    second = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)
    vl, va = evaluate(params)
    history = [{"step": 0, "loss": float(loss(params, tx, ty)),
                "val_loss": float(vl), "val_accuracy": float(va)}]
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(tx), size=batch_size)
        params, first, second, value = update(params, first, second, tx[indices], ty[indices], step)
        row = {"step": step, "loss": float(value)}
        if step % 50 == 0 or step == steps:
            vl, va = evaluate(params)
            row.update(val_loss=float(vl), val_accuracy=float(va))
        history.append(row)
    return Model(params, cfg), history


def parameter_count(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(model.params))


def save_model(model, path):
    """Save architecture and arrays in NPZ without executable pickle data."""
    arrays = {key: np.asarray(model.params[key]) for key in ("token_embed", "pos_embed", "unembed")}
    if "unembed_bias" in model.params:
        arrays["unembed_bias"] = np.asarray(model.params["unembed_bias"])
    for i, layer in enumerate(model.params["layers"]):
        for name, weights in layer.items():
            for key, value in weights.items():
                arrays[f"layer/{i}/{name}/{key}"] = np.asarray(value)
    arrays["config"] = np.asarray(json.dumps(asdict(model.config)))
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        cfg = ModelConfig(**json.loads(str(data["config"])))
        params = {key: jnp.asarray(data[key]) for key in ("token_embed", "pos_embed", "unembed")}
        if "unembed_bias" in data:
            params["unembed_bias"] = jnp.asarray(data["unembed_bias"])
        params["layers"] = tuple({name: {key: jnp.asarray(data[f"layer/{i}/{name}/{key}"])
                                        for key in ("w", "b")}
                                  for name in ("q", "k", "v", "o", "mlp_in", "mlp_out")}
                                 for i in range(cfg.layers))
    return Model(params, cfg)

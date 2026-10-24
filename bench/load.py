"""Read an InterpBench ``ll_model.pth`` into the JAX executable model, without torch.

A torch checkpoint is a zip archive holding ``data.pkl`` (the pickled state
dict, whose tensors are references into storage files) and ``data/<key>`` raw
little-endian buffers. The pickle is read with an unpickler that never imports
torch: storage references become descriptors and tensor rebuilds become
numpy views. No code from the checkpoint is executed.

The weights are TransformerLens ``HookedTransformer`` parameters with no layer
normalisation, standard positional embeddings, exact GELU, and attention
scaled by ``1/sqrt(d_head)``. They are mapped onto ``raspformer.models.Model``.
"""

import collections
import io
import pickle
import zipfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from raspformer.models import Model, ModelConfig

_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"), "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"), "BFloat16Storage": None,
    "LongStorage": np.dtype("<i8"), "IntStorage": np.dtype("<i4"),
    "BoolStorage": np.dtype("?"), "ByteStorage": np.dtype("u1"),
}


class _Storage:
    def __init__(self, dtype_name, key, numel):
        self.dtype_name, self.key, self.numel = dtype_name, key, numel


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _storage_type(name):
    return type(name, (), {"name": name})


class _Unpickler(pickle.Unpickler):
    """Resolve only the handful of symbols a state dict needs."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *_: _Tensor(storage, offset, size, stride)
        if module == "torch" and name.endswith("Storage"):
            return _storage_type(name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a checkpoint.")

    def persistent_load(self, pid):
        tag, storage_type, key, _location, numel = pid
        if tag != "storage":
            raise pickle.UnpicklingError(f"Unexpected persistent id {tag!r}.")
        return _Storage(storage_type.name, str(key), int(numel))


def read_state_dict(path) -> dict[str, np.ndarray]:
    """Return every tensor in the checkpoint as a contiguous numpy array."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        root = names[0].split("/")[0]
        state = _Unpickler(io.BytesIO(archive.read(f"{root}/data.pkl"))).load()
        arrays = {}
        for name, tensor in state.items():
            dtype = _STORAGE_DTYPES.get(tensor.storage.dtype_name)
            if dtype is None:
                raise ValueError(f"Unsupported storage {tensor.storage.dtype_name} for {name}.")
            buffer = np.frombuffer(archive.read(f"{root}/data/{tensor.storage.key}"), dtype=dtype)
            if buffer.size != tensor.storage.numel:
                raise ValueError(f"Storage size mismatch for {name}.")
            view = np.lib.stride_tricks.as_strided(
                buffer[tensor.offset:], shape=tensor.size,
                strides=tuple(s * dtype.itemsize for s in tensor.stride), writeable=False)
            arrays[name] = np.array(view)
    return arrays


def model_from_state(state: dict, cfg: dict) -> Model:
    """Map TransformerLens parameters onto the executable residual model."""
    if cfg.get("normalization_type") not in (None, "None"):
        raise ValueError("Only models without layer normalisation are supported.")
    if cfg["act_fn"] != "gelu":
        raise ValueError(f"Unsupported activation {cfg['act_fn']!r}.")
    if cfg["positional_embedding_type"] != "standard" or not cfg["use_attn_scale"]:
        raise ValueError("Expected standard positional embeddings and scaled attention.")
    heads, d_model, d_head = cfg["n_heads"], cfg["d_model"], cfg["d_head"]
    causal = cfg["attention_dir"] == "causal"

    def f32(name):
        return jnp.asarray(state[name], dtype=jnp.float32)

    layers = []
    for i in range(cfg["n_layers"]):
        prefix = f"blocks.{i}"
        # TransformerLens stores a causal mask buffer in every block and only
        # applies it when attention_dir is "causal", so the buffer alone does
        # not say which direction was used; attention_dir from the config does.
        mask = state[f"{prefix}.attn.mask"]
        if causal and not np.array_equal(mask, np.tril(np.ones_like(mask))):
            raise ValueError(f"Causal model block {i} has a non-causal mask buffer.")
        layer = {}
        for ours, theirs in (("q", "Q"), ("k", "K"), ("v", "V")):
            layer[ours] = {"w": f32(f"{prefix}.attn.W_{theirs}").transpose(1, 0, 2).reshape(d_model, heads * d_head),
                           "b": f32(f"{prefix}.attn.b_{theirs}").reshape(heads * d_head)}
        layer["o"] = {"w": f32(f"{prefix}.attn.W_O").reshape(heads * d_head, d_model),
                      "b": f32(f"{prefix}.attn.b_O")}
        layer["mlp_in"] = {"w": f32(f"{prefix}.mlp.W_in"), "b": f32(f"{prefix}.mlp.b_in")}
        layer["mlp_out"] = {"w": f32(f"{prefix}.mlp.W_out"), "b": f32(f"{prefix}.mlp.b_out")}
        layers.append(layer)
    params = {"token_embed": f32("embed.W_E"), "pos_embed": f32("pos_embed.W_pos"),
              "layers": tuple(layers), "unembed": f32("unembed.W_U"),
              "unembed_bias": f32("unembed.b_U")}
    config = ModelConfig(width=d_model, layers=cfg["n_layers"], heads=heads, key_size=d_head,
                         mlp_hidden_size=cfg["d_mlp"], causal=causal, pad_token=None,
                         activation="gelu")
    return Model(params, config)


class _Inert:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _ConfigUnpickler(pickle.Unpickler):
    """ll_model_cfg.pkl is a plain dict whose only foreign value is a torch dtype."""

    def find_class(self, module, name):
        if module in ("builtins", "collections", "numpy", "numpy.core.multiarray", "numpy._core.multiarray"):
            return super().find_class(module, name)
        return type(name, (_Inert,), {})


def read_config(path) -> dict:
    with open(path, "rb") as handle:
        return _ConfigUnpickler(handle).load()


def load_interpbench(case_dir) -> Model:
    """Executable model for one ``bench/interpbench/<id>/`` directory."""
    case_dir = Path(case_dir)
    cfg = read_config(case_dir / "ll_model_cfg.pkl")
    return model_from_state(read_state_dict(case_dir / "ll_model.pth"), cfg)

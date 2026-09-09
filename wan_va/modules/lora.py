# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LoRA (low-rank adaptation) for parameter-efficient post-training.

`LoRALinear` is a *subclass* of ``nn.Linear`` rather than a wrapper holding a
``base_layer``. That keeps every pre-existing ``state_dict`` key in place
(``blocks.0.attn1.to_q.weight`` stays put, ``lora_A``/``lora_B`` are added
alongside), so a merged checkpoint has exactly the key set the inference server
already loads, and attribute access on the enclosing module (e.g. the
``WanAttention`` KV cache) is unaffected.

Injection must happen after ``from_pretrained`` but *before* ``apply_ac`` and
``shard_model``: activation checkpointing rewrites module paths to include
``_checkpoint_wrapped_module``, and FSDP2 turns parameters into ``DTensor``.
"""
import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

__all__ = [
    'LoRALinear',
    'inject_lora',
    'lora_scaling_map',
    'lora_state_dict',
    'merge_lora_state_dict',
    'count_parameters',
    'is_lora_key',
    'DEFAULT_LORA_TARGET_MODULES',
]

# The 10 linears per transformer block, holding 4.91B of the model's 5.09B
# parameters. Matched as suffixes of the dotted module path.
DEFAULT_LORA_TARGET_MODULES = [
    'attn1.to_q',
    'attn1.to_k',
    'attn1.to_v',
    'attn1.to_out.0',
    'attn2.to_q',
    'attn2.to_k',
    'attn2.to_v',
    'attn2.to_out.0',
    'ffn.net.0.proj',
    'ffn.net.2',
]


class LoRALinear(nn.Linear):
    """``nn.Linear`` with an additive low-rank branch ``B @ A * scaling``.

    ``lora_B`` is zero-initialised, so the module is bit-identical to the
    wrapped linear until the first optimizer step.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 rank=32,
                 alpha=32,
                 dropout=0.0,
                 device=None,
                 dtype=None):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # LoRA params are always fp32: the frozen base may be bf16, but the
        # trainable branch keeps fp32 master weights. FSDP2's uniform-dtype
        # check only inspects trainable parameters, so this mix is legal.
        self.lora_A = nn.Linear(in_features,
                                rank,
                                bias=False,
                                device=device,
                                dtype=torch.float32)
        self.lora_B = nn.Linear(rank,
                                out_features,
                                bias=False,
                                device=device,
                                dtype=torch.float32)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_linear(cls, linear, rank, alpha, dropout=0.0):
        """Build a `LoRALinear` that reuses `linear`'s weight and bias tensors."""
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device='meta',  # avoid allocating a second copy of the base weight
            dtype=linear.weight.dtype,
        )
        # Adopt the pretrained tensors rather than copying them.
        module.weight = linear.weight
        module.bias = linear.bias
        device = linear.weight.device
        module.lora_A.to_empty(device=device)
        module.lora_B.to_empty(device=device)
        module.reset_lora_parameters()
        return module

    def forward(self, x):
        out = super().forward(x)
        lora_x = self.lora_dropout(x)
        # Under FSDP2's MixedPrecisionPolicy the all-gathered lora weights are
        # bf16 at forward time (the fp32 master copy is what the optimizer
        # updates); match them so the add does not upcast.
        lora_out = self.lora_B(self.lora_A(lora_x.to(self.lora_A.weight.dtype)))
        return out + lora_out.to(out.dtype) * self.scaling


def _set_child(parent, child_name, module):
    """`setattr`, but index-assign for `nn.ModuleList` / `nn.Sequential`."""
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_lora(model,
                rank=32,
                alpha=32,
                dropout=0.0,
                target_modules=None):
    """Replace every matching ``nn.Linear`` in `model` with a `LoRALinear`.

    Returns the ``{module_path: LoRALinear}`` mapping of what was replaced.
    Call before `apply_ac` / `shard_model`.
    """
    if target_modules is None:
        target_modules = DEFAULT_LORA_TARGET_MODULES
    target_modules = list(target_modules)

    for name, param in model.named_parameters():
        if '_checkpoint_wrapped_module' in name:
            raise RuntimeError(
                "inject_lora must run before apply_ac(): found an activation-"
                f"checkpointed module at '{name}'")
        if isinstance(param, DTensor):
            raise RuntimeError(
                "inject_lora must run before shard_model(): "
                f"'{name}' is already a DTensor")

    matched = {}
    # Materialise the list first: we mutate the tree while iterating.
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear)]
    for name, module in candidates:
        if not any(name == t or name.endswith('.' + t) for t in target_modules):
            continue
        parent_path, child_name = name.rpartition('.')[0], name.rpartition('.')[2]
        parent = model.get_submodule(parent_path) if parent_path else model
        lora_module = LoRALinear.from_linear(module, rank, alpha, dropout)
        _set_child(parent, child_name, lora_module)
        matched[name] = lora_module

    if not matched:
        raise ValueError(
            f"inject_lora matched no modules for target_modules={target_modules}")

    # Only the low-rank branches train.
    for lora_module in matched.values():
        lora_module.weight.requires_grad_(False)
        if lora_module.bias is not None:
            lora_module.bias.requires_grad_(False)
        lora_module.lora_A.weight.requires_grad_(True)
        lora_module.lora_B.weight.requires_grad_(True)

    return matched


def lora_scaling_map(model):
    """``{module_path: (rank, alpha, scaling)}`` for every `LoRALinear`."""
    return {
        name: (mod.rank, mod.alpha, mod.scaling)
        for name, mod in model.named_modules() if isinstance(mod, LoRALinear)
    }


def is_lora_key(key):
    return '.lora_A.' in key or '.lora_B.' in key


def lora_state_dict(state_dict, extra_prefixes=()):
    """Subset of `state_dict` holding the adapter (plus any `extra_prefixes`).

    `extra_prefixes` is for fully-trained modules that must travel with the
    adapter, e.g. ``('action_embedder', 'action_proj_out')``.
    """
    out = {}
    for key, value in state_dict.items():
        if is_lora_key(key) or any(
                key == p or key.startswith(p + '.') for p in extra_prefixes):
            out[key] = value
    return out


def merge_lora_state_dict(state_dict, scaling_map):
    """Fold adapters into their base weights and drop the adapter keys.

    `state_dict` is the gathered full state dict (CPU); `scaling_map` comes
    from `lora_scaling_map` on the live model. Merging happens in fp32 — the
    caller casts to bf16 afterwards, since folding a low-rank update in bf16
    loses precision.

    Returns a new dict whose key set matches the un-adapted model.
    """
    merged = {
        key: value
        for key, value in state_dict.items() if not is_lora_key(key)
    }

    for path, (rank, _alpha, scaling) in scaling_map.items():
        weight_key = f'{path}.weight'
        a_key = f'{path}.lora_A.weight'
        b_key = f'{path}.lora_B.weight'
        missing = [k for k in (weight_key, a_key, b_key) if k not in state_dict]
        if missing:
            raise KeyError(
                f"cannot merge LoRA at '{path}': missing {missing} in state dict")

        lora_a = state_dict[a_key].float()
        lora_b = state_dict[b_key].float()
        if lora_a.shape[0] != rank or lora_b.shape[1] != rank:
            raise ValueError(
                f"rank mismatch at '{path}': expected {rank}, got "
                f"A{tuple(lora_a.shape)} B{tuple(lora_b.shape)}")

        base = state_dict[weight_key]
        merged[weight_key] = (base.float() + (lora_b @ lora_a) * scaling).to(
            base.dtype)

    return merged


def count_parameters(model):
    """``(trainable, total)`` parameter counts, DTensor-safe."""
    trainable = total = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return trainable, total

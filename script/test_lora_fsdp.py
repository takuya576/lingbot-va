"""2-GPU test with the REAL WanTransformer3DModel (2 layers, ~1GB) to check the
one thing the toy model cannot: diffusers' `_keep_in_fp32_modules` leaves some
FROZEN params in fp32 while the rest load as bf16, so a shard group can hold
mixed frozen dtypes. Verifies that all-gathers correctly and that the real
forward_train + activation checkpointing + LoRA trains end to end.
"""
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (get_model_state_dict,
                                                     StateDictOptions)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wan_va'))
from configs import VA_CONFIGS  # noqa: E402
from distributed.fsdp import apply_ac, shard_model  # noqa: E402
from modules.lora import (count_parameters, inject_lora, lora_scaling_map,  # noqa: E402
                          lora_state_dict, merge_lora_state_dict)
from modules.model import WanTransformer3DModel  # noqa: E402
from utils import get_mesh_id  # noqa: E402

NLAYERS = 2


def log(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


def make_inputs(device):
    B, C, F, H, W = 1, 48, 4, 4, 4
    AC_, AF, AN = 30, 4, 2
    ps = (1, 2, 2)

    def mk(shape, f, action):
        lat = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        pf, ph, pw = (1, 1, 1) if action else ps
        gid = get_mesh_id(shape[-3] // pf, shape[-2] // ph, shape[-1] // pw,
                          t=1 if action else 0, f_w=1, f_shift=0,
                          action=action).to(device)
        return dict(
            timesteps=torch.randint(0, 1000, (B, f), device=device).float(),
            cond_timesteps=torch.zeros(B, f, device=device).float(),
            noisy_latents=lat, latent=lat.clone(),
            targets=torch.zeros_like(lat),
            grid_id=gid[None].repeat(B, 1, 1),
            text_emb=torch.randn(B, 512, 4096, device=device,
                                 dtype=torch.bfloat16))

    ld = mk((B, C, F, H, W), F, False)
    ad = mk((B, AC_, AF, AN, 1), AF, True)
    ad['actions_mask'] = torch.ones(B, AC_, AF, AN, 1, device=device,
                                    dtype=torch.bool)
    return {'latent_dict': ld, 'action_dict': ad,
            'chunk_size': 2, 'window_size': 8}


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{local_rank}')
    cfg = VA_CONFIGS['yam_place_cube_train']

    torch.manual_seed(0)
    # Mimic load_transformer(torch_dtype=bfloat16): build in bf16, then force
    # the _keep_in_fp32_modules patterns to fp32 exactly as from_pretrained does.
    torch.set_default_dtype(torch.bfloat16)
    model = WanTransformer3DModel(num_layers=NLAYERS, attn_mode='flex')
    torch.set_default_dtype(torch.float32)
    kept = 0
    for name, mod in model.named_modules():
        if any(pat in name for pat in model._keep_in_fp32_modules):
            mod.to(torch.float32)
            kept += 1
    for name, param in model.named_parameters(recurse=True):
        if any(pat in name for pat in model._keep_in_fp32_modules):
            param.data = param.data.float()
    log(f"[0] real model, {NLAYERS} layers, "
        f"{sum(p.numel() for p in model.parameters()):,} params; "
        f"{kept} modules forced to fp32 by _keep_in_fp32_modules")

    base_keys = set(model.state_dict().keys())
    model.requires_grad_(False)
    matched = inject_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                          dropout=cfg.lora_dropout,
                          target_modules=cfg.lora_target_modules)
    smap = lora_scaling_map(model)
    full_prefixes = ['action_embedder', 'action_proj_out']
    for p in full_prefixes:
        sub = model.get_submodule(p)
        sub.to(torch.float32)
        sub.requires_grad_(True)

    tr_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    fr_dtypes = {p.dtype for p in model.parameters() if not p.requires_grad}
    log(f"[1] injected {len(matched)} modules; trainable dtypes={tr_dtypes}, "
        f"frozen dtypes={fr_dtypes}")
    assert tr_dtypes == {torch.float32}
    assert len(fr_dtypes) > 1, "expected MIXED frozen dtypes -- test is not " \
                               "exercising the case it exists for"

    apply_ac(model)
    model = shard_model(model)
    model = model.to(device)
    model.train()
    log("[2] apply_ac + shard_model OK with mixed frozen dtypes "
        "(no uniform-dtype assertion, no uint8 all-gather fallback)")

    ntr, ntot = count_parameters(model)
    log(f"[3] trainable {ntr:,} / {ntot:,} ({100 * ntr / ntot:.2f}%)")

    inp = make_inputs(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.0, fused=True,
                            foreach=False)

    losses = []
    for step in range(3):
        for micro in range(2):
            model.set_requires_gradient_sync((micro + 1) % 2 == 0)
            v, a = model(inp, train_mode=True)
            loss = (v.float().pow(2).mean() + a.float().pow(2).mean()) / 2
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 2.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        log(f"[4] step {step}: loss={loss.item():.5f} grad_norm={norm.item():.4f}")
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    log("[4] real forward_train + AC + FSDP2 + LoRA trains; loss decreasing")

    sd = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    # FSDP2 forwards are collectives, so this must run on every rank -- never
    # inside the rank-0 branch below.
    model.eval()
    with torch.no_grad():
        live_v, live_a = model(inp, train_mode=True)

    if rank == 0:
        adapter = lora_state_dict(sd, extra_prefixes=tuple(full_prefixes))
        merged = merge_lora_state_dict(sd, smap)
        log(f"[5] gathered {len(sd)} -> adapter {len(adapter)}, merged {len(merged)}")
        assert set(merged) == base_keys, sorted(set(merged) ^ base_keys)[:6]
        log("[5] merged key set == real un-adapted model key set: True")

        # a merged bf16 checkpoint must load into a stock model
        torch.set_default_dtype(torch.bfloat16)
        ref = WanTransformer3DModel(num_layers=NLAYERS, attn_mode='flex')
        torch.set_default_dtype(torch.float32)
        missing, unexpected = ref.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in merged.items()}, strict=True)
        log(f"[6] stock model loaded merged checkpoint strictly "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")

        # the merged weights must reproduce the adapted model's own output
        ref = ref.to(device).eval()
        with torch.no_grad():
            rep_v, rep_a = ref(inp, train_mode=True)
        rel_v = ((rep_v.float() - live_v.float()).norm() /
                 live_v.float().norm()).item()
        rel_a = ((rep_a.float() - live_a.float()).norm() /
                 live_a.float().norm()).item()
        log(f"[6] merged round-trip rel err: video {rel_v:.2e}, "
            f"action {rel_a:.2e} (bf16 tolerance)")
        assert rel_v < 5e-2 and rel_a < 5e-2

    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"[7] peak VRAM {peak:.2f} GiB for {NLAYERS} layers")
    log("\nREAL-MODEL FSDP2 + LoRA CHECKS PASSED")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

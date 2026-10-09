#!/usr/bin/env bash
# Assemble a servable model directory for a YAM LoRA checkpoint.
#
# Training writes only `transformer/` (already LoRA-merged by train.py:450-456, so the
# safetensors has exactly the key set the unmodified inference server expects). The
# server additionally loads `vae/`, `tokenizer/` and `text_encoder/` as siblings under
# `wan22_pretrained_model_name_or_path` (wan_va_server.py:61-86), so those are
# symlinked in from the base checkpoint.
#
# Symlinks, not copies: the merged transformer alone is 10.2 GB.
#
# The links are absolute and point into train_out/, so renaming or pruning a run's
# save_root afterwards silently rots an already-assembled serving directory. `ls` still
# lists the entries (the links exist), and the failure surfaces much later from
# diffusers as `OSError: Error no file named config.json found in directory .../
# transformer` -- which reads like a missing file rather than a broken link. The final
# validation block below therefore resolves every link, so that breakage is caught here
# rather than at server launch.
#
# Usage:
#   script/assemble_yam_serving_model.sh [checkpoint_dir] [out_dir] [base_dir]
#   script/assemble_yam_serving_model.sh --check <out_dir>
#
# checkpoint_dir defaults to the most recent checkpoint under train_out/yam_place_cube/,
# across all timestamped runs, and out_dir to the place_cube serving directory. Serving
# place_cube_pre (the first place_cube recording) means passing both explicitly -- its
# checkpoints are under train_out/yam_place_cube_pre/ and its norm_stat differs, so
# crossing the two silently denormalizes against the wrong quantiles:
# train_out/yam_place_cube_pre/ -> Pretrained_models/lingbot-va-posttrain-yam-place-cube-pre.
# And for place_cube post-trained from lingbot-va-base (yam_place_cube_base_train):
# train_out/yam_place_cube_base/<run>/checkpoints/checkpoint_step_N ->
# Pretrained_models/lingbot-va-posttrain-yam-place-cube-base, both passed explicitly.
# base_dir only supplies vae/tokenizer/text_encoder, which are byte-identical between
# lingbot-va-base and lingbot-va-posttrain-robotwin, so either works as base_dir.
#
# --check validates an ALREADY-assembled directory without rebuilding it. That is the
# mode that catches rot, since rot happens after assembly: run it before launching the
# server, against the wan22_pretrained_model_name_or_path of the config you are serving.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve every required path, following symlinks. Shared by --check and by the
# post-assembly validation, so both report breakage the same way.
validate_serving_dir() {
    python3 - "$1" <<'PY'
import json, os, sys
out = sys.argv[1]

# os.path.exists follows symlinks, so this catches a link whose target has been moved
# or deleted -- the failure mode that otherwise only shows up at server launch.
required = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae", "tokenizer", "text_encoder",
]
broken = []
for rel in required:
    path = os.path.join(out, rel)
    if os.path.exists(path):
        continue
    target = os.readlink(path) if os.path.islink(path) else None
    broken.append(f"{rel} -> {target}" if target else f"{rel} (absent)")

if broken:
    print(f"  FAILED -- these do not resolve under {out}:", file=sys.stderr)
    for b in broken:
        print(f"    {b}", file=sys.stderr)
    print("  Re-assemble against a checkpoint that still exists.", file=sys.stderr)
    sys.exit(1)

cfg = json.load(open(os.path.join(out, "transformer", "config.json")))
print(f"  action_dim={cfg['action_dim']}  num_layers={cfg['num_layers']}  "
      f"in_channels={cfg['in_channels']}")
print(f"  all {len(required)} required paths resolve")
PY
}

if [ "${1:-}" = "--check" ]; then
    [ $# -eq 2 ] || { echo "usage: $0 --check <out_dir>" >&2; exit 2; }
    CHECK_OUT="$(realpath -m "$2")"
    [ -d "$CHECK_OUT" ] || { echo "ERROR: no such directory: $CHECK_OUT" >&2; exit 1; }
    echo "Checking $CHECK_OUT"
    validate_serving_dir "$CHECK_OUT"
    exit 0
fi

DEFAULT_ROOT="$REPO/train_out/yam_place_cube"
if [ -n "${1:-}" ]; then
    CKPT="$1"
else
    # train.py gives each launch its own dated save_root, so checkpoints live at
    # <root>/<timestamp>/checkpoints/checkpoint_step_N. The newest wins -- echoed,
    # because "newest" is a guess at which run you meant.
    #
    # `|| true` because an unmatched glob makes ls exit nonzero, and under
    # `set -euo pipefail` that kills the script inside this assignment -- silently,
    # before the diagnostic below can say why.
    CKPT="$(ls -1dt "$DEFAULT_ROOT"/*/checkpoints/checkpoint_step_* 2>/dev/null \
            | head -1 || true)"
    [ -n "$CKPT" ] || { echo "ERROR: no checkpoint_step_* under $DEFAULT_ROOT" >&2
        echo "Pass a checkpoint directory explicitly." >&2; exit 1; }
    echo "No checkpoint given; using most recent: $CKPT"
fi
OUT="${2:-$REPO/Pretrained_models/lingbot-va-posttrain-yam-place-cube}"
BASE="${3:-$REPO/Pretrained_models/lingbot-va-posttrain-robotwin}"

# Resolve to absolute paths: the symlinks below are written into $OUT, so a relative
# target would be resolved relative to that directory and dangle.
CKPT="$(realpath -m "$CKPT")"
OUT="$(realpath -m "$OUT")"
BASE="$(realpath -m "$BASE")"

[ -d "$CKPT/transformer" ] || { echo "ERROR: no transformer/ in $CKPT" >&2
    echo "Available checkpoints:" >&2; ls "$(dirname "$CKPT")" 2>/dev/null >&2; exit 1; }
[ -f "$CKPT/transformer/diffusion_pytorch_model.safetensors" ] || {
    echo "ERROR: merged weights missing in $CKPT/transformer" >&2; exit 1; }

mkdir -p "$OUT/transformer"
# Per-file for the transformer so the adapter-only files (lora_config.json,
# lora_weights.safetensors) stay out of the serving dir -- they are for resuming
# training, and leaving them here invites loading the wrong one.
ln -sfn "$CKPT/transformer/config.json"                        "$OUT/transformer/config.json"
ln -sfn "$CKPT/transformer/diffusion_pytorch_model.safetensors" "$OUT/transformer/diffusion_pytorch_model.safetensors"

for sub in vae tokenizer text_encoder; do
    [ -d "$BASE/$sub" ] || { echo "ERROR: $BASE/$sub missing" >&2; exit 1; }
    ln -sfn "$BASE/$sub" "$OUT/$sub"
done

echo "Assembled $OUT"
echo "  transformer  <- $CKPT/transformer  (LoRA-merged)"
echo "  vae/tokenizer/text_encoder <- $BASE"
validate_serving_dir "$OUT"

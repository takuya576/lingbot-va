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
# Usage:
#   script/assemble_yam_serving_model.sh [checkpoint_dir] [out_dir] [base_dir]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:-$REPO/train_out/yam_lora/checkpoints/checkpoint_step_1000}"
OUT="${2:-$REPO/Pretrained_models/lingbot-va-posttrain-yam}"
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
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
cfg = json.load(open(os.path.join(out, "transformer", "config.json")))
print(f"  action_dim={cfg['action_dim']}  num_layers={cfg['num_layers']}  "
      f"in_channels={cfg['in_channels']}")
missing = [s for s in ("transformer", "vae", "tokenizer", "text_encoder")
           if not os.path.isdir(os.path.join(out, s))]
print("  FAILED, missing: " + ", ".join(missing) if missing else "  all 4 subfolders present")
PY

#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29502"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"yam_place_cube_base"}
# GPU 0 is shared with other users' jobs; pick a free one with e.g. GPU=1.
GPU=${GPU:-"0"}
# Per-config by default, so rollouts of differently-initialized checkpoints of the
# same task (yam_place_cube vs yam_place_cube_base) do not interleave in one directory.
SAVE_ROOT=${SAVE_ROOT:-"example/${CONFIG_NAME}"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
save_root=${SAVE_ROOT}
## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
CUDA_VISIBLE_DEVICES=${GPU} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.wan_va_server --config-name ${config_name} $overrides \
    --save_root ${save_root}

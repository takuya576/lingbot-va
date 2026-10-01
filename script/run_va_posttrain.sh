#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"yam_place_cube_base_train"} # robotwin_train, yam_cover_blocks_train, libero_train

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# WANDB_API_KEY must come from the environment (e.g. `source .env.local`), not
# from this file -- it is committed to git.
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="takuya50719-the-university-of-tokyo"
export WANDB_PROJECT="lingbot-va"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides

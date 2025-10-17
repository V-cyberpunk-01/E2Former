#!/usr/bin/env bash
# scripts/run_single.sh
set -ex

# --------------- Paths & Conda ---------------
cd /mnt/shared-storage-user/chenshuizhou/E2Former
export PATH="/mnt/shared-storage-user/chenshuizhou/miniconda3/bin:$PATH"

if [ -f /mnt/shared-storage-user/chenshuizhou/miniconda3/etc/profile.d/conda.sh ]; then
  . /mnt/shared-storage-user/chenshuizhou/miniconda3/etc/profile.d/conda.sh
  conda activate e2former
fi
conda activate e2former 

# --------------- W&B: offline to local disk ---------------
export WANDB_MODE=offline
export WANDB_PROJECT="e2former-mptrj-training"
export WANDB_DIR="/mnt/shared-storage-user/chenshuizhou/wandb/${JOB_ID:-local_job}/node${NODE_RANK:-0}"
export WANDB_RUN_GROUP="${JOB_ID:-mp_group}"

ts="$(date +%Y%m%d_%H%M%S)"

NNODES="${NODE_COUNT:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR not set}"
MASTER_PORT="${MASTER_PORT:-29500}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"


torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${JOB_ID:-oc22_${ts}}" \
  main.py \
    --config-yml=configs/s2ef/MPTrj/e2former/e2former_67M.yaml \
    --mode=train \
    --run-dir=runs \
    --identifier="mptrj_lr${LR:-}bs${BATCH_SIZE:-}nbr${MAX_NEIGHBORS:-}rad${MAX_RADIUS:-}_${ts}" \
    --timestamp-id="mptrj_${ts}" \
    --num-gpus="$((NNODES * NPROC_PER_NODE))" \
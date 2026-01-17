#!/bin/bash

# Configuration
WANDB_OFFLINE=false
OVERFIT=false

# Model Architecture - OWT uses larger vocab
VOCAB_SIZE=32000
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
ROPE_THETA=10000.0
NUM_LAYERS=4
NUM_HEADS=16

# Paths
TRAIN_PATH="./data/owt_train.npy"
VAL_PATH="./data/owt_valid.npy"
CHECKPOINT_DIR="./checkpoints"

# Training args
BATCH_SIZE=128
NUM_TOKENS=327_680_000
EVAL_FREQ=500

MAX_LEARNING_RATE=0.005
WARMUP_ITERS=500

WEIGHT_DECAY=0.3

# Conditional Flags
FLAGS=""
if [ "$WANDB_OFFLINE" = true ]; then
    FLAGS="$FLAGS --wandb_offline"
fi

if [ "$OVERFIT" = true ]; then
    FLAGS="$FLAGS --overfit"
fi

# Activate virtual environment
source .venv/bin/activate

python train.py \
    --vocab_size "$VOCAB_SIZE" \
    --context_length "$CONTEXT_LENGTH" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --rope_theta "$ROPE_THETA" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --max_learning_rate "$MAX_LEARNING_RATE" \
    --warmup_iters "$WARMUP_ITERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --eval_freq "$EVAL_FREQ" \
    $FLAGS \
    "$@"

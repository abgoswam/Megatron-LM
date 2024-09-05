#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)

export CUDA_DEVICE_MAX_CONNECTIONS=1

# local vars. should be set by amulet config.
GPUS_PER_NODE=2
HIDDEN_SIZE=128
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=${SEQ_LENGTH}
TP=1
CP=2
BASE_DIR="/mnt/synthdatastore/agoswami"
DATA_DIR="/mnt/synthdatastore/agoswami/my_long_corpus"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

if [ -z "$CHECKPOINT_PATH" ]; then
  CHECKPOINT_PATH="${BASE_DIR}/checkpoint_saving/ft_$(date +%F-%H%M.%S)"
fi

if [ -z "$TENSORBOARD_LOGS_PATH" ]; then
  TENSORBOARD_LOGS_PATH="${BASE_DIR}/tensorboard_logs/ft_$(date +%F-%H%M.%S)"
fi

echo "CHECKPOINT_PATH:$CHECKPOINT_PATH"
echo "TENSORBOARD_LOGS_PATH:$TENSORBOARD_LOGS_PATH"
echo "OUTPUT_DIR:$OUTPUT_DIR"
echo "DATA_DIR:$DATA_DIR"
echo "SEQ_LENGTH:$SEQ_LENGTH"
echo "MAX_POSITION_EMBEDDINGS:$MAX_POSITION_EMBEDDINGS"
echo "GPUS_PER_NODE:$GPUS_PER_NODE"
echo "AMLT_OUTPUT_DIR:$AMLT_OUTPUT_DIR"
echo "AMLT_DATA_DIR:$AMLT_DATA_DIR"

VOCAB_FILE="./gpt2-vocab.json"
MERGE_FILE="./gpt2-merges.txt"

DATA_PATH="${DATA_DIR}/my_long_corpus_${MAX_POSITION_EMBEDDINGS}_gpt2_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    # --transformer-impl local # local will not work for RMSNorm
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --num-layers 24 
    --hidden-size ${HIDDEN_SIZE} 
    --num-attention-heads 16 
    --seq-length ${SEQ_LENGTH} 
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} 
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --swiglu
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --train-iters 128 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 0.00015 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 128
    --use-flash-attn
    --attention-dropout 0.1
    --use-distributed-optimizer
    # --tp-comm-overlap-rs-dgrad  # this seems to yield worse result
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1 
)

MODEL_PARALLEL_ARGS=(
    --context-parallel-size ${CP}
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 0 #10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# GPU==1
# torchrun pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${DATA_ARGS[@]} \
#     ${EVAL_AND_LOGGING_ARGS[@]}

# > 1 GPU
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
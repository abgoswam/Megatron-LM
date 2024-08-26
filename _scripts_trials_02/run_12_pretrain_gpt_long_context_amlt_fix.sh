#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)

export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPUS_PER_NODE=2
# NUM_NODES=1
MASTER_ADDR=node-0
MASTER_PORT=6000
RDZV_ID=f4694954-bb45-495b-9452-789e6028374a
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank 0
    --rdzv_id $RDZV_ID
    --rdzv_backend c10d
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
)

# HIDDEN_SIZE=128
# SEQ_LENGTH=131072
# SEQ_LENGTH=262144
# SEQ_LENGTH=524288
# SEQ_LENGTH=1048576
MAX_POSITION_EMBEDDINGS=${SEQ_LENGTH}
GPT_MODEL_ARGS=(
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

# TP=1
# CP=2
MODEL_PARALLEL_ARGS=(
    --context-parallel-size ${CP}
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size 1
)

DATA_DIR="/mnt/synthdatastore/agoswami/my_long_corpus"
DATA_PATH="${DATA_DIR}/my_long_corpus_${MAX_POSITION_EMBEDDINGS}_gpt2_text_document"
VOCAB_FILE="./gpt2-vocab.json"
MERGE_FILE="./gpt2-merges.txt"

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --no-create-attention-mask-in-dataloader
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --train-iters 512 
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
    --lr-decay-iters 512
    --use-flash-attn
    --attention-dropout 0.1
    --use-distributed-optimizer
    # --tp-comm-overlap-rs-dgrad  # this seems to yield worse result
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 0 #10
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# GPU==1
# torchrun pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${DATA_ARGS[@]} \
#     ${EVAL_AND_LOGGING_ARGS[@]}

# > 1 GPU (1 Node)
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

# # > 1 Nodes
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_port=6105 --master_addr=node-0 --rdzv_id=f4694954-bb45-495b-9452-789e6028374a --rdzv_backend=c10d --rdzv_endpoint=node-0:6105 pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${MODEL_PARALLEL_ARGS[@]} \
#     ${DATA_ARGS[@]} \
#     ${EVAL_AND_LOGGING_ARGS[@]}
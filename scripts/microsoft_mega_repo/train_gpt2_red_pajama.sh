#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

VOCAB_FILE=./examples/tiny_local_gpt2/gpt2-vocab.json
MERGE_FILE=./examples/tiny_local_gpt2/gpt2-merges.txt

DATA_PATH=./my_red_pajama_10k_gpt2/my_red_pajama_10k_gpt2_text_document


TP_SIZE=1
PP_SIZE=1
SEQ_LEN=8192 # 4096 was used in original phi-3-min
TARGET_GLOBAL_BATCH_SIZE=8
TRAIN_ITERS=10000
DATE=$(date '+%Y-%m-%d')

echo "DATE:"$DATE
CHECKPOINT_PATH=/mnt/synthdatastore/agoswami/models_04_postlaborday/out_gpt2_red_pajama #<Specify path>
TENSORBOARD_LOG_PATH=/mnt/synthdatastore/agoswami/models_04_postlaborday/out_gpt2_red_pajama/logs #<Specify path>

mkdir -p $CHECKPOINT_PATH $TENSORBOARD_LOG_PATH


GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=localhost
MASTER_PORT=6000
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
    --num-layers 32 
    --hidden-size 3072 
    --num-attention-heads 32 
    --seq-length $SEQ_LEN 
    --max-position-embeddings $SEQ_LEN 
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --swiglu
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size $TARGET_GLOBAL_BATCH_SIZE 
    --train-iters $TRAIN_ITERS 
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
    --lr-decay-iters $TRAIN_ITERS
    --use-flash-attn
    --attention-dropout 0.1
    --use-distributed-optimizer
    # --tp-comm-overlap-rs-dgrad  # this seems to yield worse result
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP_SIZE}
	--pipeline-model-parallel-size ${PP_SIZE}
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 99,1,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --save-interval 500 
    --eval-interval 10000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOG_PATH 
)

# more env vars
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# python pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${MODEL_PARALLEL_ARGS[@]} \
#     ${DATA_ARGS[@]} \
#     ${EVAL_AND_LOGGING_ARGS[@]}

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt

# TOKENIZER=CL100kBaseBPETokenizer
TOKENIZER=GPT4oTokenizer
# TOKENIZER=GPT2BPETokenizer
# DATA_CONFIG_PATH=./examples/tiny_local_phi3/my_long_context_8k_phi3_config.json
DATA_CONFIG_PATH=./examples/tiny_local_phi3/my_red_pajama_10k_phi3_config.json


TP_SIZE=1
PP_SIZE=1
# DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE)))
# TOKENS_PER_GLOBAL_BATCH=$((1024 * 1024 * 4)) # 4M tokens as in phi-3-min
SEQ_LEN=8192 # 4096 was used in original phi-3-min
TARGET_GLOBAL_BATCH_SIZE=8
# MICRO_BATCH_SIZE=1


########################
### training horizon ###
########################
# B=1000000000 # 1B
# TOKENS_IN_BILL=200  # 200B for ablation study
# TOKENS=$(( TOKENS_IN_BILL * B ))  # 300B tokens
TRAIN_ITERS=10000

# if [[ $((MICRO_BATCH_SIZE * DP_SIZE)) -le $TARGET_GLOBAL_BATCH_SIZE ]]; then
#     echo "TARGET_GLOBAL_BATCH_SIZE ($TARGET_GLOBAL_BATCH_SIZE) is smaller than micro_batch ($MICRO_BATCH_SIZE) * dp_size ($DP_SIZE)"
# fi


##############################
### NAME/CHECKPOINT/TB/LOG ###
##############################

# MODEL_ROOT=/mnt/std-cache/users/xihlin/tmp/megatron_lm/
# LOG_ROOT=/data/users/xihlin/tmp/megatron_lm_tb
DATE=$(date '+%Y-%m-%d')
# PROJECT_NAME="${DATE}-phi3min-tp${TP_SIZE}pp${PP_SIZE}-${TOKENS_IN_BILL}b"
CHECKPOINT_PATH=/mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi3_red_pajama #<Specify path>
TENSORBOARD_LOG_PATH=/mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi3_red_pajama/logs #<Specify path>

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
    --data-path $DATA_CONFIG_PATH 
    --tokenizer-type $TOKENIZER
    --num-workers 2
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

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
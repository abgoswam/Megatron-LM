#!/bin/bash

###############
# NOTE:
#   1. Please only run this script from the main node, from the root diretory (Megatron-LM)
#   2. Please use docker: turingdev.azurecr.io/megatron-lm:202408010
#   3. To kill the task: ds_ssh pkill torchrun (kill all torchrun)
################

# hostfile is default to /job/hostfile in singularity

# a hostfile shoud look like
# "
# node-0 slots=8
# node-1 slots=8
# ....
# "

if [[ -z $HOSTFILE ]]; then
    HOSTFILE=/job/hostfile
fi

if [[ ! -f $HOSTFILE ]]; then
    echo "hostfile at $HOSTFILE is expected."
fi
NUM_NODES=$(cat $HOSTFILE| wc -l)
GPUS_PER_NODE=8

# config for multinode
MASTER_ADDR=node-0
MASTER_PORT=$((RANDOM/1000+6010)) # use random port 
RDZV_ID=$((RANDOM+1000)) # random id
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))


# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt

# TOKENIZER=CL100kBaseBPETokenizer
TOKENIZER=GPT4oTokenizer
DATA_CONFIG_PATH=./examples/phi-mini/fineweb-edu-sample-350B.json


TP_SIZE=1
PP_SIZE=1
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE)))
TOKENS_PER_GLOBAL_BATCH=$((1024 * 1024 * 4)) # 4M tokens as in phi-3-min
SEQ_LEN=8192 # 4096 was used in original phi-3-min
TARGET_GLOBAL_BATCH_SIZE=$((TOKENS_PER_GLOBAL_BATCH / SEQ_LEN))
MICRO_BATCH_SIZE=1


########################
### training horizon ###
########################
B=1000000000 # 1B
TOKENS_IN_BILL=200  # 200B for ablation study
TOKENS=$(( TOKENS_IN_BILL * B ))  # 300B tokens
TRAIN_ITERS=$(( TOKENS / (TARGET_GLOBAL_BATCH_SIZE * SEQ_LEN) ))

if [[ $((MICRO_BATCH_SIZE * DP_SIZE)) -le $TARGET_GLOBAL_BATCH_SIZE ]]; then
    echo "TARGET_GLOBAL_BATCH_SIZE ($TARGET_GLOBAL_BATCH_SIZE) is smaller than micro_batch ($MICRO_BATCH_SIZE) * dp_size ($DP_SIZE)"
fi


##############################
### NAME/CHECKPOINT/TB/LOG ###
##############################

MODEL_ROOT=/mnt/std-cache/users/xihlin/tmp/megatron_lm/
LOG_ROOT=/data/users/xihlin/tmp/megatron_lm_tb
DATE=$(date '+%Y-%m-%d')
PROJECT_NAME="${DATE}-phi3min-tp${TP_SIZE}pp${PP_SIZE}-${TOKENS_IN_BILL}b"
CHECKPOINT_PATH=$MODEL_ROOT/${PROJECT_NAME} #<Specify path>
TENSORBOARD_LOG_PATH=$LOG_ROOT/${PROJECT_NAME} #<Specify path>

mkdir -p $CHECKPOINT_PATH $TENSORBOARD_LOG_PATH


DISTRIBUTED_ARGS=(
    --rdzv_id=${RDZV_ID}
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
    --rdzv_backend=c10d
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$NUM_NODES 
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
    --init-method-std 0.01 # originally initializer_range: 0.02, intepret as 95% percentile, so std=0.01
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-4 
    --lr-decay-style linear  
    --min-lr 0
    --lr-warmup-iters 3000 
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
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOG_PATH 
)

ENV_VARS=(
    NCCL_DEBUG=WARN
    CUDA_DEVICE_MAX_CONNECTIONS=1
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1
)

cmd="${ENV_VARS[@]} \
    torchrun ${DISTRIBUTED_ARGS[@]}  pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"


###########################
### LAUNCH ON ALL NODES ###
###########################

if [[ ! `which ds_ssh` ]]; then
    # only install to get ds_ssh
    pip install --user deepspeed
fi

echo -e "\n>> Comand to be ran on all nodes"
echo -e "=================="
echo -e $cmd | sed 's/ --/ \\\n  --/g'
echo -e "==================\n"
sleep 3 # allow time to cancel if spot anything wrong

echo -e "Hostfile:"
cat $HOSTFILE


if [[ `which ds_ssh` ]]; then
    # better terminal logging
    ds_ssh "cd $PWD && unset NCCL_ASYNC_ERROR_HANDLING && $cmd" | tee -a  $TENSORBOARD_LOG_PATH/log.txt
else
    for host in `cat $HOSTFILE | awk '{print $1}'`; do
        ssh $host "cd $PWD && unset NCCL_ASYNC_ERROR_HANDLING && $cmd" | sed "s/^/$host: /" | tee -a $TENSORBOARD_LOG_PATH/log.txt &
    done
    wait
fi
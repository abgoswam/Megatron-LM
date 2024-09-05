#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
NUM_NODES=1
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

TP=1
CP=1
MODEL_PARALLEL_ARGS=(
    --context-parallel-size ${CP}
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size 1
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
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 0 #10
	--load ./weights_conversion/out_llama2_7b/
    --finetune
	--save ./weights_conversion/out_llama2_7b_save/
)

DATA_ARGS=(
    --data-path ./my_long_corpus_llama2/my_long_corpus_4096_llama2_text_document
    --tokenizer-type SentencePieceTokenizer
    --vocab-file ./weights_conversion/out_llama2_7b/tokenizer.model
    --split 949,50,1
    --no-create-attention-mask-in-dataloader
)

# > 1 GPU (1 Node)
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

# LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
# TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"
# DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"

# LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"

# torchrun $DISTRIBUTED_ARGS finetune.py \
# 	--tensor_model_parallel_size 1 \
# 	--pipeline_model_parallel_size 1 \
# 	--load ./weights_conversion/out_llama2_7b/ \
# 	--save ./weights_conversion/out_llama2_7b_save/ \
# 	--tensorboard_dir ./weights_conversion/out_llama2_7b_save/tensorboard/ \
# 	--data_path ./my_long_corpus_llama2/my_long_corpus_4096_llama2_text_document \
# 	--model_name llama2 \
# 	--tokenizer_type SentencePieceTokenizer \
#   --vocab_file=./weights_conversion/out_llama2_7b/tokenizer.model \
# 	--bf16 \
# 	--use_flash_attn \
# 	--micro_batch_size 1 \
# 	--global_batch_size 1000 \
# 	--sequence_parallel \
# 	--recompute_granularity selective \
# 	--use_checkpoint_args \
# 	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS
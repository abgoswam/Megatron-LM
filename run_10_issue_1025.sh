#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS=(
    --nproc_per_node 4 
    --nnodes 1 
    --master_addr localhost 
    --master_port 6000
)

GPT_MODEL_ARGS=(
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --num-layers 24 
    --hidden-size 128 
    --num-attention-heads 16 
    --seq-length 131072 
    --max-position-embeddings 131072 
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
)

MODEL_PARALLEL_ARGS=(
    --context-parallel-size 4
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path "./my_long_corpus_131072_gpt2_text_document"
    --vocab-file "./gpt2-vocab.json" 
    --merge-file "./gpt2-merges.txt" 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 0
)
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
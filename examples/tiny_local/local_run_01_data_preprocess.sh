
#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

#  RedPajama
python ./tools/preprocess_data.py \
    --input /mnt/syntheticpipelinetrainerv1/omni_unified_v1/data_raw/agoswami_redpajama/red_pajama_10k.jsonl \
    --output-prefix agoswami_gpt2_red_pajama \
    --vocab-file /mnt/syntheticpipelinetrainerv1/omni_unified_v1/gpt2_tokenizer_files/gpt2-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file /mnt/syntheticpipelinetrainerv1/omni_unified_v1/gpt2_tokenizer_files/gpt2-merges.txt \
    --workers 32
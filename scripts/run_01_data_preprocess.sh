
#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

python ./tools/preprocess_data.py \
    --input ./my_red_pajama_10k_gpt2/data_10k.jsonl \
    --output-prefix my_red_pajama_10k_gpt2 \
    --vocab-file ./examples/tiny_local_gpt2_4_red_pajama/gpt2-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ./examples/tiny_local_gpt2_4_red_pajama/gpt2-merges.txt \
    --workers 32
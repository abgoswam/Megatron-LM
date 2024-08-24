# python tools/preprocess_data.py \
#     --input my-corpus.json \
#     --output-prefix my-gpt2 \
#     --vocab-file gpt2-vocab.json \
#     --tokenizer-type GPT2BPETokenizer \
#     --merge-file gpt2-merges.txt \
#     --append-eod

# #if nltk isn't installed
# pip install nltk

python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --workers 32 \
       --append-eod

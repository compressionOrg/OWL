
#!/bin/bash

# Set common variables
model="Enoch/llama-7b-hf"
# sparsity_ratio=0.5
cuda_device=3

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..
python   main.py \
 --model_name_or_path "Enoch/llama-7b-hf" \
 --model "Enoch/llama-7b-hf" \
 --prune_method "wanda_zscores" \
 --sparsity_ratio 0.7 \
 --sparsity_type "unstructured" \
 --save save_test/ \
 --save_log 
#!/bin/bash

# Set common variables

model="meta-llama/Meta-Llama-3-8B"
# sparsity_ratio=0.7
sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
model_name=$(echo "$model" | awk -F'/' '{print $2}')

cuda_device=3

# Lamda_ratio=(0.01 0.02 0.05 0.08 0.1 0.2)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..
 run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 5 \
    --model $model \
    --prune_method "sparsegpt" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/sparsegpt/${model_name}_$1.log
}

for sparsity_ratio in "${sparsity_ratios[@]}"
do
    run_python_command   ${sparsity_ratio}
done 
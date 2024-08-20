#!/bin/bash

# Set common variables

model="Enoch/llama-7b-hf"

model_name=$(echo "$model" | awk -F'/' '{print $2}')

cuda_device=2

# sparsity_ratios=(0.4 0.5 0.6 0.7 0.8)
sparsity_ratio=0.7
alpha_ratios=(0.04 0.05 0.06  0.08  0.1 0.12  0.15 0.2)
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

cd ..
 run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --alpha $1 \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "wanda_csl" \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/csl/${model_name}_${sparsity_ratio}_$1.log
}

for alpha_ratio in "${alpha_ratios[@]}"
do
    run_python_command   ${alpha_ratio}
done 

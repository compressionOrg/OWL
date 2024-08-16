#!/bin/bash

# Set common variables

model="Enoch/llama-7b-hf"
# sparsity_ratio=0.7
sparsity_ratio=0.7
# alphas=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.2)

# alphas=(0.04 0.05 0.06  0.08  0.1  0.15 0.2)
alphas=(0.11 0.12 0.13 0.14)
model_name=$(echo "$model" | awk -F'/' '{print $2}')

cuda_device=3

# Lamda_ratio=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..
 run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --model $model \
    --prune_method "wanda_zscores" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --alpha $1 \
    --save_log > logs/llama/wanda_zscores/${model_name}_alpha$1.log
}

for alpha in "${alphas[@]}"
do
    run_python_command   ${alpha}
done
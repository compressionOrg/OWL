#!/bin/bash

# Set common variables
model="Enoch/llama-7b-hf"

model_name=$(echo "$model" | awk -F'/' '{print $2}')

# sparsity_ratio=0.5
cuda_device=3
sparsity_ratio=0.7

# alpha_ratios=(0.01 0.02 0.03 0.04 0.05 0.06 0.07  0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.25)

# alpha_ratios=(0.04 0.05 0.06  0.08  0.1  0.15 0.2)
alpha_ratios=(0.11 0.12 0.13 0.14)
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..

run_python_command () {
    python   lib/imp.py \
    --model "Enoch/llama-7b-hf" \
    --nsamples 5 \
    --alpha $1 \
    --sparse_ratio ${sparsity_ratio} >logs/run/${model_name}_${sparsity_ratio}_$1.log
}

for alpha_ratio in "${alpha_ratios[@]}"
do
    run_python_command $alpha_ratio
done
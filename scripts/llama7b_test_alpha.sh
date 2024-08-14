#!/bin/bash

# Set common variables
# model="Enoch/llama-7b-hf"
# sparsity_ratio=0.5
cuda_device=2,3
sparsity_ratio=0.5

alpha_ratios=(0.01 0.02 0.05 0.08 0.1 0.2)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..

run_python_command () {
    python   lib/imp.py \
    --model "Enoch/llama-7b-hf" \
    --nsamples 5 \
    --alpha $1 \
    --sparse_ratio ${sparsity_ratio} >logs/${model_name}_${sparsity_ratio}_$1.log
}

for alpha_ratio in "${alpha_ratios[@]}"
do
    run_python_command $alpha_ratio
done
#!/bin/bash

# Set common variables

# model="meta-llama/Llama-2-7b-hf"
model="Enoch/llama-7b-hf"
sparsity_ratio=0.7
# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
model_name=$(echo "$model" | awk -F'/' '{print $2}')

ks=(0.5)

cuda_device=2

# Lamda_ratio=(0.01 0.02 0.05 0.08 0.1 0.2)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..


run_wanda_csl () {
    python   main.py \
    --model_name_or_path $model \
    --grad_nsamples 10 \
    --k $1 \
    --force_compute_ratios \
    --model $model \
    --prune_method "wanda_csl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/wanda_csl/${model_name}_${sparsity_ratio}_k$1_seqlen_32.log
}



for k in "${ks[@]}"
do
    run_wanda_csl   ${k}
done 




  


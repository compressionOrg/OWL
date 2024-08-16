#!/bin/bash

# Set common variables

model="Enoch/llama-7b-hf"

model_name=$(echo "$model" | awk -F'/' '{print $2}')

cuda_device=2,3

# sparsity_ratios=(0.4 0.5 0.6 0.7 0.8)
sparsity_ratios=(0.7)
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

cd ..
 run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --alpha 0.15 \
    --grad_nsamples 5 \
    --model $model \
    --prune_method "csl" \
    --sparsity_ratio $1 \
    --conn_ratio 0.7 \
    --node_ratio 0.3 \
    --sparsity_type "unstructured" \
    --save_log 
    # --save_log > logs/llama/csl/${model_name}_$1.log
}

for sparsity_ratio in "${sparsity_ratios[@]}"
do
    run_python_command   ${sparsity_ratio}
done 



#!/bin/bash

# Set common variables

model="Enoch/llama-7b-hf"
sparsity_ratio=0.5

model_name=$(echo "$model" | awk -F'/' '{print $2}')

cuda_device=1,2

Lamda_ratio=(0.01 0.02 0.05 0.08 0.1 0.2)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..
 run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda $1 \
    --Scale 0.2 \
    --Hyper_m 5 \
    --model $model \
    --prune_method "wanda_zscores" \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "unstructured" \
    --save save_test/ \
    --save_log > logs/llama/${model_name}_$1_${sparsity_ratio}.log
}

for Lamda in "${Lamda_ratio[@]}"
do
    run_python_command   ${Lamda}
done 
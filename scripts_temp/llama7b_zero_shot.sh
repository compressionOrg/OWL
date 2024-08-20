#!/bin/bash

# Set common variables

model="Enoch/llama-7b-hf"
# model="meta-llama/Llama-2-7b-hf"
# model="meta-llama/Meta-Llama-3-8B"
# prune_method="wanda"
prune_methods=("magnitude" "sparsegpt")

model_name=$(echo "$model" | awk -F'/' '{print $2}')
# save_model="pruned/${model_name}_${prune_method}_sparsity_0.7"

cuda_device=2,3

# sparsity_ratios=(0.4 0.5 0.6 0.7 0.8)
sparsity_ratio=0.7
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

cd ..
run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --grad_nsamples 10 \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_model "pruned/${model_name}_${1}_sparsity_0.7" \
    --eval_zero_shot \
    --save_log > logs/llama/${1}/${model_name}_zero_shot_sparsity_ratio_$1.log
    # --save_log 
}

for prune_method in "${prune_methods[@]}"
do
    run_python_command   ${prune_method}
done 



#!/bin/bash

# Set common variables

model="meta-llama/Llama-2-7b-hf"

prune_method="sparsegpt_csl"


model_name=$(echo "$model" | awk -F'/' '{print $2}')
save_model="pruned/${model_name}_${prune_method}"

cuda_device=3

sparsity_ratios=(0.4 0.5 0.6 0.7 0.8)
# sparsity_ratios=(0.7)
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

cd ..
run_python_command () {
    python   main.py \
    --model_name_or_path $model \
    --grad_nsamples 10 \
    --model $model \
    --prune_method ${prune_method} \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_model "${save_model}_$1" \
    --eval_zero_shot \
    --save_log > logs/llama/${prune_method}/${model_name}_zero_shot_$1.log
}

for sparsity_ratio in "${sparsity_ratios[@]}"
do
    run_python_command   ${sparsity_ratio}
done 






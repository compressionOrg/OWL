#!/bin/bash

# Set common variables
model="Enoch/llama-7b-hf"

# sparsity_ratio=0.7
# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
sparsity_ratios=(0.1 0.2)
model_name=$(echo "$model" | awk -F'/' '{print $2}')
alphas=(0.01 0.02 0.04 0.06 0.08 0.1)
cuda_device=2,3

# Lamda_ratio=(0.01 0.02 0.05 0.08 0.1 0.2)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
cd ..
run_wanda () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "wanda" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/wanda/${model_name}_$1.log
}

run_wanda_owl () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "wanda_owl" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/wanda_owl/${model_name}_$1.log
}

run_sparsegpt () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "sparsegpt" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/sparsegpt/${model_name}_$1.log
}

run_sparsegpt_owl () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "sparsegpt_owl" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/sparsegpt_owl/${model_name}_$1.log
}

run_magnitude () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "magnitude" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/magnitude/${model_name}_$1.log
}

run_magnitude_owl () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "magnitude_owl" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/magnitude_owl/${model_name}_$1.log
}

run_wanda_csl () {
    python   main.py \
    --model_name_or_path $model \
    --alpha $2 \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "wanda_csl" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/wanda_csl/${model_name}_alpha${alpha}_sparsity$1.log
}


run_magnitude_csl () {
    python   main.py \
    --model_name_or_path $model \
    --alpha ${alpha} \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "magnitude_csl" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/magnitude_csl/${model_name}_alpha${alpha}_sparsity$1.log
}

run_sparsegpt_csl () {
    python   main.py \
    --model_name_or_path $model \
    --alpha ${alpha} \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "sparsegpt_csl" \
    --sparsity_ratio $1 \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/sparsegpt_csl/${model_name}_alpha${alpha}_sparsity$1.log
}


for sparsity_ratio in "${sparsity_ratios[@]}"
do
    for alpha in "${alphas[@]}"
    do
        run_wanda_csl   ${sparsity_ratio} ${alpha}
        # run_sparsegpt   ${sparsity_ratio}
        # run_wanda   ${sparsity_ratio}
        # run_wanda_owl   ${sparsity_ratio}
        
        # run_sparsegpt   ${sparsity_ratio}
        # run_sparsegpt_owl   ${sparsity_ratio}
        # run_magnitude   ${sparsity_ratio}
        # run_magnitude_owl   ${sparsity_ratio}
        # run_magnitude_csl   ${sparsity_ratio}
        # run_sparsegpt_csl   ${sparsity_ratio}
    done
done
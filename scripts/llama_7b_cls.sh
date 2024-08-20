#!/bin/bash

# Set common variables

# model="meta-llama/Llama-2-7b-hf"
model="Enoch/llama-7b-hf"
sparsity_ratio=0.7
# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
model_name=$(echo "$model" | awk -F'/' '{print $2}')

cuda_device=2

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
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/wanda/${model_name}_${sparsity_ratio}.log
}

run_wanda_owl () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "wanda_owl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/wanda_owl/${model_name}_${sparsity_ratio}.log
}

run_sparsegpt () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "sparsegpt" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/sparsegpt/${model_name}_${sparsity_ratio}.log
}

run_sparsegpt_owl () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "sparsegpt_owl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/sparsegpt_owl/${model_name}_${sparsity_ratio}.log
}

run_magnitude () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "magnitude" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/magnitude/${model_name}_${sparsity_ratio}.log
}

run_magnitude_owl () {
    python   main.py \
    --model_name_or_path $model \
    --Lamda 0.08 \
    --Hyper_m 7 \
    --model $model \
    --prune_method "magnitude_owl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_log > logs/llama/magnitude_owl/${model_name}_${sparsity_ratio}.log
}

run_magnitude_csl () {
    python   main.py \
    --model_name_or_path $model \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "magnitude_csl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_model "pruned/${model_name}_magnitude_csl_sparsity_${sparsity_ratio}" \
    --eval_zero_shot \
    --save_log > logs/llama/magnitude_csl/${model_name}_${sparsity_ratio}_zero_shot.log
}

run_wanda_csl () {
    python   main.py \
    --model_name_or_path $model \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "wanda_csl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_model "pruned/${model_name}_wanda_csl_sparsity_${sparsity_ratio}" \
    --eval_zero_shot \
    --save_log > logs/llama/wanda_csl/${model_name}_${sparsity_ratio}_zero_shot.log
}

run_sparsegpt_csl () {
    python   main.py \
    --model_name_or_path $model \
    --grad_nsamples 10 \
    --model $model \
    --prune_method "sparsegpt_csl" \
    --sparsity_ratio ${sparsity_ratio} \
    --sparsity_type "unstructured" \
    --save_model "pruned/${model_name}_sparsegpt_csl_sparsity_${sparsity_ratio}" \
    --eval_zero_shot \
    --save_log > logs/llama/sparsegpt_csl/${model_name}_${sparsity_ratio}_zero_shot.log
}


# run_wanda   
# run_magnitude 
# run_sparsegpt   

# run_wanda_owl 
# run_magnitude_owl 
# run_sparsegpt_owl

  

run_wanda_csl
run_magnitude_csl
run_sparsegpt_csl
   


  


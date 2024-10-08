import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer,GPT2Tokenizer
# from importlib.metadata import version
from collections import defaultdict
from lib.prune_all import  prune_wanda,prune_magnitude,prune_sparsegpt,prune_wanda_csl,get_layer_ratios,prune_sparsegpt_csl,prune_mag_csl
from lib.prune_all import prune_wanda_outlier_structure,prune_sparsegpt_outlier,prune_wanda_outlier,prune_mag_outlier
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity, find_layers
import sys
print('# of gpus: ', torch.cuda.device_count())
from pdb import set_trace as st

# spartio ratio and alpha
sparsity_mapping = {
    "wanda": {
        "0.1": 0.02,
        "0.2": 0.04,
        "0.3": 0.04,
        "0.4": 0.06,
        "0.5": 0.06,
        "0.6": 0.1,
        "0.7": 0.15,
        "0.8": 0.2
    },
    "magnitude": {
        "0.1": 0.1,
        "0.2": 0.01,
        "0.3": 0.01,
        "0.4": 0.01,
        "0.5": 0.01,
        "0.6": 0.2,
        "0.7": 0.1,
        "0.8": 0.2
        
    },
    "sparsegpt": {
        "0.1": 0.01,
        "0.2": 0.02,
        "0.3": 0.02,
        "0.4": 0.04,
        "0.5": 0.02,
        "0.6": 0.1,
        "0.7": 0.13,
        "0.8": 0.15 
    },
}

import json
import logging
import math

import random
from itertools import chain
from pathlib import Path

import datasets

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        # cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def main():


    ########################## for prune ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--model_name', type=str, help='LLaMA model_name')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--grad_nsamples', type=int, default=10, help='grad_nsamples')
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="result", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--alpha', type=float, default=0.15, help='alpha')
    parser.add_argument('--k', type=float, default=0.5, help='k-threshold')
    parser.add_argument('--use_alpha', action="store_true", help="whether to use custom alpha")
    parser.add_argument('--force_compute_ratios', action="store_true", help="whether to force compute the ratio")
    # parser.add_argument('--conn_ratio', type=float, default=0.5, help='conn_ratio')
    # parser.add_argument('--node_ratio', type=float, default=0.5, help='node_ratio')
    parser.add_argument('--eval_zero_shot', action="store_true", help="whether to zero-shot eval")
    
    
########################################### for train
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )


    #### saving parameters #####
    
    parser.add_argument(
        "--method",
        type=str,
        default=None,

    )   
    

    
    parser.add_argument(
    "--save_log", action="store_true", help="save log")
    
    parser.add_argument(
    "--layer_method", type=str, default="weight", help='layer-wise method')
    
    #### data parameters #####
    
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )
    
    
     
    parser.add_argument(
        '--Hyper_m', 
        type=float,
        default=5, )
    
    parser.add_argument(
    "--outlier_by_activation", action="store_true", help="outlier_by_activation")  
    
    
    parser.add_argument(
    "--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")  
    
    
    args = parser.parse_args()

    # use mapping sparsity ratio to alpha
    if args.use_alpha:
        args.alpha = args.alpha
    else:
        if "wanda" in args.prune_method:
            args.alpha = sparsity_mapping['wanda'][str(args.sparsity_ratio)]
        elif "magnitude" in args.prune_method:
            args.alpha = sparsity_mapping['magnitude'][str(args.sparsity_ratio)]
        elif "sparsegpt" in args.prune_method:
            args.alpha = sparsity_mapping['sparsegpt'][str(args.sparsity_ratio)]
    
    print("args.alpha",args.alpha)
    print ("args.nsamples",args.nsamples)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print("args.sparsity_type",args.sparsity_type)
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        # assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))


    model_name = args.model.split("/")[-1]
    args.model_name = model_name
    print(f"loading llm model {args.model}")
    
    # Offline load moodel
    args.model = args.cache_dir + "/models--" + args.model.replace("/", "--") + "/model"

    model = get_llm(args.model, args.cache_dir)
    
    
    print ("model is =================================================================================")
    print (model.__class__.__name__)
    print (model)
    
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)



    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    print ("target sparsity", args.sparsity_ratio)   
    
    if "csl" in args.prune_method:
        ratios = get_layer_ratios(args, model, tokenizer, device)
        
    model.eval()

    print("pruning starts")


    ############################ baseline   ############################
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ############################ csl   ############################
    elif args.prune_method == "wanda_csl":
        prune_wanda_csl(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
        
    elif args.prune_method == "sparsegpt_csl":
        prune_sparsegpt_csl(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
        
    elif args.prune_method == "magnitude_csl":
        prune_mag_csl(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
    
    ############################ csl   ############################
    
    ############################ owl   ############################
    elif args.prune_method == "wanda_owl":

        prune_wanda_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ############################ owl   ############################
    elif args.prune_method == "magnitude_owl":

        prune_mag_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)




    elif args.prune_method == "sparsegpt_owl":
    
        prune_sparsegpt_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    elif args.prune_method == "wanda_owl_structure":


        prune_wanda_outlier_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    elif args.prune_method == "dense":
        pass


    print(f" prune method is {args.prune_method}")  
    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl_test}")

    sys.stdout.flush()

    # print(f"final ppl on wikitext {ppl}")



    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")

    if args.save_log:
        dirname = "results/{}".format(args.model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        filename = f"log_{args.prune_method}_.txt"
        save_filepath = os.path.join(dirname, filename)
        with open(save_filepath, "a") as f:
            print("method\tactual_sparsity\tsparsity_pattern\talpha\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.alpha}\t{ppl_test:.4f}", file=f, flush=True)
                

    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.eval_zero_shot:
        accelerate=True
        task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "mnli",  "openbookqa"]
        num_shot = 0
        
        
        if args.save_model:
            eval_model = args.save_model
        else:
            eval_model = args.model
        results = eval_zero_shot(eval_model, task_list, num_shot, accelerate)
        model_name = eval_model.split("/")[-1]
        dirname = "eval_zero_shot"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open('{}/results_zero_shot_{}.json'.format(dirname, model_name), 'a') as file:
            json.dump(results, file, indent=2)

if __name__ == '__main__':
    main()

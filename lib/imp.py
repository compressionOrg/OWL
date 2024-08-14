import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from importlib.metadata import version
from transformers import AdamW
from datasets import load_dataset
from data import get_loaders
import torch.nn as nn 
from tqdm import tqdm
import argparse
import os
import json
from pdb import set_trace as st 

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


## TODO:
## 1. 多个nsample
## 2.

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    print("printing gpu allocation for all the layers")
    print(model.hf_device_map)
    model.seqlen = 2048
    return model

class data:
    def __init__(self, model, scale):
        self.model = model
        self.node = dict()
        self.conn_l1 = dict()
        self.conn_l2 = dict()
        self.total_conn = dict()
        self.total_node = dict()
        self.nsample = 0
        self.scale = scale
        self.device = torch.device("cpu") 
        self.data_init()

    def data_init(self):
        layers = self.model.model.layers
        for i in tqdm(range(len(layers)), desc=f"initializing the gradient list ...."):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.node[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float32, device=self.device)
                self.conn_l1[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float16, device=self.device)
                self.conn_l2[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float32, device=self.device)
    
    def update_data(self, model, nsample):
        assert nsample - self.nsample == 1, "number of samples must be incremented by 1"
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"updating the gradient of sample no: {self.nsample}"):
            layer = layers[i]
            subset = find_layers(layer)
            per_layer_conn = []
            per_layer_node = []
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.node[indexed_name] = subset[name].weight.data.clone().to(dtype=torch.float32).to(device=self.device)
                per_layer_node.append(self.node[indexed_name])
                if subset[name].weight.grad is None:
                    print(f"Error: {name} has none gradient")
                if subset[name].weight.grad is not None:
                    assert subset[name].weight.requires_grad == True, f"Required grad must be true ( {name}: {subset[name].weight.requires_grad})"
                    grad = subset[name].weight.grad.detach().clone().to(dtype=torch.float32)  # Cast to float32
                    all_zero = (torch.abs(grad)==0).all()
                    assert int(all_zero) == 0, f"all the elements in the tensor are zero.: {all_zero}"
                    assert self.conn_l1[indexed_name].shape == grad.shape, "shape mismatch"
                    
                    # 去掉scale
                    # self.conn_l1[indexed_name] = self.conn_l1[indexed_name] + torch.abs(grad*self.scale).to(device=self.device).to(dtype=torch.float16)
                    # self.conn_l2[indexed_name] = self.conn_l2[indexed_name] + torch.abs((grad*self.scale)**2).to(device=self.device)
                    
                    self.conn_l1[indexed_name] = self.conn_l1[indexed_name] + torch.abs(grad).to(device=self.device).to(dtype=torch.float16)
                    self.conn_l2[indexed_name] = self.conn_l2[indexed_name] + torch.abs((grad)**2).to(device=self.device)
                    
                    per_layer_conn.append(self.conn_l2[indexed_name])
                    
            self.total_conn[i] =  torch.cat([torch.flatten(x.cpu()) for x in per_layer_conn])
            self.total_node[i] =  torch.cat([torch.flatten(torch.abs(x.cpu())) for x in per_layer_node])
        self.nsample = nsample
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, default=5, help='no of samples used')
    parser.add_argument('--scale', type=int, default=100, help='no of samples used')
    parser.add_argument('--alpha', type=float, default=0.01, help='alpha')
    parser.add_argument('--llama_version', type=int, default=1, help='llama version used')
    parser.add_argument('--sparse_ratio', type=float, default=0.5, help='llama version used')
    parser.add_argument('--model', type=str,default='Enoch/llama-7b-hf', help='model to used') ## change
    args = parser.parse_args()
    print(f"Obtaining gradients for no of samples {args.nsamples}, scale {args.scale}")
    
    model_args = args.model
    cache_dir_args = "llm_weights"
    args.model = cache_dir_args + "/models--" + args.model.replace("/", "--") + "/model"
    model = get_llm(args.model, cache_dir_args)
    if args.llama_version == 2:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False) ## change


    layers = model.model.layers 
    # device=torch.device("cuda:0")
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    print("loading calibdation data")
    nsamples=args.nsamples
    seed=0
    dataloader, _ = get_loaders("c4",nsamples=nsamples,seed=seed,seqlen=64,tokenizer=tokenizer)
    print("dataset loading complete")
    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    optimizer.zero_grad()
    scale = args.scale
    grad_up = data(model, scale)
    nsample = 0
    model.train()
    for input_ids, labels in dataloader:
        nsample+=1
        print("making gradient computation on sample: ", nsample)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels) 
        loss = outputs.loss
        print("Printing the loss:", loss)
        loss.backward()
        grad_up.update_data(model, nsample)
        optimizer.zero_grad()
    print("Done")
    gradients_l2 = grad_up.conn_l2
    for name in gradients_l2:
        grad_sqrt = torch.sqrt(gradients_l2[name])
        gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)
    metric_conn = grad_up.total_conn
    metric_node = grad_up.total_node

    # 计算每一层sum值
    for layer in metric_conn:
         metric_conn[layer] = torch.sum(metric_conn[layer]).item()

    for layer in metric_node:
         metric_node[layer] = torch.sum(metric_node[layer]).item()
         
    # 将字典保存为JSON文件
    with open('metric_node.json', 'w') as json_file:
        json.dump(metric_node, json_file, indent=4, ensure_ascii=False)
    
    sensitivities = torch.tensor([metric_conn[x] for x in metric_conn])
    
    normalized_sensitivities = sensitivities / torch.sum(sensitivities)
    
    # 计算归一化敏感度的标准差
    std_S = torch.std(normalized_sensitivities)

    # 使用标准差来确定调节因子 beta
    beta = 1.0 / (std_S + 1e-8)  # 1e-6 是为了避免标准差为零时除零错误
    
    # 计算每层的剪枝比例
    pruning_ratios = beta * normalized_sensitivities
    
    # 将剪枝范围映射到 [-x,x] 之间
    min_ratio = torch.min(pruning_ratios)
    max_ratio = torch.max(pruning_ratios)
    pruning_ratios = (pruning_ratios - min_ratio) * (1/(max_ratio - min_ratio)*args.alpha*2)
    
    all_layer_ratio = pruning_ratios - torch.mean(pruning_ratios) + (1-args.sparse_ratio)
    
    print(all_layer_ratio.tolist())
    with open('all_layer_ratio_{}_{}.json'.format(args.sparse_ratio, args.alpha), 'w') as json_file:
        json.dump(all_layer_ratio.tolist(), json_file, indent=4, ensure_ascii=False)
    # S_max = torch.max(normalized_sensitivities)
    # # 全局剪枝比例上限和每层剪枝比例下限
    # P_global = 0.5
    # P_min = 0.1
    
    # # 找到最大归一化敏感度
    # S_max = torch.max(normalized_sensitivities)

    # # 计算归一化敏感度的期望值
    # E_S = torch.mean(normalized_sensitivities)
    
    # # 计算调节因子 beta
    # beta = E_S / S_max
    
    # # 计算每层的剪枝比例，使用对数映射公式
    # # pruning_ratios = P_global - beta * torch.log(1 + normalized_sensitivities)
    
    # # 计算每层的剪枝比例
    # pruning_ratios = P_global - beta * normalized_sensitivities

    # 确保剪枝比例在合理范围内
    # pruning_ratios = torch.clamp(pruning_ratios, P_min, P_global)

    
    # Gather all scores in a single vector and normalise
    # metric = [metric_conn[x] for x in metric_conn]
    # all_scores = torch.cat([torch.flatten(x.cpu()) for x in metric])
    
    # st()
    
    # total_mean = torch.mean(all_scores)
    # total_std = torch.std(all_scores)
    # print ("before adjustment", total_mean, total_std)
    
    # norm_factor = torch.sum(all_scores)
    # all_scores.div_(norm_factor)
    
    # norm_score = (all_scores - total_mean) / total_std
    
    # all_layer_ratio=np.array(norm_score)
    
    # all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * 0.08*2))
    
    # all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-0.5)
   
    # print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
   
    # print ("after adjustment",all_layer_ratio)
    
    # with open('all_layer_ratio.json', 'w') as json_file:
    #     json.dump(all_layer_ratio, json_file, indent=4, ensure_ascii=False)
    
    # st()
    # for name in gradients_l2:
    #     grad_sqrt = torch.sqrt(gradients_l2[name])
    #     gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)
    # model_name = os.path.basename(args.model)
    # if not os.path.exists(f'./gradients/llama{args.llama_version}'):
    #     os.makedirs(f'./gradients/llama{args.llama_version}')
    # with open(f'./gradients/llama{args.llama_version}/gradients_aggregrate_norm_l2_model_{model_name}.pth', 'wb') as f:
    #     torch.save(gradients_l2, f)
    # with open(f'./gradients/llama{args.llama_version}/gradients_aggregrate_norm_l1_model_{model_name}.pth', 'wb') as f:
    #     torch.save(grad_up.gradients_l1, f)
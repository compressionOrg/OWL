import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
from pdb import set_trace as st 
from collections import defaultdict
from .utils import get_entropy,normalize,get_cv, cal_mhl_dis, get_z_scores_sum, get_z_scores
import json
from transformers import AdamW
from .layer import Layer
from .utils import find_layers, prepare_calibration_input_opt,return_given_alpha,prepare_calibration_input,check_sparsity,check_outlier_mean,check_outlier
from pdb import set_trace as st 


def get_layer_metric(args, model, tokenizer, device):
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.grad_nsamples,seed=args.seed,seqlen=64,tokenizer=tokenizer)
    print("dataset loading complete")
    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    optimizer.zero_grad()
    
    layer_metric = Layer(model)
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
        layer_metric.update_layer(model, nsample)
        optimizer.zero_grad()
    print("Done")
    return layer_metric


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    print ("model",model)
    
    layers = model.model.layers
    
    print (layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def get_layer_ratio_by_grad(args, model, tokenizer, device=torch.device("cuda:0")):
    all_layer_ratio=[]
    all_layer_res = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
        # layer_ametric = []
        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            # Wanda
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            # RIA
            # W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**0.5
            
            # Activation 
            # activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            # layer_ametric.append(activation_data) 
            
            layer_wmetric.append(W_metric) 
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
                 
        layer_metric = torch.cat([torch.flatten(x.cpu()) for x in layer_metric])
        # layer_ametic = torch.cat([torch.flatten(x.cpu()) for x in layer_ametric])
        # OWL
        # for out_ratio in [args.Hyper_m]:
            
        #     out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
        #     print ("layer outlier ratio",out_ratio,out_ratio_layer)
            
        all_layer_res.append(layer_metric)
        
    all_layer_metric = torch.cat([torch.flatten(x.cpu()) for x in all_layer_res])
    total_mean = torch.mean(all_layer_metric)
    total_std = torch.std(all_layer_metric)
    print ("before adjustment", total_mean, total_std)
    
    
    for layer_wmetric in all_layer_res:
        per_layer_ratio = get_z_scores_sum(layer_wmetric, total_mean.item(), total_std.item())
        print ("layer zscores ratio",per_layer_ratio)
        all_layer_ratio.append(per_layer_ratio)
    print ("all layer zscores ratio",all_layer_ratio)
 
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
   
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
   
    print ("after adjustment",all_layer_ratio)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    return all_layer_ratio


def get_layer_ratio_by_imp(args, model, tokenizer, device=torch.device("cuda:0")):
    
    all_layer_ratio=[]
    all_layer_res = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if "opt" in args.model:
        layers=model.model.decoder.layers
    
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
                    
        layer_wmetric=[]

        for name in subset:
            W_metric = subset[name].weight.data.clone()
            layer_wmetric.append(W_metric) 
                   
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        all_layer_res.append(layer_wmetric)
        
    all_layer_metric = torch.cat([torch.flatten(x.cpu()) for x in all_layer_res])
    total_mean = torch.mean(all_layer_metric)
    total_std = torch.std(all_layer_metric)
    print ("before adjustment", total_mean, total_std)
    
    all_norm_wmetric = []
    for layer_wmetric in all_layer_res:
        norm_layer_wmetric = (layer_wmetric - total_mean.item()) / total_std.item()
        all_norm_wmetric.append(norm_layer_wmetric)
        
        per_layer_ratio = get_z_scores_sum(layer_wmetric, total_mean.item(), total_std.item())
        print ("layer zscores ratio",per_layer_ratio)
        all_layer_ratio.append(per_layer_ratio)
    
    print ("all layer zscores ratio",all_layer_ratio)
 
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio= all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
   
    print (all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
   
    print ("normalization ratio is",all_layer_ratio)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    return all_layer_ratio
  

def get_layer_ratio_by_weight(args, model, tokenizer, device=torch.device("cuda:0")):
    
    all_layer_ratio=[]
    all_layer_res = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if "opt" in args.model:
        layers=model.model.decoder.layers
    
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
                    
        layer_wmetric=[]

        for name in subset:
            W_metric = subset[name].weight.data.clone()
            layer_wmetric.append(W_metric) 
                   
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        all_layer_res.append(layer_wmetric)
        
    all_layer_metric = torch.cat([torch.flatten(x.cpu()) for x in all_layer_res])
    total_mean = torch.mean(all_layer_metric)
    total_std = torch.std(all_layer_metric)
    print ("before adjustment", total_mean, total_std)
    
    all_norm_wmetric = []
    for layer_wmetric in all_layer_res:
        norm_layer_wmetric = (layer_wmetric - total_mean.item()) / total_std.item()
        all_norm_wmetric.append(norm_layer_wmetric)
        
        per_layer_ratio = get_z_scores_sum(layer_wmetric, total_mean.item(), total_std.item())
        print ("layer zscores ratio",per_layer_ratio)
        all_layer_ratio.append(per_layer_ratio)
    
    print ("all layer zscores ratio",all_layer_ratio)
 
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio= all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
   
    print (all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
   
    print ("normalization ratio is",all_layer_ratio)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    return all_layer_ratio

def get_layer_ratios(args, model, tokenizer, metric, device=torch.device("cuda:0")):
    metric_conn = metric.total_conn
    metric_node = metric.total_node
    # st()
     
    ## TODO:================C2========
    # 1. 对每一层进行zscore计算（或比值）
    # 2. 计算每一层的z分数绝对值大于 1 的层中权重的比例，评估重要性
    # 3.根据重要性分配剪枝比例
    # 初始化字典来保存剪枝比例
    layer_conn = {}
    # 1. 对每一层进行z-score计算

    for layer_name, metric in metric_conn.items():
        per_layer_mean = torch.mean(metric_conn[layer_name])
        per_layer_std = torch.std(metric_conn[layer_name])
        z_scores = (metric - per_layer_mean) / per_layer_std
        # 3. 计算每一层的z分数绝对值小于1的权重比例
        imp_ratios = torch.sum(torch.abs(z_scores) < 1).float() / metric.numel()
        layer_conn[layer_name] = imp_ratios
    
    # 4. 根据重要性分配剪枝比例
    total_conn = sum(layer_conn.values())
    ratio_conn = {}
    for layer in layer_conn:
        conn_ratio = layer_conn[layer] / total_conn
        ratio_conn[layer] = conn_ratio
    
    
    layer_node = {}
    # 1. 对每一层进行z-score计算
    for layer_name, metric in metric_node.items():
        per_layer_mean = torch.mean(metric_node[layer_name])
        per_layer_std = torch.std(metric_node[layer_name])
        z_scores = (metric - per_layer_mean) / per_layer_std
        # 3. 计算每一层的z分数绝对值小于1的权重比例
        imp_ratios = torch.sum(torch.abs(z_scores) < 1).float() / metric.numel()
        layer_node[layer_name] = imp_ratios
    
    # 4. 根据重要性分配剪枝比例
    total_node = sum(layer_node.values())
    ratio_node = {}
    for layer in layer_node:
        node_ratio = layer_node[layer] / total_node
        ratio_node[layer] = node_ratio
    
    
    # Step 2: Compute the  sum
    ratio = {}
    for key in ratio_conn:
        ratio[key] = ratio_conn[key] * args.conn_ratio + ratio_node[key] * args.node_ratio
    # st()
    imp_ratios = torch.tensor(list(ratio.values()))
    min_ratio = torch.min(imp_ratios)
    max_ratio = torch.max(imp_ratios)
    scaled_ratios = (imp_ratios - min_ratio) * (1/(max_ratio - min_ratio)*args.alpha*2)

    
    all_layer_ratio = scaled_ratios - torch.mean(scaled_ratios) + (1-args.sparsity_ratio)
    print(all_layer_ratio.tolist())
    return all_layer_ratio.tolist()

def prune_wanda_csl(args, model, tokenizer, metric, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    ratio = get_layer_ratios(args, model, tokenizer, metric, device=device)
    # with open("all_layer_ratio_{}_{}.json".format(args.sparsity_ratio, args.alpha), 'r', encoding='utf-8') as json_file:
    #     ratio = json.load(json_file)
    # ratio = []
    all_layer_ratio = np.array(ratio)
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__: 
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:  
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()  

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # layer_sparsity_ratio= all_layer_ratio[i]
            layer_sparsity_ratio= 1 - all_layer_ratio[i]
            
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda_zscores(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    # all_layer_ratio = get_layer_ratio_by_weight(args, model, tokenizer, device=torch.device("cuda:0"))
    with open("all_layer_ratio_{}_{}.json".format(args.sparsity_ratio, args.alpha), 'r', encoding='utf-8') as json_file:
        ratio = json.load(json_file)
    # ratio = []
    all_layer_ratio = np.array(ratio)
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__: 
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:  
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()  

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            # TODO:add entropy
            # pr = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=0)
            # pc = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=1).reshape(-1, 1)
            # W_metric = torch.abs((-pr * torch.log(pr)) - (pc * torch.log(pc)))
            
            # TODO:add wentropy
            # # 定义一个很小的常数 epsilon 避免计算对数时的问题
            # epsilon = 1e-10
            # W_metric += epsilon
            # # 归一化张量使其元素和为1
            # probabilities = W_metric / W_metric.sum()
            # # 计算概率分布的对数
            # log_probabilities = torch.log(probabilities)
            # # 为每个元素计算熵
            # H = -probabilities * log_probabilities
            # W_metric = torch.abs(W_metric * H)
          
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # layer_sparsity_ratio= all_layer_ratio[i]
            layer_sparsity_ratio= 1 - all_layer_ratio[i]
            
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def test(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    # all_layer_ratio = get_layer_ratio_by_weight(args, model, tokenizer, device=torch.device("cuda:0"))
    # with open("all_layer_ratio_{}_{}.json".format(args.sparsity_ratio, args.alpha), 'r', encoding='utf-8') as json_file:
    #     ratio = json.load(json_file)
    ratio = [0.7102346420288086, 0.6065380573272705, 0.5505706071853638, 0.35587194561958313, 0.3565565347671509, 0.3276030719280243, 0.32468506693840027, 0.32064902782440186, 0.32487189769744873, 0.31859976053237915, 0.300719290971756, 0.29623422026634216, 0.29466789960861206, 0.27614328265190125, 0.2711314558982849, 0.2731832265853882, 0.2523508667945862, 0.2445002645254135, 0.24720817804336548, 0.23552346229553223, 0.2253788709640503, 0.22700028121471405, 0.22049733996391296, 0.2238377034664154, 0.21733179688453674, 0.21400253474712372, 0.2102346122264862, 0.21475712954998016, 0.2246401309967041, 0.2243506908416748, 0.2548102140426636, 0.2553164064884186]
    all_layer_ratio = np.array(ratio)
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__: 
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:  
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()  

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            # TODO:add entropy
            # pr = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=0)
            # pc = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=1).reshape(-1, 1)
            # W_metric = torch.abs((-pr * torch.log(pr)) - (pc * torch.log(pc)))
            
            # TODO:add wentropy
            # # 定义一个很小的常数 epsilon 避免计算对数时的问题
            # epsilon = 1e-10
            # W_metric += epsilon
            # # 归一化张量使其元素和为1
            # probabilities = W_metric / W_metric.sum()
            # # 计算概率分布的对数
            # log_probabilities = torch.log(probabilities)
            # # 为每个元素计算熵
            # H = -probabilities * log_probabilities
            # W_metric = torch.abs(W_metric * H)
          
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # layer_sparsity_ratio= all_layer_ratio[i]
            layer_sparsity_ratio= 1 - all_layer_ratio[i]
            
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()



def prune_wanda_new(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    
    
    
    all_layer_ratio=[]
    all_subset_ratio = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            # inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]
        subset_dict = {}

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)  
            subset_dict[subset[name]] = W_metric
            

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        
        
        all_subset_ratio.append(subset_dict)
    
        
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)
        # TODO
        # entropy
        # out_ratio_layer = get_entropy(layer_wmetric)
        
        # sum
        # out_ratio_layer = torch.sum(layer_wmetric)
        
        # cv
        # out_ratio_layer = get_cv(layer_wmetric)
        # st()
        # get Mahalanobis distances
        # layer_wmetric_2d = torch.cat(layer_wmetric)
        # out_ratio_layer = cal_mhl_dis(layer_wmetric_2d)
        
        all_layer_ratio.append(out_ratio_layer)
        
        
        
    print ("subset",all_subset_ratio)
    

    print ("before adjustment",all_layer_ratio)

    # st()

    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    # all_layer_ratio = normalize(all_layer_ratio, args.Scale)
    # all_layer_ratio=all_layer_ratio+(1-args.sparsity_ratio)
    # st()
    
    # owl
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio - np.mean(all_layer_ratio) +(1-args.sparsity_ratio)
   
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
    
                
        
    
    print ("after adjustment",all_layer_ratio  )
    


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    
    
    
    
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            # inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        

        for name in subset:
            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            # TODO:使用不同方法进行稀疏化
            # pr = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=0)
            # pc = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=1).reshape(-1, 1)
            # W_metric = torch.abs((-pr * torch.log(pr)) - (pc * torch.log(pc)))
            

            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            

            layer_sparsity_ratio= 1-all_layer_ratio[i]
            
            
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()






def prune_wanda_outlier_structure_special(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data)
            

            layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])


        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print ("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)
    

    all_layer_ratio=np.round(all_layer_ratio)



 

    for i in range(len(all_layer_ratio)):
        if all_layer_ratio[i] ==1.0:
            all_layer_ratio[i]=2.0
    
    all_layer_ratio=prune_n-all_layer_ratio


    

    print ("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()







def prune_mag_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    
    
    
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            
            
            
 
            layer_wmetric.append(W_metric)    
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps



        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        
        


    print ("before adjustment",all_layer_ratio)

    

    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
    
                
        
    
    print ("after adjustment",all_layer_ratio  )
    
    

    
    
    ############## prune


    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    print (layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*layer_sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
                



def prune_wanda_outlier_structure(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data)
            

            layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print ("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)

    all_layer_ratio=np.round(all_layer_ratio)

    
    all_layer_ratio=prune_n-all_layer_ratio
    

    print ("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                        
            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    
    
    
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            # inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
            # inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        # get_sensitivity(layer_wmetric)
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        
        


    print ("before adjustment",all_layer_ratio)

    
 
 
    


    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    # st()
    # st()
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
   
   
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
    # st()
    # st()
   
    
                
        
    
    print ("after adjustment",all_layer_ratio  )
    


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    
    
    
    
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            # inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        

        for name in subset:
            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            # TODO:add entropy
            # pr = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=0)
            # pc = torch.abs(W_metric)/torch.sum(torch.abs(W_metric), dim=1).reshape(-1, 1)
            # W_metric = torch.abs((-pr * torch.log(pr)) - (pc * torch.log(pc)))
            
            # TODO:add wentropy
            # # 定义一个很小的常数 epsilon 避免计算对数时的问题
            # epsilon = 1e-10
            # W_metric += epsilon
            # # 归一化张量使其元素和为1
            # probabilities = W_metric / W_metric.sum()
            # # 计算概率分布的对数
            # log_probabilities = torch.log(probabilities)
            # # 为每个元素计算熵
            # H = -probabilities * log_probabilities
            # W_metric = torch.abs(W_metric * H)

            
            
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_sparsity_ratio= 1-all_layer_ratio[i]
            
            
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()





@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()






@torch.no_grad()
def prune_sparsegpt_outlier(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    
    device=dev
    
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_wmetric.append(W_metric)
            
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        




    print ("before adjustment",all_layer_ratio)



    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
 
    
                
        
    
    
    print ("after adjustment",all_layer_ratio  )
    
    



    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    print('Starting ...')


    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)





    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):



        layer_sparsity_ratio= 1-all_layer_ratio[i]
        
        
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01


        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs,  position_ids = inps.to(dev), outs.to(dev),  position_ids.to(dev)
            # inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(layer_sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps








    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

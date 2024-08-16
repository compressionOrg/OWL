import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
from pdb import set_trace as st 


def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache
    
    position_ids=None

    return inps, outs, attention_mask, position_ids 




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



def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_sparsity_mask(mask):


    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()



    print(f" density {float(count)/total_params:.6f}")



def check_outlier(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.max(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio




def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio



def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
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
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity











def get_entropy1(weights):
    weights = torch.abs(weights)
    weights=np.array(weights)
    weights = (weights - weights.min()) * (1/(weights.max() - weights.min()))
    epsilon = 1e-20
    weights += epsilon
    probabilities = weights / weights.sum()
    log_probabilities = np.log(probabilities)
    H = -probabilities * log_probabilities
    # H = H / np.sqrt(np.sum(H ** 2))
    entropy = H.sum()
    return entropy

def get_cv(weights):
    # 计算均值和标准差
    mean = torch.mean(weights)
    std = torch.std(weights)
    return std *(1/mean) 
    

def get_entropy(weights):
    return entropy(weights)

def pearson_correlation(vector1, vector2):
    # 转换为numpy数组
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # 计算均值
    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)
    
    # 计算分子部分 (协方差)
    covariance = np.sum((vector1 - mean1) * (vector2 - mean2))
    
    # 计算分母部分 (标准差的乘积)
    std_dev1 = np.sqrt(np.sum((vector1 - mean1) ** 2))
    std_dev2 = np.sqrt(np.sum((vector2 - mean2) ** 2))
    
    # 计算皮尔逊相关系数
    correlation = covariance / (std_dev1 * std_dev2)
    
    return correlation

def normalize(data, x=1):
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 标准化数据
    z_scores = (data - mean) / std

    # 调整数据，使其总和为0
    adjusted_z_scores = z_scores - np.mean(z_scores)
    
    # 计算缩放因子，使数据范围在 [-x, x] 之间
    max_abs_value = np.max(np.abs(adjusted_z_scores))
    scale_factor = -x / max_abs_value
    
    # 缩放数据
    scaled_z_scores = adjusted_z_scores * scale_factor
    
    return scaled_z_scores

def cal_mhl_dis(X):
    # 转换为浮点类型
    X = X.float()
    
    # 计算均值向量
    mean = torch.mean(X, dim=0)
    
    # 中心化数据
    centered_X = X - mean

    # 计算协方差矩阵
    cov_matrix = torch.mm(centered_X.t(), centered_X) / (X.size(0) - 1)
    
    # 计算协方差矩阵的逆矩阵
    inv_cov_matrix = torch.inverse(cov_matrix)
    
    # 计算每个数据点的 Mahalanobis 距离
    mahalanobis_distances = []
    for x in centered_X:
        distance = torch.sqrt(torch.dot(x, torch.mv(inv_cov_matrix, x)))
        mahalanobis_distances.append(distance.item())
    
    return mahalanobis_distances

def get_z_scores(weights):
    # 计算均值和标准差
    mean = torch.mean(weights)
    std = torch.std(weights)
    return (weights - mean) / std  

def get_z_scores_sum(weights, mean, std):
    # 计算均值和标准差
    # mean = torch.mean(weights)
    # std = torch.std(weights)
    count = torch.sum((weights - mean) / std < 2).item()
    return count / weights.numel()


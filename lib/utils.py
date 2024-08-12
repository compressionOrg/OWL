import numpy as np
from scipy.stats import entropy
import torch
from pdb import set_trace as st 

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


import torch
import torch.nn as nn
import numpy as np

# 假设我们有一个简单的神经网络定义
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(50, 50)
        self.layer5 = nn.Linear(150, 150)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
# 实例化模型
model = SimpleModel()
# 假设我们有每一层的敏感度值
layer_sensitivities = torch.tensor([0.1, 0.5, 0.2, 0.05, 0.15], dtype=torch.float32)

# 全局稀疏率目标
global_sparsity = 0.5  # 剪掉50%的参数

# 只计算每层的权重参数数量，不包括偏置
layer_param_counts = torch.tensor([
    param.numel() for name, param in model.named_parameters() if 'weight' in name
], dtype=torch.float32)

# 计算每一层的权重，即基于敏感度的权重
inverse_sensitivity = 1 / (layer_sensitivities + 1e-8)  # 防止除以零
normalized_inverse_sensitivity = inverse_sensitivity / inverse_sensitivity.sum()

# 初步计算每一层的稀疏率（尚未调整）
initial_layer_sparsities = global_sparsity * normalized_inverse_sensitivity

# 调整稀疏率以确保全局稀疏率目标
adjusted_layer_sparsities = global_sparsity * (initial_layer_sparsities / (initial_layer_sparsities * layer_param_counts).sum() * layer_param_counts)

# 打印每一层的稀疏率
for i, sparsity in enumerate(adjusted_layer_sparsities):
    print(f"Layer {i + 1} Sparsity: {sparsity:.4f}")

# 根据计算出的稀疏率进行裁剪
def prune_layer(weights, sparsity):
    # 按稀疏率剪枝，保留最重要的部分
    num_params_to_prune = int(weights.numel() * sparsity)
    if num_params_to_prune > 0:
        threshold = torch.topk(torch.abs(weights.view(-1)), num_params_to_prune, largest=False).values.max()
        mask = torch.abs(weights) >= threshold
        pruned_weights = weights * mask
    else:
        pruned_weights = weights
    return pruned_weights

# 对模型的每一层权重进行裁剪
with torch.no_grad():
    pruned_weights = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:  # 只对权重进行裁剪
            pruned_weights[name] = prune_layer(param, adjusted_layer_sparsities[i])
            param.copy_(pruned_weights[name])

# 打印每层裁剪后的参数数量
for name, param in pruned_weights.items():
    print(f"{name}: Pruned Weights Count: {param.nonzero().size(0)} / {param.numel()}")
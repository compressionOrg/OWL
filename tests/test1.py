import numpy as np

# 假设我们有一个神经网络，每一层的敏感度如下
layer_sensitivities = np.array([0.1, 0.5, 0.2, 0.05, 0.15])  # 每层的Connection Sensitivity

# 全局稀疏率目标
global_sparsity = 0.5  # 剪掉50%的参数

# 计算每一层的非均匀稀疏率
# 这里我们使用线性分配策略，稀疏率与敏感度成反比
inverse_sensitivity = 1 / (layer_sensitivities + 1e-8)  # 防止除以零
normalized_inverse_sensitivity = inverse_sensitivity / np.sum(inverse_sensitivity)
layer_sparsities = global_sparsity * normalized_inverse_sensitivity

# 打印每一层的稀疏率
for i, sparsity in enumerate(layer_sparsities):
    print(f"Layer {i + 1} Sparsity: {sparsity:.4f}")

# 根据计算出的稀疏率进行裁剪
def prune_layer(weights, sparsity):
    # 按稀疏率剪枝，保留最重要的部分
    num_params_to_prune = int(weights.size * sparsity)
    if num_params_to_prune > 0:
        threshold = np.sort(np.abs(weights).ravel())[num_params_to_prune]
        mask = np.abs(weights) >= threshold
        pruned_weights = weights * mask
    else:
        pruned_weights = weights
    return pruned_weights

# 假设我们有一个简单的模型，每层的权重如下
weights = {
    'layer1': np.random.randn(100, 100),
    'layer2': np.random.randn(200, 200),
    'layer3': np.random.randn(100, 100),
    'layer4': np.random.randn(50, 50),
    'layer5': np.random.randn(150, 150),
}

# 对每一层权重进行裁剪
pruned_weights = {}
for i, (layer_name, w) in enumerate(weights.items()):
    pruned_weights[layer_name] = prune_layer(w, layer_sparsities[i])

# 打印每层裁剪后的参数数量
for layer_name, w in pruned_weights.items():
    print(f"{layer_name}: Pruned Weights Count: {np.count_nonzero(w)} / {w.size}")
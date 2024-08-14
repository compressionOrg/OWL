import numpy as np

def normalize(data, x):
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 标准化数据
    z_scores = (data - mean) / std

    # 调整数据，使其总和为0
    adjusted_z_scores = z_scores - np.mean(z_scores)
    
    # 计算缩放因子，使数据范围在 [-x, x] 之间
    max_abs_value = np.max(np.abs(adjusted_z_scores))
    scale_factor = x / max_abs_value
    
    # 缩放数据
    scaled_z_scores = adjusted_z_scores * scale_factor
    
    return scaled_z_scores

# 示例数据
data = np.array([1.983, 4.344, 5.345, 8.223, 20.222])
x = 0.2  # 将范围限制在 [-1, 1]

# 调用函数
normalized_data_within_range = normalize_to_zero_sum_within_range(data, x)

# 打印结果
print("原始数据:", data)
print("标准化并缩放后的数据:", normalized_data_within_range)
print("总和:", np.sum(normalized_data_within_range))
print("最大值:", np.max(normalized_data_within_range))
print("最小值:", np.min(normalized_data_within_range))
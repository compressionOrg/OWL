import numpy as np
from scipy.stats import entropy

def calculate_entropy(model):
    entropies = []
    for param in model.parameters():
        param_flat = param.detach().cpu().numpy().flatten()
        value, counts = np.unique(param_flat, return_counts=True)
        entropies.append(entropy(counts, base=2))
    return sum(entropies)

original_entropy = calculate_entropy(original_model)
compressed_entropy = calculate_entropy(compressed_model)

compression_rate = original_entropy / compressed_entropy
print(f"Compression Rate: {compression_rate:.2f}")
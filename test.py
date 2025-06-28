import torch

tensor_a = torch.tensor([1, 2, 3, 4, 5])

tensor_b = torch.tensor([1, 5, 3, 4, 0])

# 生成布尔掩码

mask = tensor_a == tensor_b

print(mask)
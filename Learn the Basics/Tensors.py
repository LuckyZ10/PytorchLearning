# 载入pytorch模块
import torch
import numpy as np

# 直接从数据创建张量
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# 从numpy数组创建张量
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# 从另一个张量创建张量
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# 张量的属性
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 张量的操作
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"Device tensor is stored on: {tensor.device}")


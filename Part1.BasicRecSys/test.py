import torch
import torch.nn as nn



x = torch.randn(10, 15)
x = x.unsqueeze(2) # 将输入张量扩展为 [batch_size, n_features, 1]
x_transpose = x.transpose(1, 2) # 计算输入的转置，形状为 [batch_size, 1, n_features]
x_cross = torch.bmm(x, x_transpose) # 计算每个样本的二次交叉矩阵乘法，结果形状为 [batch_size, n_features, n_features]
w2 = torch.randn(15, 15)
print(x_cross.shape)
w1 = torch.randn(15, 1)

tot = torch.sum(w2 * x_cross, dim=(1, 2),keepdim=True)

print(tot.shape)



# cross_out = torch.sum(w2 * x_cross, dim=(1, 2)).reshape(-1, 1) # 对交叉结果和 w2 做 Hadamard 乘积，并求和
# print(cross_out.shape)
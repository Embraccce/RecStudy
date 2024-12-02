"""
直接将隐向量当作特征的embedding，通过端到端的训练方式来训练FM（Factorization Machine）模型。
"""
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, accuracy_score



class FM(nn.Module):
    def __init__(self, n_features,n_factors):
        """
        初始化 FM 类，定义模型的参数。
        
        参数:
        - n_features (int): 输入特征的数量。
        - n_factors (int): 隐因子的维度。
        """
        super(FM, self).__init__()
        self.w0 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(1, 1)))
        self.w1 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(n_features, 1)))
        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(n_features, n_factors)))

    def FMcross(self, feature_embs):
        # feature_embs: [batch_size, n_features, n_factors]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2 # [batch_size, n_factors]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1) # [batch_size, n_factors]
        output = square_of_sum - sum_of_square # [batch_size, n_factors]
        output = 0.5 * torch.sum(output, dim=1, keepdim=True) # [batch_size, 1]

        return output
    

    def forward(self, x):
        lr_out = self.w0 + torch.matmul(x, self.w1)
        cross_out = self.FMcross(x)
        logits = torch.sigmoid(lr_out + cross_out)
        return logits

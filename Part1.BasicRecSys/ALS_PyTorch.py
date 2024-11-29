import numpy as np
import torch
import torch.nn as nn

class ALS(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super(ALS, self).__init__()
        
        # 随机初始化用户和物品的嵌入向量，并将它们的L2范数约束在1以内
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        
        # 使用Sigmoid激活函数，将点积相似度映射到 [0, 1] 的范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        """
        前向传播方法，用于计算用户和物品之间的相似度。
        
        参数:
        - user_ids: 用户索引，形状为 [batch_size]
        - item_ids: 物品索引，形状为 [batch_size]
        
        返回:
        - similarity_scores: 用户和物品之间的相似度分数，范围在 [0, 1]，形状为 [batch_size]
        """
        
        # 从嵌入层提取用户和物品的嵌入向量
        # user_embeddings 和 item_embeddings 的形状均为 [batch_size, dim]
        user_embeddings = self.users(user_ids)
        item_embeddings = self.items(item_ids)
        
        # 计算用户和物品嵌入的点积相似度，形状为 [batch_size]
        dot_product_similarity = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        # 使用Sigmoid将点积相似度映射到 [0, 1] 范围，得到相似度得分
        similarity_scores = self.sigmoid(dot_product_similarity)
        
        return similarity_scores
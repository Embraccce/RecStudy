import numpy as np
import torch
import torch.nn as nn
import dataloader
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score

class ALS(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(ALS, self).__init__()
        
        # 随机初始化用户和物品的嵌入向量，并将它们的L2范数约束在1以内
        self.users = nn.Embedding(n_users, n_factors, max_norm=1)
        self.items = nn.Embedding(n_items, n_factors, max_norm=1)
        
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
        # user_embeddings 和 item_embeddings 的形状均为 [batch_size, n_factors]
        user_embeddings = self.users(user_ids)
        item_embeddings = self.items(item_ids)
        
        # 计算用户和物品嵌入的点积相似度，形状为 [batch_size]
        dot_product_similarity = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        # 使用Sigmoid将点积相似度映射到 [0, 1] 范围，得到相似度得分
        similarity_scores = self.sigmoid(dot_product_similarity)
        
        return similarity_scores
    

def load_data(file_path,test_ratio):
    """加载数据集并返回相关集合"""
    user_set, item_set, train_set, test_set = dataloader.read_triples(file_path=file_path,test_ratio=test_ratio)
    return user_set, item_set, train_set, test_set


def initialize_model(user_count, item_count, factors_dim):
    """初始化 ALS 模型"""
    return ALS(n_users=user_count, n_items=item_count, n_factors=factors_dim)



def train_model(model, train_set, batch_size, epochs, lr):
    """训练 ALS 模型"""
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 开始训练过程
    for epoch in range(epochs):
        epoch_loss = 0.0  # 记录每个 epoch 的总损失
        
        # 使用 DataLoader 加载训练数据
        for u, i, r in DataLoader(train_set, batch_size=batch_size, shuffle=True):
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播计算预测评分
            r_pred = model.forward(u, i)
            
            # 计算损失
            loss = criterion(r_pred, r.float())
            epoch_loss += loss.item()  # 累加损失值
            
            # 反向传播并更新参数
            loss.backward()
            optimizer.step()
        
        # 输出当前 epoch 的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / (len(train_set)//batch_size)}")

    return model


def evaluate_model(model, test_set, threshold=0.5):
    """
    评估 ALS 模型的性能，计算 precision, recall 和 accuracy。

    参数:
    - model: 训练好的 ALS 模型
    - test_set: 测试集，形状为 [N, 3]，其中每行是 (user_id, item_id, rating)
    - threshold: 阈值，用于将预测分数转化为二分类标签

    返回:
    - metrics: 字典，包含 precision, recall 和 accuracy
    """
    test_set = np.array(test_set)

    # 拆分测试集
    u = torch.tensor(test_set[:, 0], dtype=torch.long)
    i = torch.tensor(test_set[:, 1], dtype=torch.long)
    r = test_set[:, 2]  # 测试集真实评分

    # 模型预测
    with torch.no_grad():
        r_pred = model.forward(u, i).numpy()

    # 将预测分数转为二分类标签（使用给定阈值）
    r_pred = (r_pred >= threshold).astype(int)

    # 计算评估指标
    precision = precision_score(r, r_pred)
    recall = recall_score(r, r_pred)
    accuracy = accuracy_score(r, r_pred)

    # 返回指标字典
    metrics = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }
    return metrics



if __name__ == "__main__":
    # 配置参数
    file_path = "./dataset/ml-100k/rating_index.tsv"
    factors_dim = 64
    batch_size = 1024
    epochs = 20
    lr = 0.01
    test_ratio = 0.1

    # 步骤 1: 加载数据
    user_set, item_set, train_set, test_set = load_data(file_path, test_ratio)

    # 步骤 2: 初始化模型
    model = initialize_model(len(user_set), len(item_set), factors_dim)

    # 步骤 3: 训练模型
    model = train_model(model, train_set, batch_size, epochs, lr)

    # 步骤 4: 评估模型性能
    res = evaluate_model(model, test_set)
    print(f"Evaluation Results - precision: {res['precision']}, recall: {res['recall']}, accuracy: {res['accuracy']}")



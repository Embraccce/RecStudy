import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.dataloader as dataloader
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score


class ALS_MLP(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super(ALS_MLP,self).__init__()
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        self.items = nn.Embedding(n_items, dim, max_norm=1)

        self.denseLayer1 = self.dense_layer(dim * 2, dim)
        self.denseLayer2 = self.dense_layer(dim, dim // 2)
        self.denseLayer3 = self.dense_layer(dim // 2, 1)

        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Tanh()
        )

    def forward(self, u, i):
        u = self.users(u)
        i = self.items(i)

        ui = torch.cat([u, i], dim=1)
        ui = self.denseLayer1(ui)
        ui = self.denseLayer2(ui)
        ui = self.denseLayer3(ui)

        ui = F.dropout(ui, training=self.training)
        ui = torch.squeeze(ui)

        logit = self.sigmoid(ui)
        return logit
    

def load_data(file_path,test_ratio):
    """加载数据集并返回相关集合"""
    user_set, item_set, train_set, test_set = dataloader.read_triples(file_path=file_path,test_ratio=test_ratio)
    return user_set, item_set, train_set, test_set


def initialize_model(user_count, item_count, factors_dim):
    """初始化 ALS_MLP 模型"""
    return ALS_MLP(n_users=user_count, n_items=item_count, dim=factors_dim)


def train_model(model, train_set, batch_size, epochs, lr):
    """训练 ALS_MLP 模型"""
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
    评估 ALS_MLP 模型的性能，计算 precision, recall 和 accuracy。

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

